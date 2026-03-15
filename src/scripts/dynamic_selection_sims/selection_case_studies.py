import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque
import sys
import os

# import extracted modules
from stk_utils import load_orbit_data, interpolate_orbit
from plotting_utils import plot_orbit_dynamics, plot_telemetry, plot_naive_blitz, plot_horizon_sweep

try:
    from libs.coral_tpu_characterization.src.scripts.utils.path_utils import get_repo_root
    ROOT_DIR = get_repo_root()
except ImportError:
    ROOT_DIR = Path(".").resolve()

class FrameJob:
    def __init__(self, job_id, total_inferences, timestamp):
        self.id = job_id
        self.total_inferences = total_inferences
        self.remaining_inferences = total_inferences
        self.timestamp = timestamp
        self.assigned_model = None 

class ContinuousSatSim:
    BASE_SYSTEM = {
        'pixel_pitch_um': 3.45,
        'sensor_res': 4096,
        'tpu_dim': 224,
        'system_baseload_mw': 300.0,
        'sim_dt_s': 1.0,
        
        'alt_threshold_km': 2000.0,
        'low_alt_target_km': 1.0,
        'low_alt_min_px': 5,
        'high_alt_target_km': 200.0, 
        'high_alt_min_px': 2,        
        
        'initial_charge_pct': 0.85,
        'compute_enable_pct': 0.70,
        'compute_disable_pct': 0.45,
        
        'budget_horizon_frames': 1000.0,
        'enforce_clearing': False,
        'eclipse_illumination_pct': 0.05, 
        'buffer_max_frames': 15,          
    }

    @staticmethod
    def get_heo_config():
        cfg = ContinuousSatSim.BASE_SYSTEM.copy()
        cfg.update({
            'focal_length_mm': 300.0,
            'battery_capacity_wh': 2.0,
            'solar_generation_mw': 1500.0, 
            'alt_threshold_km': 5000.0,    
            'low_alt_target_km': 0.5,      
            'buffer_max_frames': 25000     
        })
        return cfg

    @staticmethod
    def get_sso_config():
        cfg = ContinuousSatSim.BASE_SYSTEM.copy()
        cfg.update({
            'focal_length_mm': 85.0,
            'battery_capacity_wh': 1.5,
            'solar_generation_mw': 600.0, 
            'buffer_max_frames': 5000,
        })
        return cfg

    def __init__(self, orbit_data_path, model_json_path, output_dir, sat_prefix='HEO', num_orbits=1, model_source='Custom'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sat_prefix = sat_prefix
        
        print(f"--- loading models (source={model_source}) & orbit data for {sat_prefix} ---")
        self.models = self._load_models(model_json_path, model_source)
        self.raw_orbit, self.sunlight_intervals = load_orbit_data(orbit_data_path, sat_prefix, num_orbits)
        
    def _load_models(self, json_path, source_filter):
        if not Path(json_path).exists():
            raise FileNotFoundError(f"model json not found at: {json_path}")
        df = pd.read_json(json_path)
        if df.empty: raise ValueError("model data json is empty")

        if 'Source' in df.columns and source_filter is not None:
            df = df[df['Source'] == source_filter].copy()

        df['acc_decimal'] = df['Top-1 Accuracy'] / 100.0
        df['lat_s'] = df['Measured Inference Time (ms)'] / 1000.0
        df['eng_j'] = df['Energy per Inference (mJ)'] / 1000.0
        
        df['correct_infs_per_sec'] = (1.0 / df['lat_s']) * df['acc_decimal']
        df['correct_infs_per_joule'] = (1.0 / df['eng_j']) * df['acc_decimal']
        
        return df

    def _select_model(self, energy_budget_j, time_budget_s, total_inferences):
        # bounce out if resources are drained
        if energy_budget_j <= 0 or time_budget_s <= 0: 
            return None, "depleted"
        
        # calc max inferences possible given strict time and energy budgets
        max_time_infs = time_budget_s / self.models['lat_s']
        max_eng_infs = energy_budget_j / self.models['eng_j']
        
        infs_possible = np.minimum(max_time_infs, max_eng_infs)
        clearing_mask = infs_possible >= total_inferences
        
        if clearing_mask.any():
            capable_models = self.models[clearing_mask].copy()
            best_idx = capable_models['acc_decimal'].idxmax()
            return capable_models.loc[best_idx], "clears_frame"
            
        else:
            expected_correct = infs_possible * self.models['acc_decimal']
            best_idx = expected_correct.idxmax()
            best_model = self.models.loc[best_idx]
            
            lim_time = max_time_infs[best_idx]
            lim_eng = max_eng_infs[best_idx]
            reason = "partial_frame_energy" if lim_eng < lim_time else "partial_frame_time"
            
            return best_model, reason

    def run_case_study(self, case_name, config_overrides=None, events=None):
        if self.raw_orbit.empty: return

        print(f"\n>>> running case study: {case_name}")
        cfg = self.BASE_SYSTEM.copy()
        if config_overrides: cfg.update(config_overrides)
        
        sim_data = interpolate_orbit(self.raw_orbit, self.sunlight_intervals, cfg['sim_dt_s'], cfg)
        if sim_data.empty: return

        # calc orbital baseline budget for smoother energy allocation
        total_lit_time = sum(end - start for start, end in self.sunlight_intervals)
        expected_solar_j = (cfg['solar_generation_mw'] / 1000.0) * total_lit_time
        expected_frames_per_orbit = len(sim_data[sim_data['demand_infs_per_sec'] > 0])
        baseline_energy_per_frame_j = expected_solar_j / max(1, expected_frames_per_orbit)

        naive_configs = {
            'High_Accuracy': self.models.loc[self.models['acc_decimal'].idxmax()],
            'High_Throughput': self.models.loc[self.models['correct_infs_per_sec'].idxmax()],
            'High_Efficiency': self.models.loc[self.models['correct_infs_per_joule'].idxmax()]
        }

        BATTERY_CAPACITY_J = cfg['battery_capacity_wh'] * 3600.0
        limit_disable_j = BATTERY_CAPACITY_J * cfg['compute_disable_pct']
        limit_enable_j = BATTERY_CAPACITY_J * cfg['compute_enable_pct']
        
        naive_states = {}
        for name, model in naive_configs.items():
            naive_states[name] = {
                'battery_j': BATTERY_CAPACITY_J * cfg['initial_charge_pct'],
                'recharging': False,
                'buffer': deque(),
                'total_correct': 0.0,
                'model': model,
                'power_w': model['eng_j'] / model['lat_s'],
                'ips': 1.0 / model['lat_s'],
                'logs_battery_wh': [],
                'logs_cum_correct': []
            }

        current_battery_j = BATTERY_CAPACITY_J * cfg['initial_charge_pct']
        frame_buffer = deque() 
        current_job = None 
        total_infs_correct = 0

        report_stats = {'time_limited_count': 0, 'energy_limited_count': 0, 'buffered_count': 0, 'unprocessed_count': 0, 'total_steps': len(sim_data)}
        logs = {
            'time_rel': [], 'battery_wh': [], 'buffer_count': [], 
            'throughput_infs': [], 'backlog_infs': [], 'model_name': [],
            'avg_accuracy': [], 'active_power_w': [], 'is_lit': [], 
            'demand_infs': [], 'alt_km': [], 'speed_km_s': [], 'cum_correct': [],
            'dwell_time_s': []
        }
        
        t_start = sim_data['Time (EpSec)'].iloc[0]
        dynamic_recharging = False

        for i, row in sim_data.iterrows():
            t_rel = row['Time (EpSec)'] - t_start
            dt = cfg['sim_dt_s']
            
            disturb_power_w = 0.0
            extra_demand_ips = 0.0
            cpu_blocked = False
            
            if events:
                for e in events:
                    if e['start'] <= t_rel < (e['start'] + e['duration']):
                        disturb_power_w += e.get('power_w', 0.0)
                        extra_demand_ips += e.get('extra_demand_ips', 0.0)
                        if e.get('blocked', False): cpu_blocked = True

            eclipse_pct = cfg.get('eclipse_illumination_pct', 0.0)
            effective_light = row['is_lit'] + (1.0 - row['is_lit']) * eclipse_pct
            solar_w = (cfg['solar_generation_mw'] / 1000.0) * effective_light
            base_w = (cfg['system_baseload_mw'] / 1000.0) + disturb_power_w
            env_energy_j = (solar_w - base_w) * dt 
            
            infs_for_this_second = (row['demand_infs_per_sec'] + extra_demand_ips) * dt
            buffered_this_step = False
            
            if infs_for_this_second <= 0.0:
                active_model_name = "Blind" 
                report_stats['unprocessed_count'] += 1 
            else:
                active_model_name = "Idle"
                if len(frame_buffer) < cfg['buffer_max_frames']:
                    frame_job = FrameJob(i, infs_for_this_second, t_rel)
                    frame_buffer.append(frame_job)
                    buffered_this_step = True
                    
                    for ns in naive_states.values():
                        if len(ns['buffer']) < cfg['buffer_max_frames']:
                            ns['buffer'].append(FrameJob(i, infs_for_this_second, t_rel))
                else:
                    report_stats['unprocessed_count'] += 1
            
            if buffered_this_step and len(frame_buffer) > 1:
                report_stats['buffered_count'] += 1

            # smoothed energy budget calculation 
            surplus_j = max(0.0, current_battery_j - limit_disable_j)
            step_energy_budget_j = min(baseline_energy_per_frame_j + (surplus_j * 0.05), surplus_j)

            time_available_s = dt
            processed_infs_step = 0
            processing_energy_j = 0
            current_accuracy = 0.0
            step_limitation = None 

            if current_battery_j < limit_disable_j:
                dynamic_recharging = True
            elif dynamic_recharging and current_battery_j > limit_enable_j:
                dynamic_recharging = False

            while time_available_s > 0:
                if cpu_blocked: active_model_name = "BLOCKED"; break
                if dynamic_recharging or step_energy_budget_j <= 1e-6: 
                    active_model_name = "RECHARGE"
                    break
                
                if current_job is None:
                    if len(frame_buffer) > 0:
                        current_job = frame_buffer.popleft()
                        total_workload = current_job.remaining_inferences + sum(j.remaining_inferences for j in frame_buffer)
                        
                        model, reason = self._select_model(step_energy_budget_j, time_available_s, total_workload)
                        
                        if reason == "partial_frame_time": step_limitation = "time"
                        elif reason == "partial_frame_energy": step_limitation = "energy"
                        
                        if model is None: 
                            frame_buffer.appendleft(current_job)
                            current_job = None
                            active_model_name = "RECHARGE"
                            break
                        current_job.assigned_model = model
                    else: break
                
                if current_job:
                    model = current_job.assigned_model
                    active_model_name = model['Model name']
                    current_accuracy = model['acc_decimal']
                    
                    infs_possible = min(current_job.remaining_inferences, time_available_s / model['lat_s'], step_energy_budget_j / model['eng_j'])
                    if infs_possible <= 0: active_model_name = "RECHARGE"; break

                    current_job.remaining_inferences -= infs_possible
                    e_spent = infs_possible * model['eng_j']
                    t_spent = infs_possible * model['lat_s']
                    
                    current_battery_j -= e_spent
                    step_energy_budget_j -= e_spent
                    time_available_s -= t_spent
                    processed_infs_step += infs_possible
                    processing_energy_j += e_spent
                    
                    if current_job.remaining_inferences <= 1e-6: current_job = None

            if step_limitation == "time": report_stats['time_limited_count'] += 1
            elif step_limitation == "energy": report_stats['energy_limited_count'] += 1
            
            total_load_w = base_w + (processing_energy_j / dt)
            current_battery_j = np.clip(current_battery_j + env_energy_j, 0, BATTERY_CAPACITY_J)
            total_infs_correct += processed_infs_step * current_accuracy

            for name, ns in naive_states.items():
                if ns['battery_j'] < limit_disable_j: ns['recharging'] = True
                elif ns['recharging'] and ns['battery_j'] > limit_enable_j: ns['recharging'] = False
                
                if cpu_blocked or ns['recharging']:
                    ns['battery_j'] = np.clip(ns['battery_j'] + env_energy_j, 0, BATTERY_CAPACITY_J)
                else:
                    available_energy_j = max(0, ns['battery_j'] - limit_disable_j)
                    step_energy_demand_j = dt * ns['power_w']
                    
                    time_runnable_s = dt
                    if available_energy_j < step_energy_demand_j:
                        time_runnable_s = available_energy_j / ns['power_w']
                        ns['recharging'] = True 
                    
                    ns['battery_j'] -= (time_runnable_s * ns['power_w'])
                    potential_infs = time_runnable_s * ns['ips']
                    
                    processed_naive_infs = 0
                    while potential_infs > 0 and len(ns['buffer']) > 0:
                        job = ns['buffer'][0]
                        taken = min(potential_infs, job.remaining_inferences)
                        job.remaining_inferences -= taken
                        potential_infs -= taken
                        processed_naive_infs += taken
                        if job.remaining_inferences <= 1e-6: ns['buffer'].popleft()
                    
                    ns['battery_j'] = np.clip(ns['battery_j'] + env_energy_j, 0, BATTERY_CAPACITY_J)
                    ns['total_correct'] += processed_naive_infs * ns['model']['acc_decimal']
                
                ns['logs_battery_wh'].append(ns['battery_j'] / 3600.0)
                ns['logs_cum_correct'].append(ns['total_correct'])
            
            buffer_backlog = sum(j.remaining_inferences for j in frame_buffer)
            if current_job: buffer_backlog += current_job.remaining_inferences

            logs['time_rel'].append(t_rel)
            logs['battery_wh'].append(current_battery_j / 3600.0)
            logs['buffer_count'].append(len(frame_buffer))
            logs['backlog_infs'].append(buffer_backlog)
            logs['throughput_infs'].append(processed_infs_step)
            logs['model_name'].append(active_model_name)
            logs['avg_accuracy'].append(current_accuracy if processed_infs_step > 0 else np.nan)
            logs['active_power_w'].append(total_load_w)
            logs['is_lit'].append(row['is_lit'])
            logs['demand_infs'].append(infs_for_this_second)
            logs['alt_km'].append(row['Alt (km)'])
            logs['speed_km_s'].append(row['v_ground_km_s'])
            logs['dwell_time_s'].append(row['dwell_time_s'])
            logs['cum_correct'].append(total_infs_correct)

        plot_orbit_dynamics(logs, case_name, self.output_dir)
        plot_telemetry(logs, case_name, cfg, self.output_dir)
        plot_naive_blitz(logs, naive_states, case_name, cfg, self.output_dir)
        
        return logs

if __name__ == "__main__":
    model_json = ROOT_DIR / "data/compiled_characterization.json" 
    orbit_path = ROOT_DIR / "data/stk"
    out_dir = ROOT_DIR / "results/case_studies"

    # sim_heo = ContinuousSatSim(orbit_path, model_json, out_dir, sat_prefix='HEO', num_orbits=1)
    # sim_heo.run_case_study("HEO_01_Standard", config_overrides=ContinuousSatSim.get_heo_config())
    
    sim_sso = ContinuousSatSim(orbit_path, model_json, out_dir, sat_prefix='SSO', num_orbits=20)
    sso_cfg = ContinuousSatSim.get_sso_config()
    sim_sso.run_case_study("SSO_01_Standard", config_overrides=sso_cfg)