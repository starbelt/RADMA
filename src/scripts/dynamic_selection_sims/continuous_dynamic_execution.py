import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from enum import Enum
import sys
import json
import math
import copy

# Try to import repo root, else fallback
try:
    from libs.coral_tpu_characterization.src.scripts.utils.path_utils import get_repo_root
    ROOT_DIR = get_repo_root()
except ImportError:
    ROOT_DIR = Path(".").resolve()


class FlightRegime(Enum):
    SUNLIGHT = 1
    ECLIPSE = 2

@dataclass
class OrbitalEvent:
    time_start: float
    time_end: float
    regime: FlightRegime

class FrameJob:
    """
    Represents a single frame (or tile-set) to be processed.
    """
    def __init__(self, job_id, total_inferences, timestamp):
        self.id = job_id
        self.total_inferences = total_inferences
        self.remaining_inferences = total_inferences
        self.timestamp = timestamp
        self.assigned_model = None 

class PowerManager:
    """
    Predictive Energy Budget Controller.
    Calculates the maximum safe energy expenditure per frame to ensure
    the satellite meets specific State of Charge (SoC) waypoints.
    """
    def __init__(self, battery_capacity_j, hysteresis_conf, events):
        self.capacity_j = battery_capacity_j
        self.events = events 
        self.current_event_idx = 0
        
        # Target: Enter eclipse fully charged (95%) to maximize survival time
        self.TARGET_ECLIPSE_ENTRY_SOC = 0.95 
        
        # Target: Exit eclipse slightly ABOVE the disable limit. 
        self.TARGET_ECLIPSE_EXIT_SOC = hysteresis_conf['compute_disable_pct'] + 0.05

    def _get_current_event(self, t_now):
        while (self.current_event_idx < len(self.events) and 
               t_now > self.events[self.current_event_idx].time_end):
            self.current_event_idx += 1
            
        if self.current_event_idx >= len(self.events):
            return None
        return self.events[self.current_event_idx]

    def get_allowed_budget(self, t_now, current_battery_j, solar_input_w, dt):
        event = self._get_current_event(t_now)
        if not event: return solar_input_w * dt 

        time_remaining = event.time_end - t_now
        if time_remaining <= 1e-3: time_remaining = dt

        if event.regime == FlightRegime.SUNLIGHT:
            target_j = self.capacity_j * self.TARGET_ECLIPSE_ENTRY_SOC
        else:
            target_j = self.capacity_j * self.TARGET_ECLIPSE_EXIT_SOC

        energy_diff_j = current_battery_j - target_j
        correction_power_w = energy_diff_j / time_remaining
        total_allowed_w = solar_input_w + correction_power_w

        return max(0.0, total_allowed_w * dt)

class ContinuousSatSim:
    """
    Simulates satellite operation over an orbit, managing power, data buffers, 
    and compute workloads using both a dynamic model selection strategy and 
    a naive baseline for comparison.
    """
    
    BASE_SYSTEM = {
        'pixel_pitch_um': 3.45,
        'sensor_res': 4096,
        'tpu_dim': 224,
        'system_baseload_mw': 300.0,
        'sim_dt_s': 1.0,
        
        # dual regime sensing defaults
        'alt_threshold_km': 2000.0,
        'low_alt_target_km': 1.0,
        'low_alt_min_px': 5,
        'high_alt_target_km': 200.0, 
        'high_alt_min_px': 2,        
        
        # power management
        'initial_charge_pct': 0.85,
        'compute_enable_pct': 0.70,
        'compute_disable_pct': 0.45,
    }

    @staticmethod
    def get_heo_config():
        # HEO: Time-domain focus. Massive target resolution swing, sufficient power.
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
        # SSO: Power-domain focus. Low solar overhead, frequent eclipses.
        cfg = ContinuousSatSim.BASE_SYSTEM.copy()
        cfg.update({
            'focal_length_mm': 85.0,
            'battery_capacity_wh': 1.5,
            'solar_generation_mw': 1300.0, 
            'buffer_max_frames': 5000,
        })
        return cfg

    def __init__(self, orbit_data_path, model_json_path, output_dir, sat_prefix='HEO', num_orbits=1, model_source='Custom'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sat_prefix = sat_prefix
        
        print(f"--- Loading Models (Source={model_source}) & Orbit Data for {sat_prefix} ---")
        self.models = self._load_models(model_json_path, model_source)
        self.raw_orbit, self.sunlight_intervals = self._load_orbit_data(orbit_data_path, num_orbits)
        
    def _load_models(self, json_path, source_filter):
        if not Path(json_path).exists():
            raise FileNotFoundError(f"Model JSON not found at: {json_path}")
        df = pd.read_json(json_path)
        if df.empty: raise ValueError("Model data JSON is empty.")

        if 'Source' in df.columns and source_filter is not None:
            df = df[df['Source'] == source_filter].copy()

        df['acc_decimal'] = df['Top-1 Accuracy'] / 100.0
        df['lat_s'] = df['Measured Inference Time (ms)'] / 1000.0
        df['eng_j'] = df['Energy per Inference (mJ)'] / 1000.0
        
        df['correct_infs_per_sec'] = (1.0 / df['lat_s']) * df['acc_decimal']
        df['correct_infs_per_joule'] = (1.0 / df['eng_j']) * df['acc_decimal']
        
        return df

    def _load_orbit_data(self, data_path, num_orbits):
        root = Path(data_path)
        p_path = root / f'{self.sat_prefix}_Sat_Fixed_Position_Velocity.csv'
        c_path = root / f'{self.sat_prefix}_Sat_Classical_Orbit_Elements.csv'
        l_path = root / f'{self.sat_prefix}_Sat_LLA_Position.csv'
        
        if not p_path.exists() or not c_path.exists() or not l_path.exists():
            print(f"[WARN] Missing STK Files for {self.sat_prefix}. Simulation skipped.")
            return pd.DataFrame(), []

        v = pd.read_csv(p_path)
        k = pd.read_csv(c_path)
        l = pd.read_csv(l_path)
        
        for d in [v, k, l]: 
            d.columns = d.columns.str.strip()
            if "Time (EpSec)" in d.columns:
                d["Time (EpSec)"] = d["Time (EpSec)"].round(4)

        try:
            df = v.merge(k, on="Time (EpSec)").merge(l, on="Time (EpSec)")
        except Exception as e:
            print(f"[ERROR] Merge failed for {self.sat_prefix}: {e}")
            return pd.DataFrame(), []

        if df.empty: return pd.DataFrame(), []
        
        if 'True Anomaly (deg)' in df.columns:
            ta = df['True Anomaly (deg)'].values
            diffs = np.diff(ta) 
            wrap_indices = np.where(diffs < -300)[0]
            if len(wrap_indices) > 0 and len(wrap_indices) >= num_orbits:
                cutoff_idx = wrap_indices[num_orbits - 1]
                df = df.iloc[:cutoff_idx+1].copy()

        light_path = root / f'{self.sat_prefix}_Sat_Lighting_Times.csv'
        sunlight_intervals = []
        if light_path.exists():
            sunlight_intervals = self._parse_lighting_schedule(light_path)
        else:
            t_min, t_max = df['Time (EpSec)'].min(), df['Time (EpSec)'].max()
            sunlight_intervals = [(t_min, t_max)]

        print(f"[INFO] Loaded {len(df)} rows of orbit data for {self.sat_prefix}.")
        return df, sunlight_intervals

    def _parse_lighting_schedule(self, file_path):
        intervals = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        capture = False
        header_found_count = 0
        for line in lines:
            if "Start Time (EpSec)" in line:
                header_found_count += 1
                if header_found_count > 1: break
                capture = True
                continue
            if capture:
                if "Statistics" in line or line.strip() == "": break 
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        start = float(parts[0].replace('"', ''))
                        stop = float(parts[1].replace('"', ''))
                        intervals.append((start, stop))
                    except ValueError: continue
        return intervals

    def _build_event_schedule(self, t_start_abs, t_end_abs, sunlight_intervals):
        events = []
        current_t = t_start_abs
        sorted_intervals = sorted(sunlight_intervals, key=lambda x: x[0])

        for (sun_start, sun_end) in sorted_intervals:
            if sun_start > current_t:
                start_seg = max(current_t, t_start_abs)
                end_seg = min(sun_start, t_end_abs)
                if end_seg > start_seg:
                    events.append(OrbitalEvent(start_seg - t_start_abs, end_seg - t_start_abs, FlightRegime.ECLIPSE))
            
            start_seg = max(sun_start, t_start_abs)
            end_seg = min(sun_end, t_end_abs)
            
            if start_seg >= t_end_abs: break
            
            if end_seg > start_seg:
                events.append(OrbitalEvent(start_seg - t_start_abs, end_seg - t_start_abs, FlightRegime.SUNLIGHT))
            
            current_t = max(current_t, end_seg)
            
        if current_t < t_end_abs:
             events.append(OrbitalEvent(current_t - t_start_abs, t_end_abs - t_start_abs, FlightRegime.ECLIPSE))
             
        return events

    def _select_model(self, energy_budget_j, time_budget_s, workload_infs):
        if energy_budget_j <= 0: return None, "Energy_Depleted"
        
        max_inf_time = time_budget_s / self.models['lat_s']
        max_inf_energy = energy_budget_j / self.models['eng_j']
        capacity_infs = np.minimum(max_inf_time, max_inf_energy)
        actual_processing_infs = np.minimum(capacity_infs, workload_infs)
        expected_correct = actual_processing_infs * self.models['acc_decimal']
        
        best_idx = expected_correct.idxmax()
        best_model = self.models.loc[best_idx]
        
        lim_time = max_inf_time[best_idx]
        lim_eng = max_inf_energy[best_idx]
        if lim_eng < workload_infs and lim_eng < lim_time: reason = "Energy_Limited"
        elif lim_time < workload_infs and lim_time < lim_eng: reason = "Time_Limited"
        elif workload_infs < lim_time and workload_infs < lim_eng: reason = "Workload_Clearing" 
        else: reason = "Balanced"
        return best_model, reason

    def run_case_study(self, case_name, config_overrides=None, events=None):
        if self.raw_orbit.empty: return

        print(f"\n>>> Running Case Study: {case_name}")
        cfg = self.BASE_SYSTEM.copy()
        if config_overrides: cfg.update(config_overrides)
        
        sim_data = self._interpolate_orbit(self.raw_orbit, self.sunlight_intervals, cfg['sim_dt_s'], cfg)
        if sim_data.empty: return

        # Power Manager Setup
        flight_schedule = self._build_event_schedule(
            sim_data['Time (EpSec)'].min(), 
            sim_data['Time (EpSec)'].max(), 
            self.sunlight_intervals
        )
        
        power_manager = PowerManager(cfg['battery_capacity_wh'] * 3600.0, cfg, flight_schedule)

        # Baseline Configs (Three distinct Naive Models to compare)
        naive_configs = {
            'High_Accuracy': self.models.loc[self.models['acc_decimal'].idxmax()],
            'High_Throughput': self.models.loc[self.models['correct_infs_per_sec'].idxmax()],
            'High_Efficiency': self.models.loc[self.models['correct_infs_per_joule'].idxmax()]
        }

        BATTERY_CAPACITY_J = cfg['battery_capacity_wh'] * 3600.0
        limit_disable_j = BATTERY_CAPACITY_J * cfg['compute_disable_pct']
        limit_enable_j = BATTERY_CAPACITY_J * cfg['compute_enable_pct']
        
        # Initialize Naive States
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

        # Dynamic State Initialization
        current_battery_j = BATTERY_CAPACITY_J * cfg['initial_charge_pct']
        frame_buffer = deque() 
        current_job = None 
        total_infs_correct = 0

        # Reporting
        report_stats = {'time_limited_count': 0, 'energy_limited_count': 0, 'buffered_count': 0, 'unprocessed_count': 0, 'total_steps': len(sim_data)}
        logs = {
            'time_rel': [], 'battery_wh': [], 'buffer_count': [], 
            'throughput_infs': [], 'backlog_infs': [], 'model_name': [],
            'avg_accuracy': [], 'active_power_w': [], 'is_lit': [], 
            'demand_infs': [], 'alt_km': [], 'speed_km_s': [], 'cum_correct': [],
            'dwell_time_s': []
        }
        
        t_start = sim_data['Time (EpSec)'].iloc[0]

        for i, row in sim_data.iterrows():
            t_rel = row['Time (EpSec)'] - t_start
            dt = cfg['sim_dt_s']
            
            # --- Environment & Events ---
            disturb_power_w = 0.0
            extra_demand_ips = 0.0
            cpu_blocked = False
            
            if events:
                for e in events:
                    if e['start'] <= t_rel < (e['start'] + e['duration']):
                        disturb_power_w += e.get('power_w', 0.0)
                        extra_demand_ips += e.get('extra_demand_ips', 0.0)
                        if e.get('blocked', False): cpu_blocked = True

            solar_w = (cfg['solar_generation_mw'] / 1000.0) * row['is_lit']
            base_w = (cfg['system_baseload_mw'] / 1000.0) + disturb_power_w
            env_energy_j = (solar_w - base_w) * dt 
            
            # Base physics demand + Artificial event injection
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
                    
                    # Feed the exact same job to all naive baselines
                    for ns in naive_states.values():
                        if len(ns['buffer']) < cfg['buffer_max_frames']:
                            ns['buffer'].append(FrameJob(i, infs_for_this_second, t_rel))
                else:
                    report_stats['unprocessed_count'] += 1
            
            if buffered_this_step and len(frame_buffer) > 1:
                 report_stats['buffered_count'] += 1

            # --- DYNAMIC SIMULATION LOGIC ---
            safe_budget_j = power_manager.get_allowed_budget(t_rel, current_battery_j, solar_w, dt)
            hard_budget_j = current_battery_j - limit_disable_j
            energy_budget_j = min(safe_budget_j, hard_budget_j)

            time_available_s = dt
            processed_infs_step = 0
            processing_energy_j = 0
            current_accuracy = 0.0
            step_limitation = None 

            while time_available_s > 0:
                if cpu_blocked: active_model_name = "BLOCKED"; break
                if energy_budget_j <= 1e-6: active_model_name = "RECHARGE"; break
                
                if current_job is None:
                    if len(frame_buffer) > 0:
                        current_job = frame_buffer.popleft()
                        total_workload = current_job.remaining_inferences + sum(j.remaining_inferences for j in frame_buffer)
                        model, reason = self._select_model(energy_budget_j, time_available_s, total_workload)
                        
                        if reason == "Time_Limited": step_limitation = "Time"
                        elif reason == "Energy_Limited": step_limitation = "Energy"
                        
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
                    
                    infs_possible = min(current_job.remaining_inferences, time_available_s / model['lat_s'], energy_budget_j / model['eng_j'])
                    if infs_possible <= 0: active_model_name = "RECHARGE"; break

                    current_job.remaining_inferences -= infs_possible
                    e_spent = infs_possible * model['eng_j']
                    t_spent = infs_possible * model['lat_s']
                    
                    current_battery_j -= e_spent
                    energy_budget_j -= e_spent
                    time_available_s -= t_spent
                    processed_infs_step += infs_possible
                    processing_energy_j += e_spent
                    
                    if current_job.remaining_inferences <= 1e-6: current_job = None

            if step_limitation == "Time": report_stats['time_limited_count'] += 1
            elif step_limitation == "Energy": report_stats['energy_limited_count'] += 1
            
            total_load_w = base_w + (processing_energy_j / dt)
            current_battery_j = np.clip(current_battery_j + env_energy_j, 0, BATTERY_CAPACITY_J)
            total_infs_correct += processed_infs_step * current_accuracy

            # --- NAIVE BASELINES LOGIC ---
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
            
            # --- LOGGING ---
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

        # near the end of run_case_study
        self._plot_orbit_dynamics(logs, case_name)
        self._plot_telemetry(logs, case_name, cfg)
        self._plot_naive_blitz(logs, naive_states, case_name, cfg)
        
        # update this line to include naive_states
        self._print_verbose_report(case_name, report_stats, logs, sim_data, cfg, naive_states)
        
        return logs


    def _print_verbose_report(self, case_name, stats, logs, sim_data, cfg, naive_states=None):
        print(f"\n{'='*60}")
        print(f"case report: {case_name}")
        print(f"{'='*60}")
        
        best_throughput = self.models.loc[self.models['correct_infs_per_sec'].idxmax()]
        best_efficiency = self.models.loc[self.models['correct_infs_per_joule'].idxmax()]
        best_acc = self.models.loc[self.models['acc_decimal'].idxmax()]
        
        print("model landscape:")
        print(f"  * highest throughput: {best_throughput['Model name']:<20} ({best_throughput['correct_infs_per_sec']:.1f} correct inf/s)")
        print(f"  * best efficiency:    {best_efficiency['Model name']:<20} ({best_efficiency['correct_infs_per_joule']:.1f} correct inf/j)")
        print(f"  * max accuracy:       {best_acc['Model name']:<20} ({best_acc['acc_decimal']*100:.1f}%)")
        print("-" * 60)
        
        tiles_per_frame = sim_data['tiles_per_frame']

        print("figures of merit (min / max):")
        print(f"  * swath (km):             {sim_data['swath_km'].min():.2f} / {sim_data['swath_km'].max():.2f}")
        print(f"  * gsd (m):                {sim_data['gsd_m'].min():.2f} / {sim_data['gsd_m'].max():.2f}")
        print(f"  * ground velocity (km/s): {sim_data['v_ground_km_s'].min():.2f} / {sim_data['v_ground_km_s'].max():.2f}")
        print(f"  * tiles per frame:        {tiles_per_frame.min():.2f} / {tiles_per_frame.max():.2f}")
        print("-" * 60)
        
        total_frames = stats['total_steps']
        print("simulation statistics:")
        if total_frames > 0:
            print(f"  * time limited frames:   {stats['time_limited_count']:5d} ({stats['time_limited_count']/total_frames*100:5.1f}%)")
            print(f"  * energy limited frames: {stats['energy_limited_count']:5d} ({stats['energy_limited_count']/total_frames*100:5.1f}%)")
            print(f"  * frames buffered:       {stats['buffered_count']:5d} ({stats['buffered_count']/total_frames*100:5.1f}%)")
            print(f"  * frames dropped/missed: {stats['unprocessed_count']:5d} ({stats['unprocessed_count']/total_frames*100:5.1f}%)")
        print("-" * 60)

        print("model utilization (% of active processing time):")
        df_log = pd.DataFrame({'model': logs['model_name']})
        
        compute_models = df_log[~df_log['model'].str.lower().isin(['idle', 'recharge', 'blocked', 'blind'])] 
        
        if not compute_models.empty:
            breakdown = compute_models['model'].value_counts(normalize=True) * 100
            for name, pct in breakdown.items():
                print(f"  * {name:<30}: {pct:5.1f}%")
        else:
            print("  (no models executed)")
        print("-" * 60)

        # new block to calculate and print the gains
        if naive_states:
            print("dynamic vs naive performance gain:")
            dynamic_total = logs['cum_correct'][-1] if logs['cum_correct'] else 0
            
            for name, ns in naive_states.items():
                clean_name = name.replace('_', ' ').lower()
                naive_total = ns['total_correct']
                
                # avoid division by zero just in case a naive baseline completely fails
                if naive_total > 0:
                    gain_pct = ((dynamic_total - naive_total) / naive_total) * 100.0
                    print(f"  * vs {clean_name:<20}: +{gain_pct:.1f}%")
                else:
                    print(f"  * vs {clean_name:<20}: +inf% (baseline scored 0)")
                    
        print(f"{'='*60}\n")

        print("model utilization (% of active processing time):")
        df_log = pd.DataFrame({'model': logs['model_name']})
        
        compute_models = df_log[~df_log['model'].str.lower().isin(['idle', 'recharge', 'blocked', 'blind'])] 
        
        if not compute_models.empty:
            breakdown = compute_models['model'].value_counts(normalize=True) * 100
            for name, pct in breakdown.items():
                print(f"  * {name:<30}: {pct:5.1f}%")
        else:
            print("  (no models executed)")
        print(f"{'='*60}\n")

    def _interpolate_orbit(self, df, sunlight_intervals, dt, cfg):
        if df.empty: return pd.DataFrame()
        
        t_start = df['Time (EpSec)'].min()
        t_end = df['Time (EpSec)'].max()
        new_times = np.arange(t_start, t_end, dt)
        
        cols = ['x (km)', 'y (km)', 'z (km)', 'vx (km/sec)', 'vy (km/sec)', 'vz (km/sec)', 'Alt (km)']
        new_data = {'Time (EpSec)': new_times}
        for c in cols:
            new_data[c] = np.interp(new_times, df['Time (EpSec)'], df[c])
        
        is_lit = np.zeros_like(new_times)
        for s, e in sunlight_intervals:
            mask = (new_times >= s) & (new_times <= e)
            is_lit[mask] = 1.0
        new_data['is_lit'] = is_lit
        new_df = pd.DataFrame(new_data)

        new_df['gsd_m'] = (new_df['Alt (km)'] * cfg['pixel_pitch_um']) / cfg['focal_length_mm']
        new_df['swath_km'] = (new_df['gsd_m'] * cfg['sensor_res']) / 1000.0
        
        r = new_df[['x (km)', 'y (km)', 'z (km)']].values
        v = new_df[['vx (km/sec)', 'vy (km/sec)', 'vz (km/sec)']].values
        r_norm = np.linalg.norm(r, axis=1, keepdims=True)
        v_vert = np.sum(v * (r/r_norm), axis=1, keepdims=True) * (r/r_norm)
        v_ground = np.linalg.norm(v - v_vert, axis=1)
        
        new_df['v_ground_km_s'] = v_ground
        new_df['dwell_time_s'] = new_df['swath_km'] / (v_ground + 1e-9)

        is_low_alt = new_df['Alt (km)'] <= cfg['alt_threshold_km']
        target_size_km = np.where(is_low_alt, cfg['low_alt_target_km'], cfg['high_alt_target_km'])
        min_px = np.where(is_low_alt, cfg['low_alt_min_px'], cfg['high_alt_min_px'])
        
        target_size_m = target_size_km * 1000.0
        new_df['px_per_object'] = target_size_m / new_df['gsd_m']
        
        max_tiles_per_frame = (cfg['sensor_res'] / cfg['tpu_dim'])**2
        
        new_df['tiles_per_frame'] = np.where(new_df['px_per_object'] < min_px, 0.0, max_tiles_per_frame)
        new_df['demand_infs_per_sec'] = new_df['tiles_per_frame'] / new_df['dwell_time_s']
        new_df['current_target_km'] = target_size_km
        
        return new_df

    def _set_plot_style(self):
        # boost all font sizes and line widths for presentation slides
        plt.rcParams.update({
            'font.size': 16,
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'lines.linewidth': 3.0,
            'figure.titlesize': 24,
            'figure.titleweight': 'bold'
        })

    def _plot_orbit_dynamics(self, logs, case_name):
        self._set_plot_style()
        clean_name = case_name.replace('_', ' ')
        t_plot = np.array(logs['time_rel'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f"case study: {clean_name}\norbit & data dynamics", y=0.98)

        # panel 1: altitude and speed
        ax1.plot(t_plot, logs['alt_km'], color='gray', label='altitude')
        ax1.set_ylabel('altitude (km)')
        
        ax1_t = ax1.twinx()
        ax1_t.plot(t_plot, logs['speed_km_s'], 'r--', label='ground speed')
        ax1_t.set_ylabel('speed (km/s)', color='r')
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax1_t.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        ax1.grid(True, alpha=0.3)

        # panel 2: dwell time and demand
        demand = np.array(logs['demand_infs'])
        dwell = np.array(logs['dwell_time_s'])
        
        ax2.plot(t_plot, dwell, color='tab:blue', label='frame dwell time (s)')
        ax2.set_ylabel('available time per frame (s)', color='tab:blue')
        ax2.set_xlabel('mission time (s)')
        
        ax2_t = ax2.twinx()
        ax2_t.fill_between(t_plot, demand, color='tab:orange', alpha=0.3, label='inference demand')
        ax2_t.plot(t_plot, demand, color='tab:orange', linewidth=2)
        ax2_t.set_ylabel('workload demand (infs/sec)', color='tab:orange')
        
        lines_3, labels_3 = ax2.get_legend_handles_labels()
        lines_4, labels_4 = ax2_t.get_legend_handles_labels()
        ax2.legend(lines_3 + lines_4, labels_3 + labels_4, loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = f"{case_name}_orbit_dynamics.png"
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300)
        print(f"saved orbit dynamics plot to: {save_path}")
        plt.close()

    def _plot_telemetry(self, logs, case_name, cfg):
        self._set_plot_style()
        clean_name = case_name.replace('_', ' ')
        t_plot = np.array(logs['time_rel'])
        
        fig = plt.figure(figsize=(14, 18))
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1.5, 1.5, 1], hspace=0.3)
        fig.suptitle(f"case study: {clean_name}\ndynamic telemetry", y=0.98)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax4 = fig.add_subplot(gs[3])

        # 1. orbit context
        ax1.plot(t_plot, logs['alt_km'], color='gray', label='altitude')
        ax1.set_ylabel('altitude (km)')
        
        ax1_t = ax1.twinx()
        ax1_t.plot(t_plot, logs['speed_km_s'], 'r--', label='speed')
        ax1_t.set_ylabel('speed (km/s)', color='r')
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax1_t.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.get_xticklabels(), visible=False)

        # 2. energy states
        ax2.plot(t_plot, logs['battery_wh'], 'g', label='dynamic battery')
        ax2.axhline(cfg['battery_capacity_wh']*cfg['compute_disable_pct'], color='r', linestyle=':', label='hard min (shutoff)')
        ax2.axhline(cfg['battery_capacity_wh']*cfg['compute_enable_pct'], color='g', linestyle=':', label='ref resume')
        
        lit = np.array(logs['is_lit'])
        ax2.fill_between(t_plot, 0, 1, where=(lit > 0.5), transform=ax2.get_xaxis_transform(), 
                         color='gold', alpha=0.2, label='sunlight')
        ax2.set_ylabel('battery (wh)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.get_xticklabels(), visible=False)

        # 3. performance and backlog
        ax3.plot(t_plot, logs['cum_correct'], 'k', label='dynamic correct infs')
        ax3.set_ylabel('total correct inferences')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        ax3_t = ax3.twinx()
        ax3_t.fill_between(t_plot, logs['backlog_infs'], color='tab:orange', alpha=0.15, label='backlog')
        ax3_t.plot(t_plot, logs['backlog_infs'], color='tab:orange', linewidth=1.5, alpha=0.6)
        ax3_t.set_ylabel('dynamic backlog size', color='tab:orange')
        ax3_t.tick_params(axis='y', labelcolor='tab:orange')
        ax3.set_xlabel('time (s)')

        # 4. total utilization
        df_log = pd.DataFrame({'model': logs['model_name'], 'infs': logs['throughput_infs']})
        stats = df_log[df_log['infs'] > 0].groupby('model')['infs'].sum().sort_values()
        
        if not stats.empty:
            bars = ax4.barh(stats.index, stats.values, color='tab:purple')
            ax4.bar_label(bars, fmt='{:,.0f}', padding=5, fontsize=12)
        else:
            ax4.text(0.5, 0.5, "no inferences performed", ha='center', va='center', fontsize=16)
        
        ax4.set_xlabel('total inferences processed')
        ax4.grid(True, axis='x', alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename = f"{case_name}.png"
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300)
        print(f"saved primary plot to: {save_path}")
        plt.close()

    def _plot_naive_blitz(self, logs, naive_states, case_name, cfg):
        self._set_plot_style()
        clean_name = case_name.replace('_', ' ')
        t_plot = np.array(logs['time_rel'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True, gridspec_kw={'hspace': 0.2})
        fig.suptitle(f"case study: {clean_name}\nnaive blitz comparison", y=0.96)

        colors = {'High_Accuracy': 'tab:blue', 'High_Throughput': 'tab:red', 'High_Efficiency': 'tab:purple'}
        
        # top panel: battery comparison
        ax1.plot(t_plot, logs['battery_wh'], color='black', linewidth=3.5, label='dynamic (predictive)')
        for name, ns in naive_states.items():
            clean_naive = name.replace('_', ' ')
            ax1.plot(t_plot, ns['logs_battery_wh'], color=colors[name], linestyle='--', alpha=0.8, label=f'naive ({clean_naive})')

        ax1.axhline(cfg['battery_capacity_wh']*cfg['compute_disable_pct'], color='red', linestyle=':', alpha=0.5, label='shutoff limit')
        
        lit = np.array(logs['is_lit'])
        ax1.fill_between(t_plot, 0, 1, where=(lit > 0.5), transform=ax1.get_xaxis_transform(), 
                         color='gold', alpha=0.2, label='sunlight')
        
        ax1.set_ylabel('battery (wh)')
        ax1.set_title('battery management strategies')
        ax1.legend(loc='lower right', framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        # bottom panel: inference yield
        ax2.plot(t_plot, logs['cum_correct'], color='black', linewidth=3.5, label='dynamic system')
        for name, ns in naive_states.items():
            clean_naive = name.replace('_', ' ')
            ax2.plot(t_plot, ns['logs_cum_correct'], color=colors[name], linestyle='--', alpha=0.8, label=f'naive ({clean_naive})')

        ax2.set_xlabel('mission time (s)')
        ax2.set_ylabel('cumulative correct inferences')
        ax2.set_title('inference yield')
        ax2.legend(loc='upper left', framealpha=0.9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        filename = f"{case_name}_naive_blitz.png"
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300)
        print(f"saved blitz plot to: {save_path}")
        plt.close()

def run_all_case_studies():
    model_json = ROOT_DIR / "data/compiled_characterization.json" 
    orbit_path = ROOT_DIR / "data/stk"
    out_dir = ROOT_DIR / "results/case_studies"

    sim_heo = ContinuousSatSim(orbit_path, model_json, out_dir, sat_prefix='HEO', num_orbits=1)
    
    # ==========================
    # HEO Cases (Time Domain)
    # ==========================
    # Baseline
    sim_heo.run_case_study("HEO_01_Standard", config_overrides=ContinuousSatSim.get_heo_config())
    
    # 02. Data Deluge: Massive backlog forcing rapid time-limited processing at perigee
    heo_deluge = ContinuousSatSim.get_heo_config()
    heo_deluge['low_alt_target_km'] = 0.25 # Double the target resolution requirement
    sim_heo.run_case_study("HEO_02_Data_Deluge", config_overrides=heo_deluge)
    
    # 03. Power Starved (Injection): Turn on massive power drain exactly when data ingestion spikes
    # HEO perigee roughly corresponds to the start (t=0 to 1000s) based on orbit geometry. 
    radar_events = [{'start': 0, 'duration': 1500, 'power_w': 3.5, 'blocked': False}]
    sim_heo.run_case_study("HEO_03_Power_Starved", config_overrides=ContinuousSatSim.get_heo_config(), events=radar_events)

    sim_sso = ContinuousSatSim(orbit_path, model_json, out_dir, sat_prefix='SSO', num_orbits=20)
    
    # ==========================
    # SSO Cases (Power Domain)
    # ==========================
    # Baseline
    sso_cfg = ContinuousSatSim.get_sso_config()
    sim_sso.run_case_study("SSO_01_Standard", config_overrides=sso_cfg)
    
    # 02. Eclipse Crisis: Degraded solar arrays leave thin margins for eclipse survival
    sso_crisis = sso_cfg.copy()
    sso_crisis['solar_generation_mw'] = 1000.0  
    sim_sso.run_case_study("SSO_02_Eclipse_Crisis", config_overrides=sso_crisis)
    
    # 03. Target Rich (Injection): Passing over intelligence hotspots generating huge time-limited burst demands
    burst_events = [
        {'start': 5000, 'duration': 300, 'extra_demand_ips': 150.0},
        {'start': 25000, 'duration': 400, 'extra_demand_ips': 200.0},
        {'start': 60000, 'duration': 300, 'extra_demand_ips': 250.0},
    ]
    sim_sso.run_case_study("SSO_03_Target_Rich", config_overrides=sso_cfg, events=burst_events)

if __name__ == "__main__":
    run_all_case_studies()