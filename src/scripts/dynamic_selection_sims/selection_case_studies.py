import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

from stk_utils import load_orbit_data, interpolate_orbit
from plotting_utils import plot_orbit_dynamics, plot_mission, plot_naive_blitz, plot_horizon_sweep

try:
    from libs.coral_tpu_characterization.src.scripts.utils.path_utils import get_repo_root
    ROOT_DIR = get_repo_root()
except ImportError:
    ROOT_DIR = Path(".").resolve()

class ContinuousSatSim:
    BASE_SYSTEM = {
        'pixel_pitch_um': 3.45,
        'sensor_res': 4096,
        'tpu_dim': 224,
        'system_baseload_mw': 200.0,
        'sim_dt_s': 1.0,
        
        'alt_threshold_km': 2000.0,
        'low_alt_target_km': 1.0,
        'low_alt_min_px': 5,
        'high_alt_target_km': 200.0, 
        'high_alt_min_px': 2,        
        
        'initial_charge_pct': 0.85,
        'compute_enable_pct': 0.65,
        'compute_disable_pct': 0.45,
        
        'budget_horizon_frames': 1000.0,
        'eclipse_illumination_pct': 0.05, 
        
        'hard_min_infs': 0.0,
        'hard_min_infj': 0.0,
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
        })
        return cfg

    @staticmethod
    def get_sso_config():
        cfg = ContinuousSatSim.BASE_SYSTEM.copy()
        cfg.update({
            'focal_length_mm': 85.0,
            'battery_capacity_wh': 1.5,
            'solar_generation_mw': 600.0, 
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

    def _select_model(self, viable_models, energy_budget_j, time_budget_s, total_inferences):
        if energy_budget_j <= 0 or time_budget_s <= 0: 
            return None, "depleted"
        
        max_time_infs = time_budget_s / viable_models['lat_s']
        max_eng_infs = energy_budget_j / viable_models['eng_j']
        
        infs_possible = np.minimum(max_time_infs, max_eng_infs)
        clearing_mask = infs_possible >= total_inferences
        
        if clearing_mask.any():
            capable_models = viable_models[clearing_mask].copy()
            best_idx = capable_models['acc_decimal'].idxmax()
            return capable_models.loc[best_idx], "clears_frame"
        else:
            expected_correct = infs_possible * viable_models['acc_decimal']
            best_idx = expected_correct.idxmax()
            best_model = viable_models.loc[best_idx]
            return best_model, "partial_frame"

    def _get_step_conditions(self, t_rel, dt, row, cfg, events, current_battery_j, limit_disable_j):
        # calculate the current environment and budget logic for this timestep
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

        n = cfg['budget_horizon_frames']
        surplus_j = max(0.0, current_battery_j - limit_disable_j)
        battery_alloc_j = surplus_j / n if n > 1e-6 else surplus_j
        
        net_power_w = solar_w - base_w
        step_incoming_j = max(0.0, net_power_w * dt)
        step_energy_budget_j = min(battery_alloc_j + step_incoming_j, surplus_j)

        return env_energy_j, base_w, infs_for_this_second, step_energy_budget_j, cpu_blocked

    def _process_dynamic_step(self, viable_models, infs_for_sec, step_budget_j, dt, cpu_blocked, dynamic_recharging):
        # execute the dynamic model selection and calculate drops
        drops = {'power': 0.0, 'time': 0.0, 'energy': 0.0}

        if infs_for_sec <= 0:
            return "blind", 0.0, 0.0, 0.0, drops

        if cpu_blocked:
            drops['power'] += infs_for_sec
            return "blocked", 0.0, 0.0, 0.0, drops

        if dynamic_recharging or step_budget_j <= 1e-6:
            drops['power'] += infs_for_sec
            return "recharge", 0.0, 0.0, 0.0, drops

        model, reason = self._select_model(viable_models, step_budget_j, dt, infs_for_sec)
        
        if model is None:
            drops['power'] += infs_for_sec
            return "recharge", 0.0, 0.0, 0.0, drops

        active_model = model['Model name']
        current_acc = model['acc_decimal']
        
        max_t_infs = dt / model['lat_s']
        max_e_infs = step_budget_j / model['eng_j']
        
        processed_infs = min(infs_for_sec, max_t_infs, max_e_infs)
        dropped = infs_for_sec - processed_infs
        
        if dropped > 0:
            if max_t_infs < max_e_infs and max_t_infs < infs_for_sec:
                drops['time'] += dropped
            else:
                drops['energy'] += dropped
                
        proc_energy_j = processed_infs * model['eng_j']
        return active_model, current_acc, processed_infs, proc_energy_j, drops

    def _process_naive_step(self, naive_states, env_energy_j, infs_for_sec, dt, cpu_blocked, limit_disable_j, limit_enable_j, bat_cap_j):
        # execute all naive baselines for the current timestep
        for name, ns in naive_states.items():
            if ns['battery_j'] < limit_disable_j: 
                ns['recharging'] = True
            elif ns['recharging'] and ns['battery_j'] > limit_enable_j: 
                ns['recharging'] = False
            
            if cpu_blocked or ns['recharging']:
                ns['battery_j'] = np.clip(ns['battery_j'] + env_energy_j, 0, bat_cap_j)
            else:
                available_energy_j = max(0, ns['battery_j'] - limit_disable_j)
                step_energy_demand_j = dt * ns['power_w']
                
                time_runnable_s = dt
                if available_energy_j < step_energy_demand_j:
                    time_runnable_s = available_energy_j / ns['power_w']
                    ns['recharging'] = True 
                
                ns['battery_j'] -= (time_runnable_s * ns['power_w'])
                potential_infs = time_runnable_s * ns['ips']
                taken = min(potential_infs, infs_for_sec)
                
                ns['battery_j'] = np.clip(ns['battery_j'] + env_energy_j, 0, bat_cap_j)
                ns['total_correct'] += taken * ns['model']['acc_decimal']
            
            ns['logs_battery_wh'].append(ns['battery_j'] / 3600.0)
            ns['logs_cum_correct'].append(ns['total_correct'])

    def run_case_study(self, case_name, config_overrides=None, events=None):
        if self.raw_orbit.empty: return

        print(f"\n>>> running case study: {case_name}")
        cfg = self.BASE_SYSTEM.copy()
        if config_overrides: cfg.update(config_overrides)
        
        viable_models = self.models[
            ((1.0 / self.models['lat_s']) >= cfg['hard_min_infs']) &
            (self.models['correct_infs_per_joule'] >= cfg['hard_min_infj'])
        ].copy()

        if viable_models.empty:
            print("[error] no models meet the hard barriers. check config.")
            return

        sim_data = interpolate_orbit(self.raw_orbit, self.sunlight_intervals, cfg['sim_dt_s'], cfg)
        if sim_data.empty: return

        naive_configs = {
            'High_Accuracy': viable_models.loc[viable_models['acc_decimal'].idxmax()],
            'High_Throughput': viable_models.loc[viable_models['correct_infs_per_sec'].idxmax()],
            'High_Efficiency': viable_models.loc[viable_models['correct_infs_per_joule'].idxmax()]
        }

        battery_cap_j = cfg['battery_capacity_wh'] * 3600.0
        limit_disable_j = battery_cap_j * cfg['compute_disable_pct']
        limit_enable_j = battery_cap_j * cfg['compute_enable_pct']
        
        naive_states = {}
        for name, model in naive_configs.items():
            naive_states[name] = {
                'battery_j': battery_cap_j * cfg['initial_charge_pct'],
                'recharging': False,
                'total_correct': 0.0,
                'model': model,
                'power_w': model['eng_j'] / model['lat_s'],
                'ips': 1.0 / model['lat_s'],
                'logs_battery_wh': [],
                'logs_cum_correct': []
            }

        current_battery_j = battery_cap_j * cfg['initial_charge_pct']
        total_infs_correct = 0

        report_stats = {
            'total_demand_infs': 0.0,
            'dropped_power_infs': 0.0, 
            'dropped_time_infs': 0.0,
            'dropped_energy_infs': 0.0,
            'processed_infs': 0.0
        }
        
        logs = {
            'time_rel': [], 'battery_wh': [], 'throughput_infs': [], 'backlog_infs': [], 'model_name': [],
            'avg_accuracy': [], 'active_power_w': [], 'is_lit': [], 
            'demand_infs': [], 'alt_km': [], 'speed_km_s': [], 'cum_correct': [], 'dwell_time_s': []
        }
        
        t_start = sim_data['Time (EpSec)'].iloc[0]
        dynamic_recharging = False

        for i, row in sim_data.iterrows():
            t_rel = row['Time (EpSec)'] - t_start
            dt = cfg['sim_dt_s']
            
            # get environmental conditions and constraints
            env_energy_j, base_w, infs_for_sec, step_budget_j, cpu_blocked = self._get_step_conditions(
                t_rel, dt, row, cfg, events, current_battery_j, limit_disable_j
            )
            report_stats['total_demand_infs'] += infs_for_sec

            # check power lock hysteresis
            if current_battery_j < limit_disable_j:
                dynamic_recharging = True
            elif dynamic_recharging and current_battery_j > limit_enable_j:
                dynamic_recharging = False

            # process the dynamic model
            active_model, acc, processed, proc_energy_j, drops = self._process_dynamic_step(
                viable_models, infs_for_sec, step_budget_j, dt, cpu_blocked, dynamic_recharging
            )

            # update dynamic states
            report_stats['dropped_power_infs'] += drops['power']
            report_stats['dropped_time_infs'] += drops['time']
            report_stats['dropped_energy_infs'] += drops['energy']
            report_stats['processed_infs'] += processed

            total_load_w = base_w + (proc_energy_j / dt)
            current_battery_j = np.clip(current_battery_j - proc_energy_j + env_energy_j, 0, battery_cap_j)
            total_infs_correct += processed * acc

            # process naive/static deployments
            self._process_naive_step(
                naive_states, env_energy_j, infs_for_sec, dt, cpu_blocked, 
                limit_disable_j, limit_enable_j, battery_cap_j
            )

            # logs
            logs['time_rel'].append(t_rel)
            logs['battery_wh'].append(current_battery_j / 3600.0)
            logs['backlog_infs'].append(0)
            logs['throughput_infs'].append(processed)
            logs['model_name'].append(active_model)
            logs['avg_accuracy'].append(acc if processed > 0 else np.nan)
            logs['active_power_w'].append(total_load_w)
            logs['is_lit'].append(row['is_lit'])
            logs['demand_infs'].append(infs_for_sec)
            logs['alt_km'].append(row['Alt (km)'])
            logs['speed_km_s'].append(row['v_ground_km_s'])
            logs['dwell_time_s'].append(row['dwell_time_s'])
            logs['cum_correct'].append(total_infs_correct)

        self._print_verbose_report(case_name, report_stats, logs, sim_data, cfg, naive_states)
        
        plot_orbit_dynamics(logs, case_name, self.output_dir)
        plot_mission(logs, naive_states, case_name, cfg, self.output_dir,
                    plot_accuracy_baseline=True, 
                    plot_efficiency_baseline=False, 
                    plot_throughput_baseline=False)
        plot_naive_blitz(logs, naive_states, case_name, cfg, self.output_dir)
        
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
        
        total_demand = stats['total_demand_infs']
        print("inference processing statistics:")
        if total_demand > 0:
            print(f"  * total demand:          {total_demand:,.0f} infs")
            print(f"  * processed:             {stats['processed_infs']:,.0f} ({stats['processed_infs']/total_demand*100:5.1f}%)")
            print(f"  * dropped (power lock):  {stats['dropped_power_infs']:,.0f} ({stats['dropped_power_infs']/total_demand*100:5.1f}%)")
            print(f"  * dropped (time limit):  {stats['dropped_time_infs']:,.0f} ({stats['dropped_time_infs']/total_demand*100:5.1f}%)")
            print(f"  * dropped (eng limit):   {stats['dropped_energy_infs']:,.0f} ({stats['dropped_energy_infs']/total_demand*100:5.1f}%)")
        print("-" * 60)

        if naive_states:
            print("dynamic vs naive performance gain:")
            dynamic_total = logs['cum_correct'][-1] if logs['cum_correct'] else 0
            
            for name, ns in naive_states.items():
                clean_name = name.replace('_', ' ').lower()
                naive_total = ns['total_correct']
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

if __name__ == "__main__":
    model_json = ROOT_DIR / "data/compiled_characterization.json" 
    orbit_path = ROOT_DIR / "data/stk"
    out_dir = ROOT_DIR / "results/case_studies"

    sim_sso = ContinuousSatSim(orbit_path, model_json, out_dir, sat_prefix='SSO', num_orbits=5)
    sso_cfg = ContinuousSatSim.get_sso_config()
    
    sim_sso.run_case_study("SSO_01_Baseline", config_overrides=sso_cfg, events=None)

    burst_events = [
        {'start': 500, 'duration': 100, 'extra_demand_ips': 200.0},
        {'start': 3000, 'duration': 100, 'extra_demand_ips': 200.0},
    ]
    sim_sso.run_case_study("SSO_02_Time_Crunch", config_overrides=sso_cfg, events=burst_events)

    # case 3: early power drain
    drain_events = [
        {'start': 1000, 'duration': 1500, 'power_w': 0.5},
    ]
    sim_sso.run_case_study("SSO_03_Power_Starved", config_overrides=sso_cfg, events=drain_events)

    # case 4 remains the same
    strict_cfg = sso_cfg.copy()
    strict_cfg['hard_min_infs'] = 70.0 
    sim_sso.run_case_study("SSO_04_Strict_Limits", config_overrides=strict_cfg, events=None)

    # case 5: storm hits early while it is trying to use fast models
    storm_events = [
        {'start': 2000, 'duration': 1000, 'power_w': 0.4},                     
        {'start': 2500, 'duration': 200, 'extra_demand_ips': 150.0},           
        {'start': 2600, 'duration': 50, 'power_w': 0.0, 'blocked': True},     
    ]
    sim_sso.run_case_study("SSO_05_Perfect_Storm", config_overrides=strict_cfg, events=storm_events)