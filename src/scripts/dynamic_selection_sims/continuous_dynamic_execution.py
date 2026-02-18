import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
import sys
import json
import math

# Try to import repo root, else fallback
try:
    from libs.coral_tpu_characterization.src.scripts.utils.path_utils import get_repo_root
    ROOT_DIR = get_repo_root()
except ImportError:
    ROOT_DIR = Path(".").resolve()

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

class ContinuousSatSim:
    DEFAULT_SYSTEM = {
        'focal_length_mm': 500.0,
        'pixel_pitch_um': 3.45,
        'sensor_res': 4096,
        'target_tile_km': 20.0,
        'tpu_dim': 224,
        'min_pixels': 1,
        
        'battery_capacity_wh': 1.1,
        'solar_generation_mw': 200.0,
        'system_baseload_mw': 80.0,
        'buffer_max_frames': 50, 
        'sim_dt_s': 1.0,
        'initial_charge_pct': 0.9,
        'compute_enable_pct': 0.65,
        'compute_disable_pct': 0.20,
    }

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
        
        return df

    def _load_orbit_data(self, data_path, num_orbits):
        root = Path(data_path)
        p_path = root / f'{self.sat_prefix}_Sat_Fixed_Position_Velocity.csv'
        c_path = root / f'{self.sat_prefix}_Sat_Classical_Orbit_Elements.csv'
        l_path = root / f'{self.sat_prefix}_Sat_LLA_Position.csv'
        
        # 1. Check File Existence
        if not p_path.exists() or not c_path.exists() or not l_path.exists():
            print(f"[WARN] Missing STK Files for {self.sat_prefix}. Simulation will be skipped.")
            return pd.DataFrame(), []

        v = pd.read_csv(p_path)
        k = pd.read_csv(c_path)
        l = pd.read_csv(l_path)
        
        # 2. Fix Column Names and Round Timestamps (Fixes Merge Issues)
        for d in [v, k, l]: 
            d.columns = d.columns.str.strip()
            # Round time to 4 decimal places to prevent float mismatch during merge
            if "Time (EpSec)" in d.columns:
                d["Time (EpSec)"] = d["Time (EpSec)"].round(4)

        # 3. Robust Merge
        try:
            df = v.merge(k, on="Time (EpSec)").merge(l, on="Time (EpSec)")
        except Exception as e:
            print(f"[ERROR] Merge failed for {self.sat_prefix}: {e}")
            return pd.DataFrame(), []

        if df.empty:
            print(f"[WARN] Orbit data for {self.sat_prefix} resulted in empty DataFrame (Check timestamp alignment).")
            return pd.DataFrame(), []
        
        # Slice Orbits logic
        if 'True Anomaly (deg)' in df.columns:
            ta = df['True Anomaly (deg)'].values
            diffs = np.diff(ta) 
            wrap_indices = np.where(diffs < -300)[0]
            if len(wrap_indices) > 0 and len(wrap_indices) >= num_orbits:
                cutoff_idx = wrap_indices[num_orbits - 1]
                df = df.iloc[:cutoff_idx+1].copy()

        # Lighting
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

    def _select_model(self, energy_budget_j, time_budget_s, workload_infs):
        if energy_budget_j <= 0: return None, "Energy_Depleted"
        
        max_inf_time = time_budget_s / self.models['lat_s']
        max_inf_energy = energy_budget_j / self.models['eng_j']
        capacity_infs = np.minimum(max_inf_time, max_inf_energy)
        actual_processing_infs = np.minimum(capacity_infs, workload_infs)
        expected_correct = actual_processing_infs * self.models['acc_decimal']
        
        best_idx = expected_correct.idxmax()
        best_model = self.models.loc[best_idx]
        
        # Reason
        lim_time = max_inf_time[best_idx]
        lim_eng = max_inf_energy[best_idx]
        if lim_eng < workload_infs and lim_eng < lim_time: reason = "Energy_Limited"
        elif lim_time < workload_infs and lim_time < lim_eng: reason = "Time_Limited"
        elif workload_infs < lim_time and workload_infs < lim_eng: reason = "Workload_Clearing" 
        else: reason = "Balanced"
        return best_model, reason

    def run_case_study(self, case_name, config_overrides=None, events=None):
        # --- FIX: SAFETY CHECK FOR EMPTY DATA ---
        if self.raw_orbit.empty:
            print(f"[SKIP] Skipping {case_name} (Orbit data is empty or failed to load)")
            return
        # ----------------------------------------

        print(f"\n>>> Running Case Study: {case_name}")
        cfg = self.DEFAULT_SYSTEM.copy()
        if config_overrides: cfg.update(config_overrides)
        
        sim_data = self._interpolate_orbit(self.raw_orbit, self.sunlight_intervals, cfg['sim_dt_s'], cfg)
        
        if sim_data.empty:
            print(f"[ERROR] Interpolation resulted in empty data for {case_name}. Skipping.")
            return

        # State
        BATTERY_CAPACITY_J = cfg['battery_capacity_wh'] * 3600.0
        current_battery_j = BATTERY_CAPACITY_J * cfg['initial_charge_pct']
        limit_enable_j = BATTERY_CAPACITY_J * cfg['compute_enable_pct']
        limit_disable_j = BATTERY_CAPACITY_J * cfg['compute_disable_pct']
        
        is_recharging = False 
        frame_buffer = deque() 
        current_job = None     
        
        logs = {
            'time_rel': [], 'battery_wh': [], 'buffer_count': [], 
            'throughput_infs': [], 'backlog_infs': [], 'model_name': [],
            'avg_accuracy': [], 'active_power_w': [], 'is_lit': [], 
            'demand_infs': [], 'alt_km': [], 'speed_km_s': []
        }
        
        t_start = sim_data['Time (EpSec)'].iloc[0]
        total_infs_processed = 0
        total_infs_correct = 0

        for i, row in sim_data.iterrows():
            t_rel = row['Time (EpSec)'] - t_start
            dt = cfg['sim_dt_s']
            
            # 1. External Events (ISR / Disturbances)
            disturb_power_w = 0.0
            cpu_blocked = False
            if events:
                for e in events:
                    if e['start'] <= t_rel < (e['start'] + e['duration']):
                        disturb_power_w += e.get('power_w', 0)
                        if e.get('blocked', False): cpu_blocked = True

            # 2. Power Gen & Baseload
            solar_w = (cfg['solar_generation_mw'] / 1000.0) * row['is_lit']
            base_w = (cfg['system_baseload_mw'] / 1000.0) + disturb_power_w
            
            # 3. Frame Ingestion
            infs_for_this_second = row['demand_infs_per_sec'] * dt
            
            if row['px_per_object'] < cfg['min_pixels']:
                # Blind (Too high altitude)
                active_model_name = "Blind"
            else:
                active_model_name = "Idle"
                if infs_for_this_second > 1.0: 
                    if len(frame_buffer) < cfg['buffer_max_frames']:
                        new_job = FrameJob(i, infs_for_this_second, t_rel)
                        frame_buffer.append(new_job)

            # 4. Hysteresis Check
            if current_battery_j < limit_disable_j: is_recharging = True
            elif is_recharging and current_battery_j > limit_enable_j: is_recharging = False

            # 5. Processing Loop
            time_available_s = dt
            processed_infs_step = 0
            processing_energy_j = 0
            current_accuracy = 0.0

            while time_available_s > 0:
                if cpu_blocked:
                    active_model_name = "BLOCKED (ISR)"
                    break
                
                if is_recharging:
                    active_model_name = "RECHARGE"
                    break 
                
                if current_battery_j <= limit_disable_j:
                    active_model_name = "RECHARGE"
                    is_recharging = True
                    break
                
                if current_job is None:
                    if len(frame_buffer) > 0:
                        current_job = frame_buffer.popleft()
                        energy_budget_j = current_battery_j - limit_disable_j 
                        model, _ = self._select_model(energy_budget_j, time_available_s, current_job.total_inferences)
                        
                        if model is None: 
                            frame_buffer.appendleft(current_job) # Return to queue
                            current_job = None
                            active_model_name = "RECHARGE"
                            is_recharging = True
                            break

                        current_job.assigned_model = model
                    else:
                        # Idle (name set at top)
                        break 
                
                # Execute Job
                if current_job:
                    model = current_job.assigned_model
                    active_model_name = model['Model name']
                    current_accuracy = model['acc_decimal']
                    
                    latency_s = model['lat_s']
                    joules_per_inf = model['eng_j']
                    
                    available_energy_j = max(0, current_battery_j - limit_disable_j)
                    
                    infs_time = time_available_s / latency_s
                    infs_eng = available_energy_j / joules_per_inf
                    
                    infs_possible = min(current_job.remaining_inferences, infs_time, infs_eng)
                    
                    if infs_possible <= 0:
                        is_recharging = True; active_model_name = "RECHARGE"; break

                    # Update Counters
                    current_job.remaining_inferences -= infs_possible
                    energy_spent = infs_possible * joules_per_inf
                    time_spent = infs_possible * latency_s
                    
                    current_battery_j -= energy_spent
                    time_available_s -= time_spent
                    processed_infs_step += infs_possible
                    processing_energy_j += energy_spent
                    
                    if current_job.remaining_inferences <= 1e-6:
                        current_job = None

            # 6. Finalize Step
            avg_processing_power_w = processing_energy_j / dt
            total_load_w = base_w + avg_processing_power_w
            
            env_energy_j = (solar_w - base_w) * dt # Note: Base load subtracted from battery here
            current_battery_j = np.clip(current_battery_j + env_energy_j, 0, BATTERY_CAPACITY_J)
            total_infs_processed += processed_infs_step
            total_infs_correct+= processed_infs_step*current_accuracy
            
            # Calculate Total Backlog
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

        self._plot_telemetry(logs, case_name, cfg)
        print(f"[{case_name}] Complete. Total Correct Inferences: {total_infs_correct:,.0f}")
        return logs

    def _interpolate_orbit(self, df, sunlight_intervals, dt, cfg):
        if df.empty: return pd.DataFrame()
        
        t_start = df['Time (EpSec)'].min()
        t_end = df['Time (EpSec)'].max()
        new_times = np.arange(t_start, t_end, dt)
        
        # Geometry
        cols = ['x (km)', 'y (km)', 'z (km)', 'vx (km/sec)', 'vy (km/sec)', 'vz (km/sec)', 'Alt (km)']
        new_data = {'Time (EpSec)': new_times}
        for c in cols:
            new_data[c] = np.interp(new_times, df['Time (EpSec)'], df[c])
        
        # Lighting
        is_lit = np.zeros_like(new_times)
        for s, e in sunlight_intervals:
            mask = (new_times >= s) & (new_times <= e)
            is_lit[mask] = 1.0
        new_data['is_lit'] = is_lit
        new_df = pd.DataFrame(new_data)

        # Workload Physics
        new_df['gsd_m'] = (new_df['Alt (km)'] * cfg['pixel_pitch_um']) / cfg['focal_length_mm']
        new_df['swath_km'] = (new_df['gsd_m'] * cfg['sensor_res']) / 1000.0
        
        r = new_df[['x (km)', 'y (km)', 'z (km)']].values
        v = new_df[['vx (km/sec)', 'vy (km/sec)', 'vz (km/sec)']].values
        r_norm = np.linalg.norm(r, axis=1, keepdims=True)
        v_vert = np.sum(v * (r/r_norm), axis=1, keepdims=True) * (r/r_norm)
        v_ground = np.linalg.norm(v - v_vert, axis=1)
        new_df['v_ground_km_s'] = v_ground
        
        new_df['dwell_time_s'] = new_df['swath_km'] / (v_ground + 1e-9)

        target_size_m = cfg['target_tile_km'] * 1000.0
        new_df['px_per_object'] = target_size_m / new_df['gsd_m']
        
        tpu_dim = cfg['tpu_dim']
        infs_per_object = np.where(new_df['px_per_object'] < cfg['min_pixels'], 0.0, (new_df['px_per_object'] / tpu_dim)**2)
        
        objects_in_fov = (new_df['swath_km'] / cfg['target_tile_km'])**2
        total_infs_per_frame = objects_in_fov * infs_per_object
        new_df['demand_infs_per_sec'] = total_infs_per_frame / new_df['dwell_time_s']
        
        return new_df

    def _plot_telemetry(self, logs, case_name, cfg):
        t_plot = np.array(logs['time_rel'])
        
        # Setup Figure (Disable Global sharex)
        fig = plt.figure(figsize=(12, 16))
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1.5, 1.5, 1])
        fig.suptitle(f"Case Study: {case_name}", fontsize=16)

        # Create Axes
        # ax1 is the master X-axis for time plots
        ax1 = fig.add_subplot(gs[0])
        # ax2 and ax3 share X with ax1
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        # ax4 is independent (Bar chart)
        ax4 = fig.add_subplot(gs[3])

        # Orbit
        ax1.plot(t_plot, logs['alt_km'], color='gray', label='Altitude')
        ax1.set_ylabel('Altitude (km)')
        ax1_t = ax1.twinx()
        ax1_t.plot(t_plot, logs['speed_km_s'], 'r--', label='Speed')
        ax1_t.set_ylabel('Speed (km/s)', color='r')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.get_xticklabels(), visible=False) # Hide x-labels

        # Energy
        ax2.plot(t_plot, logs['battery_wh'], 'g', label='Battery')
        ax2.axhline(cfg['battery_capacity_wh']*cfg['compute_disable_pct'], color='r', linestyle=':', label='Crit')
        ax2.axhline(cfg['battery_capacity_wh']*cfg['compute_enable_pct'], color='g', linestyle=':', label='Resume')
        
        # Sunlight Shade
        lit = np.array(logs['is_lit'])
        ax2.fill_between(t_plot, 0, 1, where=(lit > 0.5), transform=ax2.get_xaxis_transform(), 
                         color='gold', alpha=0.2, label='Sunlight')
        ax2.set_ylabel('Battery (Wh)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.get_xticklabels(), visible=False) # Hide x-labels

        # Buffer & Demand
        backlog = np.array(logs['backlog_infs'])
        ax3.fill_between(t_plot, backlog, color='tab:orange', alpha=0.3, label='Backlog')
        ax3.plot(t_plot, backlog, color='tab:orange', label='Backlog Count')
        
        ax3_t = ax3.twinx()
        # Throughput (Black) vs Demand (Blue Dotted)
        ax3_t.plot(t_plot, logs['demand_infs'], 'b:', alpha=0.6, label='Demand (In)')
        ax3_t.plot(t_plot, logs['throughput_infs'], 'k', linewidth=1, alpha=0.8, label='Throughput (Out)')
        ax3_t.set_ylabel('Inferences / Sec')
        
        ax3.set_ylabel('Backlog Size')
        ax3.set_xlabel('Time (s)') # Only bottom plot gets x-label
        ax3.legend(loc='upper left')
        ax3_t.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        # Model Usage (Bar Chart - Independent Axis)
        df_log = pd.DataFrame({'model': logs['model_name'], 'infs': logs['throughput_infs']})
        stats = df_log[df_log['infs'] > 0].groupby('model')['infs'].sum().sort_values()
        
        if not stats.empty:
            bars = ax4.barh(stats.index, stats.values, color='tab:purple')
            ax4.bar_label(bars, fmt='{:,.0f}', padding=3)
        else:
            ax4.text(0.5, 0.5, "No Inferences Performed", ha='center', va='center')
        
        ax4.set_xlabel('Total Inferences Processed')
        ax4.set_title('Total Model Utilization')
        ax4.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        filename = f"{case_name}.png"
        save_path = self.output_dir / filename
        plt.savefig(save_path)
        print(f"Saved plot to: {save_path}")
        plt.close()
        
def run_all_case_studies():
    # model_json = ROOT_DIR / "libs/coral_tpu_characterization/data/compiled_characterization.json" 
    # orbit_path = ROOT_DIR / "libs/coral_tpu_characterization/data/stk"
    model_json = ROOT_DIR / "data/compiled_characterization.json" 
    orbit_path = ROOT_DIR / "data/stk"
    out_dir = ROOT_DIR / "results/case_studies"

    # # --- HEO Setup ---
    # sim_heo = ContinuousSatSim(orbit_path, model_json, out_dir, sat_prefix='HEO', num_orbits=1)
    
    # # HEO Standard
    # sim_heo.run_case_study("HEO_01_Standard")
    
    # # HEO with Start low battery 
    # sim_heo.run_case_study("HEO_02_PowerCrisis", config_overrides={'initial_charge_pct': 0.25})

    # # HEO with Interruption at Perigee
    # # Assuming Perigee happens around t=1000s to 3000s in this specific orbit
    # isr_events = [{
    #     'start': 20000, 'duration': 600,  # 10 minute interruption
    #     'power_w': 120.0/1000.0,          # High power draw
    #     'blocked': True                  # TPU unavailable (taking pics)
    # }]
    # sim_heo.run_case_study("HEO_03_ISR_Interruption", events=isr_events)


    sim_sso = ContinuousSatSim(orbit_path, model_json, out_dir, sat_prefix='SSO', num_orbits=20)
    
    # Case 4: SSO Standard (High Duty Cycle)
    sim_sso.run_case_study("SSO_01_Standard", config_overrides={'compute_enable_pct': 0.65,'compute_disable_pct': 0.45,}) 

if __name__ == "__main__":
    run_all_case_studies()