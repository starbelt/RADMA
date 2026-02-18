import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
import sys
import json
import math


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
        'focal_length_mm': 50.0,
        'pixel_pitch_um': 3.45,
        'sensor_res': 4096,
        'target_tile_km': 10.0,
        'tpu_dim': 224,
        'min_pixels': 10,
        'battery_capacity_wh': 1.1,
        'solar_generation_mw': 200.0,
        'system_baseload_mw': 80.0,
        'buffer_max_frames': 50, 
        'sim_dt_s': 1.0,
        'initial_charge_pct': 0.9,
        'compute_enable_pct': 0.65,  # Resume
        'compute_disable_pct': 0.20, # Stop
    }

    def __init__(self, orbit_data_path, model_json_path, output_dir, sat_prefix='HEO', num_orbits=1, model_source='Custom'):
        """
        sat_prefix: 'HEO', 'SSO', etc. Matches the start of the STK csv files.
        num_orbits: Number of full orbits to simulate (slices data based on True Anomaly).
        model_source: Filter models by 'Source' column (default: 'Custom').
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sat_prefix = sat_prefix
        
        print(f"--- Loading Models (Source={model_source}) & Orbit Data for {sat_prefix} ---")
        self.models = self._load_models(model_json_path, model_source)
        
        # Load Orbit Geometry & Lighting Intervals
        self.raw_orbit, self.sunlight_intervals = self._load_orbit_data(orbit_data_path, num_orbits)
        
        valid_models = self.models.dropna(subset=['Correct_Inf_per_Sec', 'Correct_Inf_per_Joule'])
        if not valid_models.empty:
            self.model_acc_ref = valid_models.loc[valid_models['acc_decimal'].idxmax()]['Model name']
            print(f"Reference Max Accuracy Model: {self.model_acc_ref}")
        else:
            print("[WARN] No valid models found after filtering!")

    def _load_models(self, json_path, source_filter):
        if not Path(json_path).exists():
            raise FileNotFoundError(f"Model JSON not found at: {json_path}")
        df = pd.read_json(json_path)
        if df.empty: raise ValueError("Model data JSON is empty.")

        # Filter by Source if column exists
        if 'Source' in df.columns and source_filter is not None:
            original_count = len(df)
            df = df[df['Source'] == source_filter].copy()
            print(f"Filtered models from {original_count} to {len(df)} using Source='{source_filter}'")

        # Standardize units
        df['acc_decimal'] = df['Top-1 Accuracy'] / 100.0
        df['lat_s'] = df['Measured Inference Time (ms)'] / 1000.0
        df['eng_j'] = df['Energy per Inference (mJ)'] / 1000.0
        
        # Metrics
        df["Inf_per_Sec"] = 1.0 / df['lat_s']
        df["Inf_per_Joule"] = 1.0 / df['eng_j']
        df["Correct_Inf_per_Sec"] = df["Inf_per_Sec"] * df['acc_decimal']
        df["Correct_Inf_per_Joule"] = df["Inf_per_Joule"] * df['acc_decimal']
        return df

    def _load_orbit_data(self, data_path, num_orbits):
        root = Path(data_path)
        
        # Load CSVs using EpSec
        p_path = root / f'{self.sat_prefix}_Sat_Fixed_Position_Velocity.csv'
        c_path = root / f'{self.sat_prefix}_Sat_Classical_Orbit_Elements.csv'
        l_path = root / f'{self.sat_prefix}_Sat_LLA_Position.csv'
        
        if not p_path.exists(): raise FileNotFoundError(f"Missing STK File: {p_path}")

        # merge on "Time (EpSec)"
        v = pd.read_csv(p_path)
        k = pd.read_csv(c_path)
        l = pd.read_csv(l_path)
        
        # clean column names
        for d in [v, k, l]: d.columns = d.columns.str.strip()
            
        df = v.merge(k, on="Time (EpSec)").merge(l, on="Time (EpSec)")
        
        # slice to requested number of orbits (check True Anomaly wrap)
        if 'True Anomaly (deg)' in df.columns:
            ta = df['True Anomaly (deg)'].values
            diffs = np.diff(ta) # type: ignore womp womp
            # find where angle drops significantly for orbit wrap
            wrap_indices = np.where(diffs < -300)[0]
            
            if len(wrap_indices) > 0:
                if len(wrap_indices) >= num_orbits:
                    # slice up to the Nth wrap index
                    cutoff_idx = wrap_indices[num_orbits - 1]
                    print(f"[INFO] Slicing to {num_orbits} orbit(s). End index: {cutoff_idx}")
                    df = df.iloc[:cutoff_idx+1].copy()
                else:
                    print(f"[WARN] Requested {num_orbits} orbits, but only {len(wrap_indices)} full orbits found. Using all available data.")
            else:
                print("[INFO] No orbit wraps detected (single partial orbit or linear time). Using full dataset.")

        # Parse Lighting binary of Lit/Eclipse
        light_path = root / f'{self.sat_prefix}_Sat_Lighting_Times.csv'
        sunlight_intervals = []
        if light_path.exists():
            sunlight_intervals = self._parse_lighting_schedule(light_path)
            print(f"[INFO] Parsed {len(sunlight_intervals)} sunlight intervals from lighting report.")
        else:
            print("[WARN] No lighting file found. Assuming 100% sunlight.")
            t_min, t_max = df['Time (EpSec)'].min(), df['Time (EpSec)'].max()
            sunlight_intervals = [(t_min, t_max)]

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
                    except ValueError:
                        continue
        return intervals

    def _select_model(self, energy_budget_j, time_budget_s, workload_infs, include_buffer=False):
        """
        Selects the model that maximizes TOTAL CORRECT INFERENCES given the constraints.
        """
        if energy_budget_j <= 0:
            return None, "Energy_Depleted"
        
        max_inf_time = time_budget_s / self.models['lat_s']
        max_inf_energy = energy_budget_j / self.models['eng_j']
        
        # feasible inferences for each model
        capacity_infs = np.minimum(max_inf_time, max_inf_energy)
        
        # workload demand
        actual_processing_infs = np.minimum(capacity_infs, workload_infs)
        
        # figure of merit: Correct Inferences per frame
        expected_correct = actual_processing_infs * self.models['acc_decimal']
        
        # Select best
        best_idx = expected_correct.idxmax()
        best_model = self.models.loc[best_idx]
        
        # Generate Reason String
        lim_time = max_inf_time[best_idx]
        lim_eng = max_inf_energy[best_idx]
        
        if lim_eng < workload_infs and lim_eng < lim_time:
            reason = "Energy_Limited"
        elif lim_time < workload_infs and lim_time < lim_eng:
            reason = "Time_Limited"
        elif workload_infs < lim_time and workload_infs < lim_eng:
            reason = "Workload_Clearing" 
        else:
            reason = "Balanced"
            
        return best_model, reason

    def run_case_study(self, case_name, config_overrides=None, events=None):
        print(f"\n>>> Running Case Study: {case_name}")
        
        cfg = self.DEFAULT_SYSTEM.copy()
        if config_overrides: cfg.update(config_overrides)
        
        # Interpolate Orbit & Lighting
        sim_data = self._interpolate_orbit(self.raw_orbit, self.sunlight_intervals, cfg['sim_dt_s'], cfg)
        
        # Setup Simulation State
        BATTERY_CAPACITY_J = cfg['battery_capacity_wh'] * 3600.0
        current_battery_j = BATTERY_CAPACITY_J * cfg['initial_charge_pct']
        
        limit_enable_j = BATTERY_CAPACITY_J * cfg['compute_enable_pct']
        limit_disable_j = BATTERY_CAPACITY_J * cfg['compute_disable_pct']
        
        is_recharging = False # Hysteresis state
        
        # Buffers and Jobs
        frame_buffer = deque() # Holds FrameJob objects
        current_job = None     # Currently processing FrameJob
        
        logs = {
            'time_rel': [], 'battery_wh': [], 'buffer_count': [], 
            'throughput_infs': [], 'events': [], 'model_name': [],
            'avg_accuracy': [], 'active_power_w': [], 'selection_reason': [],
            'is_lit': [], 'demand_infs': [], 'alt_km': [], 'speed_km_s': []
        }
        
        total_frames_generated = 0
        total_infs_processed = 0

        # define start time
        t_start = sim_data['Time (EpSec)'].iloc[0]
        
        # environment step
        for i, row in sim_data.iterrows():
            t_abs = row['Time (EpSec)']
            t_rel = t_abs - t_start
            dt = cfg['sim_dt_s']
            
            # Check external events
            disturb_power_w = 0.0
            cpu_blocked = False
            if events:
                for e in events:
                    if e['start'] <= t_rel < (e['start'] + e['duration']):
                        disturb_power_w += e.get('power_w', 0)
                        if e.get('blocked', False): cpu_blocked = True

            # Solar Generation
            solar_w = (cfg['solar_generation_mw'] / 1000.0) * row['is_lit']
            
            # Base Load
            base_w = (cfg['system_baseload_mw'] / 1000.0) + disturb_power_w
            
            # Frame generation
            # Calculate inferences needed for a frame at this moment
            infs_for_this_second = row['demand_infs_per_sec'] * dt

            if infs_for_this_second > 1.0: # Filter out tiny noise
                if len(frame_buffer) < cfg['buffer_max_frames']:
                    new_job = FrameJob(total_frames_generated, infs_for_this_second, t_rel)
                    frame_buffer.append(new_job)
                    total_frames_generated += 1
            
            # Power charging/hysteresis
            if current_battery_j < limit_disable_j:
                is_recharging = True
            elif is_recharging and current_battery_j > limit_enable_j:
                is_recharging = False

            # Inference Logic
            # We have 'dt' seconds available to process work
            time_available_s = dt
            processed_infs_step = 0
            processing_energy_j = 0
            
            if row['px_per_object'] < cfg['min_pixels']:
                active_model_name = "Blind"
            else:
                active_model_name = "Idle"
            selection_reason = "None"
            current_accuracy = 0.0

            while time_available_s > 0:
                # State Check (Blocked or Recharging)
                if cpu_blocked:
                    active_model_name = "BLOCKED"
                    break
                
                if is_recharging:
                    active_model_name = "RECHARGE"
                    break # Cannot process in recharge mode
                
                if current_battery_j <= limit_disable_j:
                    active_model_name = "RECHARGE" # Safety catch if it drops mid-step
                    is_recharging = True
                    break
                
                # Get Job if Idle
                if current_job is None:
                    if len(frame_buffer) > 0:
                        current_job = frame_buffer.popleft()
                        
                        # select model
                        energy_budget_j = current_battery_j - limit_disable_j # gots to base budget off computer power floor, not kill the sat
                        model, reason = self._select_model(
                            energy_budget_j=energy_budget_j,
                            time_budget_s=time_available_s,
                            workload_infs=current_job.total_inferences
                        )
                        
                        if model is None: 
                            active_model_name = "RECHARGE"
                            is_recharging = True
                            break

                        current_job.assigned_model = model
                        selection_reason = reason
                    else:
                        if processed_infs_step == 0:
                            active_model_name = "Idle"

                        break # Nothing to do
                
                # Process Current Job
                if current_job:
                    model = current_job.assigned_model
                    active_model_name = model['Model name']
                    current_accuracy = model['acc_decimal']
                    if selection_reason == "None": selection_reason = "Continuing"
                    
                    # Params
                    latency_s = model['lat_s']
                    joules_per_inf = model['eng_j']
                    
                    # Limits
                    max_infs_time = time_available_s / latency_s
                    
                    # Energy Limit: Can only burn down to disable limit
                    available_energy_j = max(0, current_battery_j - limit_disable_j)
                    max_infs_energy = available_energy_j / joules_per_inf
                    
                    # Limiting factor
                    infs_possible = min(current_job.remaining_inferences, max_infs_time, max_infs_energy)
                    
                    if infs_possible <= 0:
                        # Out of usable energy
                        is_recharging = True
                        active_model_name = "RECHARGE"
                        break

                    # Execute
                    infs_done = infs_possible
                    time_spent = infs_done * latency_s
                    energy_spent = infs_done * joules_per_inf
                    
                    # Update State
                    current_job.remaining_inferences -= infs_done
                    current_battery_j -= energy_spent
                    time_available_s -= time_spent
                    processed_infs_step += infs_done
                    processing_energy_j += energy_spent
                    
                    # Job Completion
                    if current_job.remaining_inferences <= 1e-6: # Float tolerance
                        current_job = None
                    
            avg_processing_power_w = processing_energy_j / dt
            total_load_w = base_w + avg_processing_power_w
            
            # Charge Battery (Net)
            env_energy_j = (solar_w - base_w) * dt
            current_battery_j = np.clip(current_battery_j + env_energy_j, 0, BATTERY_CAPACITY_J)

            # log df
            total_infs_processed += processed_infs_step
            
            logs['time_rel'].append(t_rel)
            logs['battery_wh'].append(current_battery_j / 3600.0)
            logs['buffer_count'].append(len(frame_buffer) + (1 if current_job else 0))
            logs['throughput_infs'].append(processed_infs_step)
            logs['events'].append(1 if cpu_blocked or disturb_power_w > 0 else 0)
            logs['model_name'].append(active_model_name)
            logs['selection_reason'].append(selection_reason)
            logs['avg_accuracy'].append(current_accuracy if processed_infs_step > 0 else np.nan)
            logs['active_power_w'].append(total_load_w)
            logs['is_lit'].append(row['is_lit'])
            logs['demand_infs'].append(infs_for_this_second)
            logs['alt_km'].append(row['Alt (km)'])
            logs['speed_km_s'].append(row['v_ground_km_s'])

        self._plot_telemetry(logs, case_name, cfg)
        print(f"[{case_name}] Complete. Total Inferences: {total_infs_processed:,.0f}")
        return logs

    def _interpolate_orbit(self, df, sunlight_intervals, dt, cfg):
        t_start = df['Time (EpSec)'].min()
        t_end = df['Time (EpSec)'].max()
        new_times = np.arange(t_start, t_end, dt)
        
        # Geometry Interpolation
        cols_to_interp = ['x (km)', 'y (km)', 'z (km)', 
                          'vx (km/sec)', 'vy (km/sec)', 'vz (km/sec)', 
                          'Alt (km)']
        
        new_data = {'Time (EpSec)': new_times}
        for c in cols_to_interp:
            if c in df.columns:
                new_data[c] = np.interp(new_times, df['Time (EpSec)'], df[c])
            else:
                new_data[c] = np.zeros_like(new_times)
        
        # Lighting Logic (Binary)
        is_lit = np.zeros_like(new_times, dtype=float)
        for start, end in sunlight_intervals:
            mask = (new_times >= start) & (new_times <= end)
            is_lit[mask] = 1.0
        new_data['is_lit'] = is_lit
        new_df = pd.DataFrame(new_data)

        # 1. Ground Sample Distance (GSD)
        #    How many meters does one pixel represent?
        #    Higher Altitude = Larger GSD = Lower Resolution
        new_df['gsd_m'] = (new_df['Alt (km)'] * cfg['pixel_pitch_um']) / cfg['focal_length_mm']
        
        # 2. Swath Width
        #    Total width of ground visible to the sensor
        new_df['swath_km'] = (new_df['gsd_m'] * cfg['sensor_res']) / 1000.0
        
        # 3. Velocity Calculation (Keep existing logic)
        r = new_df[['x (km)', 'y (km)', 'z (km)']].values
        v = new_df[['vx (km/sec)', 'vy (km/sec)', 'vz (km/sec)']].values
        r_norm = np.linalg.norm(r, axis=1, keepdims=True)
        v_vert = np.sum(v * (r/r_norm), axis=1, keepdims=True) * (r/r_norm)
        v_ground = np.linalg.norm(v - v_vert, axis=1)
        new_df['v_ground_km_s'] = v_ground
        
        # 4. Dwell Time
        #    How long a specific point on the ground remains in the camera's FOV.
        #    (Avoid divide by zero errors with a tiny epsilon)
        new_df['dwell_time_s'] = new_df['swath_km'] / (v_ground + 1e-9)

        # 5. Object Resolution Calculation
        #    We are looking for a physical object of size 'target_tile_km'.
        #    How many pixels does that object occupy?
        target_size_m = cfg['target_tile_km'] * 1000.0
        new_df['px_per_object'] = target_size_m / new_df['gsd_m']
        
        # 6. Smart Inference Scaling
        #    IF pixels < min_pixels: We are BLIND. Demand = 0.
        #    ELSE: We need (pixels / 224)^2 inferences to cover that object.
        #    (This simulates running a sliding window over the object)
        tpu_dim = cfg['tpu_dim']
        infs_per_object = np.where(
            new_df['px_per_object'] < cfg['min_pixels'],
            0.0, # Blind
            (new_df['px_per_object'] / tpu_dim)**2
        )
        
        # 7. Total Frame Demand
        #    How many "objects" fit in our current FOV?
        #    (Approximating FOV area as swath_km^2)
        objects_in_fov = (new_df['swath_km'] / cfg['target_tile_km'])**2
        
        #    Total inferences required to process ONE full snapshot of the ground
        total_infs_per_frame = objects_in_fov * infs_per_object
        
        # 8. Demand Rate (Inferences Per Second)
        #    We must process the frame within the dwell time to maintain continuous coverage.
        new_df['demand_infs_per_sec'] = total_infs_per_frame / new_df['dwell_time_s']
        
        return new_df
    def _plot_telemetry(self, logs, case_name, cfg):
        t = np.array(logs['time_rel'])
        t_plot = t - t[0]
        
        batt = np.array(logs['battery_wh'])
        lit = np.array(logs['is_lit'])
        demand = np.array(logs['demand_infs'])
        alt = np.array(logs['alt_km'])
        speed = np.array(logs['speed_km_s'])
        
        # Aggregate Model Usage
        df_log = pd.DataFrame({'model': logs['model_name'], 'infs': logs['throughput_infs']})
        # Filter out Idle/Blocked/Recharge for the bar chart if they have 0 infs
        model_stats = df_log[df_log['infs'] > 0].groupby('model')['infs'].sum()
        
        # Plotting
        fig = plt.figure(figsize=(12, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 2, 1])
        
        # Top Plot: Altitude & Speed
        ax0 = fig.add_subplot(gs[0])
        color_alt = 'tab:gray'
        ax0.set_ylabel('Altitude (km)', color=color_alt)
        ax0.plot(t_plot, alt, color=color_alt, linewidth=2, label='Altitude')
        ax0.tick_params(axis='y', labelcolor=color_alt)
        ax0.grid(True, alpha=0.3)
        ax0.set_title(f"Case Study: {case_name}")

        ax0_twin = ax0.twinx()
        color_speed = 'tab:red'
        ax0_twin.set_ylabel('Ground Speed (km/s)', color=color_speed)
        ax0_twin.plot(t_plot, speed, color=color_speed, linestyle='--', linewidth=1.5, label='Speed')
        ax0_twin.tick_params(axis='y', labelcolor=color_speed)
        
        # Combine legends
        l1, lab1 = ax0.get_legend_handles_labels()
        l2, lab2 = ax0_twin.get_legend_handles_labels()
        ax0.legend(l1+l2, lab1+lab2, loc='upper center', ncol=2)

        # Middle Plot: Energy vs Workload
        ax1 = fig.add_subplot(gs[1], sharex=ax0)
        color_batt = 'tab:green'
        ax1.set_ylabel('Battery Energy (Wh)', color=color_batt)
        ax1.plot(t_plot, batt, color=color_batt, linewidth=2, label='Battery')
        # Add Hysteresis Limit lines
        ax1.axhline(cfg['battery_capacity_wh']*cfg['compute_disable_pct'], color='r', linestyle=':', alpha=0.5, label='Disable Limit')
        ax1.axhline(cfg['battery_capacity_wh']*cfg['compute_enable_pct'], color='g', linestyle=':', alpha=0.5, label='Enable Limit')
        ax1.tick_params(axis='y', labelcolor=color_batt)
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        color_demand = 'tab:blue'
        ax2.set_ylabel('Workload Demand (Infs/s)', color=color_demand)
        ax2.plot(t_plot, demand, color=color_demand, alpha=0.6, linestyle='--', label='Demand')
        ax2.tick_params(axis='y', labelcolor=color_demand)
        
        # Sunlight Overlay
        ax1.fill_between(t_plot, 0, 1, where=(lit > 0.5), 
                         transform=ax1.get_xaxis_transform(), 
                         color='gold', alpha=0.2, label='Sunlight')
        
        # Combine legends
        l3, lab3 = ax1.get_legend_handles_labels()
        l4, lab4 = ax2.get_legend_handles_labels()
        ax1.legend(l3+l4, lab3+lab4, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.1))

        # Model Usage
        ax3 = fig.add_subplot(gs[2])
        if not model_stats.empty:
            model_stats = model_stats.sort_values(ascending=True)
            bars = ax3.barh(model_stats.index, model_stats.values, color='tab:purple')
            ax3.set_xlabel('Total Inferences Performed')
            ax3.set_title('Model Utilization Summary')
            ax3.bar_label(bars, fmt='{:,.0f}', padding=3)
            ax3.grid(True, axis='x', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No Inferences Performed", ha='center', va='center')
        
        ax1.set_xlabel('Time (s)') 
        
        plt.tight_layout()
        filename = f"sim_{self.sat_prefix}_{case_name.replace(' ', '_').lower()}.png"
        save_path = self.output_dir / filename
        plt.savefig(save_path)
        print(f"Saved plot to: {save_path}")
        plt.close()

def run_all_case_studies():
    model_json = ROOT_DIR / "libs/coral_tpu_characterization/data/compiled_characterization.json" 
    output_path = ROOT_DIR / "results/case_studies"
    orbit_path = ROOT_DIR / "libs/coral_tpu_characterization/data/stk"

    # Example: Filter for 'Custom' models only
    sim_heo = ContinuousSatSim(orbit_path, model_json, output_path, sat_prefix='HEO', num_orbits=1, model_source='Custom')
    sim_heo.run_case_study("Dynamic_Hysteresis_Control")

if __name__ == "__main__":
    run_all_case_studies()