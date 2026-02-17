import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
import sys

# Attempt to import get_repo_root, fallback to local dir
try:
    from libs.coral_tpu_characterization.src.scripts.utils.path_utils import get_repo_root
    ROOT_DIR = get_repo_root()
except ImportError:
    ROOT_DIR = Path(".")

import libs.coral_tpu_characterization.src.scripts.hardware_characterization.plotting.tpunet_plotting as mdl

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
        'buffer_max_images': 50,
        'sim_dt_s': 1.0,
        'initial_charge_pct': 0.9,
    }

    def __init__(self, orbit_data_path, json_dir, saleae_root, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("--- Loading Models & Orbit Data ---")
        self.models = self._load_models(json_dir, saleae_root)
        self.raw_orbit = self._load_orbit(orbit_data_path)
        
        self.model_fast = self.models.loc[self.models['Correct_Inf_per_Sec'].idxmax()]
        self.model_eff = self.models.loc[self.models['Correct_Inf_per_Joule'].idxmax()]

    def run_case_study(self, case_name, config_overrides=None, events=None):
        print(f"\n>>> Running Case Study: {case_name}")
        
        cfg = self.DEFAULT_SYSTEM.copy()
        if config_overrides: cfg.update(config_overrides)
        
        sim_data = self._interpolate_orbit(self.raw_orbit, cfg['sim_dt_s'], cfg)
        BATTERY_CAPACITY_J = cfg['battery_capacity_wh'] * 3600.0
        current_battery_j = BATTERY_CAPACITY_J * cfg['initial_charge_pct']
        
        image_buffer = deque()
        
        logs = {'time': [], 'battery_wh': [], 'buffer': [], 'throughput': [], 'events': [], 'model': []}
        total_acc_infs = 0
        
        for i, row in sim_data.iterrows():
            t = row['Time (s)']
            dt = cfg['sim_dt_s']
            
            # --- Events ---
            disturb_power_w = 0.0
            cpu_blocked = False
            if events:
                for e in events:
                    if e['start'] <= t < (e['start'] + e['duration']):
                        disturb_power_w += e.get('power_w', 0)
                        if e.get('blocked', False): cpu_blocked = True

            # --- Energy Inputs ---
            solar_w = cfg['solar_generation_mw']/1000.0 * row['Solar Intensity']
            base_w = cfg['system_baseload_mw']/1000.0 + disturb_power_w
            
            # --- Acquisition ---
            new_work = row['demand_infs_per_sec'] * dt
            if new_work > 0:
                if len(image_buffer) < cfg['buffer_max_images']:
                    image_buffer.append(new_work)

            # --- Policy & Processing ---
            processed_infs = 0
            processing_power_w = 0
            active_model_name = "Idle"
            
            is_dead = current_battery_j <= 0
            is_crit = current_battery_j < (BATTERY_CAPACITY_J * 0.05)
            is_full = len(image_buffer) > (cfg['buffer_max_images'] * 0.4)
            
            if cpu_blocked:
                active_model = None
                active_model_name = "BLOCKED"
            elif is_dead:
                active_model = None
                active_model_name = "DEAD"
            elif is_crit:
                if len(image_buffer) > (cfg['buffer_max_images'] * 0.9):
                    active_model = self.model_eff 
                    active_model_name = "Crit_Drain"
                else:
                    active_model = None
                    active_model_name = "Recharge"
            elif is_full:
                active_model = self.model_fast
                active_model_name = "Fast"
            elif len(image_buffer) > 0:
                active_model = self.model_eff
                active_model_name = "Eco"
            else:
                active_model = None
                active_model_name = "Idle"

            if active_model is not None and len(image_buffer) > 0:
                max_infs = dt / active_model['lat_s']
                work_done = 0
                
                while len(image_buffer) > 0 and work_done < max_infs:
                    packet = image_buffer[0]
                    space = max_infs - work_done
                    
                    if packet <= space:
                        work_done += packet
                        image_buffer.popleft()
                    else:
                        image_buffer[0] -= space
                        work_done += space
                        break
                
                processed_infs = work_done
                processing_power_w = (processed_infs * active_model['eng_j']) / dt
                total_acc_infs += processed_infs * active_model['acc_decimal']

            # --- Physics Update ---
            total_load_w = base_w + processing_power_w
            net_power_w = solar_w - total_load_w
            current_battery_j = np.clip(current_battery_j + (net_power_w * dt), 0, BATTERY_CAPACITY_J)

            # --- Logging ---
            logs['time'].append(t)
            logs['battery_wh'].append(current_battery_j / 3600.0)
            logs['buffer'].append(len(image_buffer))
            logs['throughput'].append(processed_infs)
            logs['events'].append(1 if cpu_blocked or disturb_power_w > 0 else 0)
            logs['model'].append(active_model_name)

        self._plot_telemetry(logs, case_name, cfg)
        print(f"[{case_name}] Complete. Total Inferences: {total_acc_infs:,.0f}")
        return logs

    def _load_models(self, json_dir, saleae_root):
        plotter = mdl.GridStatsPlotting(json_dir, saleae_root, self.output_dir)
        df = plotter.load_and_aggregate_data()
        df['acc_decimal'] = df['Top-1 Accuracy'] / 100.0
        df['lat_s'] = df['Measured Inference Time (ms)'] / 1000.0
        df['eng_j'] = df['Energy per Inference (mJ)'] / 1000.0
        return df

    def _load_orbit(self, data_path):
        root = Path(data_path)
        v = pd.read_csv(root / 'HEO_Sat_Fixed_Position_Velocity.csv')
        k = pd.read_csv(root / 'HEO_Sat_Classical_Orbit_Elements.csv')
        l = pd.read_csv(root / 'HEO_Sat_LLA_Position.csv')
        df = v.merge(k, on="Time (UTCG)").merge(l, on="Time (UTCG)")
        
        solar_path = root / 'HEO_Sat_Solar_Intensity.csv'
        if solar_path.exists():
            s = pd.read_csv(solar_path)
            if 'Solar Intensity' in s.columns:
                df = df.merge(s[['Time (UTCG)', 'Solar Intensity']], on="Time (UTCG)", how='left')
        
        df.columns = df.columns.str.strip()
        ta = df['True Anomaly (deg)'].values
        if len(np.where(np.diff(ta) < -300)[0]) > 0: 
            df = df.iloc[:np.where(np.diff(ta) < -300)[0][0]+1]
        
        df['Time (s)'] = pd.to_datetime(df['Time (UTCG)'])
        df['Time (s)'] = (df['Time (s)'] - df['Time (s)'].iloc[0]).dt.total_seconds()
        return df

    def _interpolate_orbit(self, df, dt, cfg):
        t_end = df['Time (s)'].max()
        new_times = np.arange(0, t_end, dt)
        new_df = pd.DataFrame({'Time (s)': new_times})
        
        cols = ['x (km)', 'y (km)', 'z (km)', 'vx (km/sec)', 'vy (km/sec)', 'vz (km/sec)', 'Alt (km)', 'Solar Intensity']
        for c in cols:
            new_df[c] = np.interp(new_times, df['Time (s)'], df[c].fillna(1.0)) if c in df.columns else 0

        # Physics Recalc
        new_df['gsd_m'] = (new_df['Alt (km)'] * cfg['pixel_pitch_um']) / cfg['focal_length_mm']
        new_df['swath_km'] = (new_df['gsd_m'] * cfg['sensor_res']) / 1000.0
        
        r = new_df[['x (km)', 'y (km)', 'z (km)']].values
        v = new_df[['vx (km/sec)', 'vy (km/sec)', 'vz (km/sec)']].values
        r_norm = np.linalg.norm(r, axis=1, keepdims=True)
        v_vert = np.sum(v * (r/r_norm), axis=1, keepdims=True) * (r/r_norm)
        v_ground = np.linalg.norm(v - v_vert, axis=1)
        
        target_km = cfg['target_tile_km']
        px_per_tile = (target_km * 1000.0) / new_df['gsd_m']
        inf_per_tile = np.where(px_per_tile < cfg['min_pixels'], 0.0, (px_per_tile / cfg['tpu_dim'])**2)
        
        # Tiles/sec = (Swath * V_ground) / Target_Area
        new_df['demand_infs_per_sec'] = ((new_df['swath_km'] * v_ground) / (target_km**2)) * inf_per_tile
        return new_df

    def _plot_telemetry(self, logs, case_name, cfg):
        t, batt, buf = np.array(logs['time']), np.array(logs['battery_wh']), np.array(logs['buffer'])
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Battery
        ax1.plot(t, batt, 'g', label='Battery (Wh)')
        ax1.set_ylabel('Energy (Wh)')
        ax1.set_title(f"Case Study: {case_name}")
        if np.any(np.array(logs['events']) > 0):
             ax1.fill_between(t, 0, max(batt), where=(np.array(logs['events'])>0), color='red', alpha=0.3, label="Event")
        ax1.legend(loc='upper right'); ax1.grid(True, alpha=0.3)

        # Buffer
        ax2.plot(t, buf, 'b', label='Buffer Depth')
        ax2.axhline(y=cfg['buffer_max_images'], color='r', linestyle='--', label='Max Buffer')
        ax2.set_ylabel('Queued Frames'); ax2.grid(True, alpha=0.3); ax2.legend(loc='upper right')
        
        # States
        models = logs['model']
        unique_models = sorted(list(set(models)))
        model_map = {m: i for i, m in enumerate(unique_models)}
        numeric_models = [model_map[m] for m in models]
        
        ax3.step(t, numeric_models, where='post', color='k')
        ax3.set_yticks(range(len(unique_models))); ax3.set_yticklabels(unique_models)
        ax3.set_ylabel("Active State"); ax3.set_xlabel("Time (s)"); ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"sim_{case_name.replace(' ', '_').lower()}.png")
        plt.close()

def run_all_case_studies():
    sim = ContinuousSatSim(
        orbit_data_path=ROOT_DIR / "data/stk", 
        json_dir=ROOT_DIR / "data/tpunet_acc",
        saleae_root="results/captures_1_20", 
        output_dir=ROOT_DIR / "results/case_studies"
    )
    
    # 1. Baseline
    sim.run_case_study("Baseline_Nominal")

    # 2. Radio Event (15W Power, CPU Blocked)
    sim.run_case_study("Radio_Downlink_Event", 
        events=[{'start': 3000, 'duration': 60, 'power_w': 15.0, 'blocked': True}]
    )

    # 3. High Draw / Failure (Low Start Bat, Power Leak)
    sim.run_case_study("Power_Failure_Scenario",
        config_overrides={'initial_charge_pct': 0.3},
        events=[{'start': 1000, 'duration': 2000, 'power_w': 5.0, 'blocked': False}]
    )

if __name__ == "__main__":
    run_all_case_studies()