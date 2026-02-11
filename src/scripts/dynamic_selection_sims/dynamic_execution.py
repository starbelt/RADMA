import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import libs.coral_tpu_characterization.src.scripts.dynamic_selection_sims.ground_track_stk as orb
import libs.coral_tpu_characterization.src.scripts.hardware_characterization.plotting.tpunet_plotting as mdl

# TODO: Sudden power failure case? e.g. higher base power, worse solar performance due to pointing

class SatelliteInferenceSim:
    DEFAULT_CONFIG = {
        # --- NEW SENSOR PROPERTIES ---
        'focal_length_mm': 50.0,   # Lens Focal Length
        'pixel_pitch_um': 3.45,    # Pixel size (microns), e.g., Sony IMX250 is ~3.45um
        'sensor_res': 4096,        # Camera Resolution (pixels) squared or width
        # -----------------------------

        'target_tile_km': 10.0,   # Ground feature size
        'tpu_dim': 224,            # TPU Input size
        'min_pixels': 10,          # Blindness threshold (min pixels required to resolve feature)

        'battery_capacity_wh': 1.1,    
        'solar_generation_mw': 200.0,  
        'system_baseload_mw': 80.0,    
        'initial_charge_pct': 1.0,

        'min_safe_battery_pct': 0.05,
        'naive_restart_threshold': 0.65, 
        
        # Switching
        'switch_latency_s': 0.0, 
        'switch_energy_mj': 0.0 
    }

    def __init__(self, orbit_data_path, json_dir, saleae_root, output_dir, config=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = self.DEFAULT_CONFIG.copy()
        if config: self.config.update(config)

        self.BATTERY_CAPACITY_J = self.config['battery_capacity_wh'] * 3600.0
        
        print("--- Loading Data ---")
        self.models = self._load_models(json_dir, saleae_root)
        self.df = self._load_orbit(orbit_data_path)
        
        print("--- Pre-calculating Geometry ---")
        self.df = self.calculate_geometry_and_workload(self.df)

    def _load_models(self, json_dir, saleae_root):
        plotter = mdl.GridStatsPlotting(json_dir, saleae_root, self.output_dir)
        df = plotter.load_and_aggregate_data()
        if df is None or df.empty: raise ValueError("No model data found.")
        df['acc_decimal'] = df['Top-1 Accuracy'] / 100.0
        df['lat_sec'] = df['Measured Inference Time (ms)'] / 1000.0
        return df

    def _load_orbit(self, data_path):
        root = Path(data_path)
        v = pd.read_csv(root / 'HEO_Sat_Fixed_Position_Velocity.csv')
        k = pd.read_csv(root / 'HEO_Sat_Classical_Orbit_Elements.csv')
        l = pd.read_csv(root / 'HEO_Sat_LLA_Position.csv')
        
        df = v.merge(k, on="Time (UTCG)").merge(l, on="Time (UTCG)")
        
        solar_path = root / 'HEO_Sat_Solar_Intensity.csv'
        if solar_path.exists():
            print("[INFO] Found STK Solar Intensity data. Importing...")
            s = pd.read_csv(solar_path)
            if 'Solar Intensity' in s.columns:
                df = df.merge(s[['Time (UTCG)', 'Solar Intensity']], on="Time (UTCG)", how='left')
        
        df.columns = df.columns.str.strip()

        ta = df['True Anomaly (deg)'].values
        diffs = np.diff(ta) # type: ignore
        wrap_indices = np.where(diffs < -300)[0]
        
        if len(wrap_indices) > 0:
            cutoff_idx = wrap_indices[0] + 1
            print(f"[INFO] Multi-orbit data detected. Slicing to first orbit (Indices 0 to {cutoff_idx}).")
            df = df.iloc[:cutoff_idx].copy()
        else:
            print("[INFO] Single orbit (or partial) detected. Using full dataset.")
        
        return df

    def calculate_geometry_and_workload(self, df):
        sensor_width_mm = self.config['sensor_res'] * (self.config['pixel_pitch_um'] / 1000.0)
        
        # FOV = 2 * atan(sensor_width / (2 * focal_length))
        fov_rad = 2 * np.arctan(sensor_width_mm / (2 * self.config['focal_length_mm']))
        
        # Save derived FOV for reference
        self.config['calculated_fov_deg'] = np.degrees(fov_rad)
        print(f"[INFO] Calculated FOV: {self.config['calculated_fov_deg']:.2f} degrees")

        # Ground Track Velocity from ECEF inputs
        r_vec = df[['x (km)', 'y (km)', 'z (km)']].values # pos
        v_vec = df[['vx (km/sec)', 'vy (km/sec)', 'vz (km/sec)']].values # vel
        
        # radial unit vector 
        r_norm = np.linalg.norm(r_vec, axis=1, keepdims=True)
        r_unit = r_vec / r_norm
        
        # project Velocity onto Radial 
        # dot product: (v . r_unit)
        v_vertical_mag = np.sum(v_vec * r_unit, axis=1, keepdims=True)
        v_vertical_vec = v_vertical_mag * r_unit
        
        # Subtract Vertical component to get Horizontal (Ground Track) Vector
        v_ground_vec = v_vec - v_vertical_vec
        
        # Magnitude of the ground track vector
        df['v_ground'] = np.linalg.norm(v_ground_vec, axis=1)

        # swath = 2 * altitude * tan(FOV/2)
        df['swath_km'] = 2 * df['Alt (km)'] * np.tan(fov_rad / 2)
        
        # dwell time = swath / ground track velocity
        df['dwell_time_s'] = df['swath_km'] / df['v_ground']
        
        # GSD = Swath Width in meters / Sensor Resolution in pixels 
        df['gsd_m'] = (df['swath_km'] * 1000.0) / self.config['sensor_res'] 

        # Workload Logic 
        target_km = self.config['target_tile_km']
        tpu_dim = self.config['tpu_dim']
        
        # Calculate pixels per target tile based on current GSD
        df['px_per_tile'] = (target_km * 1000.0) / df['gsd_m']
        
        # "Smart" Workload
        df['infs_per_tile'] = np.where(df['px_per_tile'] < self.config['min_pixels'], 
                                        0.0, 
                                        (df['px_per_tile'] / tpu_dim)**2)
        
        # Total Tiles
        tilees_in_view = (df['swath_km']**2) / (target_km**2)
        df['n_inferences_req'] = np.ceil(tilees_in_view * df['infs_per_tile'])
        
        # Power Logic 
        if 'Solar Intensity' in df.columns:
            df['solar_factor'] = df['Solar Intensity'].fillna(1.0)
            df['is_eclipse'] = df['solar_factor'] < 0.1
        else:
            df['solar_factor'] = 1.0
            df['is_eclipse'] = False

        df['gen_rate_mw'] = self.config['solar_generation_mw'] * df['solar_factor']
        df['energy_harvested_j'] = (df['gen_rate_mw'] * df['dwell_time_s']) / 1000.0
        df['energy_baseload_j'] = (self.config['system_baseload_mw'] * df['dwell_time_s']) / 1000.0

        return df

    def _evaluate_step_sequential(self, current_battery_j, harvested_j, baseload_j, dwell_time_s, req_inf, 
                                lat, eng, acc, switching_penalty_j=0, switching_penalty_s=0):
        if req_inf <= 0:
            return 0, 0, baseload_j, "Blind"

        available_energy_j = current_battery_j + harvested_j - baseload_j - switching_penalty_j
        available_time_s = dwell_time_s - switching_penalty_s

        if available_energy_j <= 0:
            return 0, 0, current_battery_j + harvested_j, "DeadBattery"

        eng_j = eng / 1000.0
        cap_eng = np.floor(available_energy_j / eng_j)
        cap_time = np.floor(available_time_s / lat)
        
        actual_runs = np.minimum(np.minimum(cap_eng, cap_time), req_inf)
        total_consumed_j = baseload_j + switching_penalty_j + (actual_runs * eng_j)
        score = actual_runs * acc
        
        if actual_runs >= req_inf: status = "Success"
        elif actual_runs > 0:      status = "Partial"
        else:                      status = "Fail"
            
        return score, actual_runs, total_consumed_j, status

    def run_dynamic_optimization(self):
        print("--- Running Sequential Dynamic Optimization ---")
        curr_battery_j = self.BATTERY_CAPACITY_J * self.config['initial_charge_pct']
        
        res_model, res_acc, res_raw, res_energy, res_status, res_battery = [], [], [], [], [], []
        
        prev_model_idx = -1
        model_names = self.models['Model name'].values
        m_lat = self.models['lat_sec'].values
        m_eng = self.models['Energy per Inference (mJ)'].values
        m_acc = self.models['acc_decimal'].values

        for idx, row in self.df.iterrows():
            harvest, baseload, dwell, req = row['energy_harvested_j'], row['energy_baseload_j'], row['dwell_time_s'], row['n_inferences_req']
            
            best_score, best_raw, best_idx = -1, 0, -1
            best_status, best_consumed_j = "Fail", baseload 
            
            for i in range(len(model_names)):
                # Go through each potential model, evaluate the best one for the next step
                sw_j = (self.config['switch_energy_mj'] / 1000.0) if (prev_model_idx != -1 and i != prev_model_idx) else 0
                sw_s = self.config['switch_latency_s'] if (prev_model_idx != -1 and i != prev_model_idx) else 0

                score, raw, cons_j, status = self._evaluate_step_sequential(
                    curr_battery_j, harvest, baseload, dwell, req,
                    m_lat[i], m_eng[i], m_acc[i], sw_j, sw_s
                )
                
                if score > best_score:
                    best_score, best_raw, best_idx = score, raw, i
                    best_status, best_consumed_j = status, cons_j
                
                if curr_battery_j < (self.BATTERY_CAPACITY_J * self.config['min_safe_battery_pct']): #margin
                    # Recharge behavior
                    # Don't run any models, just baseload
                    best_score = 0
                    best_raw = 0
                    best_idx = -1
                    best_status = "Recharging"
                    best_consumed_j = baseload
            
            if best_idx != -1:
                res_model.append(model_names[best_idx])
                prev_model_idx = best_idx
            else:
                res_model.append("None")
                best_consumed_j = baseload
            
            res_acc.append(best_score)
            res_raw.append(best_raw)
            res_energy.append(best_consumed_j)
            res_status.append(best_status)
            
            curr_battery_j = np.clip(curr_battery_j + harvest - best_consumed_j, 0, self.BATTERY_CAPACITY_J)
            res_battery.append(curr_battery_j)

        self.df['dynamic_model'] = res_model
        self.df['dynamic_acc_inf'] = res_acc
        self.df['dynamic_raw_inf'] = res_raw 
        self.df['dynamic_energy_j'] = res_energy 
        self.df['dynamic_status'] = res_status
        self.df['dynamic_battery_j'] = res_battery
        return self.df

    def run_baseline_sequential(self, strategy_name, model_idx):
        curr_battery_j = self.BATTERY_CAPACITY_J * self.config['initial_charge_pct']
        
        restart_threshold_j = self.BATTERY_CAPACITY_J * self.config['naive_restart_threshold']
        cutoff_j = self.BATTERY_CAPACITY_J * self.config['min_safe_battery_pct']
        
        total_acc_inf = 0
        battery_trace = []
        
        lat = self.models.at[model_idx, 'lat_sec']
        eng = self.models.at[model_idx, 'Energy per Inference (mJ)']
        acc = self.models.at[model_idx, 'acc_decimal']
        
        dead_steps = 0
        blind_steps = 0
        is_recovering = False
        
        for _, row in self.df.iterrows():
            harvest = row['energy_harvested_j']
            baseload = row['energy_baseload_j']
            dwell = row['dwell_time_s']
            req_inf = row['n_inferences_req']
            
            # recovery period
            if is_recovering:
                # Operationally dead while recovering
                dead_steps += 1 
                if req_inf > 0: blind_steps += 1
                
                # consume baseload power
                cons_j = baseload
                
                # update battery
                potential_batt = curr_battery_j + harvest - cons_j
                
                # check in on wakeup condition
                if potential_batt >= restart_threshold_j:
                    is_recovering = False
                
                # check low bound
                curr_battery_j = np.clip(potential_batt, 0, self.BATTERY_CAPACITY_J)
                battery_trace.append(curr_battery_j)
                continue 

            # Active Period
            score, _, cons_j, status = self._evaluate_step_sequential(
                curr_battery_j, harvest, baseload, dwell, req_inf,
                lat, eng, acc, 0, 0 
            )
            
            # Update Battery
            curr_battery_j = np.clip(curr_battery_j + harvest - cons_j, 0, self.BATTERY_CAPACITY_J)
            
            # if it died or dipped below the safety cutoff, force recovery.
            # This prevents the model from riding the 0% line forever.
            if status == "DeadBattery" or curr_battery_j < cutoff_j:
                is_recovering = True
                # If we actually crashed (DeadBattery), ensure battery is 0
                if status == "DeadBattery": curr_battery_j = 0
                
                # Don't count score for this step if we crashed/died mid-step?
                # actually, we get the data before the crash, so we keep the score, we'll be generous and say we're really quick at writing to storage lol .
            
            if status == "Blind": blind_steps += 1
            
            total_acc_inf += score
            battery_trace.append(curr_battery_j)
            
        return total_acc_inf, battery_trace, dead_steps, blind_steps

    def run_naive_baseline(self, model_idx):
        curr_battery_j = self.BATTERY_CAPACITY_J * self.config['initial_charge_pct']
        restart_threshold_j = self.BATTERY_CAPACITY_J * self.config['naive_restart_threshold']
        
        total_acc_inf = 0
        battery_trace = []
        
        lat = self.models.at[model_idx, 'lat_sec']
        eng = self.models.at[model_idx, 'Energy per Inference (mJ)']
        acc = self.models.at[model_idx, 'acc_decimal']
        
        dead_steps = 0
        blind_steps = 0
        is_recovering = False
        
        for _, row in self.df.iterrows():
            harvest = row['energy_harvested_j']
            baseload = row['energy_baseload_j']
            dwell = row['dwell_time_s']
            real_demand = row['n_inferences_req']
            
            if is_recovering:
                dead_steps += 1 # offline while recovering
                if real_demand > 0: blind_steps += 1
                
                cons_j = baseload
                potential_batt = curr_battery_j + harvest - cons_j
                
                if potential_batt >= restart_threshold_j:
                    is_recovering = False
                
                curr_battery_j = np.clip(potential_batt, 0, self.BATTERY_CAPACITY_J)
                battery_trace.append(curr_battery_j)
                continue

            # Naive demands MAXIMUM possible throughput, no knowledge of demand/ ground frame deadlines!
            runs_wanted = np.floor(dwell / lat)
            energy_needed_j = baseload + (runs_wanted * (eng / 1000.0))
            
            available_j = curr_battery_j + harvest
            
            if energy_needed_j > available_j:
                # CRASH
                curr_battery_j = 0 
                is_recovering = True 
                dead_steps += 1 # count the crash step as dead too? This is so marginal whatever
                actual_runs_for_score = 0 
            else:
                # SUCCESS
                curr_battery_j = available_j - energy_needed_j
                actual_runs_for_score = min(runs_wanted, real_demand)
            
            if real_demand == 0: blind_steps += 1
            total_acc_inf += (actual_runs_for_score * acc)
            battery_trace.append(curr_battery_j)
            
        return total_acc_inf, battery_trace, dead_steps, blind_steps

    def generate_report(self):
        print("\n" + "="*100)
        print(f"{'SEQUENTIAL BATTERY SIMULATION REPORT':^100}")
        print("="*100)

        # Identify Models
        idx_worst = self.models['Correct_Inf_per_Joule'].idxmin()
        idx_throughput = self.models['Correct_Inf_per_Sec'].idxmax()
        idx_efficiency = self.models['Correct_Inf_per_Joule'].idxmax()
        
        name_worst = self.models.at[idx_worst, 'Model name']
        name_fast = self.models.at[idx_throughput, 'Model name']
        name_eff = self.models.at[idx_efficiency, 'Model name']

        print(f"\n> Model Selection for Baselines:")
        print(f"  Worst Efficiency:   {name_worst}")
        print(f"  Best Throughput:    {name_fast}")
        print(f"  Best Efficiency:    {name_eff}")

        # run baselines
        bl_configs = {
            "Naive (Worst)": (idx_worst, True),
            "Static Worst (Smart)": (idx_worst, False),
            "Static Throughput (Smart)": (idx_throughput, False),
            "Static Efficiency (Smart)": (idx_efficiency, False)
        }
        
        bl_results = {}
        bl_traces = {}
        
        print(f"\n{'~'*38} PERFORMANCE COMPARISON {'~'*38}")
        print(f"{'Strategy':<26} | {'Total Acc. Inf':<14} | {'% Dead':<7} | {'% Blind':<7} | {'Improvement'}")
        print("-" * 100)
        
        score_worst_naive = 0
        score_best_static = 0
        total_steps = len(self.df)
        
        for name, (idx, is_naive) in bl_configs.items():
            if is_naive:
                score, trace, dead, blind = self.run_naive_baseline(idx)
                if name == "Naive (Worst)": score_worst_naive = score
            else:
                score, trace, dead, blind = self.run_baseline_sequential(name, idx)
                if score > score_best_static: score_best_static = score

            bl_results[name] = score
            bl_traces[name] = trace
            
            pct_dead = (dead / total_steps) * 100
            pct_blind = (blind / total_steps) * 100
            
            print(f"{name:<26} | {score:,.0f}".ljust(43) + 
                f" | {pct_dead:>5.1f}% | {pct_blind:>5.1f}% | ", end="")
            print("") 

        # dynamic Results
        dyn_total = self.df['dynamic_acc_inf'].sum()
        dyn_dead = (self.df['dynamic_status'] == 'DeadBattery').sum()
        dyn_blind = (self.df['dynamic_status'] == 'Blind').sum()
        
        pct_dyn_dead = (dyn_dead / total_steps) * 100
        pct_dyn_blind = (dyn_blind / total_steps) * 100

        print("-" * 100)
        gain_vs_best = ((dyn_total - score_best_static) / score_best_static * 100) if score_best_static > 0 else 0
        gain_vs_naive = ((dyn_total - score_worst_naive) / score_worst_naive * 100) if score_worst_naive > 0 else 0
        
        print(f"{'Dynamic Switching (Ours)':<26} | {dyn_total:,.0f}".ljust(43) + 
            f" | {pct_dyn_dead:>5.1f}% | {pct_dyn_blind:>5.1f}% | +{gain_vs_best:.1f}% (vs Best Static)")
        print(f"{'':<43}             |        |        | +{gain_vs_naive:.1f}% (vs Naive)")
        print("="*100)
        
        # utilization breakdown
        print(f"\n{'~'*36} DYNAMIC UTILIZATION DETAILS {'~'*37}")
        print(f"{'Model Name':<28} | {'% Time':<7} | {'% Infs':<7} | {'Total Infs':<11} | {'Energy (J)':<11}")
        print("-" * 100)
        
        valid_df = self.df[~self.df['dynamic_model'].isin(['None', 'Blind'])]
        total_raw_infs = valid_df['dynamic_raw_inf'].sum()
        
        if total_raw_infs > 0:
            stats = valid_df.groupby('dynamic_model').agg({
                'dynamic_raw_inf': 'sum',
                'dynamic_energy_j': 'sum',
                'dynamic_model': 'count'
            }).rename(columns={'dynamic_model': 'steps'}).sort_values('dynamic_raw_inf', ascending=True)

            for name, row in stats.iterrows():
                raw_count = row['dynamic_raw_inf']
                energy = row['dynamic_energy_j']
                steps = row['steps']
                
                pct_inf = (raw_count / total_raw_infs) * 100
                pct_time = (steps / total_steps) * 100
                
                disp_name = (name[:25] + '..') if len(name) > 27 else name
                print(f"{disp_name:<28} | {pct_time:>6.1f}% | {pct_inf:>6.1f}% | {raw_count:,.0f}".rjust(12) + 
                        f" | {energy:,.0f}".rjust(13))
        else:
            print("No inferences performed.")
            
        print("-" * 100)
        
        self.plot_results(bl_traces)

    def plot_results(self, baseline_traces):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16), sharex=True, 
                                            gridspec_kw={'height_ratios': [2, 1, 2]})
        
        x = self.df['True Anomaly (deg)']
        
        # Plot Dynamic
        ax1.plot(x, self.df['dynamic_battery_j'] / 3600.0, 'g-', linewidth=2.5, label='Dynamic (Ours)', zorder=10)
        
        # Plot Baselines
        for name, trace in baseline_traces.items():
            trace_wh = np.array(trace) / 3600.0
            style = '--'
            if "Naive" in name: color = 'purple'; alpha=0.8
            elif "Worst" in name: color = 'red'; alpha=0.5
            elif "Throughput" in name: color = 'orange'; alpha=0.6
            elif "Efficiency" in name: color = 'blue'; alpha=0.6
            else: color='gray'; alpha=0.5
            ax1.plot(x, trace_wh, linestyle=style, color=color, alpha=alpha, label=name)
            
        ax1.set_ylabel('Battery (Wh)', fontsize=14)
        ax1.set_title(f'Power Management Trace', fontsize=16)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Solar/Eclipse
        if 'is_eclipse' in self.df.columns:
            mask = self.df['is_eclipse'].values.astype(float)
            width = np.diff(x, append=x.iloc[-1])
            width[-1] = width[-2] if len(width) > 1 else 1.0
            for ax in [ax1, ax2, ax3]:
                ax.bar(x, mask * ax.get_ylim()[1], width=width, color='gray', alpha=0.15, align='edge', label='Eclipse')

        # Model Strip
        unique_models = sorted([m for m in self.df['dynamic_model'].unique() if m not in ['None', 'Blind']])
        if unique_models:
            cmap = plt.get_cmap('tab20')
            color_dict = {model: cmap(i/len(unique_models)) for i, model in enumerate(unique_models)}
        else: color_dict = {}
        color_dict['None'] = 'white'; color_dict['Blind'] = 'lightgray'
        
        orbit_colors = [color_dict.get(m, 'lightgray') for m in self.df['dynamic_model']]
        width = np.diff(x, append=x.iloc[-1]); width[-1] = width[-2] if len(width) > 1 else 1.0
        ax2.bar(x, np.ones(len(x)), width=width, color=orbit_colors, align='edge')
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_dict[m], label=m) for m in unique_models]
        legend_elements.append(Patch(facecolor='lightgray', label='Idle/Blind'))
        ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.5))
        ax2.set_ylabel('Active Model')

        # Throughput
        ax3.plot(x, self.df['n_inferences_req'], 'k:', linewidth=2, label='Demand')
        ax3.plot(x, self.df['dynamic_raw_inf'], 'g', linewidth=2, label='Delivered')
        ax3.set_yscale('log'); ax3.set_ylim(bottom=1)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "sequential_battery_sim.png", dpi=300)

if __name__ == "__main__":
    from libs.coral_tpu_characterization.src.scripts.utils.path_utils import get_repo_root
    REPO_ROOT = get_repo_root()

    sim = SatelliteInferenceSim(
        orbit_data_path=REPO_ROOT / "data/stk", 
        json_dir=REPO_ROOT / "data/tpunet_acc",
        saleae_root= "results/captures_1_20", 
        output_dir=REPO_ROOT / "results/final_analysis"
    )
    sim.run_dynamic_optimization()
    sim.generate_report()