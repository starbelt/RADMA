import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- NEW IMPORT ---
from model_data_manager import ModelDataManager 

class SatelliteInferenceSim:
    DEFAULT_CONFIG = {
        'fov': 2.0,                # Sensor FOV (deg)
        'target_patch_km': 2.0,    # Ground feature size
        'tpu_dim': 224,            # TPU Input size
        'sensor_res': 4096,        # Camera Res
        'min_pixels': 20,          # Blindness threshold
        
        # Power / Battery
        'battery_capacity_wh': 2.0,    # ~7200 Joules
        'solar_generation_mw': 600.0,  # Small deployable or good body angle
        'system_baseload_mw': 300.0,   # Highly optimized microcontroller idle
        'initial_charge_pct': 1.0,
        
        # Switching
        'switch_latency_s': 0.0, 
        'switch_energy_mj': 0.0 
    }

    def __init__(self, orbit_data_path, excel_path, saleae_root, output_dir, config=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = self.DEFAULT_CONFIG.copy()
        if config: self.config.update(config)

        self.BATTERY_CAPACITY_J = self.config['battery_capacity_wh'] * 3600.0
        
        print("--- Loading Data ---")
        self.models = self._load_models(excel_path, saleae_root)
        self.df = self._load_orbit(orbit_data_path)
        
        print("--- Pre-calculating Geometry ---")
        self.df = self.calculate_geometry_and_workload(self.df)

    def _load_models(self, excel_path, saleae_root):
        print(f"Ingesting models from Excel: {excel_path}")
        print(f"Parsing Saleae CSVs from: {saleae_root}")
        
        manager = ModelDataManager(excel_path, saleae_root)
        df = manager.get_compiled_dataframe(sheet_name="Img_Class", run_names=None)
        
        if df is None or df.empty: 
            raise ValueError("No model data found.")
            
        acc_col = "Top-1 Accuracy (measured)" if "Top-1 Accuracy (measured)" in df.columns else "Top-1 Accuracy"
        
        df['acc_decimal'] = df[acc_col] / 100.0
        
        print(f"Successfully loaded {len(df)} models.")
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
        diffs = np.diff(ta)
        wrap_indices = np.where(diffs < -300)[0]
        
        if len(wrap_indices) > 0:
            cutoff_idx = wrap_indices[0] + 1
            print(f"[INFO] Multi-orbit data detected. Slicing to first orbit (Indices 0 to {cutoff_idx}).")
            df = df.iloc[:cutoff_idx].copy()
        else:
            print("[INFO] Single orbit (or partial) detected. Using full dataset.")
        
        return df

    def calculate_geometry_and_workload(self, df):
        fov_rad = np.deg2rad(self.config['fov'])
        df['v_ground'] = np.sqrt(df['vx (km/sec)']**2 + df['vy (km/sec)']**2 + df['vz (km/sec)']**2)
        df['swath_km'] = 2 * df['Alt (km)'] * np.tan(fov_rad / 2)
        df['dwell_time_s'] = df['swath_km'] / df['v_ground']
        
        target_km = self.config['target_patch_km']
        res = self.config['sensor_res']
        tpu_dim = self.config['tpu_dim']
        
        # "Smart" Workload calculation (accounts for blindness)
        df['px_per_patch'] = (res / df['swath_km']) * target_km
        df['infs_per_patch'] = np.where(df['px_per_patch'] < self.config['min_pixels'], 
                                        0.0, 
                                        (df['px_per_patch'] / tpu_dim)**2)
        
        patches_in_view = (df['swath_km']**2) / (target_km**2)
        df['n_inferences_req'] = np.ceil(patches_in_view * df['infs_per_patch'])
        
        # Solar Logic
        if 'Solar Intensity' in df.columns:
            df['solar_factor'] = df['Solar Intensity'].fillna(1.0)
            df['is_eclipse'] = df['solar_factor'] < 0.1
        else:
            print("[INFO] No Solar Intensity data found. Assuming 100% Sunlight (HEO).")
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
                sw_j = (self.config['switch_energy_mj'] / 1000.0) if (prev_model_idx != -1 and i != prev_model_idx) else 0
                sw_s = self.config['switch_latency_s'] if (prev_model_idx != -1 and i != prev_model_idx) else 0

                score, raw, cons_j, status = self._evaluate_step_sequential(
                    curr_battery_j, harvest, baseload, dwell, req,
                    m_lat[i], m_eng[i], m_acc[i], sw_j, sw_s
                )
                
                if score > best_score:
                    best_score, best_raw, best_idx = score, raw, i
                    best_status, best_consumed_j = status, cons_j
            
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
        total_acc_inf = 0
        battery_trace = []
        
        lat = self.models.at[model_idx, 'lat_sec']
        eng = self.models.at[model_idx, 'Energy per Inference (mJ)']
        acc = self.models.at[model_idx, 'acc_decimal']
        
        for _, row in self.df.iterrows():
            score, _, cons_j, _ = self._evaluate_step_sequential(
                curr_battery_j, row['energy_harvested_j'], row['energy_baseload_j'], row['dwell_time_s'], row['n_inferences_req'],
                lat, eng, acc, 0, 0 
            )
            total_acc_inf += score
            curr_battery_j = np.clip(curr_battery_j + row['energy_harvested_j'] - cons_j, 0, self.BATTERY_CAPACITY_J)
            battery_trace.append(curr_battery_j)
            
        return total_acc_inf, battery_trace

    # --- NEW NAIVE BASELINE ---
    def run_naive_baseline(self, model_idx):
        """
        Naive = Tiling the full sensor frame regardless of altitude/blindness.
        """
        curr_battery_j = self.BATTERY_CAPACITY_J * self.config['initial_charge_pct']
        total_acc_inf = 0
        battery_trace = []
        
        lat = self.models.at[model_idx, 'lat_sec']
        eng = self.models.at[model_idx, 'Energy per Inference (mJ)']
        acc = self.models.at[model_idx, 'acc_decimal']
        
        # Constant Naive Workload: (Sensor / TPU)^2
        # e.g., (4096 / 224)^2 = ~334 tiles
        tiles_naive = (self.config['sensor_res'] / self.config['tpu_dim'])**2
        
        for _, row in self.df.iterrows():
            # Naive Demand is ALWAYS tiles_naive
            # But "Accurate Inferences" are only possible if we aren't legally blind
            is_actually_blind = row['n_inferences_req'] == 0
            
            # Use the sequential stepper to calculate energy consumption
            # We pass tiles_naive as the requirement
            _, actual_runs, cons_j, status = self._evaluate_step_sequential(
                curr_battery_j, row['energy_harvested_j'], row['energy_baseload_j'], row['dwell_time_s'], 
                tiles_naive, # FORCE NAIVE DEMAND
                lat, eng, acc, 0, 0 
            )
            
            # Logic: We burned the power to do 'actual_runs'. 
            # If we were blind, those runs yielded 0 value.
            # If we weren't blind, they yielded 'actual_runs * acc' value.
            if is_actually_blind:
                step_score = 0
            else:
                step_score = actual_runs * acc
                
            total_acc_inf += step_score
            curr_battery_j = np.clip(curr_battery_j + row['energy_harvested_j'] - cons_j, 0, self.BATTERY_CAPACITY_J)
            battery_trace.append(curr_battery_j)
            
        return total_acc_inf, battery_trace

    def generate_report(self):
        print("\n" + "="*95)
        print(f"{'SEQUENTIAL BATTERY SIMULATION REPORT':^95}")
        print("="*95)

        # Baseline definitions
        idx_worst = self.models['Correct_Inf_per_Joule'].idxmin()
        idx_throughput = self.models['Correct_Inf_per_Sec'].idxmax()
        idx_efficiency = self.models['Correct_Inf_per_Joule'].idxmax()
        
        bl_configs = {
            "Naive (Worst Model)": (idx_worst, True),   # (Index, IsNaive)
            "Static Worst (Smart)": (idx_worst, False),
            "Static Throughput (Smart)": (idx_throughput, False),
            "Static Efficiency (Smart)": (idx_efficiency, False)
        }
        
        bl_results = {}
        bl_traces = {}
        
        print(f"\n{'~'*33} PERFORMANCE COMPARISON {'~'*34}")
        print(f"\n{'Strategy':<35} | {'Total Acc. Inferences':<25} | {'Improvement'}")
        print("-" * 80)
        
        score_worst_naive = 0
        score_best_static = 0
        
        for name, (idx, is_naive) in bl_configs.items():
            if is_naive:
                score, trace = self.run_naive_baseline(idx)
                if name == "Naive (Worst Model)": score_worst_naive = score
            else:
                score, trace = self.run_baseline_sequential(name, idx)
                if score > score_best_static: score_best_static = score

            bl_results[name] = score
            bl_traces[name] = trace
            print(f"{name:<35} | {score:,.0f}")
            
        print("-" * 80)
        
        dyn_total = self.df['dynamic_acc_inf'].sum()
        gain_vs_best = ((dyn_total - score_best_static) / score_best_static * 100) if score_best_static > 0 else 0
        gain_vs_naive = ((dyn_total - score_worst_naive) / score_worst_naive * 100) if score_worst_naive > 0 else 0
        
        print(f"{'Dynamic Switching (Ours)':<35} | {dyn_total:,.0f}{'':<14} | +{gain_vs_best:.2f}% (vs Best Static)")
        print(f"{'':<35} | {'':<25} | +{gain_vs_naive:.2f}% (vs Naive)")
        print("="*95)
        
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
    from path_utils import get_repo_root
    REPO_ROOT = get_repo_root()

    sim = SatelliteInferenceSim(
        orbit_data_path=REPO_ROOT / "data/stk", 
        excel_path=REPO_ROOT / "data/Model_Stats.xlsx",
        saleae_root=REPO_ROOT / "results/captures/IMG_CLASS02", 
        output_dir=REPO_ROOT / "results/final_analysis"
    )
    sim.run_dynamic_optimization()
    sim.generate_report()