import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- NEW IMPORT ---
from model_data_manager import ModelDataManager 

class SatelliteInferenceSim:
    DEFAULT_CONFIG = {
        'fov': 2.0,                # Sensor FOV (deg)
        'target_patch_km': 5.0,    # Ground feature size
        'tpu_dim': 224,            # TPU Input size
        'sensor_res': 4096,        # Camera Res
        'min_pixels': 10,          # Blindness threshold
        
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

        # Convert Wh to Joules for internal math (1 Wh = 3600 J)
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
        df['lat_sec'] = df['Measured Inference Time (ms)'] / 1000.0
        
        print(f"Successfully loaded {len(df)} models.")
        return df

    def _load_orbit(self, data_path):
        root = Path(data_path)
        v = pd.read_csv(root / 'HEO_Sat_Fixed_Position_Velocity.csv')
        k = pd.read_csv(root / 'HEO_Sat_Classical_Orbit_Elements.csv')
        l = pd.read_csv(root / 'HEO_Sat_LLA_Position.csv')
        
        # Merge on Time
        df = v.merge(k, on="Time (UTCG)").merge(l, on="Time (UTCG)")
        df.columns = df.columns.str.strip()

        # --- SLICING LOGIC ---
        # 1. Ensure True Anomaly is in [0, 360] (usually is, but good to check)
        # 2. Find the first index where the orbit "resets" (wraps from 360 -> 0)
        #    We look for a large negative derivative in True Anomaly.
        
        ta = df['True Anomaly (deg)'].values
        # Calculate difference between consecutive steps
        diffs = np.diff(ta)
        
        # A wrap-around looks like a jump of ~ -360 degrees (e.g., 359 -> 1)
        # We find indices where the drop is significant (e.g., < -300)
        wrap_indices = np.where(diffs < -300)[0]
        
        if len(wrap_indices) > 0:
            # The first wrap happens at wrap_indices[0]. 
            # We want to include that point (end of orbit) but stop before the next start.
            # np.diff indices are 'i' vs 'i+1', so the cutoff is 'i+1'
            cutoff_idx = wrap_indices[0] + 1
            print(f"[INFO] Multi-orbit data detected. Slicing to first orbit (Indices 0 to {cutoff_idx}).")
            df = df.iloc[:cutoff_idx].copy()
        else:
            print("[INFO] Single orbit (or partial) detected. Using full dataset.")
        
        return df

    # ==========================================
    # 1. GEOMETRY & WORKLOAD (Stateless)
    # ==========================================
    def calculate_geometry_and_workload(self, df):
        # Physics
        fov_rad = np.deg2rad(self.config['fov'])
        df['v_ground'] = np.sqrt(df['vx (km/sec)']**2 + df['vy (km/sec)']**2 + df['vz (km/sec)']**2)
        df['swath_km'] = 2 * df['Alt (km)'] * np.tan(fov_rad / 2)
        df['dwell_time_s'] = df['swath_km'] / df['v_ground']
        
        # Workload
        target_km = self.config['target_patch_km']
        res = self.config['sensor_res']
        tpu_dim = self.config['tpu_dim']
        
        df['px_per_patch'] = (res / df['swath_km']) * target_km
        df['infs_per_patch'] = np.where(df['px_per_patch'] < self.config['min_pixels'], 
                                        0.0, 
                                        (df['px_per_patch'] / tpu_dim)**2)
        
        patches_in_view = (df['swath_km']**2) / (target_km**2)
        df['n_inferences_req'] = np.ceil(patches_in_view * df['infs_per_patch'])
        
        # Power Generation
        df['is_eclipse'] = (df['True Anomaly (deg)'] >= 180) & (df['True Anomaly (deg)'] <= 360)
        df['gen_rate_mw'] = np.where(df['is_eclipse'], 0.0, self.config['solar_generation_mw'])
        df['energy_harvested_j'] = (df['gen_rate_mw'] * df['dwell_time_s']) / 1000.0
        df['energy_baseload_j'] = (self.config['system_baseload_mw'] * df['dwell_time_s']) / 1000.0

        return df

    # ==========================================
    # 2. SEQUENTIAL SIMULATION ENGINE
    # ==========================================
    
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
        
        res_model = []
        res_acc = []
        res_raw = []        # Track Raw Inferences
        res_energy = []     # Track Energy Consumed
        res_status = []
        res_battery = []
        
        prev_model_idx = -1
        model_names = self.models['Model name'].values
        m_lat = self.models['lat_sec'].values
        m_eng = self.models['Energy per Inference (mJ)'].values
        m_acc = self.models['acc_decimal'].values

        for idx, row in self.df.iterrows():
            harvest = row['energy_harvested_j']
            baseload = row['energy_baseload_j']
            dwell = row['dwell_time_s']
            req = row['n_inferences_req']
            
            best_score = -1
            best_raw = 0
            best_idx = -1
            best_status = "Fail"
            best_consumed_j = baseload 
            
            for i in range(len(model_names)):
                sw_j = 0
                sw_s = 0
                if prev_model_idx != -1 and i != prev_model_idx:
                    sw_j = self.config['switch_energy_mj'] / 1000.0
                    sw_s = self.config['switch_latency_s']

                score, raw, cons_j, status = self._evaluate_step_sequential(
                    curr_battery_j, harvest, baseload, dwell, req,
                    m_lat[i], m_eng[i], m_acc[i], sw_j, sw_s
                )
                
                if score > best_score:
                    best_score = score
                    best_raw = raw
                    best_idx = i
                    best_status = status
                    best_consumed_j = cons_j
            
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
            
            curr_battery_j = np.clip(curr_battery_j + harvest - best_consumed_j, 
                                    0, self.BATTERY_CAPACITY_J)
            res_battery.append(curr_battery_j)

        self.df['dynamic_model'] = res_model
        self.df['dynamic_acc_inf'] = res_acc
        self.df['dynamic_raw_inf'] = res_raw # NEW: Store raw counts
        self.df['dynamic_energy_j'] = res_energy # NEW: Store energy
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
            score, _, cons_j, status = self._evaluate_step_sequential(
                curr_battery_j, 
                row['energy_harvested_j'], 
                row['energy_baseload_j'], 
                row['dwell_time_s'], 
                row['n_inferences_req'],
                lat, eng, acc, 
                0, 0 
            )
            total_acc_inf += score
            curr_battery_j = np.clip(curr_battery_j + row['energy_harvested_j'] - cons_j, 
                                    0, self.BATTERY_CAPACITY_J)
            battery_trace.append(curr_battery_j)
            
        return total_acc_inf, battery_trace

    # ==========================================
    # 3. REPORTING
    # ==========================================

    def generate_report(self):
        print("\n" + "="*95)
        print(f"{'SEQUENTIAL BATTERY SIMULATION REPORT':^95}")
        print("="*95)

        # --- 1. Pixel Density & Workloads ---
        px_min = self.df['px_per_patch'].min()
        px_max = self.df['px_per_patch'].max()
        px_avg = self.df['px_per_patch'].mean()
        
        # Identify longest/shortest dwells
        row_max_dwell = self.df.loc[self.df['dwell_time_s'].idxmax()]
        row_min_dwell = self.df.loc[self.df['dwell_time_s'].idxmin()]
        
        print(f"\n> Pixel Density (Pixels per Patch)")
        print(f"  Max:      {px_max:>10.2f} px")
        print(f"  Min:      {px_min:>10.2f} px")
        print(f"  Avg:      {px_avg:>10.2f} px")
        
        print(f"\n> Frame Workloads")
        print(f"  Longest Dwell:     {row_max_dwell['dwell_time_s']:>10.2f} s")
        print(f"     -> Load:        {row_max_dwell['n_inferences_req']:>10.0f} tiles")
        print(f"  Shortest Dwell:    {row_min_dwell['dwell_time_s']:>10.2f} s")
        print(f"     -> Load:        {row_min_dwell['n_inferences_req']:>10.0f} tiles")

        # --- 2. Baselines & Performance Table ---
        # Ensure we don't crash if columns are missing, though ModelDataManager should provide them
        if 'Correct_Inf_per_Joule' not in self.models.columns:
            raise ValueError("Sanity Check Failed: 'Correct_Inf_per_Joule' missing from model dataframe.")

        idx_worst = self.models['Correct_Inf_per_Joule'].idxmin()
        idx_throughput = self.models['Correct_Inf_per_Sec'].idxmax()
        idx_efficiency = self.models['Correct_Inf_per_Joule'].idxmax()
        
        bl_configs = {
            "Static Worst": idx_worst,
            "Static Best Throughput": idx_throughput,
            "Static Best Efficiency": idx_efficiency
        }
        
        bl_results = {}
        bl_traces = {}
        
        print(f"\n{'~'*33} PERFORMANCE COMPARISON {'~'*34}")
        print("--- Running Baseline Comparison ---")
        print(f"\n{'Strategy':<35} | {'Total Acc. Inferences':<25} | {'Improvement'}")
        print("-" * 80)
        
        score_worst = 0
        score_best_static = 0
        
        # Run and Print Baselines
        for name, idx in bl_configs.items():
            score, trace = self.run_baseline_sequential(name, idx)
            bl_results[name] = score
            bl_traces[name] = trace
            
            if name == "Static Worst":
                score_worst = score
            
            if score > score_best_static:
                score_best_static = score
            
            print(f"{name:<35} | {score:,.0f}")
            
        print("-" * 80)
        
        # Print Dynamic
        dyn_total = self.df['dynamic_acc_inf'].sum()
        
        # Calculate Gains
        gain_vs_best = ((dyn_total - score_best_static) / score_best_static * 100) if score_best_static > 0 else 0
        gain_vs_worst = ((dyn_total - score_worst) / score_worst * 100) if score_worst > 0 else 0
        
        print(f"{'Dynamic Switching (Ours)':<35} | {dyn_total:,.0f}{'':<14} | +{gain_vs_best:.2f}% (vs Best)")
        print(f"{'':<35} | {'':<25} | +{gain_vs_worst:.2f}% (vs Worst)")

        # --- 3. Dynamic Model Breakdown ---
        print(f"\n{'~'*32} DYNAMIC MODEL BREAKDOWN {'~'*32}")
        
        # Header with Sanity Check Columns
        header = (f"{'Model Name':<28} | {'% Infs':<7} | {'Count':<9} | {'Energy (J)':<11} | "
                  f"{'C.Inf/s':<9} | {'C.Inf/J':<9}")
        print(header)
        print("-" * 95)
        
        # Filter out 'None' or 'Blind' for the stats table
        valid_df = self.df[~self.df['dynamic_model'].isin(['None', 'Blind'])]
        total_raw_infs = valid_df['dynamic_raw_inf'].sum()
        
        if total_raw_infs > 0:
            stats = valid_df.groupby('dynamic_model').agg({
                'dynamic_raw_inf': 'sum',
                'dynamic_energy_j': 'sum'
            }).sort_values('dynamic_raw_inf', ascending=True)

            for name, row in stats.iterrows():
                raw_count = row['dynamic_raw_inf']
                energy = row['dynamic_energy_j']
                pct = (raw_count / total_raw_infs) * 100
                
                # SANITY CHECK: Lookup static stats for this model from self.models
                model_meta = self.models[self.models['Model name'] == name]
                if not model_meta.empty:
                    c_inf_s = model_meta.iloc[0]['Correct_Inf_per_Sec']
                    c_inf_j = model_meta.iloc[0]['Correct_Inf_per_Joule']
                else:
                    c_inf_s = 0
                    c_inf_j = 0

                # Truncate very long model names for the table
                disp_name = (name[:25] + '..') if len(name) > 27 else name
                
                # Formatted Row
                row_str = (f"{disp_name:<28} | {pct:>6.1f}% | {raw_count:,.0f}".rjust(9) + 
                           f" | {energy:,.0f}".rjust(11) + 
                           f" | {c_inf_s:>9.1f} | {c_inf_j:>9.1f}")
                print(row_str)
        else:
            print("No inferences performed.")
            
        print("-" * 95)
        print("="*95)
        
        self.plot_results(bl_traces)

    def plot_results(self, baseline_traces):
        # Use a 3-row layout
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16), sharex=True, 
                                            gridspec_kw={'height_ratios': [2, 1, 2]})
        
        x = self.df['True Anomaly (deg)']
        
        # --- AX1: BATTERY STATE ---
        # Plot Dynamic (Ours) with a stronger line
        ax1.plot(x, self.df['dynamic_battery_j'] / 3600.0, 'g-', linewidth=2.5, label='Dynamic (Ours)', zorder=10)
        
        # Plot Baselines
        for name, trace in baseline_traces.items():
            trace_wh = np.array(trace) / 3600.0
            style = '--'
            if "Worst" in name: color = 'red'; alpha=0.5
            elif "Throughput" in name: color = 'orange'; alpha=0.6
            elif "Efficiency" in name: color = 'blue'; alpha=0.6
            else: color='gray'; alpha=0.5
            
            ax1.plot(x, trace_wh, linestyle=style, color=color, alpha=alpha, label=name)
            
        ax1.set_ylabel('Battery State (Wh)', fontsize=14)
        ax1.set_title(f'Power Management: Battery State over Orbit (Cap: {self.config["battery_capacity_wh"]} Wh)', fontsize=16)
        ax1.axhline(y=0, color='k', linewidth=1)
        ax1.legend(loc='upper right', fontsize=12, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Eclipse Shading (180 to 360 is standard for True Anomaly eclipse definition)
        ax1.axvspan(180, 360, color='gray', alpha=0.15, label='Eclipse')
        ax2.axvspan(180, 360, color='gray', alpha=0.15)
        ax3.axvspan(180, 360, color='gray', alpha=0.15)

        # --- AX2: MODEL SELECTION STRIP (Gantt Chart) ---
        # Filter out 'None' for the color map generation so we don't waste a color
        unique_models = sorted([m for m in self.df['dynamic_model'].unique() if m not in ['None', 'Blind']])
        
        # Create a consistent color map
        if unique_models:
            cmap = plt.get_cmap('tab20')
            color_dict = {model: cmap(i/len(unique_models)) for i, model in enumerate(unique_models)}
        else:
            color_dict = {}
            
        color_dict['None'] = 'white'
        color_dict['Blind'] = 'lightgray'
        
        # Create a list of colors for every point in the orbit
        orbit_colors = [color_dict.get(m, 'lightgray') for m in self.df['dynamic_model']]
        
        # Use bar plot with width equal to step size to create a continuous strip
        # Assuming uniform time steps is risky, so we use width=x_next - x_curr
        # For simplicity in this visualization, since points are dense, we can just use fill_between or bar
        # 'width' calculation:
        width = np.diff(x, append=x.iloc[-1])
        # Fix last point width
        width[-1] = width[-2] if len(width) > 1 else 1.0

        ax2.bar(x, np.ones(len(x)), width=width, color=orbit_colors, align='edge')
        
        # Create custom legend handles
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_dict[m], label=m) for m in unique_models]
        legend_elements.append(Patch(facecolor='lightgray', label='Idle/Blind'))
        
        # Legend outside right
        ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.5), 
                   fontsize=10, title="Active Model")
        
        ax2.set_ylabel('Active Model', fontsize=14)
        ax2.set_yticks([]) 
        ax2.set_title("Dynamic Model Switching Strategy", fontsize=16)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(0, 360) # Enforce 0-360 view

        # --- AX3: SUPPLY vs DEMAND ---
        # Plot Required Inferences (Demand)
        ax3.plot(x, self.df['n_inferences_req'], color='black', linestyle=':', linewidth=2, label='Demand (Tiles in View)', alpha=0.8)
        
        # Plot Delivered Inferences (Supply)
        ax3.plot(x, self.df['dynamic_raw_inf'], color='green', linewidth=2, label='Delivered (Raw)', alpha=0.9)
        
        # Fill gap for Unmet Demand
        ax3.fill_between(x, self.df['dynamic_raw_inf'], self.df['n_inferences_req'], 
                         where=(self.df['dynamic_raw_inf'] < self.df['n_inferences_req']),
                         color='red', alpha=0.3, label='Unmet Demand')

        ax3.set_ylabel('Inferences per Frame', fontsize=14)
        ax3.set_xlabel('True Anomaly (deg)', fontsize=14)
        ax3.set_title('Throughput: Demand vs. Delivered Performance', fontsize=16)
        ax3.legend(loc='upper right', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log') # Log scale is vital here
        ax3.set_ylim(bottom=1) # Log scale can't handle 0, clip bottom
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "sequential_battery_sim.png", dpi=300)
        print(f"Plot saved to {self.output_dir / 'sequential_battery_sim.png'}")

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