import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import ground_track_stk as orb
import tpunet_plotting as mdl

class SatelliteInferenceSim:
    DEFAULT_CONFIG = {
        'fov': 2.0,              # Sensor FOV (deg)
        'target_patch_km': 5.0,  # Constant Ground Feature Size (km)
        'sun_power_mw': 2500.0,  # Power available from Solar (mW)
        'eclipse_power_mw': 150.0, # Power available from Battery (mW)
        'tpu_dim': 224,          # Pixel input size for TPU
        'sensor_res': 4096,      # Camera Resolution (px)
        'min_pixels': 100,        # Minimum pixels to attempt detection
        'solar_margin': 0.90,    # Safety margin for power fluctuations
        
        # New Hysteresis / Switching Parameters
        # TODO: Get some lab measurements for these once the RTOS app is up and running
        'switch_latency_s': 0.0, # Time cost to load new weights 
        'switch_energy_mj': 0.0 # Energy cost to load new weights
    }

    def __init__(self, orbit_data_path, model_json_dir, saleae_root, output_dir, config=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = self.DEFAULT_CONFIG.copy()
        if config: self.config.update(config)

        print("--- Loading Data ---")
        self.models = self._load_models(model_json_dir, saleae_root)
        self.df = self._load_orbit(orbit_data_path)
        
        print("--- Calculating Resource Budgets ---")
        self.df = self.calculate_physics_and_budgets(self.df)

    def _load_models(self, json_dir, saleae_root):
        plotter = mdl.GridStatsPlotting(json_dir, saleae_root, self.output_dir)
        df = plotter.load_and_aggregate_data()
        if df is None or df.empty: raise ValueError("No model data found.")
        
        # Standardize units
        df['acc_decimal'] = df['Top-1 Accuracy'] / 100.0
        df['lat_sec'] = df['Measured Inference Time (ms)'] / 1000.0
        return df

    def _load_orbit(self, data_path):
        root = Path(data_path)
        v = pd.read_csv(root / 'HEO_Sat_Fixed_Position_Velocity.csv')
        k = pd.read_csv(root / 'HEO_Sat_Classical_Orbit_Elements.csv')
        l = pd.read_csv(root / 'HEO_Sat_LLA_Position.csv')
        df = v.merge(k, on="Time (UTCG)").merge(l, on="Time (UTCG)")
        df.columns = df.columns.str.strip()
        return df

    # ==========================================
    # ORBITAL PHYSICS & RESOURCE BUDGETS
    # ==========================================

    def calculate_physics_and_budgets(self, df):
        df = self._calc_geometry(df)
        df = self._calc_dynamic_workload(df)
        df = self._calc_power_limits(df)
        return df

    def _calc_geometry(self, df):
        # Calculate how fast ground passes under the sensor
        fov_rad = np.deg2rad(self.config['fov'])
        df['v_ground'] = np.sqrt(df['vx (km/sec)']**2 + df['vy (km/sec)']**2 + df['vz (km/sec)']**2)
        
        # Calculate the physical width of the view
        df['swath_km'] = 2 * df['Alt (km)'] * np.tan(fov_rad / 2)
        
        # Time budget: How long we can dwell on this frame before it's gone
        df['budget_time_s'] = df['swath_km'] / df['v_ground']
        return df

    def _calc_dynamic_workload(self, df):
        target_km = self.config['target_patch_km']
        res = self.config['sensor_res']
        tpu_dim = self.config['tpu_dim']

        # Determine Pixel Density
        df['px_per_patch'] = (res / df['swath_km']) * target_km
        
        # Determine Regime: Are we upscaling (Tiling) or downscaling (Aggregation)?
        # Ratio > 1.0 means tiling. Ratio < 1.0 means aggregation.
        df['infs_per_patch_raw'] = (df['px_per_patch'] / tpu_dim)**2
        
        # Check Blindness: If features are too small (e.g. 5px), we can't see them.
        df['infs_per_patch'] = np.where(df['px_per_patch'] < self.config['min_pixels'], 
                                        0.0, 
                                        df['infs_per_patch_raw'])
        
        # Calculate Total TPU Operations needed for the frame
        patches_in_view = (df['swath_km']**2) / (target_km**2)
        
        # We must execute an integer number of inferences
        df['n_inferences_req'] = np.ceil(patches_in_view * df['infs_per_patch'])
        
        return df

    def _calc_power_limits(self, df):
        # TODO: This seems to reset every frame, need to track battery over time - swap to some sort of mW/sec during solar gained instead of static buffer
        # Identify Eclipse (Apogee region for this orbit)
        df['is_eclipse'] = (df['True Anomaly (deg)'] >= 180) & (df['True Anomaly (deg)'] <= 360)
        
        # Set Power Limits (Solar vs Battery)
        solar_avail = self.config['sun_power_mw'] * self.config['solar_margin']
        batt_avail  = self.config['eclipse_power_mw']
        df['power_limit_mw'] = np.where(df['is_eclipse'], batt_avail, solar_avail)
        
        # Total Energy Budget for this frame (Power * Time)
        df['budget_energy_mj'] = df['power_limit_mw'] * df['budget_time_s']
        return df

    # ==========================================
    # DECISION TREE
    # ==========================================

    def _evaluate_model_at_step(self, limit_t, limit_e, req_inf, lat, eng, acc):
        """
        Calculates how many accurate inferences a model can perform 
        given specific Time and Energy limits.
        """
        if req_inf <= 0: return 0, 0, "Blind"

        # You can't run half an inference.
        cap_time = np.floor(limit_t / lat)
        cap_eng  = np.floor(limit_e / eng)
        
        # Actual inferences run is limited by Capacity AND Requirement
        # We don't run more than the job requires.
        actual_runs = np.minimum(np.minimum(cap_time, cap_eng), req_inf)
        
        # Score Calculation
        score = actual_runs * acc
        
        # Status determination
        if actual_runs >= req_inf:
            status = "Success"
        elif actual_runs > 0:
            status = "Partial"
        else:
            status = "Fail"
            
        return score, actual_runs, status

    def run_optimization(self):
        print("--- Running Dynamic Optimization ---")
        
        # Extract model arrays for faster looping
        model_names = self.models['Model name'].values
        m_lat = self.models['lat_sec'].values
        m_eng = self.models['Energy per Inference (mJ)'].values
        m_acc = self.models['acc_decimal'].values

        results_best_model = []
        results_acc_inf = []
        results_raw_inf = [] 
        results_status = [] 
        
        # Track previous model for hysteresis costs
        prev_model_idx = -1 

        for idx, row in self.df.iterrows():
            limit_t = row['budget_time_s']
            limit_e = row['budget_energy_mj']
            req_inf = row['n_inferences_req']
            
            # Skip Blind frames immediately
            if req_inf <= 0:
                results_best_model.append("Blind")
                results_acc_inf.append(0)
                results_raw_inf.append(0)
                results_status.append("Blind")
                prev_model_idx = -1
                continue

            best_score = -1
            best_raw = 0
            best_idx = -1
            best_status = "Fail"
            
            # Competition: Check every model against the budget
            for i in range(len(model_names)):
                
                # Apply Switching Penalty if we changed models
                # We deduct the cost from the available budget *before* evaluation
                curr_limit_t = limit_t
                curr_limit_e = limit_e
                
                if prev_model_idx != -1 and i != prev_model_idx:
                    curr_limit_t -= self.config['switch_latency_s']
                    curr_limit_e -= self.config['switch_energy_mj']
                
                # Evaluate
                score, raw, status = self._evaluate_model_at_step(
                    curr_limit_t, curr_limit_e, req_inf, 
                    m_lat[i], m_eng[i], m_acc[i]
                )
                
                # Selection: Pure Score Optimization
                # We pick the model that yields the most correct answers, period.
                # (Even if it didn't finish the whole frame)
                if score > best_score:
                    best_score = score
                    best_raw = raw
                    best_idx = i
                    best_status = status
            
            # Record the winner
            if best_idx != -1:
                results_best_model.append(model_names[best_idx])
                prev_model_idx = best_idx
            else:
                results_best_model.append("None")
                prev_model_idx = -1
                
            results_acc_inf.append(best_score)
            results_raw_inf.append(best_raw)
            results_status.append(best_status)

        self.df['selected_model'] = results_best_model
        self.df['dynamic_inferences'] = results_acc_inf
        self.df['raw_inferences'] = results_raw_inf
        self.df['status'] = results_status
        return self.df

    def run_baseline_comparison(self):
        print("--- Running Baseline Comparison ---")
        
        # Identify Baseline Indices
        idx_worst = self.models['Correct_Inf_per_Joule'].idxmin()
        idx_fastest_raw = self.models['lat_sec'].idxmin()
        idx_throughput = self.models['Correct_Inf_per_Sec'].idxmax()
        idx_efficiency = self.models['Correct_Inf_per_Joule'].idxmax()
        
        baselines = {
            "Static Worst": idx_worst,
            "Static Fastest (Raw)": idx_fastest_raw,
            "Static Best Throughput": idx_throughput,
            "Static Best Efficiency": idx_efficiency
        }
        # TODO: Print all these baseline model names
        
        results = {}
        for name, idx in baselines.items():
            total_inf = 0
            lat = self.models.at[idx, 'lat_sec']
            eng = self.models.at[idx, 'Energy per Inference (mJ)']
            acc = self.models.at[idx, 'acc_decimal']
            
            # Run simulation without switching costs (static = no switching)
            for _, row in self.df.iterrows():
                inf, _, _ = self._evaluate_model_at_step(
                    row['budget_time_s'], 
                    row['budget_energy_mj'], 
                    row['n_inferences_req'],
                    lat, eng, acc
                )
                total_inf += inf
            results[name] = total_inf
            
        return results

    # ==========================================
    # 3. REPORTING & PLOTS
    # ==========================================

    def generate_report(self):
        print("\n" + "="*80)
        print(f"{'SYSTEM PERFORMANCE REPORT':^80}")
        print("="*80)

        # --- ORBIT STATS ---
        print(f"\n{' ORBIT STATISTICS ':~^80}")
        px_stats = self.df['px_per_patch'].describe()
        print(f"\n{'> Pixel Density (Pixels per Patch)':<40}")
        print(f"  Max: {px_stats['max']:>10.2f} px")
        print(f"  Min: {px_stats['min']:>10.2f} px")
        print(f"  Avg: {px_stats['mean']:>10.2f} px")

        idx_max_time = self.df['budget_time_s'].idxmax()
        idx_min_time = self.df['budget_time_s'].idxmin()

        print(f"\n{'> Frame Workloads':<40}")
        print(f"  Longest Dwell: {self.df.at[idx_max_time, 'budget_time_s']:>10.2f} s")
        print(f"     -> Load: {self.df.at[idx_max_time, 'n_inferences_req']:>10.0f} tiles")
        print(f"  Shortest Dwell: {self.df.at[idx_min_time, 'budget_time_s']:>10.2f} s")
        print(f"     -> Load: {self.df.at[idx_min_time, 'n_inferences_req']:>10.0f} tiles")

        # --- PERFORMANCE ---
        print(f"\n{' PERFORMANCE COMPARISON ':~^80}")
        dyn_total = self.df['dynamic_inferences'].sum()
        baselines = self.run_baseline_comparison()
        
        print(f"\n{'Strategy':<35} | {'Total Acc. Inferences':<25} | {'Improvement'}")
        print("-" * 75)
        
        best_static_score = max(baselines.values())
        for name, score in baselines.items():
            print(f"{name:<35} | {score:,.0f}")
        print("-" * 75)
        
        imp_pct = ((dyn_total - best_static_score) / best_static_score) * 100 if best_static_score > 0 else 0.0
        print(f"{'Dynamic Switching (Ours)':<35} | {dyn_total:,.0f}{'':<14} | +{imp_pct:.2f}%")
        
        # --- USAGE ---
        print(f"\n{' DYNAMIC MODEL BREAKDOWN ':~^80}")
        used_df = self.df[self.df['status'].isin(['Success', 'Partial'])].copy()
        
        if used_df.empty:
            print("No inferences performed.")
        else:
            stats = used_df.groupby('selected_model').agg(
                raw_count=('raw_inferences', 'sum'),
            ).reset_index()

            stats = stats.merge(self.models[['Model name', 'Energy per Inference (mJ)']], 
                                left_on='selected_model', right_on='Model name', how='left')
            
            stats['total_energy_j'] = (stats['raw_count'] * stats['Energy per Inference (mJ)']) / 1000.0
            total_raw = stats['raw_count'].sum()
            stats['pct_usage'] = (stats['raw_count'] / total_raw) * 100
            
            print(f"\n{'Model Name':<30} | {'% of Infs':<10} | {'Count (Raw)':<12} | {'Energy (J)':<12}")
            print("-" * 75)
            for _, row in stats.iterrows():
                print(f"{row['selected_model']:<30} | {row['pct_usage']:>9.1f}% | {int(row['raw_count']):>12,} | {row['total_energy_j']:>12.2f}")
            print("-" * 75)

        self.plot_results()

    def plot_results(self):
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        x = self.df['True Anomaly (deg)']
        
        # Workload
        ax1.plot(x, self.df['n_inferences_req'], color='blue', label='Required Inferences')
        ax1.set_yscale('log')
        ax1.set_ylabel('Inferences Required (Log Scale)')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(x, self.df['budget_energy_mj'], color='orange', linestyle='--', label='Energy Budget')
        ax1_twin.set_ylabel('Energy Budget (mJ)', color='orange')
        
        # Performance
        unique_models = [m for m in self.df['selected_model'].unique() if m not in ["None", "Blind"]]
        if unique_models:
            model_map = {name: i for i, name in enumerate(unique_models)}
            colors = [model_map.get(m, -1) for m in self.df['selected_model']]
            
            mask_succ = (self.df['status'] == 'Success')
            if mask_succ.any():
                ax2.scatter(x[mask_succ], self.df.loc[mask_succ, 'dynamic_inferences'], 
                           c=[colors[i] for i in np.where(mask_succ)[0]], cmap='tab10', marker='o')
            
            mask_part = (self.df['status'] == 'Partial')
            if mask_part.any():
                ax2.scatter(x[mask_part], self.df.loc[mask_part, 'dynamic_inferences'], 
                           c=[colors[i] for i in np.where(mask_part)[0]], cmap='tab10', marker='x')

            from matplotlib.patches import Patch
            cmap = plt.get_cmap('tab10')
            legend_elements = [Patch(facecolor=cmap(model_map[m]), label=m) for m in unique_models]
            ax2.legend(handles=legend_elements, title="Active Model", loc='upper right')

        ax2.set_ylabel('Accurate Inferences')
        ax2.set_xlabel('True Anomaly (deg)')
        ax2.set_title('Dynamic Performance')
        ax2.axvspan(180, 360, color='gray', alpha=0.1, label='Eclipse')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "final_system_performance.png")
        print(f"Plot saved to {self.output_dir / 'final_system_performance.png'}")

if __name__ == "__main__":
    sim_config = {
        'target_patch_km': 2.0,     
        'sun_power_mw': 1000.0,     
        'eclipse_power_mw': 500.0,  
        'fov': 2.0,
        'min_pixels': 10            
    }

    sim = SatelliteInferenceSim(
        orbit_data_path="libs/coral_tpu_characterization/data/stk", 
        model_json_dir="libs/coral_tpu_characterization/data/tpunet_acc", 
        saleae_root="results/captures_1_20", 
        output_dir="results/final_analysis",
        config=sim_config
    )
    
    sim.run_optimization()
    sim.generate_report()