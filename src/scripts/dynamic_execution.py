import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import ground_track_stk as orb
import tpunet_plotting as mdl

class SatelliteInferenceSim:
    DEFAULT_CONFIG = {
        'fov': 2.0,                # Sensor FOV (deg)
        'target_patch_km': 5.0,    # Ground feature size
        'tpu_dim': 224,            # TPU Input size
        'sensor_res': 4096,        # Camera Res
        'min_pixels': 10,          # Blindness threshold
        
        # Make real values here
        'battery_capacity_wh': 100,  # Super-Cap Capacity in Watt-Hours
        'solar_generation_mw': 100.0, # Rate of energy harvesting in Sun
        'system_baseload_mw': 100.0,  # Constant power draw (radio/heater/computer idle)
        'initial_charge_pct': 1.0,   # Start simulation at 100% battery
        
        # Switching
        'switch_latency_s': 0.0, 
        'switch_energy_mj': 0.0 
    }

    def __init__(self, orbit_data_path, model_json_dir, saleae_root, output_dir, config=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = self.DEFAULT_CONFIG.copy()
        if config: self.config.update(config)

        # Convert Wh to Joules for internal math (1 Wh = 3600 J)
        self.BATTERY_CAPACITY_J = self.config['battery_capacity_wh'] * 3600.0
        
        print("--- Loading Data ---")
        self.models = self._load_models(model_json_dir, saleae_root)
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
        df.columns = df.columns.str.strip()
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
        
        # Workload (Dynamic Tiling)
        target_km = self.config['target_patch_km']
        res = self.config['sensor_res']
        tpu_dim = self.config['tpu_dim']
        
        df['px_per_patch'] = (res / df['swath_km']) * target_km
        df['infs_per_patch'] = np.where(df['px_per_patch'] < self.config['min_pixels'], 
                                        0.0, 
                                        (df['px_per_patch'] / tpu_dim)**2)
        
        patches_in_view = (df['swath_km']**2) / (target_km**2)
        df['n_inferences_req'] = np.ceil(patches_in_view * df['infs_per_patch'])
        
        # Power Generation Potential (Not Budget yet)
        # Eclipse is 180-360 deg
        df['is_eclipse'] = (df['True Anomaly (deg)'] >= 180) & (df['True Anomaly (deg)'] <= 360)
        
        # Generation Rate (mW)
        df['gen_rate_mw'] = np.where(df['is_eclipse'], 0.0, self.config['solar_generation_mw'])
        
        # Energy In per Frame (Joules) = Rate * Time / 1000
        df['energy_harvested_j'] = (df['gen_rate_mw'] * df['dwell_time_s']) / 1000.0
        
        # Baseload Energy Cost per Frame (Joules)
        df['energy_baseload_j'] = (self.config['system_baseload_mw'] * df['dwell_time_s']) / 1000.0

        return df

    # ==========================================
    # 2. SEQUENTIAL SIMULATION ENGINE
    # ==========================================
    
    def _evaluate_step_sequential(self, current_battery_j, harvested_j, baseload_j, dwell_time_s, req_inf, 
                                  lat, eng, acc, switching_penalty_j=0, switching_penalty_s=0):
        """
        Determines if a specific model can run given the CURRENT battery state.
        Returns: (Score, Raw_Inferences, Energy_Consumed_J, Status)
        """
        # 1. Blind check
        if req_inf <= 0:
            # We still pay baseload!
            return 0, 0, baseload_j, "Blind"

        # 2. Net Available Energy
        # We start with battery + what we harvest during the frame
        # We must subtract baseload and switching costs first
        available_energy_j = current_battery_j + harvested_j - baseload_j - switching_penalty_j
        
        # 3. Available Time
        available_time_s = dwell_time_s - switching_penalty_s

        # If baseload killed the battery, we are dead.
        if available_energy_j <= 0:
            return 0, 0, current_battery_j + harvested_j, "DeadBattery"

        # 4. Capacity Calculation
        # How many inferences fit in the Remaining Energy?
        # Energy per inf is in mJ, convert to J
        eng_j = eng / 1000.0
        cap_eng = np.floor(available_energy_j / eng_j)
        
        # How many fit in Time?
        cap_time = np.floor(available_time_s / lat)
        
        # 5. Actual execution
        actual_runs = np.minimum(np.minimum(cap_eng, cap_time), req_inf)
        
        # 6. Total Consumption
        # Baseload + Switch + (Inferences * EnergyPerInf)
        total_consumed_j = baseload_j + switching_penalty_j + (actual_runs * eng_j)
        
        score = actual_runs * acc
        
        if actual_runs >= req_inf: status = "Success"
        elif actual_runs > 0:      status = "Partial"
        else:                      status = "Fail"
            
        return score, actual_runs, total_consumed_j, status

    def run_dynamic_optimization(self):
        print("--- Running Sequential Dynamic Optimization ---")
        
        # State Initialization
        curr_battery_j = self.BATTERY_CAPACITY_J * self.config['initial_charge_pct']
        
        # Results containers
        res_model = []
        res_acc = []
        res_status = []
        res_battery = [] # Track SoC over time
        
        prev_model_idx = -1
        model_names = self.models['Model name'].values
        m_lat = self.models['lat_sec'].values
        m_eng = self.models['Energy per Inference (mJ)'].values
        m_acc = self.models['acc_decimal'].values

        # SEQUENTIAL LOOP
        for idx, row in self.df.iterrows():
            
            # Step Inputs
            harvest = row['energy_harvested_j']
            baseload = row['energy_baseload_j']
            dwell = row['dwell_time_s']
            req = row['n_inferences_req']
            
            best_score = -1
            best_idx = -1
            best_status = "Fail"
            best_consumed_j = baseload # Default if we do nothing
            
            # 1. Test All Models against CURRENT Battery State
            for i in range(len(model_names)):
                
                # Check Hysteresis
                sw_j = 0
                sw_s = 0
                if prev_model_idx != -1 and i != prev_model_idx:
                    sw_j = self.config['switch_energy_mj'] / 1000.0
                    sw_s = self.config['switch_latency_s']

                # Evaluate
                score, raw, cons_j, status = self._evaluate_step_sequential(
                    curr_battery_j, harvest, baseload, dwell, req,
                    m_lat[i], m_eng[i], m_acc[i], sw_j, sw_s
                )
                
                # Logic: Maximize Score
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_status = status
                    best_consumed_j = cons_j
            
            # 2. Execute Champion
            if best_idx != -1:
                res_model.append(model_names[best_idx])
                prev_model_idx = best_idx
            else:
                res_model.append("None") # Should imply Blind or Dead
                # If we selected nothing, we still pay baseload
                best_consumed_j = baseload
            
            res_acc.append(best_score)
            res_status.append(best_status)
            
            # 3. Update Battery State
            # New = Old + In - Out
            # Clamp between 0 and Max Capacity
            curr_battery_j = np.clip(curr_battery_j + harvest - best_consumed_j, 
                                     0, self.BATTERY_CAPACITY_J)
            
            res_battery.append(curr_battery_j)

        # Store to DF
        self.df['dynamic_model'] = res_model
        self.df['dynamic_acc_inf'] = res_acc
        self.df['dynamic_status'] = res_status
        self.df['dynamic_battery_j'] = res_battery
        
        return self.df

    def run_baseline_sequential(self, strategy_name, model_idx):
        """
        Runs the full orbit sequentially for a SINGLE static model.
        Crucial: This lets us see if the static model kills the battery.
        """
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
                0, 0 # No switching costs for static
            )
            
            total_acc_inf += score
            
            # Update Battery
            curr_battery_j = np.clip(curr_battery_j + row['energy_harvested_j'] - cons_j, 
                                     0, self.BATTERY_CAPACITY_J)
            battery_trace.append(curr_battery_j)
            
        return total_acc_inf, battery_trace

    # ==========================================
    # 3. REPORTING
    # ==========================================

    def generate_report(self):
        print("\n" + "="*80)
        print(f"{'SEQUENTIAL BATTERY SIMULATION REPORT':^80}")
        print("="*80)

        # 1. Physics Stats
        print(f"\n{' ORBIT STATISTICS ':~^80}")
        print(f"Battery Capacity: {self.config['battery_capacity_wh']} Wh ({self.BATTERY_CAPACITY_J:,.0f} Joules)")
        print(f"Solar Generation: {self.config['solar_generation_mw']} mW")
        print(f"System Baseload:  {self.config['system_baseload_mw']} mW")
        
        # 2. Dynamic Results
        dyn_total = self.df['dynamic_acc_inf'].sum()
        
        # 3. Run Baselines
        baselines = {}
        # Identify Models
        idx_worst = self.models['Correct_Inf_per_Joule'].idxmin()
        idx_throughput = self.models['Correct_Inf_per_Sec'].idxmax()
        idx_efficiency = self.models['Correct_Inf_per_Joule'].idxmax()
        
        bl_configs = {
            "Static Worst": idx_worst,
            "Static Best Throughput": idx_throughput,
            "Static Best Efficiency": idx_efficiency
        }
        
        bl_traces = {} # Store battery traces for plotting
        
        print(f"\n{' PERFORMANCE COMPARISON ':~^80}")
        print(f"\n{'Strategy':<35} | {'Total Acc. Inferences':<25} | {'Battery Died?'}")
        print("-" * 75)
        
        best_static = 0
        
        for name, idx in bl_configs.items():
            score, trace = self.run_baseline_sequential(name, idx)
            baselines[name] = score
            bl_traces[name] = trace
            
            # Check if battery hit 0
            died = "YES" if min(trace) <= 1.0 else "No"
            print(f"{name:<35} | {score:,.0f}{'':<10} | {died}")
            if score > best_static: best_static = score
            
        print("-" * 75)
        
        # Dynamic Result
        died_dyn = "YES" if self.df['dynamic_battery_j'].min() <= 1.0 else "No"
        imp = ((dyn_total - best_static)/best_static)*100 if best_static > 0 else 0
        print(f"{'Dynamic Switching (Ours)':<35} | {dyn_total:,.0f}{'':<10} | {died_dyn} (+{imp:.1f}%)")
        print("="*80)
        
        self.plot_results(bl_traces)

    def plot_results(self, baseline_traces):
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        x = self.df['True Anomaly (deg)']
        
        # Plot 1: Battery State of Charge
        # Plot Dynamic
        ax1.plot(x, self.df['dynamic_battery_j'] / 3600.0, 'g-', linewidth=2, label='Dynamic (Ours)')
        
        # Plot Baselines
        for name, trace in baseline_traces.items():
            trace_wh = np.array(trace) / 3600.0
            style = '--'
            if "Worst" in name: color = 'red'
            elif "Throughput" in name: color = 'orange'
            else: color = 'blue'
            ax1.plot(x, trace_wh, linestyle=style, color=color, alpha=0.7, label=name)
            
        ax1.set_ylabel('Battery State (Wh)')
        ax1.set_title(f'Power Management: Battery State over Orbit (Cap: {self.config["battery_capacity_wh"]} Wh)')
        ax1.axhline(y=0, color='k', linewidth=1)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower left')
        
        # Eclipse Shading
        ax1.axvspan(180, 360, color='gray', alpha=0.1, label='Eclipse')
        
        # Plot 2: Inference Performance
        # Same scatter plot as before for Dynamic
        unique_models = [m for m in self.df['dynamic_model'].unique() if m not in ["None", "Blind"]]
        if unique_models:
            model_map = {name: i for i, name in enumerate(unique_models)}
            colors = [model_map.get(m, -1) for m in self.df['dynamic_model']]
            
            mask_succ = (self.df['dynamic_status'] == 'Success')
            if mask_succ.any():
                ax2.scatter(x[mask_succ], self.df.loc[mask_succ, 'dynamic_acc_inf'], 
                           c=[colors[i] for i in np.where(mask_succ)[0]], cmap='tab10', marker='o')
            
            mask_part = (self.df['dynamic_status'] == 'Partial')
            if mask_part.any():
                ax2.scatter(x[mask_part], self.df.loc[mask_part, 'dynamic_acc_inf'], 
                           c=[colors[i] for i in np.where(mask_part)[0]], cmap='tab10', marker='x')

        ax2.set_ylabel('Accurate Inferences')
        ax2.set_xlabel('True Anomaly (deg)')
        ax2.set_title('Dynamic Inference Performance')
        ax2.grid(True, alpha=0.3)
        ax2.axvspan(180, 360, color='gray', alpha=0.1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "sequential_battery_sim.png")
        print(f"Plot saved to {self.output_dir / 'sequential_battery_sim.png'}")

if __name__ == "__main__":
    sim = SatelliteInferenceSim(
        orbit_data_path="libs/coral_tpu_characterization/data/stk", 
        model_json_dir="libs/coral_tpu_characterization/data/tpunet_acc", 
        saleae_root="results/captures_1_20", 
        output_dir="results/final_analysis"
    )
    sim.run_dynamic_optimization()
    sim.generate_report()