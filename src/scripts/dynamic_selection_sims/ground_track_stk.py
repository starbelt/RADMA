import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Attempt to import get_repo_root, fallback to local dir if missing
try:
    scripts_dir = Path(__file__).resolve().parent.parent
    sys.path.append(str(scripts_dir))
    from utils.path_utils import get_repo_root
    ROOT_DIR = get_repo_root()
except ImportError:
    ROOT_DIR = Path(".")

class OrbitAnalyzer:
    """
    A physics-based analyzer for satellite orbit workloads.
    Derives GSD, Swath, and Compute requirements from sensor properties and orbital mechanics.
    """
    
    DEFAULT_CONFIG = {
        # Sensor Properties
        'focal_length_mm': 50.0,
        'pixel_pitch_um': 3.45,
        'sensor_res_px': 4096,
        
        # Compute / AI Properties
        'tpu_input_dim': 224,       # Model input size (px)
        'target_tile_km': 5.0,     # The size of the ground feature we want to classify
        'min_resolvable_px': 24,    # Standard detail threshold (e.g. for cars)
        'coarse_resolvable_px': 4,  # Coarse threshold (e.g. for fires/clouds)
        
        # Plotting Thresholds (Meters)
        'gsd_thresholds': {
            'Aircraft/Buildings': 4.0,
            'Lg. Ships': 15.0,
            'Disaster/Fire': 30.0,  
            'Regional/Climate': 375.0 
        }
    }

    def __init__(self, df_merged, config=None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config: self.config.update(config)
        self.df = self._preprocess_orbit_data(df_merged)
        self._calculate_physics()

    def _preprocess_orbit_data(self, df):
        """Standardizes columns and ensures we only analyze whole number of orbits."""
        df.columns = df.columns.str.strip()
        
        # Detect Orbit Wraps (True Anomaly resetting from ~360 to ~0)
        if 'True Anomaly (deg)' in df.columns:
            diffs = df['True Anomaly (deg)'].diff()
            # Find indices where True Anomaly drops significantly (e.g. 359 -> 1)
            reset_indices = df.index[diffs < -300].tolist()
            
            if len(reset_indices) > 0:
                first_complete_idx = reset_indices[0] - 1 
                
                print(f"[INFO] Multi-orbit data found. Slicing to first single orbit (Index {first_complete_idx}).")
                return df.loc[:first_complete_idx].copy()
            else:
                print("[INFO] Less than one full orbit detected. Using entire dataset.")
                return df.copy()
        
        return df.copy()


    def _calculate_physics(self):
        """
        The Core Physics Engine.
        Order of Ops: Altitude -> GSD -> Swath -> Ground Track Speed -> Workload
        """
        df = self.df
        c = self.config

        # GSD & Swath 
        # GSD (m) = (Alt_km * Pitch_um) / FL_mm
        df['gsd_m'] = (df['Alt (km)'] * c['pixel_pitch_um']) / c['focal_length_mm']
        
        # Swath (km) = (GSD (m) * Sensor_Res_px) / 1000
        df['swath_width_km'] = (df['gsd_m'] * c['sensor_res_px']) / 1000.0

        # Ground Track Velocity from ECEF inputs
        r_vec = df[['x (km)', 'y (km)', 'z (km)']].values # pos
        v_vec = df[['vx (km/sec)', 'vy (km/sec)', 'vz (km/sec)']].values # vel
        
        # radial unit vector 
        r_norm = np.linalg.norm(r_vec, axis=1, keepdims=True)
        r_unit = r_vec / r_norm
        
        # Project Velocity onto Radial Vector
        # Dot product: (v . r_unit)
        v_vertical_mag = np.sum(v_vec * r_unit, axis=1, keepdims=True)
        v_vertical_vec = v_vertical_mag * r_unit
        
        # Subtract Vertical component to get Horizontal (Ground Track) Vector
        v_ground_vec = v_vec - v_vertical_vec
        
        # Magnitude of the ground track vector
        df['v_ground_track'] = np.linalg.norm(v_ground_vec, axis=1)

        # Time Dynamics
        # Dwell Time = Swath / Ground Track Speed
        df['t_dwell'] = df['swath_width_km'] / df['v_ground_track']

        # Workload Regimes
        # How many pixels represent our target tile?
        df['px_per_target_tile'] = (c['target_tile_km'] * 1000.0) / df['gsd_m']
        
        # Inferences per tile: (Pixels / TPU_Input)^2
        raw_load = (df['px_per_target_tile'] / c['tpu_input_dim'])**2
        
        # Dynamic Blindness Threshold
        # If GSD > 20m (Low Res), use coarse threshold (4px) -> looking for big intense things
        # If GSD < 20m (High Res), use fine threshold (24px) -> looking for detailed things
        thresholds = np.where(df['gsd_m'] > 20.0, c['coarse_resolvable_px'], c['min_resolvable_px'])
        
        df['infs_per_tile'] = np.where(df['px_per_target_tile'] < thresholds, 0.0, raw_load)
        
        # Total Throughput
        tiles_in_view = (df['swath_width_km']**2) / (c['target_tile_km']**2)
        df['total_infs_per_frame'] = tiles_in_view * df['infs_per_tile']
        
        # Final Rate Requirement
        df['infs_per_sec'] = df['total_infs_per_frame'] / df['t_dwell']

        self.df = df

    def print_stats(self):
        """Prints key stats requested in TODOs."""
        min_alt = self.df['Alt (km)'].min()
        best_gsd = self.df['gsd_m'].min()
        max_load = self.df['infs_per_sec'].max()
        
        print("-" * 30)
        print(f"Orbit Statistics")
        print("-" * 30)
        print(f"Perigee Altitude:    {min_alt:.2f} km")
        print(f"Best GSD (Perigee):  {best_gsd:.2f} m/px")
        print(f"Peak TPU Load:       {max_load:.2f} infs/sec")
        
        # Calc FOV for verification
        sensor_width = self.config['sensor_res_px'] * (self.config['pixel_pitch_um'] / 1000.0)
        fov = 2 * np.degrees(np.arctan(sensor_width / (2 * self.config['focal_length_mm'])))
        print(f"Calculated FOV:      {fov:.2f} deg")
        print("-" * 30)

    def plot_resolution_regimes(self, filename):
        """Plots GSD and Altitude with regime thresholds."""
        df = self.df
        x = df['True Anomaly (deg)']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Altitude & Speed
        color = 'tab:blue'
        ax1.set_ylabel('Altitude (km)', color=color)
        ax1.plot(x, df['Alt (km)'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        ax1_twin = ax1.twinx()
        color = 'tab:red'
        ax1_twin.set_ylabel('Ground Speed (km/s)', color=color)
        ax1_twin.plot(x, df['v_ground_track'], color=color, linestyle='--')
        ax1_twin.tick_params(axis='y', labelcolor=color)
        ax1.set_title('Orbit Dynamics: Altitude vs Speed')

        # Resolution (GSD)
        ax2.plot(x, df['gsd_m'], 'purple', linewidth=2)
        ax2.set_ylabel('GSD (m/px)')
        ax2.set_xlabel('True Anomaly (deg)')
        ax2.set_title('Ground Sampling Distance vs Requirements')
        ax2.set_yscale('log')
        
        # Add Threshold Lines
        colors = ['red', 'orange', 'green', 'blue']
        for i, (label, val) in enumerate(self.config['gsd_thresholds'].items()):
            c = colors[i % len(colors)]
            ax2.axhline(y=val, color=c, linestyle=':', label=f'{label} ({val}m)')
            
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if filename:
            plt.savefig(filename)
            print(f"Saved Resolution Plot to {filename}")
        plt.close()

    def plot_compute_regimes(self, filename):
        """Plots the Tiling vs Aggregation regimes and TPU load."""
        df = self.df
        c = self.config
        x = df['True Anomaly (deg)']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Tiling vs Aggregation
        ax1.plot(x, df['px_per_target_tile'], 'k', label='Pixels per tile')
        ax1.axhline(y=c['tpu_input_dim'], color='r', linestyle='--', label=f'TPU Input ({c["tpu_input_dim"]}px)')
        
        # Calculate dynamic threshold for visualization
        dynamic_threshold = np.where(df['gsd_m'] > 20.0, c['coarse_resolvable_px'], c['min_resolvable_px'])
        ax1.plot(x, dynamic_threshold, color='gray', linestyle=':', label='Dynamic Blindness Limit')
        
        # Fill Regimes
        ax1.fill_between(x, dynamic_threshold, df['px_per_target_tile'], 
                        where=(df['px_per_target_tile'] >= c['tpu_input_dim']), 
                        color='green', alpha=0.1, label='Tiling Regime (High Res)')
        ax1.fill_between(x, dynamic_threshold, df['px_per_target_tile'], 
                        where=(df['px_per_target_tile'] < c['tpu_input_dim']), 
                        color='blue', alpha=0.1, label='Aggregation Regime (Low Res)')
        
        ax1.set_yscale('log')
        ax1.set_ylabel(f'Pixels per {c["target_tile_km"]}km tile')
        ax1.set_title('Compute Regimes: Tiling vs Aggregation')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Total TPU Load
        ax2.plot(x, df['infs_per_sec'], 'r', linewidth=2)
        ax2.set_ylabel('Inferences / Sec')
        ax2.set_xlabel('True Anomaly (deg)')
        ax2.set_title('Total TPU Throughput Requirement')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
            print(f"Saved Compute Plot to {filename}")
        plt.close()

if __name__ == "__main__":

    config = {
        'focal_length_mm': 400.0,
        'pixel_pitch_um': 3.45,
        'sensor_res_px': 4096, 
        'target_tile_km': 10.0,
        'min_resolvable_px': 24
    }

    data_root = ROOT_DIR / "data/stk"
    plot_dir = ROOT_DIR / "results/plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("--- Loading STK Data ---")
        v = pd.read_csv(data_root / 'HEO_Sat_Fixed_Position_Velocity.csv')
        k = pd.read_csv(data_root / 'HEO_Sat_Classical_Orbit_Elements.csv')
        l = pd.read_csv(data_root / 'HEO_Sat_LLA_Position.csv')
        
        # Merge on Time (UTCG)
        df_merged = v.merge(k, on="Time (UTCG)").merge(l, on="Time (UTCG)")
        
        # run analysis
        analyzer = OrbitAnalyzer(df_merged, config)
        analyzer.print_stats()
        
        analyzer.plot_resolution_regimes(plot_dir / "stk_resolution_analysis.png")
        analyzer.plot_compute_regimes(plot_dir / "stk_compute_regimes.png")
        
    except FileNotFoundError as e:
        print(f"\n[Error] Could not find data files in {data_root}.")
        print(f"Please check your path_utils or hardcode the path.\nDetails: {e}")