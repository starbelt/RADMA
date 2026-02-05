import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from path_utils import get_repo_root

def analyze_orbit_regimes(df, sensor_res=4096, fov=2.0, tpu_dim=224, filename='ground_track_stk.png'):
    df.columns = df.columns.str.strip()
    
    # --- Data Prep ---
    diffs = df['True Anomaly (deg)'].diff()
    reset_indices = df.index[diffs < -300].tolist()
    if not reset_indices:
        df_orbit = df.copy()
    else:
        df_orbit = df.loc[:reset_indices[0]-1].copy()

    # Calculate Physical Parameters
    df_orbit['v_ground'] = np.sqrt(df_orbit['vx (km/sec)']**2 + df_orbit['vy (km/sec)']**2 + df_orbit['vz (km/sec)']**2)
    
    half_angle = np.deg2rad(fov / 2)

    # total width of the ground seen by the sensor
    df_orbit['swath_width_km'] = 2 * df_orbit['Alt (km)'] * np.tan(half_angle)
    
    # GSD
    # (swath_width_km * 1000) / sensor_res
    df_orbit['gsd_m'] = (df_orbit['swath_width_km'] * 1000) / sensor_res

    ## Regime 1 - sample size stays the same/ full frame processed
    # We always read the full 4k sensor, downsample/tile it to 224x224 chunks for the TPU.
    # must frame fast enough so the ground doesn't slip away.
    # Time per frame = Swath Height / Ground Speed
    df_orbit['t_frame_natural'] = df_orbit['swath_width_km'] / df_orbit['v_ground']
    
    # If we tile the ENTIRE image into 224x224 chunks:
    total_tiles_per_frame = np.ceil(sensor_res / tpu_dim)**2 # approx 335 tiles for 4k->224
    df_orbit['tiles_per_sec_natural'] = total_tiles_per_frame / df_orbit['t_frame_natural']

    # Regime 2 - Constant Feature Size     # TODO: "Find GSD for one 224 feature at periapsis"
    # Let's say at Perigee, 1 pixel = X meters. We want to maintain that detection capability.
    min_alt = df_orbit['Alt (km)'].min()
    best_gsd = df_orbit['gsd_m'].min() # This is our "High Res" benchmark
    print(f"Perigee (Min Alt): {min_alt:.2f} km")
    print(f"Best GSD at Perigee: {best_gsd:.2f} m/px")

    
    # --- Plotting ---
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
    x = df_orbit['True Anomaly (deg)']

    # Altitude
    ax1.plot(x, df_orbit['Alt (km)'], 'g')
    ax1.set_ylabel('Alt (km)')
    ax1.set_title('Altitude')
    ax1.grid(True, alpha=0.3)

    # Ground Speed
    ax2.plot(x, df_orbit['v_ground'], 'b')
    ax2.set_ylabel('Speed (km/s)')
    ax2.set_title('Ground Speed')
    ax2.grid(True, alpha=0.3)

    # Resolution 
    ax3.plot(x, df_orbit['gsd_m'], 'purple')
    ax3.set_ylabel('GSD (m/px)')
    ax3.set_title('Ground Sampling Distance (Resolution)')
    ax3.axhline(y=10, color='r', linestyle=':', label='Fire/Car Threshold (10m)')
    ax3.axhline(y=50, color='orange', linestyle=':', label='Ship Threshold (50m)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Throughput (Natural)
    # shows the raw data pressure on the TPU just to keep up with the camera
    ax4.plot(x, df_orbit['tiles_per_sec_natural'], 'orange')
    ax4.set_ylabel('Tiles/sec (Full Frame)')
    ax4.set_title(f'Data Pressure (Processing full {sensor_res}x{sensor_res} frame)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

def ground_track_plots(df, sensor_res=4096, fov=2.0, tpu_dim=224, filename='ground_track_stk.png'):
    df.columns = df.columns.str.strip()
    
    # slice by true anomaly
    diffs = df['True Anomaly (deg)'].diff()
    # a huge negative jump indicates the orbit reset (e.g., 359 -> 0.1)
    reset_indices = df.index[diffs < -300].tolist()
    
    if not reset_indices:
        # if no reset found, just use the whole dataframe
        df_orbit = df.copy()
    else:
        # take everything from start to the first reset
        df_orbit = df.loc[:reset_indices[0]-1].copy()

    # ground speed (ecef)
    df_orbit['v_ground'] = np.sqrt(df_orbit['vx (km/sec)']**2 + df_orbit['vy (km/sec)']**2 + df_orbit['vz (km/sec)']**2)

    # footprint and throughput
    half_angle = np.deg2rad(fov / 2)
    df_orbit['footprint_km'] = 2 * df_orbit['Alt (km)'] * np.tan(half_angle) ## TODO: What happens if we tile to keep feature size constant - i.e. more tiles at higher alts
    df_orbit['t_frame'] = df_orbit['footprint_km'] / df_orbit['v_ground'] ## TODO: What happens if we don't do any tiling? 
    print(df_orbit['t_frame'].min()) ## TODO : select reasonable feature - find GSD for one 224 feature at periapsis
                                ## TODO: KEEP GSD CONSTANT - at some point tiling is so tiny that we have changed regimes
                                # what is smallest allowable? Start with like an order of magnitude 20x20, 10x10, etc. 
                                # then reverse - find gsd at peak, then as you go lower it become easier to tile. - for tasks like detecting fires, rocket launches, etc
                                # High and Low GSD tasks

    tiles_per_frame = np.ceil(sensor_res / tpu_dim)**2
    print(tiles_per_frame)
    df_orbit['tiles_per_sec'] = tiles_per_frame / df_orbit['t_frame']

    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    x = df_orbit['True Anomaly (deg)']

    # plot altitude 
    ax1.plot(x, df_orbit['Alt (km)'], color='green')
    print(np.min(df_orbit['Alt (km)']))
    ax1.set_ylabel('Altitude (km)')
    ax1.set_title('Orbital Altitude')
    ax1.grid(True, alpha=0.3)

    # plot ground speed
    ax2.plot(x, df_orbit['v_ground'], color='blue')
    ax2.set_ylabel('Ground Speed (km/s)')
    ax2.set_title('Speed Relative to Surface')
    ax2.grid(True, alpha=0.3)

    # plot tpu throughput required
    ax3.plot(x, df_orbit['tiles_per_sec'], color='orange')
    ax3.set_ylabel('Tiles / sec')
    ax3.set_xlabel('True Anomaly (deg)')
    ax3.set_title(f'TPU Workload ({sensor_res} px Sensor)')
    ax3.grid(True, alpha=0.3)

    # mark critical points
    for ax in [ax1, ax2, ax3]:
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.4, label='perigee')
        ax.axvline(x=180, color='black', linestyle='--', alpha=0.4, label='apogee')


    plt.tight_layout()
    plt.savefig(filename)

def analyze_dynamic_regimes(df, sensor_res=4096, fov=2.0, tpu_dim=224, target_patch_km=5.0, filename='orbit_regimes_discrete.png'):
    df.columns = df.columns.str.strip()
    
    # --- Data Prep ---
    diffs = df['True Anomaly (deg)'].diff()
    reset_indices = df.index[diffs < -300].tolist()
    if not reset_indices:
        df_orbit = df.copy()
    else:
        df_orbit = df.loc[:reset_indices[0]-1].copy()

    # --- Physics ---
    df_orbit['v_ground'] = np.sqrt(df_orbit['vx (km/sec)']**2 + df_orbit['vy (km/sec)']**2 + df_orbit['vz (km/sec)']**2)
    half_angle = np.deg2rad(fov / 2)
    
    # Geometry
    df_orbit['swath_width_km'] = 2 * df_orbit['Alt (km)'] * np.tan(half_angle)
    df_orbit['swath_area_km2'] = df_orbit['swath_width_km']**2 # Square view assumption
    
    # 1. Native Resolution Check
    # How many pixels represent our target 5km patch?
    # (Sensor Res / Swath Width) * Target Size
    df_orbit['px_per_target_patch'] = (sensor_res / df_orbit['swath_width_km']) * target_patch_km
    
    # --- REGIME LOGIC ---
    
    # Regime Oversampling (Low Alt) BIGGER than the TPU input (224px).
    # We must TILE the patch.
    # Count = (Pixels / 224)^2
    
    # Regime Undersampling (High Alt) SMALLER than the TPU input.
    # We AGGREGATE patches (fit multiple 5km patches into one 224 input).
    # Count = (Pixels / 224)^2  <-- This naturally becomes a fraction (< 1.0)
    
    # Calculate Inferences per Patch
    # If this is 4.0, we need 4 inferences to cover patch area.
    # If this is 0.25, one inference covers four patch areas.
    df_orbit['infs_per_patch'] = (df_orbit['px_per_target_patch'] / tpu_dim)**2
    
    # Clamp extreme upscaling
    # If px_per_target_patch < 10 pixels, maybe we just can't detect anything
    # Let's set a "Blindness Threshold" where resolution is too poor to be useful.
    MIN_RESOLVABLE_PIXELS = 24 # If a 5km patch is < 10px, we treat it as zero load (useless data) # TODO: Make this a parameter
    df_orbit['infs_per_patch'] = np.where(df_orbit['px_per_target_patch'] < MIN_RESOLVABLE_PIXELS, 0, df_orbit['infs_per_patch'])

    
    # Total patches visible in the frame
    df_orbit['patches_in_view'] = df_orbit['swath_area_km2'] / (target_patch_km**2)
    
    # Total Inferences for the whole frame = (Patches in View) * (Infs per Patch)
    # In the aggregation regime, 'Patches in view' goes UP, but 'Infs per patch' goes DOWN faster.
    df_orbit['total_infs_per_frame'] = df_orbit['patches_in_view'] * df_orbit['infs_per_patch']
    
    # Time available (Dwell time)
    df_orbit['t_dwell'] = df_orbit['swath_width_km'] / df_orbit['v_ground']
    
    # Final Throughput Requirement
    df_orbit['infs_per_sec'] = df_orbit['total_infs_per_frame'] / df_orbit['t_dwell']


    #  plotting
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    x = df_orbit['True Anomaly (deg)']

    # Pixel Density
    ax1 = axes[0]
    
    ax1.plot(x, df_orbit['px_per_target_patch'], 'k', label='Pixels per 5km Patch')
    ax1.axhline(y=tpu_dim, color='r', linestyle='--', label=f'TPU Input ({tpu_dim}px)')
    ax1.axhline(y=MIN_RESOLVABLE_PIXELS, color='gray', linestyle=':', label='Blindness Limit')
    ax1.set_yscale('log')
    ax1.set_ylabel('Pixels per Patch')
    ax1.set_title('Regime Detection: Resolution vs TPU Input')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add colored bands for regimes
    # Where Px > 224: Tiling Regime (Green)
    # Where Px < 224: Aggregation Regime (Blue)
    # TODO: Add legend entries for these areas
    ax1.fill_between(x, MIN_RESOLVABLE_PIXELS, df_orbit['px_per_target_patch'], 
                     where=(df_orbit['px_per_target_patch'] >= tpu_dim), 
                     color='green', alpha=0.1, label='Tiling Regime')
    ax1.fill_between(x, MIN_RESOLVABLE_PIXELS, df_orbit['px_per_target_patch'], 
                     where=(df_orbit['px_per_target_patch'] < tpu_dim), 
                     color='blue', alpha=0.1, label='Aggregation Regime')

    # Inferences per Patch (The "Efficiency" Metric)
    ax2 = axes[1]
    ax2.plot(x, df_orbit['infs_per_patch'], 'purple')
    ax2.set_ylabel('Inferences per Patch')
    ax2.set_yscale('log')
    ax2.set_title('Compute Density: Inferences required per ground unit')
    ax2.axhline(y=1.0, color='k', linestyle='--', label='1:1 Ratio')
    ax2.grid(True, alpha=0.3)

    # Total Load
    ax3 = axes[2]
    ax3.plot(x, df_orbit['infs_per_sec'], 'r', linewidth=2)
    ax3.set_ylabel('Total Inferences / Sec')
    ax3.set_title('Final TPU Throughput Requirement')
    ax3.grid(True, alpha=0.3)

    # Swath/Alt Reference
    ax4 = axes[3]
    ax4.plot(x, df_orbit['Alt (km)'], 'g')
    ax4.set_ylabel('Altitude (km)')
    ax4.set_xlabel('True Anomaly (deg)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved discrete regime analysis to {filename}")

if __name__ == "__main__":
    root = get_repo_root() / "data/stk"
    plotdir = get_repo_root() / "results/plots"
    v = pd.read_csv(root / 'HEO_Sat_Fixed_Position_Velocity.csv')
    k = pd.read_csv(root / 'HEO_Sat_Classical_Orbit_Elements.csv')
    l = pd.read_csv(root / 'HEO_Sat_LLA_Position.csv')

    df = v.merge(k, on="Time (UTCG)").merge(l, on="Time (UTCG)")

    ## Params for sensitivity study
    Sensor_Resolution = 4096 # px 4k
    Sensor_FOV = 2.0 # deg
    Model_Input_dim = 224 # px tile size

    # ground_track_plots(df, filename = str(plotdir / "ground_track.png"))
    
    # analyze_orbit_regimes(df, sensor_res=4096, fov=2.0, tpu_dim=224, filename=str(plotdir /'ground_track_regimes_stk.png'))

    analyze_dynamic_regimes(df, 
                            sensor_res=4096, 
                            fov=2.0, 
                            tpu_dim=224, 
                            target_patch_km=10.0, # 5km patches
                            filename=str(plotdir / "orbit_regimes_discrete.png"))