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
    
    # If we want to maintain this specific "Best GSD" quality, as we go higher,
    # our pixels get bigger (worse). To treat them as "constant size", we are essentially
    # looking for objects that are physically larger as we go up, OR we accept that 
    # we are looking for the same object but with fewer pixels.
    
    # Alternate Interpretation of TODO: Constant Ground Coverage
    # Let's say a 'Task' is analyzing a 1km x 1km patch.
    # At Perigee: 1km takes (1000 / best_gsd) pixels.
    # At Apogee: 1km takes (1000 / current_gsd) pixels.
    
    # Let's define two specific tasks for the TPU:
    # Task A (Low GSD): "Fire Detection" - Needs 10m resolution (Launch/Fire)
    # Task B (High GSD): "Cloud/Weather" - Needs 100m resolution (General)
    
    # Let's calculate the 'Effort' (Tiles/sec) required to scan the *same physical area* # as the perigee pass, but at current altitude.
    # Actually, usually higher altitude = easier to scan area because FOV is huge.
    # The bottleneck is usually resolution.
    
    # Let's just plot the GSD regime.
    
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
    # This shows the raw data pressure on the TPU just to keep up with the camera
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

def analyze_constant_ground_size(df, sensor_res=4096, fov=2.0, tpu_dim=224, target_patch_km=1.0, filename='orbit_workload_analysis.png'):
    df.columns = df.columns.str.strip()
    
    # Data Prep
    diffs = df['True Anomaly (deg)'].diff()
    reset_indices = df.index[diffs < -300].tolist()
    df_orbit = df.copy() if not reset_indices else df.loc[:reset_indices[0]-1].copy()

    # Physics & Geometry
    # Velocity (Ground Speed)
    df_orbit['v_ground'] = np.sqrt(df_orbit['vx (km/sec)']**2 + df_orbit['vy (km/sec)']**2 + df_orbit['vz (km/sec)']**2)
    
    # Swath Width
    half_angle = np.deg2rad(fov / 2)
    df_orbit['swath_width_km'] = 2 * df_orbit['Alt (km)'] * np.tan(half_angle)
    
    # area visible in one frame (assuming square sensor for simplicity)
    df_orbit['frame_area_km2'] = df_orbit['swath_width_km']**2

    # Regime: Constant Feature Size 
    
    # How many tiles fit in the current view?
    # (Area of View) / (Area of Target Patch)
    df_orbit['tiles_per_frame_fixed_ground'] = df_orbit['frame_area_km2'] / (target_patch_km**2)
    
    # How much time do we have before the satellite moves one swath height
    # t = Distance / Speed
    df_orbit['t_dwell'] = df_orbit['swath_width_km'] / df_orbit['v_ground']
    
    # tiles per second required to cover the ground
    df_orbit['tpu_load_fixed_ground'] = df_orbit['tiles_per_frame_fixed_ground'] / df_orbit['t_dwell']
    
    # check resolution limit
    # At high altitude, a 1km patch might only be 2 pixels wide!
    # Let's calculate "Pixels per Target Patch" to see if the TPU input is valid.
    # Pixels per km = Sensor Res / Swath Width
    df_orbit['px_per_patch'] = (sensor_res / df_orbit['swath_width_km']) * target_patch_km
    
    # If px_per_patch < tpu_dim (224), we are upscaling (bad?). 
    # If px_per_patch > tpu_dim, we are downscaling (good).

    # plotting
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    x = df_orbit['True Anomaly (deg)']

    # alt and speed
    ax1 = axes[0]
    ax1.plot(x, df_orbit['Alt (km)'], 'g', label='Altitude')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x, df_orbit['v_ground'], 'b--', label='Ground Speed')
    ax1.set_ylabel('Altitude (km)', color='g')
    ax1_twin.set_ylabel('Speed (km/s)', color='b')
    ax1.set_title('Orbital Dynamics')
    ax1.grid(True, alpha=0.3)

    # swath width
    ax2 = axes[1]
    ax2.plot(x, df_orbit['swath_width_km'], 'purple')
    ax2.set_ylabel('Swath Width (km)')
    ax2.set_title(f'Field of View Width (FOV: {fov} deg)')
    ax2.grid(True, alpha=0.3)

    # TPU Workload 
    ax3 = axes[2]
    ax3.plot(x, df_orbit['tpu_load_fixed_ground'], 'r', linewidth=2)
    ax3.set_ylabel('Inferences / Sec')
    ax3.set_title(f'TPU Workload: Processing {target_patch_km}km² Tiles (Constant Ground Size)')
    ax3.grid(True, alpha=0.3)
    
    # Data Quality 
    ax4 = axes[3]
    ax4.plot(x, df_orbit['px_per_patch'], 'k')
    ax4.axhline(y=tpu_dim, color='red', linestyle='--', label=f'Native TPU Input ({tpu_dim}px)')
    ax4.set_ylabel('Pixels per Patch')
    ax4.set_yscale('log') # Log scale because this varies wildly
    ax4.set_title('Resolution Quality (Log Scale)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('True Anomaly (deg)')

    # Add regime labels
    for ax in axes:
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=180, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Analysis saved to {filename}")
    
    # Stats
    peak_load = df_orbit['tpu_load_fixed_ground'].max()
    min_load = df_orbit['tpu_load_fixed_ground'].min()
    print(f"TPU Load Range: {min_load:.1f} to {peak_load:.1f} inferences/sec")

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

    ground_track_plots(df, filename = str(plotdir / "ground_track.png"))
    
    analyze_orbit_regimes(df, sensor_res=4096, fov=2.0, tpu_dim=224, filename=str(plotdir /'ground_track_regimes_stk.png'))

    analyze_constant_ground_size(df,target_patch_km=5.0, filename=str(plotdir / "orbit_workload_constant_ground.png"))