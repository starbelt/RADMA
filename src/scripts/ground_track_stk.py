import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from path_utils import get_repo_root

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