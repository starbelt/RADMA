import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from path_utils import get_repo_root

def ground_track_plots(df, sensor_res=4096, fov=2.0, tpu_dim=224, filename='ground_track_stk.png'):
    df.columns = df.columns.str.strip()
    
    # robust slicing: find the first perigee and the next one
    # if true anomaly isn't perfectly 0 to 360, we find the "reset" point
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
    df_orbit['footprint_km'] = 2 * df_orbit['Alt (km)'] * np.tan(half_angle)
    df_orbit['t_frame'] = df_orbit['footprint_km'] / df_orbit['v_ground']
    
    tiles_per_frame = np.ceil(sensor_res / tpu_dim)**2
    df_orbit['tiles_per_sec'] = tiles_per_frame / df_orbit['t_frame']

    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    x = df_orbit['True Anomaly (deg)']

    # plot altitude 
    ax1.plot(x, df_orbit['Alt (km)'], color='green')
    ax1.set_ylabel('altitude (km)')
    ax1.set_title('orbital altitude')
    ax1.grid(True, alpha=0.3)

    # plot ground speed
    ax2.plot(x, df_orbit['v_ground'], color='blue')
    ax2.set_ylabel('ground speed (km/s)')
    ax2.set_title('speed relative to surface')
    ax2.grid(True, alpha=0.3)

    # plot tpu throughput required
    ax3.plot(x, df_orbit['tiles_per_sec'], color='orange')
    ax3.set_ylabel('tiles / sec')
    ax3.set_xlabel('true anomaly (deg) [0=perigee]')
    ax3.set_title(f'tpu workload ({sensor_res}px sensor)')
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
    