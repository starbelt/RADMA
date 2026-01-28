import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from path_utils import get_repo_root

def ground_track_plots(df, sensor_res=4096, fov=2.0, tpu_dim=224):
    df.columns = df.columns.str.strip()
    
    #slice to a single orbit period
    # we look for where true anomaly resets (goes from ~360 back to 0)
    # or just take the first full 0-360 cycle
    start_idx = df['True Anomaly (deg)'].idxmin()
    # find where it approaches 360 again
    end_idx = df[df['True Anomaly (deg)'] > 359].index[0]
    df_orbit = df.loc[start_idx:end_idx].copy()

    # ground speed (ecef)
    df_orbit['v_ground'] = np.sqrt(df_orbit['vx (km/sec)']**2 + df_orbit['vy (km/sec)']**2 + df_orbit['vz (km/sec)']**2)

    # footprint and frame period
    half_angle = np.deg2rad(fov / 2)
    df_orbit['footprint_km'] = 2 * df_orbit['Alt (km)'] * np.tan(half_angle)
    df_orbit['t_frame'] = df_orbit['footprint_km'] / df_orbit['v_ground']

    # tpu throughput
    tiles_per_frame = np.ceil(sensor_res / tpu_dim)**2
    df_orbit['tiles_per_sec'] = tiles_per_frame / df_orbit['t_frame']

    # plotting with true anomaly on x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    x_axis = df_orbit['True Anomaly (deg)']

    # ground speed vs true anomaly
    ax1.plot(x_axis, df_orbit['v_ground'], color='blue')
    ax1.set_ylabel('ground speed (km/s)')
    ax1.set_title('speed relative to ground vs orbital position')
    ax1.grid(True, alpha=0.3)

    # throughput vs true anomaly
    ax2.plot(x_axis, df_orbit['tiles_per_sec'], color='orange')
    ax2.set_ylabel('tpu throughput (tiles/sec)')
    ax2.set_xlabel('true anomaly (degrees) - [0 is perigee]')
    ax2.set_title(f'tpu workload over one orbit ({sensor_res}px sensor)')
    ax2.grid(True, alpha=0.3)

    # mark perigee and apogee
    for ax in [ax1, ax2]:
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='perigee')
        ax.axvline(x=180, color='green', linestyle='--', alpha=0.5, label='apogee')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    root = get_repo_root() / "data/stk"
    
    v = pd.read_csv(root / 'HEO_Sat_Fixed_Position_Velocity.csv')
    k = pd.read_csv(root / 'HEO_Sat_Classical_Orbit_Elements.csv')
    l = pd.read_csv(root / 'HEO_Sat_LLA_Position.csv')

    df = v.merge(k, on="Time (UTCG)").merge(l, on="Time (UTCG)")

    ground_track_plots(df)