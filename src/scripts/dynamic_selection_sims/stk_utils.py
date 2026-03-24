import pandas as pd
import numpy as np
from pathlib import Path

def parse_lighting_schedule(file_path):
    # grab the start and stop times for sunlight intervals
    intervals = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    capture = False
    header_found_count = 0
    for line in lines:
        if "Start Time (EpSec)" in line:
            header_found_count += 1
            if header_found_count > 1: break
            capture = True
            continue
        if capture:
            if "Statistics" in line or line.strip() == "": break 
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    start = float(parts[0].replace('"', ''))
                    stop = float(parts[1].replace('"', ''))
                    intervals.append((start, stop))
                except ValueError: continue
    return intervals

def load_orbit_data(data_path, sat_prefix, num_orbits):
    # merge position, velocity, and classical orbital elements
    root = Path(data_path)
    p_path = root / f'{sat_prefix}_Sat_Fixed_Position_Velocity.csv'
    c_path = root / f'{sat_prefix}_Sat_Classical_Orbit_Elements.csv'
    l_path = root / f'{sat_prefix}_Sat_LLA_Position.csv'
    
    if not p_path.exists() or not c_path.exists() or not l_path.exists():
        print(f"[warn] missing stk files for {sat_prefix}. simulation skipped.")
        return pd.DataFrame(), []

    v = pd.read_csv(p_path)
    k = pd.read_csv(c_path)
    l = pd.read_csv(l_path)
    
    for d in [v, k, l]: 
        d.columns = d.columns.str.strip()
        if "Time (EpSec)" in d.columns:
            d["Time (EpSec)"] = d["Time (EpSec)"].round(4)

    try:
        df = v.merge(k, on="Time (EpSec)").merge(l, on="Time (EpSec)")
    except Exception as e:
        print(f"[error] merge failed for {sat_prefix}: {e}")
        return pd.DataFrame(), []

    if df.empty: return pd.DataFrame(), []
    
    light_path = root / f'{sat_prefix}_Sat_Lighting_Times.csv'
    if light_path.exists():
        sunlight_intervals = parse_lighting_schedule(light_path)
        
        if len(sunlight_intervals) > 0:
            t_min_orig = df['Time (EpSec)'].min()
            first_sun_start = sunlight_intervals[0][0]
            
            if first_sun_start <= t_min_orig and len(sunlight_intervals) > 1:
                t_new_start = sunlight_intervals[1][0]
            else:

                t_new_start = first_sun_start

            df = df[df['Time (EpSec)'] >= t_new_start].copy()
            
            df['Time (EpSec)'] = df['Time (EpSec)'] - t_new_start
            
            shifted_intervals = []
            for s, e in sunlight_intervals:
                if e > t_new_start:

                    shifted_intervals.append((max(0.0, s - t_new_start), e - t_new_start))
            sunlight_intervals = shifted_intervals
    else:
        # Fallback if no lighting file exists: zero out the timeline anyway
        t_min = df['Time (EpSec)'].min()
        df['Time (EpSec)'] = df['Time (EpSec)'] - t_min
        t_max = df['Time (EpSec)'].max()
        sunlight_intervals = [(0.0, t_max)]
    if 'True Anomaly (deg)' in df.columns:
        ta = df['True Anomaly (deg)'].values
        diffs = np.diff(ta) 
        # Wraps from ~360 back to 0 will result in a large negative difference
        wrap_indices = np.where(diffs < -300)[0]
        if len(wrap_indices) > 0 and len(wrap_indices) >= num_orbits:
            cutoff_idx = wrap_indices[num_orbits - 1]
            df = df.iloc[:cutoff_idx+1].copy()
            t_max_new = df['Time (EpSec)'].max()
            sunlight_intervals = [(s, min(e, t_max_new)) for s, e in sunlight_intervals if s <= t_max_new]

    print(f"[info] loaded {len(df)} rows of orbit data for {sat_prefix}, synchronized to first full sunlight.")
    return df, sunlight_intervals

def interpolate_orbit(df, sunlight_intervals, dt, cfg):
    # generate a high resolution timeline matching the sim dt
    if df.empty: return pd.DataFrame()
    
    t_start = df['Time (EpSec)'].min()
    t_end = df['Time (EpSec)'].max()
    new_times = np.arange(t_start, t_end, dt)
    
    cols = ['x (km)', 'y (km)', 'z (km)', 'vx (km/sec)', 'vy (km/sec)', 'vz (km/sec)', 'Alt (km)']
    new_data = {'Time (EpSec)': new_times}
    for c in cols:
        new_data[c] = np.interp(new_times, df['Time (EpSec)'], df[c])
    
    is_lit = np.zeros_like(new_times)
    for s, e in sunlight_intervals:
        mask = (new_times >= s) & (new_times <= e)
        is_lit[mask] = 1.0
    new_data['is_lit'] = is_lit
    new_df = pd.DataFrame(new_data)

    # calc optical and dynamic properties
    new_df['gsd_m'] = (new_df['Alt (km)'] * cfg['pixel_pitch_um']) / cfg['focal_length_mm']
    new_df['swath_km'] = (new_df['gsd_m'] * cfg['sensor_res']) / 1000.0
    
    r = new_df[['x (km)', 'y (km)', 'z (km)']].values
    v = new_df[['vx (km/sec)', 'vy (km/sec)', 'vz (km/sec)']].values
    r_norm = np.linalg.norm(r, axis=1, keepdims=True)
    v_vert = np.sum(v * (r/r_norm), axis=1, keepdims=True) * (r/r_norm)
    v_ground = np.linalg.norm(v - v_vert, axis=1)
    
    new_df['v_ground_km_s'] = v_ground
    new_df['dwell_time_s'] = new_df['swath_km'] / (v_ground + 1e-9)

    # set logic for altitude thresholds
    is_low_alt = new_df['Alt (km)'] <= cfg['alt_threshold_km']
    target_size_km = np.where(is_low_alt, cfg['low_alt_target_km'], cfg['high_alt_target_km'])
    min_px = np.where(is_low_alt, cfg['low_alt_min_px'], cfg['high_alt_min_px'])
    
    target_size_m = target_size_km * 1000.0
    new_df['px_per_object'] = target_size_m / new_df['gsd_m']
    
    max_tiles_per_frame = (cfg['sensor_res'] / cfg['tpu_dim'])**2 
    
    new_df['tiles_per_frame'] = np.where(new_df['px_per_object'] < min_px, 0.0, max_tiles_per_frame)
    new_df['demand_infs_per_sec'] = new_df['tiles_per_frame'] / new_df['dwell_time_s'] 
    new_df['current_target_km'] = target_size_km
    
    return new_df