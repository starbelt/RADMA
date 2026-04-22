import pandas as pd
import numpy as np
import os
import json

SHUNT_RESISTANCE = 0.2  # Ohms
BASE_DIR = "/home/jackr/Downloads/ModelSwaps"  # Set this to the parent directory
OUTPUT_FILE = "model_switching_results.json"

# Map physical directory names to the final JSON key and the subset of depths they contain
RUN_CONFIG = {
    "A25": {"alpha_key": "A25", "depths": ["02", "04", "06", "08", "10", "12"]},
    "A50": {"alpha_key": "A50", "depths": ["02", "04", "06", "08", "10", "12"]},
    "A75": {"alpha_key": "A75", "depths": ["02", "04", "06", "08", "10", "12"]},
    "A100": {"alpha_key": "A100", "depths": ["02", "04", "06", "08", "10", "12"]},
    # Split runs - rename the dictionary keys to match your actual folder names
    "A125_1": {"alpha_key": "A125", "depths": ["02", "04", "06"]},
    "A125_2": {"alpha_key": "A125", "depths": ["08", "10", "12"]},
    "A150_1": {"alpha_key": "A150", "depths": ["02", "04", "06"]},
    "A150_2": {"alpha_key": "A150", "depths": ["08", "10", "12"]},
}

def process_sweeps():
    results = {}

    for dir_name, config in RUN_CONFIG.items():
        alpha_key = config["alpha_key"]
        expected_depths = config["depths"]
        dir_path = os.path.join(BASE_DIR, dir_name)
        
        if not os.path.isdir(dir_path):
            print(f"Skipping {dir_name} - Directory not found.")
            continue
            
        digital_path = os.path.join(dir_path, "digital.csv")
        analog_path = os.path.join(dir_path, "analog.csv")
        
        if not (os.path.exists(digital_path) and os.path.exists(analog_path)):
            print(f"Skipping {dir_name} - Missing CSVs.")
            continue
            
        print(f"Processing {dir_name} (Mapping to {alpha_key})...")
        
        #  logic analyzer data
        digital_df = pd.read_csv(digital_path)
        analog_df = pd.read_csv(analog_path)
        
        # Find edges
        switching_state = digital_df['Switching'].astype(int)
        edges = switching_state.diff()
        starts = digital_df.loc[edges == 1, 'Time [s]'].values
        ends = digital_df.loc[edges == -1, 'Time [s]'].values
        
        if switching_state.iloc[0] == 1:
            starts = np.insert(starts, 0, digital_df['Time [s]'].iloc[0])
        if switching_state.iloc[-1] == 1:
            ends = np.append(ends, digital_df['Time [s]'].iloc[-1])
            
        min_len = min(len(starts), len(ends))
        starts = starts[:min_len]
        ends = ends[:min_len]

        # Software Debounce
        MIN_PULSE_DURATION_S = 0.001 
        durations = ends - starts
        valid_pulse_mask = durations > MIN_PULSE_DURATION_S
        
        clean_starts = starts[valid_pulse_mask]
        clean_ends = ends[valid_pulse_mask]
        
        # Mask out the 1-second sleep bug - should be every other high duration (starting with the clean signal off the boot)
        valid_starts = clean_starts[0::2]
        valid_ends = clean_ends[0::2]
        
        # Ensure the alpha key exists 
        if alpha_key not in results:
            results[alpha_key] = {}
            
        # analog
        for i, (t_start, t_end) in enumerate(zip(valid_starts, valid_ends)):
            if i >= len(expected_depths):
                print(f"  Warning: Found more switching pulses than expected depths in {dir_name}.")
                break
                
            depth = expected_depths[i]
            
            mask = (analog_df['Time [s]'] >= t_start) & (analog_df['Time [s]'] <= t_end)
            analog_slice = analog_df[mask].copy()
            
            if analog_slice.empty:
                print(f"  Warning: No analog data found for {dir_name} depth {depth}.")
                continue
                
            analog_slice['dt'] = analog_slice['Time [s]'].diff().fillna(0)
            analog_slice['V_drop'] = analog_slice['V_BEFORE_SHUNT'] - analog_slice['V_AFTER_SHUNT']
            analog_slice['Current'] = analog_slice['V_drop'] / SHUNT_RESISTANCE
            analog_slice['Power'] = analog_slice['VSYS'] * analog_slice['Current']
            
            energy_j = (analog_slice['Power'] * analog_slice['dt']).sum()
            avg_power_w = analog_slice['Power'].mean()
            duration_s = t_end - t_start
            
            # Stitch directly into the shared alpha key
            results[alpha_key][f"depth_{depth}"] = {
                "switching_time_s": float(duration_s),
                "avg_power_W": float(avg_power_w),
                "energy_J": float(energy_j)
            }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nExtraction complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_sweeps()