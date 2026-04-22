# model_swaps.py
import pandas as pd
import numpy as np
import os
import json

# --- Configuration ---
SHUNT_RESISTANCE = 0.2  # Ohms
BASE_DIR = "/home/jackr/Downloads/ModelSwaps/"  # Set this to the parent directory containing A25, A50, etc.
ALPHA_DIRS = ["A25", "A50", "A75", "A100", "A125", "A150"]
DEPTHS = ["02", "04", "06", "08", "10", "12"]
OUTPUT_FILE = "model_switching_results.json"

def process_sweeps():
    results = {}

    for alpha in ALPHA_DIRS:
        alpha_path = os.path.join(BASE_DIR, alpha)
        print(alpha_path)
        
        # Skip if directory or split-run components don't exist
        if not os.path.isdir(alpha_path):
            print(f"Skipping {alpha} - Directory not found.")
            continue
            
        digital_path = os.path.join(alpha_path, "digital.csv")
        analog_path = os.path.join(alpha_path, "analog.csv")
        
        if not (os.path.exists(digital_path) and os.path.exists(analog_path)):
            print(f"Skipping {alpha} - Missing CSVs.")
            continue
            
        print(f"Processing {alpha}...")
        
        # Load logic analyzer data
        digital_df = pd.read_csv(digital_path)
        analog_df = pd.read_csv(analog_path)
        
        # Find all HIGH pulses on the Switching GPIO
        switching_state = digital_df['Switching'].astype(int)
        edges = switching_state.diff()
        
        starts = digital_df.loc[edges == 1, 'Time [s]'].values
        ends = digital_df.loc[edges == -1, 'Time [s]'].values
        
        # Handle boundary conditions (if capture started/ended while HIGH)
        if switching_state.iloc[0] == 1:
            starts = np.insert(starts, 0, digital_df['Time [s]'].iloc[0])
        if switching_state.iloc[-1] == 1:
            ends = np.append(ends, digital_df['Time [s]'].iloc[-1])
            
        # Ensure array lengths match
        min_len = min(len(starts), len(ends))
        starts = starts[:min_len]
        ends = ends[:min_len]
        
        MIN_PULSE_DURATION_S = 0.001 
        durations = ends - starts
        
        # Optional: Print raw pulses to see the hidden bounces
        # for j, (s, d) in enumerate(zip(starts, durations)):
        #     print(f"  Raw Pulse {j}: Start={s:.5f}s, Duration={d:.6f}s")
            
        valid_pulse_mask = durations > MIN_PULSE_DURATION_S
        clean_starts = starts[valid_pulse_mask]
        clean_ends = ends[valid_pulse_mask]
        
        # 2. Mask out the 1-second sleep bug using the cleaned array
        valid_starts = clean_starts[0::2]
        valid_ends = clean_ends[0::2]
        
        alpha_results = {}
        
        # process the analog data for each valid switching interval
        for i, (t_start, t_end) in enumerate(zip(valid_starts, valid_ends)):
            if i >= len(DEPTHS):
                print(f"  Warning: Found more switching pulses than expected depths in {alpha}.")
                break
                
            depth = DEPTHS[i]
            
            # Mask analog data to the digital timeframe
            mask = (analog_df['Time [s]'] >= t_start) & (analog_df['Time [s]'] <= t_end)
            analog_slice = analog_df[mask].copy()
            
            if analog_slice.empty:
                print(f"  Warning: No analog data found for {alpha} depth {depth}.")
                continue
                
            # Calculate dt for the integral (fill first row with 0 to prevent NaN propagation)
            analog_slice['dt'] = analog_slice['Time [s]'].diff().fillna(0)
            
            # Calculate physical metrics
            analog_slice['V_drop'] = analog_slice['V_BEFORE_SHUNT'] - analog_slice['V_AFTER_SHUNT']
            analog_slice['Current'] = analog_slice['V_drop'] / SHUNT_RESISTANCE
            analog_slice['Power'] = analog_slice['VSYS'] * analog_slice['Current']
            
            # Perform finite summation for Energy
            energy_j = (analog_slice['Power'] * analog_slice['dt']).sum()
            avg_power_w = analog_slice['Power'].mean()
            duration_s = t_end - t_start
            
            alpha_results[f"depth_{depth}"] = {
                "switching_time_s": float(duration_s),
                "avg_power_W": float(avg_power_w),
                "energy_J": float(energy_j)
            }
            
        results[alpha] = alpha_results

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nExtraction complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_sweeps()