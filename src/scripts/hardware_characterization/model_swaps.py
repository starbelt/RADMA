# model_swaps.py
'''
Parses outputs from saleae for model_swaps.cc runs.
Take note of exepcted column names in saleae logic2 channels

Digital
    CH0:    Inference
    CH1:    Switching
Analog
    CH2:    V_BEFORE_SHUNT
    CH3:    V_AFTER_SHUNT
    CH4    VSYS

Some alpha values are split up since it can be hard to fit lots of models onto
the Coral Dev Board in one go!
'''
import os, json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path


SHUNT_RESISTANCE = 0.2  # Ohms
BASE_DIR = "/home/jackr/Downloads/ModelSwaps" # change for whatever yours is!
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

def plot_switching_metrics(json_file="model_switching_results.json", filename="switching_metrics.png", output_dir=".", show_values=True):

    # Enforce global serif font layout for native LaTeX integration
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix"
    })

    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    if not data:
        print("JSON data is empty.")
        return

    # Extract and sort Alphas numerically (e.g., 'A25' -> 25)
    alphas = sorted(list(data.keys()), key=lambda x: int(x.replace('A', '')))
    
    # Extract and sort Depths
    sample_alpha = data[alphas[0]]
    depth_keys = sorted(list(sample_alpha.keys()), key=lambda x: int(x.split('_')[1]))
    depths = [d.split('_')[1] for d in depth_keys]

    x = np.arange(len(alphas))  
    
    # Shrink the whitespace between groups by making the bars wider
    group_width = 0.9 
    width = group_width / len(depths)   

    cmap = plt.get_cmap('magma_r') 
    colors = cmap(np.linspace(0.2, 0.8, len(depths)))

    # Sized for a single-column paper layout (approx 3.5 inches wide)
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(3.5, 3.5), gridspec_kw={'hspace': 0.15})

    def plot_group_row(ax_idx, metric_key, ylabel, ylim_top=None):
        ax = axes[ax_idx]
        max_height = 0
        
        for i, depth_key in enumerate(depth_keys):
            heights = []
            for alpha in alphas:
                val = data.get(alpha, {}).get(depth_key, {}).get(metric_key, 0)
                heights.append(val)
            
            # Track the maximum height in this row for dynamic Y-axis scaling
            max_height = max(max_height, max(heights) if heights else 0)
            
            offset = (i - len(depths)/2) * width + width/2
            
            # Only add labels to the top plot for the legend
            label = f'Depth {depths[i]}' if ax_idx == 0 else ""
            ax.bar(x + offset, heights, width, label=label, color=colors[i], edgecolor='#333333', linewidth=0.5)
            
            if show_values:
                for j, h in enumerate(heights):
                    if h > 0:
                        # 90 deg rotation so they fit in the tight horizontal space
                        ax.text(x[j] + offset, h * 1.05, f"{h:.1f}", 
                                ha='center', va='bottom', fontsize=5, rotation=90)

        # Paper-ready typography
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if ylim_top: 
            ax.set_ylim(0, ylim_top)
        else: 
            # Add headroom for the rotated text if values are shown
            headroom = 1.35 if show_values else 1.1
            ax.set_ylim(0, max_height * headroom)

    # Plotting rows
    plot_group_row(0, "switching_time_s", "Time (s)")
    
    # Place legend above the first plot, spread across columns
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), fontsize=9, ncol=len(depths), frameon=False)
    
    plot_group_row(1, "energy_J", "Energy (J)")

    # X-axis formatting on the bottom axis
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{float(a.replace('A', ''))/100}" for a in alphas], fontsize=10)
    axes[1].set_xlabel(r"Width Multiplier ($\alpha$)", fontsize=11)

    # Ensure pdf extension
    pdf_filename = Path(filename).with_suffix('.pdf')
    output_path = Path(output_dir) / pdf_filename
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches="tight")
    print(f"[PLOT] Saved {output_path}")
    plt.close(fig)

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
    plot_switching_metrics()
    #process_sweeps()