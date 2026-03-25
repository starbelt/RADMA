import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def set_plot_style():
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'lines.linewidth': 2.5,
        'figure.titlesize': 20,
        'figure.titleweight': 'bold'
    })

def _get_model_colors(model_names):
    # assigns consistent colors to models, forcing non-compute states to grey
    unique_models = list(dict.fromkeys(model_names))
    cmap = plt.get_cmap('tab20')
    color_dict = {}
    
    color_idx = 0
    for m in unique_models:
        if m in ['Idle', 'RECHARGE', 'Blind', 'BLOCKED']:
            color_dict[m] = '#d3d3d3' 
        else:
            color_dict[m] = cmap(color_idx % 20)
            color_idx += 1
    return color_dict

def _plot_segmented_line(ax, t, y, categories, color_dict, ylabel="cumulative yield"):
    # plots a single continuous line, changing colors based on the category array
    start_idx = 0
    seen_labels = set()
    handles, labels = [], []
    
    for i in range(1, len(categories)):
        if categories[i] != categories[i-1] or i == len(categories) - 1:
            end_idx = i + 1 if i < len(categories) - 1 else i + 1
            current_cat = categories[start_idx]
            
            label = None
            if current_cat not in seen_labels and current_cat not in ['Idle', 'RECHARGE', 'Blind', 'BLOCKED']:
                label = current_cat
                seen_labels.add(current_cat)
                
            line, = ax.plot(t[start_idx:end_idx], y[start_idx:end_idx], 
                    color=color_dict[current_cat], linewidth=3.5, label=label)
            
            if label:
                handles.append(line)
                labels.append(label)
                
            start_idx = i
            
    ax.set_ylabel(ylabel, color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    
    return handles, labels


def plot_mission(logs, naive_states, case_name, cfg, output_dir, 
                        plot_accuracy_baseline=False, 
                        plot_efficiency_baseline=False, 
                        plot_throughput_baseline=False):
    set_plot_style()
    t_plot = np.array(logs['time_rel'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'hspace': 0.15})
    
    # bump title up to make room for floating legend
    fig.suptitle(f"Case Study Telemetry: {case_name}", y=1.05) 

    ax1.plot(t_plot, logs['alt_km'], color='dimgray', label='Altitude')
    ax1.set_ylabel('Altitude (km)')
    
    alt_span = np.max(logs['alt_km']) - np.min(logs['alt_km'])
    if alt_span == 0: alt_span = np.max(logs['alt_km']) * 0.1 
    ax1.set_ylim(np.min(logs['alt_km']) - (0.1 * alt_span), np.max(logs['alt_km']) + (0.1 * alt_span))
    
    ax1_t = ax1.twinx()
    ax1_t.plot(t_plot, logs['speed_km_s'], color='tab:red', linestyle='--', alpha=0.7, label='Ground Speed')
    ax1_t.set_ylabel('Speed (km/s)', color='tab:red')
    
    spd_span = np.max(logs['speed_km_s']) - np.min(logs['speed_km_s'])
    if spd_span == 0: spd_span = np.max(logs['speed_km_s']) * 0.1
    ax1_t.set_ylim(np.min(logs['speed_km_s']) - (0.1 * spd_span), np.max(logs['speed_km_s']) + (0.1 * spd_span))
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_t.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower center', 
            bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    ax1.grid(True, alpha=0.3)
    
    # Define baseline colors early so we can use them for both axes
    baseline_colors = {'High_Accuracy': 'tab:blue', 'High_Throughput': 'tab:red', 'High_Efficiency': 'tab:purple'}
    
    ax2.plot(t_plot, logs['battery_wh'], color='tab:green', linewidth=3, label='Battery Charge (Dynamic)')
    
    # Inject requested naive battery lines
    if plot_accuracy_baseline and 'High_Accuracy' in naive_states:
        ax2.plot(t_plot, naive_states['High_Accuracy']['logs_battery_wh'], 
                color=baseline_colors['High_Accuracy'], linestyle='-.', alpha=0.5, linewidth=1.5, label='Battery (High Accuracy)')
        
    if plot_throughput_baseline and 'High_Throughput' in naive_states:
        ax2.plot(t_plot, naive_states['High_Throughput']['logs_battery_wh'], 
                color=baseline_colors['High_Throughput'], linestyle='-.', alpha=0.5, linewidth=1.5, label='Battery (High Throughput)')
        
    if plot_efficiency_baseline and 'High_Efficiency' in naive_states:
        ax2.plot(t_plot, naive_states['High_Efficiency']['logs_battery_wh'], 
                color=baseline_colors['High_Efficiency'], linestyle='-.', alpha=0.5, linewidth=1.5, label='Battery (High Efficiency)')

    ax2.axhline(cfg['battery_capacity_wh']*cfg['compute_disable_pct'], color='tab:red', linestyle=':', label='Hard Lock Limit')
    ax2.axhline(cfg['battery_capacity_wh']*cfg['compute_enable_pct'], color='tab:green', linestyle=':', label='Resume Limit')
    
    lit = np.array(logs['is_lit'])
    ax2.fill_between(t_plot, 0, 1, where=(lit > 0.5), transform=ax2.get_xaxis_transform(), 
                    color='gold', alpha=0.15, label='Sunlight Interval')
    
    ax2.set_ylabel('Battery (Wh)')
    ax2.set_xlabel('Mission Time (s)')
    ax2.grid(True, alpha=0.3)
    
    # Moved the legend slightly to accommodate the extra lines without covering data
    ax2.legend(loc='upper left', frameon=True, framealpha=0.85, fontsize=10)

    # Segmented yield & naive baselines
    ax2_t = ax2.twinx()
    color_dict = _get_model_colors(logs['model_name'])
    handles, labels = _plot_segmented_line(ax2_t, t_plot, logs['cum_correct'], logs['model_name'], color_dict, ylabel="Cumulative Correct Inferences")
    
    # Inject requested naive yield lines
    if plot_accuracy_baseline and 'High_Accuracy' in naive_states:
        line, = ax2_t.plot(t_plot, naive_states['High_Accuracy']['logs_cum_correct'], 
                        color=baseline_colors['High_Accuracy'], linestyle='--', alpha=0.8, label='Yield (High Accuracy)')
        handles.append(line)
        labels.append('Yield (High Accuracy)')
        
    if plot_throughput_baseline and 'High_Throughput' in naive_states:
        line, = ax2_t.plot(t_plot, naive_states['High_Throughput']['logs_cum_correct'], 
                        color=baseline_colors['High_Throughput'], linestyle='--', alpha=0.8, label='Yield (High Throughput)')
        handles.append(line)
        labels.append('Yield (High Throughput)')
        
    if plot_efficiency_baseline and 'High_Efficiency' in naive_states:
        line, = ax2_t.plot(t_plot, naive_states['High_Efficiency']['logs_cum_correct'], 
                        color=baseline_colors['High_Efficiency'], linestyle='--', alpha=0.8, label='Yield (High Efficiency)')
        handles.append(line)
        labels.append('Yield (High Efficiency)')

    if handles:
        ncol = min(4, len(labels))
        ax2_t.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                    ncol=ncol, frameon=False, title="Performance & Active Models")
        
    save_path = output_dir / f"{case_name}_STATIC.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_orbit_dynamics(logs, case_name, output_dir):
    set_plot_style()
    clean_name = case_name.replace('_', ' ')
    t_plot = np.array(logs['time_rel'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"case study: {clean_name}\norbit & data dynamics", y=0.98)

    ax1.plot(t_plot, logs['alt_km'], color='gray', label='Altitude')
    ax1.set_ylabel('Altitude (km)')
    
    ax1_t = ax1.twinx()
    ax1_t.plot(t_plot, logs['speed_km_s'], 'r--', label='ground speed')
    ax1_t.set_ylabel('speed (km/s)', color='r')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_t.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    ax1.grid(True, alpha=0.3)

    demand = np.array(logs['demand_infs'])
    dwell = np.array(logs['dwell_time_s'])
    
    ax2.plot(t_plot, dwell, color='tab:blue', label='frame dwell time (s)')
    ax2.set_ylabel('available time per frame (s)', color='tab:blue')
    ax2.set_xlabel('mission time (s)')
    
    ax2_t = ax2.twinx()
    ax2_t.fill_between(t_plot, demand, color='tab:orange', alpha=0.3, label='inference demand')
    ax2_t.plot(t_plot, demand, color='tab:orange', linewidth=2)
    ax2_t.set_ylabel('workload demand (infs/sec)', color='tab:orange')
    
    lines_3, labels_3 = ax2.get_legend_handles_labels()
    lines_4, labels_4 = ax2_t.get_legend_handles_labels()
    ax2.legend(lines_3 + lines_4, labels_3 + labels_4, loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = output_dir / f"{case_name}_orbit_dynamics.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_naive_blitz(logs, naive_states, case_name, cfg, output_dir):
    set_plot_style()
    clean_name = case_name.replace('_', ' ')
    t_plot = np.array(logs['time_rel'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True, gridspec_kw={'hspace': 0.2})
    fig.suptitle(f"case study: {clean_name}\nnaive blitz comparison", y=0.96)

    colors = {'High_Accuracy': 'tab:blue', 'High_Throughput': 'tab:red', 'High_Efficiency': 'tab:purple'}
    
    ax1.plot(t_plot, logs['battery_wh'], color='black', linewidth=3.5, label='dynamic (predictive)')
    for name, ns in naive_states.items():
        clean_naive = name.replace('_', ' ')
        ax1.plot(t_plot, ns['logs_battery_wh'], color=colors[name], linestyle='--', alpha=0.8, label=f'naive ({clean_naive})')

    ax1.axhline(cfg['battery_capacity_wh']*cfg['compute_disable_pct'], color='red', linestyle=':', alpha=0.5, label='shutoff limit')
    
    lit = np.array(logs['is_lit'])
    ax1.fill_between(t_plot, 0, 1, where=(lit > 0.5), transform=ax1.get_xaxis_transform(), color='gold', alpha=0.2, label='sunlight')
    
    ax1.set_ylabel('battery (wh)')
    ax1.set_title('battery management strategies')
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_plot, logs['cum_correct'], color='black', linewidth=3.5, label='dynamic system')
    for name, ns in naive_states.items():
        clean_naive = name.replace('_', ' ')
        ax2.plot(t_plot, ns['logs_cum_correct'], color=colors[name], linestyle='--', alpha=0.8, label=f'naive ({clean_naive})')

    ax2.set_xlabel('mission time (s)')
    ax2.set_ylabel('cumulative correct inferences')
    ax2.set_title('inference yield')
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    save_path = output_dir / f"{case_name}_naive_blitz.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_horizon_sweep(results, best_horizon, frames_per_orbit, case_name, output_dir):
    set_plot_style()
    clean_name = case_name.replace('_', ' ')
    
    horizons = [r[0] for r in results]
    infs = [r[1] for r in results]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(horizons, infs, marker='o', color='tab:blue', linewidth=2.5, label='total correct inferences')
    
    best_inf = next(r[1] for r in results if r[0] == best_horizon)
    ax.plot(best_horizon, best_inf, marker='*', color='gold', markersize=18, label=f'optimal horizon ({best_horizon})')
    ax.axvline(x=frames_per_orbit, color='tab:red', linestyle='--', alpha=0.7, label=f'frames per orbit (~{frames_per_orbit})')
    
    ax.set_title(f"decision horizon optimization\n{clean_name}")
    ax.set_xlabel("budget horizon (frames)")
    ax.set_ylabel("total inferences yield")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / f"{case_name}_optimization_curve.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_single(logs, case_name, output_dir):
    """
    Plots exactly one orbit of altitude and ground speed, 
    sliced to start and end exactly between apogee and perigee.
    """
    set_plot_style()


    COLOR_ORBIT = '#008855'
    COLOR_SPEED = '#224477'
    
    t = np.array(logs['time_rel'])
    alt = np.array(logs['alt_km'])
    speed = np.array(logs['speed_km_s'])
    

    mean_alt = np.mean(alt)
    

    # This isolates the exact midpoints of the altitude curve
    alt_centered = alt - mean_alt
    crossings = np.where(np.diff(np.sign(alt_centered)))[0]
    
    # We need at least 3 crossings to capture a full orbit from the exact same phase
    if len(crossings) >= 3:
        # Pick the first crossing as start, and the crossing 2 steps ahead as end (one full period)
        start_idx = crossings[0]
        end_idx = crossings[2] 
    else:
        # Fallback if the data is shorter than one orbit
        start_idx = 0
        end_idx = len(t) - 1
        print(f"[warn] Could not isolate a full orbit for {case_name}. Plotting available data.")
        
    t_slice = t[start_idx:end_idx]
    t_slice_norm = t_slice - t_slice[0] # Normalize time to start at 0 for the plot
    alt_slice = alt[start_idx:end_idx]
    speed_slice = speed[start_idx:end_idx]
    
    fig, ax1 = plt.subplots(figsize=(3.5, 2.625))
    
    # Plot Altitude on the left axis
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Orbital Altitude (km)', color=COLOR_ORBIT)
    line1, = ax1.plot(t_slice_norm, alt_slice, color=COLOR_ORBIT, linewidth=3.5, label='Altitude')
    ax1.tick_params(axis='y', labelcolor=COLOR_ORBIT)
    ax1.grid(True, alpha=0.3)
    
    # Plot Speed on the right axis
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Ground Track Speed (km/s)', color=COLOR_SPEED)  
    line2, = ax2.plot(t_slice_norm, speed_slice, color=COLOR_SPEED, linewidth=3.5, linestyle='--', label='Ground Speed')
    ax2.tick_params(axis='y', labelcolor=COLOR_SPEED)
    
    # Clean up limits to make the inverse relationship pop visually
    alt_span = np.max(alt_slice) - np.min(alt_slice)
    ax1.set_ylim(np.min(alt_slice) - (0.1 * alt_span), np.max(alt_slice) + (0.1 * alt_span))
    
    spd_span = np.max(speed_slice) - np.min(speed_slice)
    ax2.set_ylim(np.min(speed_slice) - (0.1 * spd_span), np.max(speed_slice) + (0.1 * spd_span))
    
    # Combined Floating Legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
    
    plt.tight_layout()
    save_path = output_dir / f"{case_name}_orbit_motivation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()