import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def set_plot_style():
    plt.rcdefaults()  # reset any cached state before applying overrides
    plt.rcParams.update({
        'font.size': 7,
        'axes.titlesize': 9,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6,
        'lines.linewidth': 1.5,
        'figure.titlesize': 9,
        'figure.titleweight': 'bold'
    })

def _get_model_colors(model_names):
    # defines states that shouldn't get a magma color
    non_compute = ['idle', 'recharge', 'blind', 'blocked', 'Idle', 'RECHARGE', 'Blind', 'BLOCKED']
    
    # assigns consistent colors to models using magma
    unique_models = [m for m in dict.fromkeys(model_names) if m not in non_compute]
    
    # Sort alphabetically as a proxy for model size/complexity
    unique_models.sort()
    
    cmap = plt.get_cmap('magma')
    color_dict = {}
    
    # Set all non-compute states to a distinct dark gray
    for m in non_compute:
        color_dict[m] = '#707070' 
        
    n_models = len(unique_models)
    for idx, m in enumerate(unique_models):
        if n_models == 1:
            val = 0.5
        else:
            # Scale from 0.2 to 0.8 to avoid pure black (invisible) or pure white
            val = 0.2 + (0.6 * idx / (n_models - 1))
        color_dict[m] = cmap(val)
        
    return color_dict

def _plot_segmented_line(ax, t, y, categories, color_dict, ylabel="cumulative yield"):
    start_idx = 0
    seen_labels = set()
    handles, labels = [], []
    non_compute = ['idle', 'recharge', 'blind', 'blocked', 'Idle', 'RECHARGE', 'Blind', 'BLOCKED']
    
    for i in range(1, len(categories)):
        if categories[i] != categories[i-1] or i == len(categories) - 1:
            end_idx = i + 1 if i < len(categories) - 1 else i + 1
            current_cat = categories[start_idx]
            
            label = None
            # Only add to legend if it's a real model we haven't seen yet
            if current_cat not in seen_labels and current_cat not in non_compute:
                label = current_cat
                seen_labels.add(current_cat)
                
            # Thinned out linewidth for single-column width
            line, = ax.plot(t[start_idx:end_idx], y[start_idx:end_idx], 
                    color=color_dict[current_cat], linewidth=1.5, label=label)
            
            if label:
                handles.append(line)
                labels.append(label)
                
            start_idx = i
            
    ax.set_ylabel(ylabel)
    
    return handles, labels

def plot_mission(logs, naive_states, case_name, cfg, output_dir, 
                plot_accuracy_baseline=False, 
                plot_efficiency_baseline=False, 
                plot_throughput_baseline=False,
                plot_true_naive_baseline=False):
    set_plot_style()
    t_plot = np.array(logs['time_rel'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.5), sharex=True, gridspec_kw={'hspace': 0.15}) 

    baseline_colors = {
        'True_Naive': 'tab:gray',
        'High_Accuracy': 'tab:blue', 
        'High_Throughput': 'tab:red', 
        'High_Efficiency': 'tab:purple'
    }
    
    color_dict = _get_model_colors(logs['model_name'])
    
    yield_scaled = np.array(logs['cum_correct']) / 100000.0
    handles_yield, labels_yield = _plot_segmented_line(ax1, t_plot, yield_scaled, logs['model_name'], color_dict, ylabel="Yield ($10^5$ infs)")
    
    if plot_true_naive_baseline and 'True_Naive' in naive_states:
        line, = ax1.plot(t_plot, np.array(naive_states['True_Naive']['logs_cum_correct']) / 100000.0, 
                        color=baseline_colors['True_Naive'], linestyle='--', alpha=0.8, linewidth=1.2, label='Yield (True Naive)')
        handles_yield.append(line)
        labels_yield.append('Yield (True Naive)')

    if plot_accuracy_baseline and 'High_Accuracy' in naive_states:
        line, = ax1.plot(t_plot, np.array(naive_states['High_Accuracy']['logs_cum_correct']) / 100000.0, 
                        color=baseline_colors['High_Accuracy'], linestyle='--', alpha=0.8, linewidth=1.2, label='Yield (High Acc)')
        handles_yield.append(line)
        labels_yield.append('Yield (High Acc)')
        
    if plot_throughput_baseline and 'High_Throughput' in naive_states:
        line, = ax1.plot(t_plot, np.array(naive_states['High_Throughput']['logs_cum_correct']) / 100000.0, 
                        color=baseline_colors['High_Throughput'], linestyle='--', alpha=0.8, linewidth=1.2, label='Yield (High Thr)')
        handles_yield.append(line)
        labels_yield.append('Yield (High Thr)')
        
    if plot_efficiency_baseline and 'High_Efficiency' in naive_states:
        line, = ax1.plot(t_plot, np.array(naive_states['High_Efficiency']['logs_cum_correct']) / 100000.0, 
                        color=baseline_colors['High_Efficiency'], linestyle='--', alpha=0.8, linewidth=1.2, label='Yield (High Eff)')
        handles_yield.append(line)
        labels_yield.append('Yield (High Eff)')

    ax1.grid(True, alpha=0.3)

    handles_batt, labels_batt = _plot_segmented_line(ax2, t_plot, logs['battery_wh'], logs['model_name'], color_dict, ylabel="Battery (Wh)")
    
    if plot_true_naive_baseline and 'True_Naive' in naive_states:
        ax2.plot(t_plot, naive_states['True_Naive']['logs_battery_wh'], 
                color=baseline_colors['True_Naive'], linestyle='-.', alpha=0.5, linewidth=1.0)
                
    if plot_accuracy_baseline and 'High_Accuracy' in naive_states:
        ax2.plot(t_plot, naive_states['High_Accuracy']['logs_battery_wh'], 
                color=baseline_colors['High_Accuracy'], linestyle='-.', alpha=0.5, linewidth=1.0)
        
    if plot_throughput_baseline and 'High_Throughput' in naive_states:
        ax2.plot(t_plot, naive_states['High_Throughput']['logs_battery_wh'], 
                color=baseline_colors['High_Throughput'], linestyle='-.', alpha=0.5, linewidth=1.0)
        
    if plot_efficiency_baseline and 'High_Efficiency' in naive_states:
        ax2.plot(t_plot, naive_states['High_Efficiency']['logs_battery_wh'], 
                color=baseline_colors['High_Efficiency'], linestyle='-.', alpha=0.5, linewidth=1.0)

    line_lock = ax2.axhline(cfg['battery_capacity_wh']*cfg['compute_disable_pct'], color='tab:red', linestyle=':', linewidth=1.0, label='Lock Limit')
    line_resume = ax2.axhline(cfg['battery_capacity_wh']*cfg['compute_enable_pct'], color='tab:green', linestyle=':', linewidth=1.0, label='Resume Limit')
    
    lit = np.array(logs['is_lit'])
    fill_sun = ax2.fill_between(t_plot, 0, 1, where=(lit > 0.5), transform=ax2.get_xaxis_transform(), 
                    color='gold', alpha=0.15, label='Sunlight')
    
    ax2.set_xlabel('Mission Time (s)')
    ax2.grid(True, alpha=0.3)
    

    unique_labels = []
    unique_handles = []
    
    all_raw_handles = handles_yield + [line_lock, line_resume, fill_sun]
    all_raw_labels = labels_yield + ['Lock Limit', 'Resume Limit', 'Sunlight']
    
    # Deduplicate
    for h, l in zip(all_raw_handles, all_raw_labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)

    # Categorize into logical groups
    system_names = ['Lock Limit', 'Resume Limit', 'Sunlight']
    sys_group = []
    model_group = []
    baseline_group = []
    
    for h, l in zip(unique_handles, unique_labels):
        if l in system_names:
            sys_group.append((h, l))
        elif l.startswith('Yield'):
            baseline_group.append((h, l))
        else:
            model_group.append((h, l))
            
    # Sort each group
    sys_group.sort(key=lambda x: system_names.index(x[1])) # Preserve defined exact order
    model_group.sort(key=lambda x: x[1])                   # Alphabetical (groups alpha, then depth natively)
    baseline_group.sort(key=lambda x: x[1])                # Alphabetical 
    
    # Recombine (System Limits -> Dynamic Models -> Baselines)
    sorted_legend = sys_group + model_group + baseline_group
    
    final_handles = [x[0] for x in sorted_legend]
    final_labels = [x[1] for x in sorted_legend]

    ax2.legend(final_handles, final_labels, loc='upper center', bbox_to_anchor=(0.5, -0.25), 
            ncol=3, frameon=False, fontsize=6)
            
    plt.tight_layout()
    plt.subplots_adjust(top=0.90) 
    
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

    # --- Hokie colors ---
    COLOR_ORBIT = '#861F41'  # Hokie Stone maroon
    COLOR_SPEED = '#E5751F'  # Hokie orange
    
    t = np.array(logs['time_rel'])
    alt = np.array(logs['alt_km'])
    speed = np.array(logs['speed_km_s'])
    
    # 1. Find the mean altitude to identify the midpoints between apo/peri
    mean_alt = np.mean(alt)
    
    # 2. Find zero-crossings of (alt - mean_alt)
    alt_centered = alt - mean_alt
    crossings = np.where(np.diff(np.sign(alt_centered)))[0]
    
    # We need at least 3 crossings to capture a full orbit from the exact same phase
    if len(crossings) >= 3:
        start_idx = crossings[0]
        end_idx = crossings[2] 
    else:
        start_idx = 0
        end_idx = len(t) - 1
        print(f"[warn] Could not isolate a full orbit for {case_name}. Plotting available data.")
        
    t_slice = t[start_idx:end_idx]
    t_slice_norm = t_slice - t_slice[0]
    alt_slice = alt[start_idx:end_idx]
    speed_slice = speed[start_idx:end_idx]
    
    fig, ax1 = plt.subplots(figsize=(3.5, 2.4))
    
    # Plot Altitude on the left axis
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Orbital Altitude (km)', color=COLOR_ORBIT)
    line1, = ax1.plot(t_slice_norm, alt_slice, color=COLOR_ORBIT, linewidth=1.5, label='Altitude')
    ax1.tick_params(axis='y', labelcolor=COLOR_ORBIT)
    ax1.grid(True, alpha=0.3)
    
    # Plot Speed on the right axis
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Ground Track Speed (km/s)', color=COLOR_SPEED)  
    line2, = ax2.plot(t_slice_norm, speed_slice, color=COLOR_SPEED, linewidth=1.5, linestyle='--', label='Ground Speed')
    ax2.tick_params(axis='y', labelcolor=COLOR_SPEED)
    
    alt_span = np.max(alt_slice) - np.min(alt_slice)
    ax1.set_ylim(np.min(alt_slice) - (0.1 * alt_span), np.max(alt_slice) + (0.1 * alt_span))
    
    spd_span = np.max(speed_slice) - np.min(speed_slice)
    ax2.set_ylim(np.min(speed_slice) - (0.1 * spd_span), np.max(speed_slice) + (0.1 * spd_span))
    
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
    
    plt.tight_layout()
    save_path = output_dir / f"{case_name}_orbit_motivation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_static_failure_motivation(logs, naive_states, case_name, cfg, output_dir):
    """
    Isolates and plots the failure mode of a static, high-accuracy model deployment 
    during power constraints. Uses Hokie colors.
    """
    set_plot_style()
    
    if 'High_Accuracy' not in naive_states:
        print(f"[warn] High_Accuracy baseline missing for {case_name}. Cannot plot failure.")
        return
        
    t_plot = np.array(logs['time_rel'])
    battery = np.array(naive_states['High_Accuracy']['logs_battery_wh'])
    yield_arr = np.array(naive_states['High_Accuracy']['logs_cum_correct'])
    lit = np.array(logs['is_lit'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.0), sharex=True, gridspec_kw={'hspace': 0.15})
    
    # --- Hokie colors ---
    COLOR_BATT = '#861F41'   # Hokie maroon
    COLOR_YIELD = '#E5751F'  # Hokie orange
    COLOR_LINES = '#75787b'  # neutral gray
    
    # --- Top Pane: Battery & Thresholds ---
    ax1.plot(t_plot, battery, color=COLOR_BATT, linewidth=1.5, label='Battery (Static)')
    
    lock_limit = cfg['battery_capacity_wh'] * cfg['compute_disable_pct']
    resume_limit = cfg['battery_capacity_wh'] * cfg['compute_enable_pct']
    
    ax1.axhline(lock_limit, color='#51c29b', linestyle=':', linewidth=1.0, label='Lockout Threshold')
    ax1.axhline(resume_limit, color='#cd1d5b', linestyle='--', linewidth=1.0, label='Resume Threshold')
    
    failure_idx = np.where(battery < lock_limit)[0]
    if len(failure_idx) > 0:
        ax1.axvline(x=15000, color=COLOR_LINES, linestyle='-.', linewidth=1.0, label='Power Degradation')
        ax2.axvline(x=15000, color=COLOR_LINES, linestyle='-.', linewidth=1.0)
    
    ax1.fill_between(t_plot, 0, 1, where=(lit > 0.5), transform=ax1.get_xaxis_transform(), 
                    color='gold', alpha=0.15, label='Sunlight')
    
    ax1.set_ylabel('Battery (Wh)')
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # --- Bottom Pane: Cumulative Yield ---
    ax2.plot(t_plot, yield_arr, color=COLOR_YIELD, linewidth=1.5, label='Cumulative Yield')
    
    ax2.set_ylabel('Total Inferences')
    ax2.set_xlabel('Mission Time (s)')
    ax2.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / f"{case_name}_static_failure_motivation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()