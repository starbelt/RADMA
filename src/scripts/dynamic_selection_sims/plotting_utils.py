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

def plot_energy(logs, naive_states, case_name, cfg, output_dir, 
                plot_accuracy_baseline=False, 
                plot_efficiency_baseline=False, 
                plot_throughput_baseline=False,
                plot_true_naive_baseline=False,
                plot_cheapest_baseline=False,
                plot_fastest_baseline=False):
    """
    Plots a single-pane energy management trace for a mission, comparing 
    dynamic model swapping against static baselines. Formatted for single-column.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Enforce global serif font layout for native LaTeX integration
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix"
    })
    
    set_plot_style()
    t_plot = np.array(logs['time_rel'])
    
    # Single, condensed subplot for a column-width layout
    fig, ax = plt.subplots(figsize=(3.5, 4.5)) 

    baseline_colors = {
        'True_Naive': 'tab:gray',
        'High_Accuracy': 'tab:blue', 
        'High_Throughput': 'tab:green',
        'High_Efficiency': 'tab:yellow',
        'Cheapest': 'tab:purple',
        'Fastest': 'tab:red'
    }
    
    
    color_dict = _get_model_colors(logs['model_name'])
    
    # dynamic
    handles_batt, labels_batt = _plot_segmented_line(ax, t_plot, logs['battery_wh'], logs['model_name'], color_dict, ylabel="Battery (Wh)")
    
    # baselines
    if plot_true_naive_baseline and 'True_Naive' in naive_states:
        line, = ax.plot(t_plot, naive_states['True_Naive']['logs_battery_wh'], 
                color=baseline_colors['True_Naive'],  alpha=0.5, linewidth=1.5, label='Batt (True Naive)')
        handles_batt.append(line)
        labels_batt.append('Batt (True Naive)')
                
    if plot_accuracy_baseline and 'High_Accuracy' in naive_states:
        line, = ax.plot(t_plot, naive_states['High_Accuracy']['logs_battery_wh'], 
                color=baseline_colors['High_Accuracy'],  alpha=0.5, linewidth=1.5, label='Batt (High Acc)')
        handles_batt.append(line)
        labels_batt.append('Batt (High Acc)')
        
    if plot_throughput_baseline and 'High_Throughput' in naive_states:
        line, = ax.plot(t_plot, naive_states['High_Throughput']['logs_battery_wh'], 
                color=baseline_colors['High_Throughput'], alpha=0.5, linewidth=1.5, label='Batt (High Thr)')
        handles_batt.append(line)
        labels_batt.append('Batt (High Thr)')
        
    if plot_efficiency_baseline and 'High_Efficiency' in naive_states:
        line, = ax.plot(t_plot, naive_states['High_Efficiency']['logs_battery_wh'], 
                color=baseline_colors['High_Efficiency'], alpha=0.5, linewidth=1.5, label='Batt (High Eff)')
        handles_batt.append(line)
        labels_batt.append('Batt (High Eff)')

    if plot_cheapest_baseline and 'Cheapest' in naive_states:
        line, = ax.plot(t_plot, naive_states['Cheapest']['logs_battery_wh'], 
                color=baseline_colors['Cheapest'], alpha=0.5, linewidth=1.5, label='Batt (Max Inf/J)')
        handles_batt.append(line)
        labels_batt.append('Batt (Max Inf/J)')

    if plot_fastest_baseline and 'Fastest' in naive_states:
        line, = ax.plot(t_plot, naive_states['Cheapest']['logs_battery_wh'], 
                color=baseline_colors['Cheapest'], alpha=0.5, linewidth=1.5, label='Batt (Max Inf/s)')
        handles_batt.append(line)
        labels_batt.append('Batt (Max Inf/s)')

    # System Thresholds
    line_lock = ax.axhline(cfg['battery_capacity_wh']*cfg['compute_disable_pct'], color='tab:red', linestyle=':', linewidth=1.0, label='Lock Limit')
    line_resume = ax.axhline(cfg['battery_capacity_wh']*cfg['compute_enable_pct'], color='tab:green', linestyle=':', linewidth=1.0, label='Resume Limit')
    
    # Sunlight Fill
    lit = np.array(logs['is_lit'])
    fill_sun = ax.fill_between(t_plot, 0, 1, where=(lit > 0.5), transform=ax.get_xaxis_transform(), 
                    color='gold', alpha=0.15, label='Sunlight')
    
    # Paper-ready styling updates
    ax.set_xlabel('Mission Time (s)', fontsize=12)
    ax.set_ylabel('Battery (Wh)', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    unique_labels = []
    unique_handles = []
    
    all_raw_handles = handles_batt + [line_lock, line_resume, fill_sun]
    all_raw_labels = labels_batt + ['Lock Limit', 'Resume Limit', 'Sunlight']
    
    for h, l in zip(all_raw_handles, all_raw_labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)

    system_names = ['Lock Limit', 'Resume Limit', 'Sunlight']
    sys_group = []
    model_group = []
    baseline_group = []
    
    # Categorize handles for grouped legend display
    for h, l in zip(unique_handles, unique_labels):
        if l in system_names:
            sys_group.append((h, l))
        elif l.startswith('Batt ('):
            baseline_group.append((h, l))
        else:
            model_group.append((h, l))
            
    sys_group.sort(key=lambda x: system_names.index(x[1])) 
    model_group.sort(key=lambda x: x[1])                   
    baseline_group.sort(key=lambda x: x[1])                
    

    sys_handles = [x[0] for x in sys_group]
    sys_labels = [x[1] for x in sys_group]
    
    sys_legend = ax.legend(sys_handles, sys_labels, loc='upper left', 
                           frameon=True, framealpha=0.9, edgecolor='none', fontsize=8)
    ax.add_artist(sys_legend) # Prevents the second legend from overwriting this one
    
    bottom_items = model_group + baseline_group
    bottom_handles = [x[0] for x in bottom_items]
    bottom_labels = [x[1] for x in bottom_items]
    
    ax.legend(bottom_handles, bottom_labels, loc='upper center', bbox_to_anchor=(0.5, -0.22), 
            ncol=2, frameon=False, fontsize=8) # Changed ncol=3 to 2 to narrow the footprint
            
    plt.subplots_adjust(bottom=0.45) 

    save_path = Path(output_dir) / f"{case_name}_STATIC.pdf"
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

def plot_energy_margin(logs, naive_states, case_name, output_dir, 
                        plot_accuracy_baseline=False, 
                        plot_efficiency_baseline=False, 
                        plot_throughput_baseline=False,
                        plot_true_naive_baseline=False,
                        plot_cheapest_baseline=False,
                        plot_fastest_baseline=False):
    """
    Plots the Energy Margin (Surplus vs Deficit) against 
    the dynamic requirements of the orbit, color-coded by active model.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Enforce global serif font layout for native LaTeX integration
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix"
    })

    set_plot_style()
    t_plot = np.array(logs['time_rel'])
    demand = np.array(logs['demand_infs'])
    budget = np.array(logs['step_budget_j'])

    # Standard single-column sizing
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    baseline_colors = {
        'True_Naive': 'tab:gray',
        'High_Accuracy': 'tab:blue',
        'High_Throughput': 'tab:green',
        'High_Efficiency': 'tab:yellow',
        'Cheapest': 'tab:purple',
        'Fastest': 'tab:red'
    }

    handles_base = []
    labels_base = []

    # --- Static Baselines (Togglable) ---
    # Margin = Budget - Required Energy (Demand * Model Energy Cost)
    if plot_true_naive_baseline and 'True_Naive' in naive_states:
        margin = budget - (demand * naive_states['True_Naive']['model']['eng_j'])
        line, = ax.plot(t_plot, margin, color=baseline_colors['True_Naive'],
                alpha=0.8, linewidth=1.0, label='True Naive')
        handles_base.append(line)
        labels_base.append('True Naive')

    if plot_accuracy_baseline and 'High_Accuracy' in naive_states:
        margin = budget - (demand * naive_states['High_Accuracy']['model']['eng_j'])
        line, = ax.plot(t_plot, margin, color=baseline_colors['High_Accuracy'],
                alpha=0.8, linewidth=1.0, label='High Accuracy')
        handles_base.append(line)
        labels_base.append('High Accuracy')
        
    if plot_throughput_baseline and 'High_Throughput' in naive_states:
        margin = budget - (demand * naive_states['High_Throughput']['model']['eng_j'])
        line, = ax.plot(t_plot, margin, color=baseline_colors['High_Throughput'],
                alpha=0.8, linewidth=1.0, label='High Throughput')
        handles_base.append(line)
        labels_base.append('High Throughput')
        
    if plot_efficiency_baseline and 'High_Efficiency' in naive_states:
        margin = budget - (demand * naive_states['High_Efficiency']['model']['eng_j'])
        line, = ax.plot(t_plot, margin, color=baseline_colors['High_Efficiency'],
                alpha=0.8, linewidth=1.0, label='High Efficiency')
        handles_base.append(line)
        labels_base.append('High Efficiency')
    
    if plot_cheapest_baseline and 'Cheapest' in naive_states:
        margin = budget - (demand * naive_states['Cheapest']['model']['eng_j'])
        line, = ax.plot(t_plot, margin, color=baseline_colors['Cheapest'],
                alpha=0.8, linewidth=1.0, label='Maximum inf/J')
        handles_base.append(line)
        labels_base.append('Maximum inf/J')
    
    if plot_fastest_baseline and 'Fastest' in naive_states:
        margin = budget - (demand * naive_states['Fastest']['model']['eng_j'])
        line, = ax.plot(t_plot, margin, color=baseline_colors['Fastest'],
                alpha=0.8, linewidth=1.0, label='Maximum inf/s')
        handles_base.append(line)
        labels_base.append('Maximum inf/s')

    # --- Dynamic System (RADMA) --- 
    active_eng_j = np.array(logs['active_eng_j'])
    dynamic_margin = budget - (demand * active_eng_j)
    
    # Generate the color mapping and plot the segmented dynamic line
    color_dict = _get_model_colors(logs['model_name'])
    handles_dyn, labels_dyn = _plot_segmented_line(
        ax, t_plot, dynamic_margin, logs['model_name'], color_dict, ylabel='Energy Margin (J)'
    )

    # Zero Line (Requirement exactly met)
    line_zero = ax.axhline(0, color='black', linestyle=':', linewidth=1.0, alpha=0.5, label='Budget Met (0 Margin)')

    # Highlighting
    fill_surplus = ax.fill_between(t_plot, 0, dynamic_margin, where=(dynamic_margin > 0),
                    color='tab:green', alpha=0.15, label='Surplus Energy')
    fill_deficit = ax.fill_between(t_plot, 0, dynamic_margin, where=(dynamic_margin < 0),
                    color='tab:red', alpha=0.15, label='Energy Deficit')

    # Formatting overrides
    ax.set_ylabel('Energy Margin (J/step)', fontsize=10)
    ax.set_xlabel('Mission Time (s)', fontsize=10)
    ax.tick_params(axis='both', labelsize=8)

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    all_raw_handles = handles_dyn + handles_base + [line_zero, fill_surplus, fill_deficit]
    all_raw_labels = labels_dyn + labels_base + ['Budget Met (0 Margin)', 'Surplus Energy', 'Energy Deficit']
    
    unique_labels = []
    unique_handles = []
    
    # Deduplicate legend items
    for h, l in zip(all_raw_handles, all_raw_labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)

    ax.legend(unique_handles, unique_labels, loc='upper center',
            bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=8)
    
    ax.set_xlim(0,30000)
    ax.set_ylim(-0.6
                ,0.6)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35) # Make room for the legend underneath

    save_path = Path(output_dir) / f"{case_name}_energy_margin.pdf"
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

def plot_inference_margin(logs, naive_states, case_name, output_dir, 
                        plot_accuracy_baseline=False, 
                        plot_efficiency_baseline=False, 
                        plot_throughput_baseline=False,
                        plot_true_naive_baseline=False,
                        plot_cheapest_baseline=False,
                        plot_fastest_baseline=False):
    """
    Plots the Inference Throughput Margin (Surplus vs Deficit) against 
    the dynamic requirements of the orbit, color-coded by active model.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Enforce global serif font layout for native LaTeX integration
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix"
    })

    set_plot_style()
    t_plot = np.array(logs['time_rel'])
    demand = np.array(logs['demand_infs'])

    # Standard single-column sizing
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    baseline_colors = {
        'True_Naive': 'tab:gray',
        'High_Accuracy': 'tab:blue',
        'High_Throughput': 'tab:green',
        'High_Efficiency': 'tab:yellow',
        'Cheapest': 'tab:purple',
        'Fastest': 'tab:red'
    }

    handles_base = []
    labels_base = []

    # --- Static Baselines (Togglable) ---
    if plot_true_naive_baseline and 'True_Naive' in naive_states:
        margin = naive_states['True_Naive']['ips'] - demand
        line, = ax.plot(t_plot, margin, color=baseline_colors['True_Naive'],
                alpha=0.8, linewidth=1.0, label='True Naive')
        handles_base.append(line)
        labels_base.append('True Naive')

    if plot_accuracy_baseline and 'High_Accuracy' in naive_states:
        margin = naive_states['High_Accuracy']['ips'] - demand
        line, = ax.plot(t_plot, margin, color=baseline_colors['High_Accuracy'],
                alpha=0.8, linewidth=1.0, label='High Accuracy')
        handles_base.append(line)
        labels_base.append('High Accuracy')
        
    if plot_throughput_baseline and 'High_Throughput' in naive_states:
        margin = naive_states['High_Throughput']['ips'] - demand
        line, = ax.plot(t_plot, margin, color=baseline_colors['High_Throughput'],
                alpha=0.8, linewidth=1.0, label='High Throughput')
        handles_base.append(line)
        labels_base.append('High Throughput')
        
    if plot_efficiency_baseline and 'High_Efficiency' in naive_states:
        margin = naive_states['High_Efficiency']['ips'] - demand
        line, = ax.plot(t_plot, margin, color=baseline_colors['High_Efficiency'],
                alpha=0.8, linewidth=1.0, label='High Efficiency')
        handles_base.append(line)
        labels_base.append('High Efficiency')
    
    if plot_cheapest_baseline and 'Cheapest' in naive_states:
        margin = naive_states['Cheapest']['ips'] - demand
        line, = ax.plot(t_plot, margin, color=baseline_colors['Cheapest'],
                alpha=0.8, linewidth=1.0, label='Maximum inf/J')
        handles_base.append(line)
        labels_base.append('Maximum inf/J')
    
    if plot_fastest_baseline and 'Fastest' in naive_states:
        margin = naive_states['Fastest']['ips'] - demand
        line, = ax.plot(t_plot, margin, color=baseline_colors['Fastest'],
                alpha=0.8, linewidth=1.0, label='Maximum inf/s')
        handles_base.append(line)
        labels_base.append('FMaximum inf/s')

    # --- Dynamic System (RADMA) --- 
    active_ips = np.array(logs['active_ips'])
    dynamic_margin = active_ips - demand
    
    # Generate the color mapping and plot the segmented dynamic line
    color_dict = _get_model_colors(logs['model_name'])
    handles_dyn, labels_dyn = _plot_segmented_line(
        ax, t_plot, dynamic_margin, logs['model_name'], color_dict, ylabel='Inference Margin (Inf/s)'
    )

    # Zero Line (Requirement exactly met)
    line_zero = ax.axhline(0, color='black', linestyle=':', linewidth=1.0, alpha=0.5, label='Demand Met (0 Margin)')

    # Highlighting
    fill_surplus = ax.fill_between(t_plot, 0, dynamic_margin, where=(dynamic_margin > 0),
                    color='tab:green', alpha=0.15, label='Surplus Capacity')
    fill_deficit = ax.fill_between(t_plot, 0, dynamic_margin, where=(dynamic_margin < 0),
                    color='tab:red', alpha=0.15, label='Processing Deficit')

    # Formatting overrides
    ax.set_ylabel('Throughput Margin (inf/s)', fontsize=10)
    ax.set_xlabel('Mission Time (s)', fontsize=10)
    ax.tick_params(axis='both', labelsize=8)

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    all_raw_handles = handles_dyn + handles_base + [line_zero, fill_surplus, fill_deficit]
    all_raw_labels = labels_dyn + labels_base + ['Demand Met (0 Margin)', 'Surplus Capacity', 'Processing Deficit']
    
    unique_labels = []
    unique_handles = []
    
    # Deduplicate legend items
    for h, l in zip(all_raw_handles, all_raw_labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)

    ax.legend(unique_handles, unique_labels, loc='upper center',
            bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35) # Make room for the legend underneath

    save_path = Path(output_dir) / f"{case_name}_throughput_margin.pdf"
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

def plot_mission(logs, naive_states, case_name, cfg, output_dir, 
                plot_accuracy_baseline=False, 
                plot_efficiency_baseline=False, 
                plot_throughput_baseline=False,
                plot_true_naive_baseline=False):
    import numpy as np
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix"
    })
    
    set_plot_style()
    t_plot = np.array(logs['time_rel'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.5), sharex=True, gridspec_kw={'hspace': 0.15}) 

    baseline_colors = {
        'True_Naive': 'tab:gray',
        'High_Accuracy': 'tab:blue', 
        'High_Throughput': 'tab:green',
        'High_Efficiency': 'tab:yellow',
        'Cheapest': 'tab:purple',
        'Fastest': 'tab:red'
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

    # Styling updates
    ax1.set_ylabel("Yield ($10^5$ infs)", fontsize=12)
    ax1.tick_params(axis='y', labelsize=10)
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

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
    
    # Styling updates
    ax2.set_xlabel('Mission Time (s)', fontsize=12)
    ax2.set_ylabel('Battery (Wh)', fontsize=12)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    

    unique_labels = []
    unique_handles = []
    
    all_raw_handles = handles_yield + [line_lock, line_resume, fill_sun]
    all_raw_labels = labels_yield + ['Lock Limit', 'Resume Limit', 'Sunlight']
    
    for h, l in zip(all_raw_handles, all_raw_labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)

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
            
    sys_group.sort(key=lambda x: system_names.index(x[1])) 
    model_group.sort(key=lambda x: x[1])                   
    baseline_group.sort(key=lambda x: x[1])                
    
    sorted_legend = sys_group + model_group + baseline_group
    
    final_handles = [x[0] for x in sorted_legend]
    final_labels = [x[1] for x in sorted_legend]

    # Bumped legend fontsize from 6 to 9
    ax2.legend(final_handles, final_labels, loc='upper center', bbox_to_anchor=(0.5, -0.25), 
            ncol=3, frameon=False, fontsize=9)
            
    plt.tight_layout()
    plt.subplots_adjust(top=0.90) 
    
    # Save as PDF
    save_path = output_dir / f"{case_name}_STATIC.pdf"
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

def plot_delivered_yield(logs, naive_states, case_name, output_dir, 
                        plot_accuracy_baseline=False, 
                        plot_efficiency_baseline=False, 
                        plot_throughput_baseline=False,
                        plot_true_naive_baseline=False,
                        plot_cheapest_baseline=False,
                        plot_fastest_baseline=False):
    """
    Plots the instantaneous correct inferences delivered versus the ideal demand.
    Shows the actual accuracy-adjusted productivity of the system over time.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Enforce global serif font layout for native LaTeX integration
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix"
    })

    set_plot_style()
    t_plot = np.array(logs['time_rel'])
    demand = np.array(logs['demand_infs'])

    # Standard single-column sizing
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    baseline_colors = {
        'True_Naive': 'tab:gray',
        'High_Accuracy': 'tab:blue',
        'High_Throughput': 'tab:green',
        'High_Efficiency': 'tab:yellow',
        'Cheapest': 'tab:purple',
        'Fastest': 'tab:red'
    }

    handles_base = []
    labels_base = []

    # 1. Plot Ideal Demand (The Envelope)
    # This represents 100% accuracy on all demanded inferences
    fill_ideal = ax.fill_between(t_plot, 0, demand, color='tab:gray', alpha=0.10, label='Ideal Yield (Demand)')
    line_ideal, = ax.plot(t_plot, demand, color='black', linestyle=':', linewidth=1.0, alpha=0.5, label='Demand Envelope')

    # Helper function to extract step-by-step correct inferences from cumulative logs
    def get_baseline_step_yield(ns):
        cum = np.array(ns['logs_cum_correct'])
        # Prepend the first value so the array length matches t_plot after differencing
        return np.concatenate(([cum[0]], np.diff(cum)))

    # --- Static Baselines (Togglable) ---
    if plot_true_naive_baseline and 'True_Naive' in naive_states:
        step_yield = get_baseline_step_yield(naive_states['True_Naive'])
        line, = ax.plot(t_plot, step_yield, color=baseline_colors['True_Naive'],
                alpha=0.8, linewidth=1.0, label='True Naive')
        handles_base.append(line)
        labels_base.append('True Naive')

    if plot_accuracy_baseline and 'High_Accuracy' in naive_states:
        step_yield = get_baseline_step_yield(naive_states['High_Accuracy'])
        line, = ax.plot(t_plot, step_yield, color=baseline_colors['High_Accuracy'],
                alpha=0.8, linewidth=1.0, label='High Accuracy')
        handles_base.append(line)
        labels_base.append('High Accuracy')
        
    if plot_throughput_baseline and 'High_Throughput' in naive_states:
        step_yield = get_baseline_step_yield(naive_states['High_Throughput'])
        line, = ax.plot(t_plot, step_yield, color=baseline_colors['High_Throughput'],
                alpha=0.8, linewidth=1.0, label='High Throughput')
        handles_base.append(line)
        labels_base.append('High Throughput')
        
    if plot_efficiency_baseline and 'High_Efficiency' in naive_states:
        step_yield = get_baseline_step_yield(naive_states['High_Efficiency'])
        line, = ax.plot(t_plot, step_yield, color=baseline_colors['High_Efficiency'],
                alpha=0.8, linewidth=1.0, label='High Efficiency')
        handles_base.append(line)
        labels_base.append('High Efficiency')
    
    if plot_cheapest_baseline and 'Cheapest' in naive_states:
        step_yield = get_baseline_step_yield(naive_states['Cheapest'])
        line, = ax.plot(t_plot, step_yield, color=baseline_colors['Cheapest'],
                linestyle='--', alpha=0.8, linewidth=1.0, label='Maximum inf/J')
        handles_base.append(line)
        labels_base.append('Maximum inf/J')
    
    if plot_fastest_baseline and 'Fastest' in naive_states:
        step_yield = get_baseline_step_yield(naive_states['Fastest'])
        line, = ax.plot(t_plot, step_yield, color=baseline_colors['Fastest'],
                linestyle='--', alpha=0.8, linewidth=1.0, label='Maximum inf/s')
        handles_base.append(line)
        labels_base.append('Maximum inf/s')

    # --- Dynamic System (RADMA) --- 
    # Handle NaNs in accuracy where processed infs were 0
    throughput = np.array(logs['throughput_infs'])
    accuracy = np.nan_to_num(np.array(logs['avg_accuracy']), nan=0.0)
    dynamic_yield = throughput * accuracy
    
    # Generate the color mapping and plot the segmented dynamic line
    color_dict = _get_model_colors(logs['model_name'])
    handles_dyn, labels_dyn = _plot_segmented_line(
        ax, t_plot, dynamic_yield, logs['model_name'], color_dict, ylabel='Correct Inferences / Step'
    )

    # Formatting overrides
    ax.set_xlabel('Mission Time (s)', fontsize=10)
    ax.tick_params(axis='both', labelsize=8)

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Dual Legend Construction ---
    all_raw_handles = handles_dyn + handles_base + [line_ideal, fill_ideal]
    all_raw_labels = labels_dyn + labels_base + ['Demand Envelope', 'Ideal Yield']
    
    unique_labels = []
    unique_handles = []
    
    # Deduplicate legend items
    for h, l in zip(all_raw_handles, all_raw_labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)

    system_names = ['Demand Envelope', 'Ideal Yield']
    sys_group, model_group, baseline_group = [], [], []
    
    for h, l in zip(unique_handles, unique_labels):
        if l in system_names:
            sys_group.append((h, l))
        elif l in labels_base:
            baseline_group.append((h, l))
        else:
            model_group.append((h, l))
            
    sys_group.sort(key=lambda x: system_names.index(x[1])) 
    
    # 1. System Legend (Inside Upper Right, to stay out of the way of the low valleys)
    sys_handles = [x[0] for x in sys_group]
    sys_labels = [x[1] for x in sys_group]
    
    sys_legend = ax.legend(sys_handles, sys_labels, loc='upper right', 
                        frameon=True, framealpha=0.9, edgecolor='none', fontsize=8)
    ax.add_artist(sys_legend)
    
    # 2. Models & Baselines Legend (Outside Bottom)
    bottom_items = model_group + baseline_group
    bottom_handles = [x[0] for x in bottom_items]
    bottom_labels = [x[1] for x in bottom_items]
    
    ax.legend(bottom_handles, bottom_labels, loc='upper center', bbox_to_anchor=(0.5, -0.22), 
            ncol=2, frameon=False, fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.40) # Make room for the legend underneath

    save_path = Path(output_dir) / f"{case_name}_delivered_yield.pdf"
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
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
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Force paper-quality serif fonts globally
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix"
    })
    
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
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Orbital Altitude (km)', color=COLOR_ORBIT, fontsize=12)
    line1, = ax1.plot(t_slice_norm, alt_slice, color=COLOR_ORBIT, linewidth=1.5, label='Altitude')
    ax1.tick_params(axis='both', labelsize=10)
    ax1.tick_params(axis='y', labelcolor=COLOR_ORBIT)
    ax1.grid(False) # Removed grid for pure white background
    ax1.spines['top'].set_visible(False)
    
    # Plot Speed on the right axis
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Ground Track Speed (km/s)', color=COLOR_SPEED, fontsize=12)  
    line2, = ax2.plot(t_slice_norm, speed_slice, color=COLOR_SPEED, linewidth=1.5, linestyle='--', label='Ground Speed')
    ax2.tick_params(axis='y', labelsize=10, labelcolor=COLOR_SPEED)
    ax2.spines['top'].set_visible(False)
    
    alt_span = np.max(alt_slice) - np.min(alt_slice)
    ax1.set_ylim(np.min(alt_slice) - (0.1 * alt_span), np.max(alt_slice) + (0.1 * alt_span))
    
    spd_span = np.max(speed_slice) - np.min(speed_slice)
    ax2.set_ylim(np.min(speed_slice) - (0.1 * spd_span), np.max(speed_slice) + (0.1 * spd_span))
    
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False, fontsize=10)
    
    plt.tight_layout()
    # Save as PDF
    save_path = output_dir / f"{case_name}_orbit_motivation.pdf"
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

def plot_static_failure_motivation(logs, naive_states, case_name, cfg, output_dir):
    """
    Isolates and plots the failure mode of a static, high-accuracy model deployment 
    during power constraints. Uses Hokie colors. Formatted for paper layout.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Enforce global serif font layout for native LaTeX integration
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix"
    })
    
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
    
    # --- Top Pane: Cumulative Yield ---
    ax1.plot(t_plot, yield_arr, color=COLOR_YIELD, linewidth=1.5, label='Cumulative Yield')
    
    # Styling updates for top pane
    ax1.set_ylabel('Total Inferences', fontsize=12)
    ax1.tick_params(axis='both', labelsize=10)
    
    # Force scientific notation on the y-axis
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # Ensure the '1e5' multiplier matches the tick font size
    ax1.yaxis.get_offset_text().set_fontsize(10)
    
    ax1.grid(False) # Removed grid
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # Legend directly on the subplot
    ax1.legend(loc='upper left', frameon=True, fontsize=10) 
    
    # --- Bottom Pane: Battery & Thresholds ---
    ax2.plot(t_plot, battery, color=COLOR_BATT, linewidth=1.5, label='Battery (Static)')
    
    lock_limit = cfg['battery_capacity_wh'] * cfg['compute_disable_pct']
    resume_limit = cfg['battery_capacity_wh'] * cfg['compute_enable_pct']
    
    ax2.axhline(lock_limit, color='#51c29b', linestyle=':', linewidth=1.0, label='Lockout Threshold')
    ax2.axhline(resume_limit, color='#cd1d5b', linestyle='--', linewidth=1.0, label='Resume Threshold')
    
    failure_idx = np.where(battery < lock_limit)[0]
    if len(failure_idx) > 0:
        # Plotted on both panes for visual continuity, but NO labels so they don't enter the legends
        ax1.axvline(x=10000, color=COLOR_LINES, linestyle='-.', linewidth=1.0)
        ax2.axvline(x=10000, color=COLOR_LINES, linestyle='-.', linewidth=1.0)
    
    ax2.fill_between(t_plot, 0, 1, where=(lit > 0.5), transform=ax2.get_xaxis_transform(), 
                    color='gold', alpha=0.15, label='Sunlight')
    
    # Styling updates for bottom pane
    ax2.set_ylabel('Battery (Wh)', fontsize=12)
    ax2.set_xlabel('Mission Time (s)', fontsize=12)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.grid(False) # Removed grid
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Battery Legend underneath the bottom pane (2 columns fits 4 items perfectly)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), 
            ncol=2, frameon=False, fontsize=10)
    
    # Use tight_layout and add extra room at the bottom so the external legend isn't clipped
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25) 
    
    # Save as PDF for vector graphics
    save_path = output_dir / f"{case_name}_static_failure_motivation.pdf"
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()