import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def set_plot_style():
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'lines.linewidth': 3.0,
        'figure.titlesize': 24,
        'figure.titleweight': 'bold'
    })

def plot_orbit_dynamics(logs, case_name, output_dir):
    set_plot_style()
    clean_name = case_name.replace('_', ' ')
    t_plot = np.array(logs['time_rel'])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"case study: {clean_name}\norbit & data dynamics", y=0.98)

    ax1.plot(t_plot, logs['alt_km'], color='gray', label='altitude')
    ax1.set_ylabel('altitude (km)')
    
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

def plot_telemetry(logs, case_name, cfg, output_dir):
    set_plot_style()
    clean_name = case_name.replace('_', ' ')
    t_plot = np.array(logs['time_rel'])
    
    fig = plt.figure(figsize=(14, 18))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1.5, 1.5, 1], hspace=0.3)
    fig.suptitle(f"case study: {clean_name}\ndynamic telemetry", y=0.98)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3])

    ax1.plot(t_plot, logs['alt_km'], color='gray', label='altitude')
    ax1.set_ylabel('altitude (km)')
    
    ax1_t = ax1.twinx()
    ax1_t.plot(t_plot, logs['speed_km_s'], 'r--', label='speed')
    ax1_t.set_ylabel('speed (km/s)', color='r')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_t.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.plot(t_plot, logs['battery_wh'], 'g', label='dynamic battery')
    ax2.axhline(cfg['battery_capacity_wh']*cfg['compute_disable_pct'], color='r', linestyle=':', label='hard min (shutoff)')
    ax2.axhline(cfg['battery_capacity_wh']*cfg['compute_enable_pct'], color='g', linestyle=':', label='ref resume')
    
    lit = np.array(logs['is_lit'])
    ax2.fill_between(t_plot, 0, 1, where=(lit > 0.5), transform=ax2.get_xaxis_transform(), color='gold', alpha=0.2, label='sunlight')
    ax2.set_ylabel('battery (wh)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax3.plot(t_plot, logs['cum_correct'], 'k', label='dynamic correct infs')
    ax3.set_ylabel('total correct inferences')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    ax3_t = ax3.twinx()
    ax3_t.fill_between(t_plot, logs['backlog_infs'], color='tab:orange', alpha=0.15, label='backlog')
    ax3_t.plot(t_plot, logs['backlog_infs'], color='tab:orange', linewidth=1.5, alpha=0.6)
    ax3_t.set_ylabel('dynamic backlog size', color='tab:orange')
    ax3_t.tick_params(axis='y', labelcolor='tab:orange')
    ax3.set_xlabel('time (s)')

    df_log = pd.DataFrame({'model': logs['model_name'], 'infs': logs['throughput_infs']})
    stats = df_log[df_log['infs'] > 0].groupby('model')['infs'].sum().sort_values()
    
    if not stats.empty:
        bars = ax4.barh(stats.index, stats.values, color='tab:purple')
        ax4.bar_label(bars, fmt='{:,.0f}', padding=5, fontsize=12)
    else:
        ax4.text(0.5, 0.5, "no inferences performed", ha='center', va='center', fontsize=16)
    
    ax4.set_xlabel('total inferences processed')
    ax4.grid(True, axis='x', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = output_dir / f"{case_name}.png"
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

    plt.tight_layout(rect=[0, 0, 1, 0.94])
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