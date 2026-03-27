import sys
import json
import pathlib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('TkAgg')

# Climb up until we find the project root 'CoralGUI'
current_file = pathlib.Path(__file__).resolve()
project_root = None

for parent in current_file.parents:
    if parent.name == "CoralGUI":
        project_root = parent
        break

if project_root:
    sys.path.insert(0, str(project_root))
else:
    
    sys.path.insert(0, str(current_file.parents[5]))


from libs.coral_tpu_characterization.src.scripts.utils.saleae_parsing import SaleaeOutputParsing
from libs.coral_tpu_characterization.src.scripts.utils.path_utils import get_repo_root
from libs.coral_tpu_characterization.src.scripts.utils.ParamCounts import ParamCounts


plt.rcParams.update({'font.size': 16})  # Base font size

def lighten_color(color, factor=0.5):
    """Lightens the given color."""
    r, g, b = matplotlib.colors.to_rgb(color)
    return (1 - factor) + factor * r, (1 - factor) + factor * g, (1 - factor) + factor * b

class GridStatsPlotting:
    def __init__(self, json_dir, saleae_root, output_dir):
        self.json_dir = pathlib.Path(json_dir)
        self.modeldir = self.json_dir.parent / "models/custom/tfliteCPU"
        self.saleae_root = pathlib.Path(saleae_root)
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None

    def load_and_aggregate_data(self):
        alphas = ["0.25", "0.50", "0.75", "1.0", "1.25", "1.50"]
        depths = [2, 4, 6, 8, 10, 12]
        
        data_rows = []
        psu_dc_volts = 5.0
        r_shunt = 0.2

        print(f"\n[INFO] Starting Data Aggregation...")
        print(f"       Repo Root:   {get_repo_root()}")
        print(f"       Saleae Root: {self.saleae_root}")
        print(f"       JSON Dir:    {self.json_dir}")

        if not self.saleae_root.exists():
            print(f"[CRITICAL] Saleae root does not exist: {self.saleae_root}")
            return None
        if not self.modeldir.exists():
            print(f"[CRITICAL] model root does not exist: {self.modeldir}")
            return None
        
        # Check param counts
        pc_list, pc_dict = ParamCounts(self.modeldir).scan_models()
        
        for alpha in alphas:
            for depth in depths:
                # Load JSON Metrics
                json_name = f"Grid_A{float(alpha)}_D{depth:02d}_quant_eval.json" 
                json_path = self.json_dir / json_name
                top1 = 0.0
                
                if json_path.exists():
                    try:
                        with open(json_path, 'r') as f:
                            jdata = json.load(f)
                        top1 = jdata.get('top1', 0)
                    except Exception as e:
                        print(f"[WARN] Failed to read JSON {json_name}: {e}")
                else:
                    print(f"[WARN] Missing JSON file: {json_name}")

                # Load Saleae Metrics
                subdir_path = self.saleae_root / f"A_{alpha}" / f"D_{depth:02d}"
                
                inf_ms_hardware = np.nan
                power_mw = np.nan
                energy_mj = np.nan

                if not subdir_path.exists():
                    continue

                raw_folders = list(subdir_path.rglob("saleae_raw"))
                if not raw_folders:
                    print(f"[SKIP] No 'saleae_raw' found inside: {subdir_path}")
                    continue
                
                target_dir = raw_folders[0].parent
                
                try:
                    parsed = SaleaeOutputParsing(target_dir)
                    print(f"  > Processing: {target_dir.name}...", flush=True)
                    
                    if parsed.avg_inference_time() is None:
                        print(f"[SKIP] {target_dir.name}: parsed.avg_inference_time() returned None")
                        continue
                        
                    inf_ms_hardware = parsed.avg_inference_time() * 1e3
                    mean_pwr, _, mean_energy, _ = parsed.avg_power_measurement(psu_dc_volts, r_shunt)
                    
                    if mean_pwr is None or mean_energy is None:
                         print(f"[SKIP] {target_dir.name}: Power parsing returned None")
                         continue

                    power_mw = mean_pwr * 1e3
                    energy_mj = mean_energy * 1e3

                except Exception as e:
                    print(f"[ERR]  Exception parsing {target_dir.name}: {e}")
                    continue
                
                # success
                model_name = f"Grid A{float(alpha)} D{depth:02d}"
                
                data_rows.append({
                    "Model name": model_name,
                    "Alpha": float(alpha),
                    "Depth": depth,
                    "Top-1 Accuracy": top1*100.0,
                    "Measured Inference Time (ms)": inf_ms_hardware,
                    "Energy per Inference (mJ)": energy_mj,
                    "Average Power (mW)": power_mw,
                    "Parameter Count (M)" : pc_dict.get(f"Grid_A{float(alpha)}_D{depth:02d}_quant.tflite", np.nan)*1e-6 # in millions
                })

        self.df = pd.DataFrame(data_rows)
        
        if self.df.empty:
            print("\n[CRITICAL] No valid runs were loaded! Check paths and debug messages above.")
            return self.df

        # Derived Metrics
        self.df["Inf_per_Sec"] = 1000.0 / self.df["Measured Inference Time (ms)"]
        self.df["Inf_per_Joule"] = 1000.0 / self.df["Energy per Inference (mJ)"]
        acc_frac = self.df["Top-1 Accuracy"]*0.01
        self.df["Correct_Inf_per_Sec"] = self.df["Inf_per_Sec"] * acc_frac
        self.df["Correct_Inf_per_Joule"] = self.df["Inf_per_Joule"] * acc_frac
        
        print(f"\n[SUCCESS] Loaded {len(self.df)} runs successfully.")
        return self.df

    def find_champions(self):
        if self.df is None or self.df.empty: return None, None
        best_eff_idx = self.df["Correct_Inf_per_Joule"].idxmax()
        name_eff = self.df.loc[best_eff_idx, "Model name"]
        best_rate_idx = self.df["Correct_Inf_per_Sec"].idxmax()
        name_rate = self.df.loc[best_rate_idx, "Model name"]
        print(f"\n[ANALYSIS] Champions:")
        print(f"  Efficiency: {name_eff}")
        print(f"  Throughput: {name_rate}")
        return name_eff, name_rate

    # standard metrics, sorted by energy per inference
    def plot_standard_metrics(self, filename="grid_standard_metrics.png"):
        if self.df is None or self.df.empty: return
        subset = self.df.sort_values("Energy per Inference (mJ)", ascending=True).copy()
        names = subset["Model name"].tolist()
        power = subset["Average Power (mW)"].to_numpy()
        energy = subset["Energy per Inference (mJ)"].to_numpy()
        latency = subset["Measured Inference Time (ms)"].to_numpy()
        accuracy = subset["Top-1 Accuracy"].to_numpy()
        paramcount = subset["Parameter Count (M)"].to_numpy()

        # param_counts = [x/1e6 for x in pc.scan_models()]
        # TODO: Plot param count
        
        cmap = matplotlib.colormaps["viridis"]
        colors = [cmap(i / len(names)) for i in range(len(names))]
        x_pos = np.arange(len(names))
        
        # INCREASED FIGSIZE
        fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(20, 22))

        def plot_row(ax_idx, data, title, unit, ylim_top=None):
            ax = axes[ax_idx]
            ax.bar(x_pos, data, color=colors)
            ax.set_title(title, fontsize=28) 
            ax.set_ylabel(unit, fontsize=24)
            ax.tick_params(axis="y", labelsize=20) 
            if ylim_top: ax.set_ylim(0, ylim_top)
            else: ax.set_ylim(0, max(data) * 1.15)
            
            # Value labels
            for i, v in enumerate(data):
                ax.text(x_pos[i], v * 1.02, f"{v:.1f}", ha="center", va="bottom", fontsize=18, rotation=45)

        plot_row(0, paramcount, "Parameter Count", "Millions")
        plot_row(1, power, "Average Power", "mW",1200)
        plot_row(2, energy, "Energy per Inference", "mJ")
        plot_row(3, latency, "Measured Latency", "ms")
        plot_row(4, accuracy, "Top-1 Accuracy", "%", ylim_top=100)
        
        axes[3].set_xticks(x_pos)
        axes[3].set_xticklabels(names, rotation=45, ha="right", fontsize=18)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        print(f"[PLOT] Saved {filename}")
        plt.close()

    # Standard Metrics grouped by alpha, ordered by depth
    def plot_grouped_metrics(self, filename="grid_grouped_metrics.png"):
        """
        Plots 4 rows (Power, Energy, Latency, Accuracy) grouped by Alpha.
        Bars represent different Depths.
        """
        if self.df is None or self.df.empty: return

        alphas = sorted(self.df['Alpha'].unique())
        depths = sorted(self.df['Depth'].unique())
        
        x = np.arange(len(alphas))  # label locations for alphas
        width = 0.8 / len(depths)   # width of individual bars

        # Colors for depth levels
        cmap = plt.get_cmap('magma_r') 
        colors = cmap(np.linspace(0.2, 0.8, len(depths)))

        fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(20, 22))

        # Helper to plot one metric
        def plot_group_row(ax_idx, metric_col, title, ylabel, ylim_top=None):
            if self.df is None or self.df.empty: return # How do i handle var scope here?
            ax = axes[ax_idx]
            
            for i, depth in enumerate(depths):
                subset = self.df[self.df['Depth'] == depth]
                
                # Align values to correct alpha index
                heights = []
                for alpha in alphas:
                    val = subset[subset['Alpha'] == alpha][metric_col]
                    heights.append(val.item() if not val.empty else 0)
                
                offset = (i - len(depths)/2) * width + width/2
                # Only label the first plot for the legend
                label = f'Depth={depth}' if ax_idx == 0 else ""
                rects = ax.bar(x + offset, heights, width, label=label, color=colors[i])
                
                # Optional: Add text labels on bars if needed (can get crowded)
                for j, h in enumerate(heights):
                    if h > 0: ax.text(x[j] + offset, h*1.01, f"{h:.1f}", ha='center', va='bottom', fontsize=18, rotation=45)

            ax.set_title(title, fontsize=28)
            ax.set_ylabel(ylabel, fontsize=24)
            ax.tick_params(axis="y", labelsize=20)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            
            if ylim_top: ax.set_ylim(0, ylim_top)
            else: ax.autoscale(enable=True, axis='y', tight=False)

        # Parameter Count
        plot_group_row(0, "Parameter Count (M)", "Parameter Count", "Millions", ylim_top = 12)
        axes[0].legend(title="Depthwise Seperable Layer Repeats", loc='upper left', fontsize=18, title_fontsize=20, ncol=len(depths))

        # Power
        plot_group_row(1, "Average Power (mW)", "Average Power", "mW", ylim_top = 1400)
        
        # Energy
        plot_group_row(2, "Energy per Inference (mJ)", "Energy per Inference", "mJ",80)

        # Latency
        plot_group_row(3, "Measured Inference Time (ms)", "Measured Latency", "ms",90)

        # Accuracy
        plot_group_row(4, "Top-1 Accuracy", "Top-1 Accuracy", "%", ylim_top=100)

        # X-Axis
        axes[3].set_xticks(x)
        axes[3].set_xticklabels([f"Alpha {a}" for a in alphas], fontsize=22)
        axes[3].set_xlabel("Width Multiplier (Alpha)", fontsize=24)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        print(f"[PLOT] Saved {filename}")
        plt.close()

    # Efficiency Overview
    def plot_efficiency_overview(self):
        if self.df is None or self.df.empty:
            return

        subset = self.df.sort_values("Correct_Inf_per_Joule", ascending=True).copy()
        names = subset["Model name"].tolist()
        cmap = plt.get_cmap("viridis")
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 12))

        def plot_stack(ax, sort_col, total_col, correct_col, title, unit):
            sorted_df = self.df.sort_values(sort_col, ascending=True)
            local_names = sorted_df["Model name"].tolist()
            total = sorted_df[total_col].to_numpy()
            correct = sorted_df[correct_col].to_numpy()

            x = np.arange(len(local_names))
            local_colors = [cmap(i / len(local_names)) for i in range(len(local_names))]

            for i in range(len(x)):
                c_base = local_colors[i]
                c_light = lighten_color(c_base, 0.5)

                ax.bar(x[i], correct[i], color=c_base)
                ax.bar(x[i], total[i] - correct[i], bottom=correct[i], color=c_light)

                # --- value annotations (1 decimal place) ---
                ax.text(
                    x[i],
                    correct[i] / 2,
                    f"{correct[i]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white",
                    fontweight="bold",
                )

                ax.text(
                    x[i],
                    total[i],
                    f"{total[i]:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="black",
                )

            ax.set_title(title, fontsize=28)
            ax.set_ylabel(unit, fontsize=24)
            ax.tick_params(axis="y", labelsize=20)
            ax.set_xticks(x)
            ax.set_xticklabels(local_names, rotation=45, ha="right", fontsize=14)

        plot_stack(
            axes[0],
            "Correct_Inf_per_Sec",
            "Inf_per_Sec",
            "Correct_Inf_per_Sec",
            "Throughput",
            "Inf/s",
        )

        plot_stack(
            axes[1],
            "Correct_Inf_per_Joule",
            "Inf_per_Joule",
            "Correct_Inf_per_Joule",
            "Efficiency",
            "Inf/J",
        )

        plt.tight_layout()
        plt.savefig(self.output_dir / "grid_efficiency_overview.png", dpi=300)
        plt.close()

    # 3D Surface

    def plot_3d_surface(self, specific_models=None, filename="grid_3d_surface.png", title_suffix=""):
        if self.df is None or self.df.empty: return
        fig = plt.figure(figsize=(16, 14))
        ax = fig.add_subplot(111, projection='3d')
        buffers = np.linspace(0.1, 10.0, 30) 
        times = np.linspace(0.1, 10.0, 30)   
        B_grid, T_grid = np.meshgrid(buffers, times)
        
        if specific_models:
            subset = self.df[self.df["Model name"].isin(specific_models)]
            alpha_val = 0.8
        else:
            subset = self.df
            alpha_val = 0.15

        cmap = plt.get_cmap("turbo") 
        
        for idx, row in subset.iterrows():
            name = row["Model name"]
            acc_frac = row["Top-1 Accuracy"] / 100.0
            energy_j = row["Energy per Inference (mJ)"] * 1e-3
            lat_s = row["Measured Inference Time (ms)"] * 1e-3
            rate_e = acc_frac / energy_j 
            rate_t = acc_frac / lat_s    
            Z = np.minimum(B_grid * rate_e, T_grid * rate_t)
            c_val = (row["Alpha"] - 0.25) / 1.0
            color = cmap(c_val)
            ax.plot_surface(B_grid, T_grid, Z, alpha=alpha_val, color=color, label=name)

        ax.set_title(f"3D Correct Inference Surface {title_suffix}", fontsize=28)
        ax.set_xlabel("Energy Buffer (Joules)", fontsize=20, labelpad=15)
        ax.set_ylabel("Pass Time (Seconds)", fontsize=20, labelpad=15)
        ax.set_zlabel("Correct Inferences", fontsize=20, labelpad=15)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.view_init(elev=30, azim=135)
        
        if specific_models:
            legend_elements = [Patch(facecolor=cmap((self.df.loc[self.df["Model name"]==m, "Alpha"].values[0]-0.25)), label=m) for m in specific_models]
            ax.legend(handles=legend_elements, fontsize=18)
            
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        print(f"[PLOT] Saved {filename}")
        plt.close()

    # Decision Frontier

    def plot_decision_frontier(self, model_a_name, model_b_name):
        if self.df is None or self.df.empty:
            print("No data to plot decision frontier")
            return  
        subset = self.df[self.df["Model name"].isin([model_a_name, model_b_name])].copy()
        if len(subset) != 2: print("[ANALYSIS] The one True Model has been found");return
        subset["Slope"] = (subset["Top-1 Accuracy"]/100) / (subset["Energy per Inference (mJ)"]*1e-3)
        subset["Rate"]  = (subset["Top-1 Accuracy"]/100) / (subset["Measured Inference Time (ms)"]*1e-3)
        subset.sort_values("Slope", ascending=False, inplace=True)
        m_eff, m_pwr = subset.iloc[0], subset.iloc[1]
        name_eff, name_pwr = m_eff["Model name"], m_pwr["Model name"]
        eff_rate, pwr_rate = m_eff["Rate"], m_pwr["Rate"]
        eff_slope, pwr_slope = m_eff["Slope"], m_pwr["Slope"]

        fig, ax = plt.subplots(figsize=(14, 10))
        max_t = 10.0
        time_range = np.linspace(0, max_t, 200)

        if eff_rate >= pwr_rate:
            ax.fill_between(time_range, 0, max_t*1000, color='tab:blue', alpha=0.2)
            ax.text(max_t*0.5, max_t*10, f"{name_eff} DOMINATES", ha='center', fontweight='bold', fontsize=24, bbox=dict(facecolor='white', alpha=0.8))
            title = f"{name_eff} Wins Globally"
        else:
            k = eff_rate / pwr_slope
            boundary_energy = time_range * k
            ax.plot(time_range, boundary_energy, 'k--', linewidth=3)
            ax.fill_between(time_range, 0, boundary_energy, color='tab:blue', alpha=0.2)
            ax.text(max_t*0.75, max(boundary_energy)*0.25, f"Use {name_eff}\n(Energy Limited)", ha='center', color='tab:blue', fontsize=20, fontweight='bold')
            ax.fill_between(time_range, boundary_energy, max(boundary_energy)*1.5, color='tab:orange', alpha=0.2)
            ax.text(max_t*0.25, max(boundary_energy)*1.25, f"Use {name_pwr}\n(Time Limited)", ha='center', color='tab:orange', fontsize=20, fontweight='bold')
            ax.set_ylim(0, max(boundary_energy)*1.5)
            title = f"Decision Frontier: {name_eff} vs {name_pwr}"

        ax.set_title(title, fontsize=28)
        ax.set_xlabel("Time (s)", fontsize=24)
        ax.set_ylabel("Energy (J)", fontsize=24)
        ax.tick_params(axis='both', labelsize=20)
        ax.set_xlim(0, max_t)
        plt.savefig(self.output_dir / "grid_decision_frontier.png", dpi=300)
        print(f"[PLOT] Saved grid_decision_frontier.png")
        plt.close()

    # Budget Loop to generate excel tables
    def run_budget_loop(self, buffers, frame_times):
        if self.df is None or self.df.empty: return
        xl_path = self.output_dir / "grid_budget_tables.xlsx"
        with pd.ExcelWriter(xl_path) as writer:
            for _, row in self.df.iterrows():
                name = row["Model name"]
                ej = row["Energy per Inference (mJ)"] * 1e-3
                ls = row["Measured Inference Time (ms)"] * 1e-3
                acc = row["Top-1 Accuracy"] / 100.0
                grid = np.empty((len(buffers), len(frame_times)), dtype=object)
                for i, b in enumerate(buffers):
                    for j, t in enumerate(frame_times):
                        val = min(int((t/ls)*acc), int((b/ej)*acc))
                        lim = "T" if (t/ls) < (b/ej) else "E"
                        grid[i, j] = f"{val} ({lim})"
                pd.DataFrame(grid, index=[f"E:{b:.2f}" for b in buffers], columns=[f"T:{t:.2f}" for t in frame_times]).to_excel(writer, sheet_name=name.replace("Grid ", "").replace(" ", "_"))
        print(f"[EXCEL] Saved {xl_path}")

    def plot_model_selection_heatmap(self, resolution=300, filename="grid_selection_heatmap.png"):
        """
        Generates a 2D heatmap showing which model to select based on required 
        Inferences per Second and Inferences per Joule constraints.
        """
        if self.df is None or self.df.empty:
            print("[WARN] No data available for heatmap.")
            return

        print(f"\n[PLOT] Generating Selection Heatmap (Resolution: {resolution}x{resolution})...")

        # Define the grid space based on our max achievable metrics
        max_inf_sec = self.df["Inf_per_Sec"].max() * 1.05
        max_inf_joule = self.df["Inf_per_Joule"].max() * 1.05
        
        req_inf_sec = np.linspace(0, max_inf_sec, resolution)
        req_inf_joule = np.linspace(0, max_inf_joule, resolution)
        X, Y = np.meshgrid(req_inf_sec, req_inf_joule)

        # Extract model stats as numpy arrays for fast vectorized comparisons
        models = self.df["Model name"].values
        inf_sec = self.df["Inf_per_Sec"].values
        inf_joule = self.df["Inf_per_Joule"].values
        
        # Tie-breaker: If multiple models meet the throughput and efficiency needs, 
        # we select the one with the highest Top-1 Accuracy.
        tie_breaker_scores = self.df["Top-1 Accuracy"].values 

        # Create a 3D boolean mask: (num_models, res_y, res_x)
        # True if a model satisfies the grid point's minimum requirements
        valid_mask = (inf_sec[:, None, None] >= X) & (inf_joule[:, None, None] >= Y)

        # Apply the tie-breaker
        # Initialize scores with -1 (unachievable)
        scores = np.full_like(valid_mask, -1.0, dtype=float)
        # Broadcast the tie-breaker scores onto the valid coordinates
        scores[valid_mask] = np.broadcast_to(tie_breaker_scores[:, None, None], valid_mask.shape)[valid_mask]

        # Find the index of the model with the highest score at each grid point
        best_idx = np.argmax(scores, axis=0)
        max_scores = np.max(scores, axis=0)
        
        # Identify regions where NO model can meet the requirements
        best_idx[max_scores == -1] = -1

        # Map indices to colors for plotting
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Create an RGB image array (Default to light grey for 'Unachievable')
        Z_color = np.full((resolution, resolution, 3), 0.9) 
        
        unique_winners = np.unique(best_idx)
        valid_winners = [uid for uid in unique_winners if uid != -1]
        
        # Generate distinct colors for our models (scalable to 25+)
        cmap = matplotlib.colormaps["turbo"]
        model_colors = cmap(np.linspace(0.05, 0.95, len(models)))
        
        legend_patches = []
        for uid in valid_winners:
            mask = (best_idx == uid)
            Z_color[mask] = model_colors[uid][:3]
            legend_patches.append(Patch(color=model_colors[uid], label=models[uid]))
            
        legend_patches.append(Patch(color=(0.9, 0.9, 0.9), label='Unachievable Constraints'))

        # Draw the heatmap
        ax.imshow(
            Z_color, 
            origin='lower', 
            extent=[0, max_inf_sec, 0, max_inf_joule], 
            aspect='auto'
        )

        ax.set_title("Model Selection Heatmap", fontsize=28)
        ax.set_xlabel("Required Inferences / Second", fontsize=24)
        ax.set_ylabel("Required Inferences / Joule", fontsize=24)
        ax.tick_params(axis='both', labelsize=18)
        
        ax.legend(
            handles=legend_patches, 
            loc='center left', 
            bbox_to_anchor=(1.02, 0.5), 
            fontsize=14, 
            title="Selected Model", 
            title_fontsize=16
        )

        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[PLOT] Saved {filename}")
        plt.close()

    def plot_3d_accuracy_surface(self, resolution=100, filename="grid_3d_accuracy_surface.png", interactive=False):
        # TODO: 4 side videw of cube + sliced version
        if self.df is None or self.df.empty:
            print("[warn] no data available for 3d plot.")
            return

        print(f"\n[plot] generating 3d accuracy surface (resolution: {resolution}x{resolution})...")

        # define the grid space
        max_inf_sec = self.df["Inf_per_Sec"].max() * 1.05
        max_inf_joule = self.df["Inf_per_Joule"].max() * 1.05
        
        req_inf_sec = np.linspace(0, max_inf_sec, resolution)
        req_inf_joule = np.linspace(0, max_inf_joule, resolution)
        X, Y = np.meshgrid(req_inf_sec, req_inf_joule)

        # extract arrays
        models = self.df["Model name"].values
        inf_sec = self.df["Inf_per_Sec"].values
        inf_joule = self.df["Inf_per_Joule"].values
        accuracies = self.df["Top-1 Accuracy"].values 

        # boolean mask for valid constraints
        valid_mask = (inf_sec[:, None, None] >= X) & (inf_joule[:, None, None] >= Y)

        # broadcast accuracies and find the winner
        scores = np.full_like(valid_mask, -1.0, dtype=float)
        scores[valid_mask] = np.broadcast_to(accuracies[:, None, None], valid_mask.shape)[valid_mask]

        Z = np.max(scores, axis=0)
        best_idx = np.argmax(scores, axis=0)

        # mask unachievable space
        best_idx[Z == -1] = -1
        Z[Z == -1] = 0.0  

        # build the color grid matching the 2d heatmap
        cmap = plt.get_cmap("turbo")
        model_colors = cmap(np.linspace(0.05, 0.95, len(models)))
        
        # default everything to light grey (rgba)
        color_grid = np.full((resolution, resolution, 4), [0.9, 0.9, 0.9, 1.0])
        
        unique_winners = np.unique(best_idx)
        valid_winners = [uid for uid in unique_winners if uid != -1]
        
        legend_patches = []
        for uid in valid_winners:
            color_grid[best_idx == uid] = model_colors[uid]
            legend_patches.append(Patch(color=model_colors[uid], label=models[uid]))
            
        legend_patches.append(Patch(color=(0.9, 0.9, 0.9), label='Unachievable'))

        # pad the geometry for solid volume walls
        X_pad = np.pad(X, pad_width=1, mode='edge')
        Y_pad = np.pad(Y, pad_width=1, mode='edge')
        Z_pad = np.pad(Z, pad_width=1, mode='constant', constant_values=0.0)
        
        # pad the colors using 'edge' so the model colors drag down the cliff faces
        color_grid_pad = np.pad(color_grid, ((1, 1), (1, 1), (0, 0)), mode='edge')

        def render_subplot(ax, elev, azim, hide_axis=None):
            ax.plot_surface(
                X_pad, Y_pad, Z_pad, 
                facecolors=color_grid_pad,
                edgecolor='black',
                linewidth=0.3,
                antialiased=True,
                rcount=45,
                ccount=45
            )


            # Push the back-panes out by 10% so the data doesn't swallow the axis lines
            ax.set_xlim(-8, max_inf_sec)
            ax.set_ylim(-8, max_inf_joule)
            ax.set_zlim(0, 100) 
            
            # Subplot titles moved closer
            ax.set_xlabel("Required Inferences / Sec" if hide_axis != 'x' else "", fontsize=12, labelpad=5)
            ax.set_ylabel("Required Inferences / Joule" if hide_axis != 'y' else "", fontsize=12, labelpad=5)
            ax.set_zlabel("Top-1 Acc (%)" if hide_axis != 'z' else "", fontsize=12, labelpad=5)
            
            # Completely remove the ticks for the hidden axis to avoid the thick black overlapping line
            if hide_axis == 'x':
                ax.set_xticks([])
            elif hide_axis == 'y':
                ax.set_yticks([])
            elif hide_axis == 'z':
                ax.set_zticks([])
            
            ax.tick_params(axis='both', which='major', labelsize=9, pad=0)
            
            # set camera angle
            ax.view_init(elev=elev, azim=azim) 

            # ZOOM HACK: Force the 3D plot to fill more of its invisible bounding box
            try:
                ax.set_box_aspect(None, zoom=1.2) # For modern matplotlib versions
            except AttributeError:
                ax.dist = 7 # Fallback for older matplotlib versions (default is 10)
            ax.margins(0) # Strip extra margins

        if interactive:
            # Single large interactive plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            render_subplot(ax, elev=35, azim=230)
            ax.set_title("Maximum Achievable Accuracy vs. Resource Constraints", fontsize=20, pad=15)
            
            # Shared bottom legend for interactive view
            fig.legend(
                handles=legend_patches, loc='lower center', 
                ncol=min(len(legend_patches), 6), bbox_to_anchor=(0.5, 0.02),
                title="Selected Model", title_fontsize=14, fontsize=12
            )
            
            plt.subplots_adjust(bottom=0.15)
            print("[plot] opening interactive 3d window. close the window to continue...")
            plt.show()
            
        else:
            fig = plt.figure(figsize=(20.5, 6)) 
            fig.suptitle("Maximum Achievable Accuracy vs. Resource Constraints", fontsize=22, y=1.05)

            # top 
            ax1 = fig.add_subplot(141, projection='3d')
            render_subplot(ax1, elev=90, azim=-90, hide_axis='z')

            ax2 = fig.add_subplot(142, projection='3d')
            render_subplot(ax2, elev=0, azim=180, hide_axis='x')


            ax3 = fig.add_subplot(143, projection='3d')
            render_subplot(ax3, elev=0, azim=-90, hide_axis='y')

            # iso view -> hide nothing
            ax4 = fig.add_subplot(144, projection='3d')
            render_subplot(ax4, elev=35, azim=230)

            plt.subplots_adjust(wspace=-0.25, bottom=0.25, top=0.9, left=0.0, right=1.0)
            
            # Shared bottom legend
            fig.legend(
                handles=legend_patches, loc='lower center', 
                ncol=min(len(legend_patches), 8), bbox_to_anchor=(0.5, -0.05),
                title="Selected Model", title_fontsize=14, fontsize=12
            )
            
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
            print(f"[plot] saved {filename}")
            
        plt.close()

if __name__ == "__main__":
    REPO_ROOT = get_repo_root()
    
    # RELATIVE PATHS
    JSON_DIR = REPO_ROOT / "data/tpunet_acc"
    SALEAE_ROOT = REPO_ROOT / "../../results/captures_1_20" # so cursed
    OUTPUT_DIR = REPO_ROOT / "results/plots/grid_analysis"

    plotter = GridStatsPlotting(JSON_DIR, SALEAE_ROOT, OUTPUT_DIR)
    plotter.load_and_aggregate_data()
    
    if plotter.df is not None and not plotter.df.empty:

        # plotter.plot_standard_metrics()
        # plotter.plot_grouped_metrics()
        # plotter.plot_efficiency_overview()
        # eff, rate = plotter.find_champions()

        plotter.plot_model_selection_heatmap(resolution=500)
        plotter.plot_3d_accuracy_surface(resolution = 300, interactive=False)

        # plotter.plot_3d_surface(specific_models=None, filename="grid_3d_all.png", title_suffix="(All Models)")
        # if eff and rate:
        #     champs = list(set([eff, rate]))
        #     plotter.plot_3d_surface(specific_models=champs, filename="grid_3d_champions.png", title_suffix="(Champions Only)")
        #     plotter.plot_decision_frontier(eff, rate)

        # buffers = [0.5 * c * 25 for c in [0.01, 0.1, 1.0, 5.0]]
        # times = [1.0, 5.0, 10.0, 30.0, 60.0]
        # plotter.run_budget_loop(sorted(buffers), times)
        
        print("\n[DONE] Script completed successfully.")
    else:
        print("\n[FAIL] Script ended without generating plots due to missing data.")