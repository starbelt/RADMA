import sys
import json
import pathlib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D

# --- User Imports ---
from saleae_parsing import SaleaeOutputParsing
from path_utils import get_repo_root

# --- Utility Functions ---

def lighten_color(color, factor=0.5):
    """Lightens the given color."""
    r, g, b = matplotlib.colors.to_rgb(color)
    return (1 - factor) + factor * r, (1 - factor) + factor * g, (1 - factor) + factor * b

class GridStatsPlotting:
    def __init__(self, json_dir, saleae_root, output_dir):
        self.json_dir = pathlib.Path(json_dir)
        self.saleae_root = pathlib.Path(saleae_root)
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None

    def load_and_aggregate_data(self):
        alphas = [0.25, 0.5, 0.75, 1.0, 1.25]
        depths = [2, 4, 6, 8, 10]
        
        data_rows = []
        psu_dc_volts = 5.0
        r_shunt = 0.2

        print(f"[INFO] Aggregating data from {self.saleae_root}...")

        for alpha in alphas:
            for depth in depths:
                # 1. Load JSON Metrics
                json_name = f"Grid_A{alpha}_D{depth:02d}_quant_eval.json"
                json_path = self.json_dir / json_name
                top1 = 0.0
                
                if json_path.exists():
                    try:
                        with open(json_path, 'r') as f:
                            jdata = json.load(f)
                        top1 = jdata.get('top1', 0)
                    except Exception:
                        pass

                # 2. Load Saleae Metrics
                subdir_path = self.saleae_root / f"A_{alpha}" / f"D_{depth:02d}"
                
                inf_ms_hardware = np.nan
                power_mw = np.nan
                energy_mj = np.nan

                if subdir_path.exists():
                    raw_folders = list(subdir_path.rglob("saleae_raw"))
                    if raw_folders:
                        target_dir = raw_folders[0].parent
                        try:
                            parsed = SaleaeOutputParsing(target_dir)
                            if parsed.avg_inference_time() is not None:
                                inf_ms_hardware = parsed.avg_inference_time() * 1e3
                                mean_pwr, _, mean_energy, _ = parsed.avg_power_measurement(psu_dc_volts, r_shunt)
                                if mean_pwr and mean_energy:
                                    power_mw = mean_pwr * 1e3
                                    energy_mj = mean_energy * 1e3
                        except Exception as e:
                            print(f"[ERR] {target_dir.name}: {e}")
                
                model_name = f"Grid A{alpha} D{depth:02d}"
                
                if not np.isnan(inf_ms_hardware) and not np.isnan(energy_mj):
                    data_rows.append({
                        "Model name": model_name,
                        "Alpha": alpha,
                        "Depth": depth,
                        "Top-1 Accuracy": top1,
                        "Measured Inference Time (ms)": inf_ms_hardware,
                        "Energy per Inference (mJ)": energy_mj,
                        "Average Power (mW)": power_mw
                    })

        self.df = pd.DataFrame(data_rows)
        
        if not self.df.empty:
            self.df["Inf_per_Sec"] = 1000.0 / self.df["Measured Inference Time (ms)"]
            self.df["Inf_per_Joule"] = 1000.0 / self.df["Energy per Inference (mJ)"]
            acc_frac = self.df["Top-1 Accuracy"]
            self.df["Correct_Inf_per_Sec"] = self.df["Inf_per_Sec"] * acc_frac
            self.df["Correct_Inf_per_Joule"] = self.df["Inf_per_Joule"] * acc_frac
            print(f"[INFO] Loaded {len(self.df)} runs.")
        return self.df

    def find_champions(self):
        """Identifies and returns the names of the two best models."""
        if self.df is None or self.df.empty: return None, None
        
        # Best Efficiency
        best_eff_idx = self.df["Correct_Inf_per_Joule"].idxmax()
        name_eff = self.df.loc[best_eff_idx, "Model name"]
        
        # Best Throughput
        best_rate_idx = self.df["Correct_Inf_per_Sec"].idxmax()
        name_rate = self.df.loc[best_rate_idx, "Model name"]

        print(f"\n[ANALYSIS] Champions:")
        print(f"  Efficiency: {name_eff}")
        print(f"  Throughput: {name_rate}")
        return name_eff, name_rate

    # ------------------------------------------------------------------
    # Plot 3D Surface (Flexible)
    # ------------------------------------------------------------------
    def plot_3d_surface(self, specific_models=None, filename="grid_3d_surface.png", title_suffix=""):
        if self.df is None or self.df.empty: return

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        buffers = np.linspace(0.1, 10.0, 30) 
        times = np.linspace(0.1, 10.0, 30)   
        B_grid, T_grid = np.meshgrid(buffers, times)

        # Filter Data
        if specific_models:
            subset = self.df[self.df["Model name"].isin(specific_models)]
            alpha_val = 0.8  # Higher opacity for specific models
        else:
            subset = self.df # All models
            alpha_val = 0.15 # Lower opacity to see through the clutter

        cmap = plt.get_cmap("turbo") # High contrast colormap
        unique_names = subset["Model name"].unique()
        
        for idx, row in subset.iterrows():
            name = row["Model name"]
            acc_frac = row["Top-1 Accuracy"] / 100.0
            energy_j = row["Energy per Inference (mJ)"] * 1e-3
            lat_s = row["Measured Inference Time (ms)"] * 1e-3
            
            rate_e = acc_frac / energy_j 
            rate_t = acc_frac / lat_s    
            
            Z = np.minimum(B_grid * rate_e, T_grid * rate_t)
            
            # Color based on alpha value in the model name (visual grouping)
            c_val = (row["Alpha"] - 0.25) / 1.0  # Normalize 0.25-1.25 to 0-1
            color = cmap(c_val)
            
            surf = ax.plot_surface(B_grid, T_grid, Z, alpha=alpha_val, color=color, label=name)

        ax.set_title(f"3D Correct Inference Surface {title_suffix}", fontsize=16)
        ax.set_xlabel("Energy Buffer (Joules)")
        ax.set_ylabel("Pass Time (Seconds)")
        ax.set_zlabel("Correct Inferences")
        ax.view_init(elev=30, azim=135)
        
        # Legend (Simplify if too many)
        if specific_models:
            legend_elements = [Patch(facecolor=cmap((self.df.loc[self.df["Model name"]==m, "Alpha"].values[0]-0.25)), label=m) for m in specific_models]
            ax.legend(handles=legend_elements)

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[PLOT] Saved {out_path}")
        plt.close()

    # ------------------------------------------------------------------
    # Other Plotting Functions (Overview, Frontier, Budget)
    # ------------------------------------------------------------------
    def plot_efficiency_overview(self):
        if self.df is None or self.df.empty: return
        subset = self.df.sort_values("Correct_Inf_per_Joule", ascending=True).copy()
        names = subset["Model name"].tolist()
        cmap = plt.get_cmap("viridis")
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        
        def plot_stack(ax, sort_col, total_col, correct_col, title, unit):
            sorted_df = self.df.sort_values(sort_col, ascending=True)
            local_names = sorted_df["Model name"].tolist()
            total = sorted_df[total_col].to_numpy()
            correct = sorted_df[correct_col].to_numpy()
            x = np.arange(len(local_names))
            local_colors = [cmap(i/len(local_names)) for i in range(len(local_names))]
            for i in range(len(x)):
                c_base = local_colors[i]
                c_light = lighten_color(c_base, 0.5)
                ax.bar(x[i], correct[i], color=c_base)
                ax.bar(x[i], total[i] - correct[i], bottom=correct[i], color=c_light)
            ax.set_title(title, fontsize=18)
            ax.set_ylabel(unit, fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(local_names, rotation=90, fontsize=10)
        
        plot_stack(axes[0], "Correct_Inf_per_Sec", "Inf_per_Sec", "Correct_Inf_per_Sec", "Throughput", "Inf/s")
        plot_stack(axes[1], "Correct_Inf_per_Joule", "Inf_per_Joule", "Correct_Inf_per_Joule", "Efficiency", "Inf/J")
        plt.tight_layout()
        plt.savefig(self.output_dir / "grid_efficiency_overview.png", dpi=300)
        plt.close()

    def plot_decision_frontier(self, model_a_name, model_b_name):
        subset = self.df[self.df["Model name"].isin([model_a_name, model_b_name])].copy()
        if len(subset) != 2: return

        subset["Slope"] = (subset["Top-1 Accuracy"]/100) / (subset["Energy per Inference (mJ)"]*1e-3)
        subset["Rate"]  = (subset["Top-1 Accuracy"]/100) / (subset["Measured Inference Time (ms)"]*1e-3)
        subset.sort_values("Slope", ascending=False, inplace=True)
        
        m_eff, m_pwr = subset.iloc[0], subset.iloc[1]
        name_eff, name_pwr = m_eff["Model name"], m_pwr["Model name"]
        eff_rate, pwr_rate = m_eff["Rate"], m_pwr["Rate"]
        eff_slope, pwr_slope = m_eff["Slope"], m_pwr["Slope"]

        fig, ax = plt.subplots(figsize=(10, 6))
        max_t = 10.0
        time_range = np.linspace(0, max_t, 200)

        if eff_rate >= pwr_rate:
            ax.fill_between(time_range, 0, max_t*1000, color='tab:blue', alpha=0.2)
            ax.text(max_t*0.5, max_t*10, f"{name_eff} DOMINATES", ha='center', fontweight='bold')
            title = f"{name_eff} Wins Globally"
        else:
            k = eff_rate / pwr_slope
            boundary_energy = time_range * k
            ax.plot(time_range, boundary_energy, 'k--', linewidth=2)
            ax.fill_between(time_range, 0, boundary_energy, color='tab:blue', alpha=0.2)
            ax.text(max_t*0.75, max(boundary_energy)*0.25, f"Use {name_eff}\n(Energy Limited)", ha='center', color='tab:blue')
            ax.fill_between(time_range, boundary_energy, max(boundary_energy)*1.5, color='tab:orange', alpha=0.2)
            ax.text(max_t*0.25, max(boundary_energy)*1.25, f"Use {name_pwr}\n(Time Limited)", ha='center', color='tab:orange')
            ax.set_ylim(0, max(boundary_energy)*1.5)
            title = f"Decision Frontier: {name_eff} vs {name_pwr}"

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Energy (J)")
        ax.set_xlim(0, max_t)
        plt.savefig(self.output_dir / "grid_decision_frontier.png", dpi=300)
        plt.close()

    def run_budget_loop(self, buffers, frame_times):
        if self.df is None or self.df.empty: return
        with pd.ExcelWriter(self.output_dir / "grid_budget_tables.xlsx") as writer:
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

if __name__ == "__main__":
    REPO_ROOT = get_repo_root()
    JSON_DIR = "/home/jackr/Repos/CoralGUI/libs/coral_tpu_characterization/data/tpunet_acc"
    SALEAE_ROOT = "/home/jackr/Repos/CoralGUI/results/captures_1_20"
    OUTPUT_DIR = REPO_ROOT / "results/plots/grid_analysis"

    plotter = GridStatsPlotting(JSON_DIR, SALEAE_ROOT, OUTPUT_DIR)
    plotter.load_and_aggregate_data()
    
    # 1. Overview
    plotter.plot_efficiency_overview()
    
    # 2. Find Champions
    eff, rate = plotter.find_champions()

    # 3. Plot 3D: ALL Models
    plotter.plot_3d_surface(specific_models=None, filename="grid_3d_all.png", title_suffix="(All Models)")
    
    # 4. Plot 3D: Champions Only
    if eff and rate:
        # Use a list to ensure set logic doesn't remove duplicates if champions are same
        champs = list(set([eff, rate]))
        plotter.plot_3d_surface(specific_models=champs, filename="grid_3d_champions.png", title_suffix="(Champions Only)")
        plotter.plot_decision_frontier(eff, rate)

    # 5. Budget Tables
    buffers = [0.5 * c * 25 for c in [0.01, 0.1, 1.0, 5.0]]
    times = [1.0, 5.0, 10.0, 30.0, 60.0]
    plotter.run_budget_loop(sorted(buffers), times)
    
    print("[DONE] Generated all plots.")