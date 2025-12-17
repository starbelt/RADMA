import sys

import matplotlib, pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path
from adjustText import adjust_text
from ParamCounts import ParamCounts
from saleae_parsing import SaleaeOutputParsing
from path_utils import get_repo_root

REPO_ROOT = get_repo_root()


# Data Collection and Sorting Functions ## 
def inference_from_csvs(rundir):
    """
    If you only need the inference time and no power metrics (e.g. if you only have a digital.csv file)
    """
    top_dir = REPO_ROOT / "results/captures" / rundir
    inference_time_ms = []
    for root, dirs, files in top_dir.walk():
        for d in sorted(dirs):
            parsed = SaleaeOutputParsing(root / d)
            if parsed.avg_inference_time() is not None:
                inference_time_ms.append(parsed.avg_inference_time() * 1e3)
            else:
                inference_time_ms.append(None)
    return inference_time_ms

def collect_results(run_dir:pathlib.Path, psu_dc_volts=5.0, r_shunt=0.2):
    """
    Collects primary results of digital and analog Saleae results into a dict for ease of use.
    Point to directory that directly parents the .CSVs
    """
    results = {}
    for subdir in sorted(run_dir.iterdir()):
        if not subdir.is_dir():
            continue
        try:
            parsed = SaleaeOutputParsing(subdir)
        except FileNotFoundError:
            continue
        if parsed.avg_inference_time() is not None:
            avg_time = parsed.avg_inference_time() * 1e3  # ms
            mean_pwr, std_pwr, mean_energy, all_energy = parsed.avg_power_measurement(
                psu_dc_volts, r_shunt
            )
            # only add if both digital+analog are present
            if mean_pwr is not None:
                results[subdir.name] = {
                    "inference_time_ms": avg_time,
                    "avg_power_mW": mean_pwr * 1e3,
                    "energy_mJ": mean_energy * 1e3 if mean_energy else 0
                }
        else:
            raise Exception("No average inference time found")
    return results

## Plot generation Functions ##

def lighten_color(color, factor=0.5):
    """
    Lightens the given color.
    factor=0 -> white, factor=1 -> original color.
    """
    r, g, b = matplotlib.colors.to_rgb(color) # pyright: ignore[reportAttributeAccessIssue] 
    return (1 - factor) + factor * r, (1 - factor) + factor * g, (1 - factor) + factor * b

def make_figure(titles,names,values,units,filename):
    cmap = matplotlib.colormaps["tab20"]
    colors = [cmap(i) for i in range(len(names))]

    fig, ax = plt.subplots(nrows=len(titles), ncols=1, sharex=True, figsize=(16, 12))

    for i, (title, unit, y) in enumerate(zip(titles, units, values)):
        x_pos = range(len(names))
        ax[i].bar(x_pos, y, color=colors)
        ax[i].set_title(title,fontsize=20)
        ax[i].set_ylabel(unit,fontsize=18)
        ax[i].set_xticks(x_pos)
        ax[i].set_xticklabels(names, rotation=45,fontsize=18, ha="right")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")

def make_combined_latency_figure(names, param_counts, accuracy, measured, quoted, units, filename):
    """
    Create a 3-row figure:
    1. Parameter count
    2. Accuracy (Top-1 or mAP)
    3. Stacked latency: measured + (quoted - measured)
    """
    cmap = matplotlib.colormaps["tab20"]
    colors = [cmap(i) for i in range(len(names))]

    x_pos = np.arange(len(names))

    # 3x1 stacked plot
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(16, 12))

    # param count
    axes[0].bar(x_pos, param_counts, color=colors)
    axes[0].set_title("Parameter Count", fontsize=20)
    axes[0].set_ylabel(units[0], fontsize=18)

    # accuracy
    axes[1].bar(x_pos, accuracy, color=colors)
    axes[1].set_title("Accuracy", fontsize=20)
    axes[1].set_ylabel(units[1], fontsize=18)
    axes[1].set_ylim(0,100)

    # latency
    measured = np.array(measured)
    quoted = np.array(quoted)
    diff = np.clip(quoted - measured, 0, None)



    bars1 = []
    bars2 = []

    for i, (m, q) in enumerate(zip(measured, quoted)):
        if m >= q:
            # Base = quoted, top = measured - quoted
            b1 = axes[2].bar(x_pos[i], q, color=colors[i], label="Quoted Latency" if i == 0 else "") 
            b2 = axes[2].bar(x_pos[i], m - q, bottom=q, color=lighten_color(colors[i],0.5) , label="Measured Latency" if i == 0 else "")
            axes[2].text(x_pos[i], m , f"{m:.1f}", ha="center", va="bottom", fontsize=16)
            axes[2].text(x_pos[i], q , f"{q:.1f}", ha="center", va="bottom", fontsize=16)
        else:
            # Base = measured, top = quoted - measured
            b1 = axes[2].bar(x_pos[i], m, color=colors[i], label="Measured Latency" if i == 0 else "")
            b2 = axes[2].bar(x_pos[i], q - m, bottom=m, color=lighten_color(colors[i],0.5),label="Quoted Latency" if i == 0 else "")
            axes[2].text(x_pos[i], q , f"{q:.1f}", ha="center", va="bottom", fontsize=14)
            axes[2].text(x_pos[i], m , f"{m:.1f}", ha="center", va="bottom", fontsize=14)

    axes[2].set_title("Quoted vs Measured Latency", fontsize=20)
    axes[2].set_ylabel("ms", fontsize=18)
    axes[2].set_ylim(0, max(measured.max(), quoted.max()) * 1.15)
    axes[2].legend()


    # shared x axis labels
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(names, rotation=45, ha="right", fontsize=16)


    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

def param_latency_scatter(names,paramcount, latency, filename):
    cmap = matplotlib.colormaps["tab20"]
    colors = [cmap(i) for i in range(len(names))]

    fig,ax = plt.subplots(figsize = (12,10))

    ax.scatter(paramcount, latency, s=100,color=colors)
    ax.set_xlabel("Parameter count (Millions)", fontsize=18)
    ax.set_ylabel("Latency (ms)", fontsize=18)
    ax.set_ylabel("Latency (ms)")

    texts = []
    for i, name in enumerate(names):
        texts.append(
            ax.text(paramcount[i], latency[i], name, fontsize=12)
        )

    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="->", color="gray"),
        expand=(1.2, 1.4),
        # only_move={"points": "y", "text": "y"}
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")

def jacknet_sweep_plot(self, results_dir: pathlib.Path, filename=quikndirty):
        """
        Quick plot for JackNet 0.75 (Depth 2,4,6,8,10) sweep.
        
        Plots 5 rows:
        1) Operation Count (M)
        2) Average Power (mW)
        3) Energy per Inference (mJ)
        4) Measured Latency (ms)
        5) Accuracy (%)
        """
        
        # Hardcoded specific run names for this sweep
        run_names = ["A075D02", "A075D04", "A075D06", "A075D08", "A075D10"]
        sheet_name = "Img_Class"

        print(f"[DEBUG] Collecting results from: {results_dir}")
        # Assuming collect_results returns dict keyed by folder name (e.g. "A075D02")
        results_dict = collect_results(results_dir)
        
        # Load Excel Metadata
        df = pd.read_excel(self.sheet, sheet_name)
        
        # Filter for just these models
        subset = df[df["Model name"].isin(run_names)].copy()
        
        # Sort based on the hardcoded list order (Depth 2 -> 10)
        # Create a categorical type to enforce order
        subset["Model name"] = pd.Categorical(subset["Model name"], categories=run_names, ordered=True)
        subset.sort_values("Model name", inplace=True)
        subset.reset_index(drop=True, inplace=True)

        if subset.empty:
            raise ValueError(f"No matching models found in Excel for: {run_names}")

        # --- Map Saleae Results ---
        # Note: If collect_results keys don't match exactly, we might need a fallback, 
        # but usually folder name = model name.
        subset["Measured Inference Time (ms)"] = subset["Model name"].map(
            lambda m: results_dict.get(m, {}).get("inference_time_ms")
        )
        subset["Energy per Inference (mJ)"] = subset["Model name"].map(
            lambda m: results_dict.get(m, {}).get("energy_mJ")
        )
        subset["Average Power (mW)"] = subset["Model name"].map(
            lambda m: results_dict.get(m, {}).get("avg_power_mW")
        )

        # --- Prepare Plot Data ---
        names = subset["Model name"].tolist()
        
        # 1. Operations
        # Ensure column name matches Excel exactly (from your image)
        ops_col = "Operation Count (M)" 
        if ops_col not in subset.columns:
            print(f"[WARNING] '{ops_col}' not found. Looking for alternatives...")
            # Fallback search if name varies slightly
            found = [c for c in subset.columns if "Operation" in c]
            if found: ops_col = found[0]
            
        ops = pd.to_numeric(subset[ops_col], errors='coerce').fillna(0).to_numpy()

        # 2. Power
        power = pd.to_numeric(subset["Average Power (mW)"], errors='coerce').fillna(0).to_numpy()
        
        # 3. Energy
        energy = pd.to_numeric(subset["Energy per Inference (mJ)"], errors='coerce').fillna(0).to_numpy()
        
        # 4. Latency
        latency = pd.to_numeric(subset["Measured Inference Time (ms)"], errors='coerce').fillna(0).to_numpy()
        
        # 5. Accuracy
        # Try to find "Measured" first, fallback to "Top-1 Accuracy"
        acc_col = "Top-1 Accuracy"
        acc_label = "Top-1 Accuracy"
        if "Top-1 Accuracy (measured)" in subset.columns:
            acc_col = "Top-1 Accuracy (measured)"
            acc_label = "Measured Accuracy"
            
        accuracy = pd.to_numeric(subset[acc_col], errors='coerce').fillna(0).to_numpy()

        # --- Plotting ---
        cmap = matplotlib.colormaps["tab10"]
        colors = [cmap(i) for i in range(len(names))]
        x_pos = np.arange(len(names))
        
        fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(10, 18))
        
        # Helper to plot bar and add text
        def plot_bar(ax, data, title, ylabel, ylim_buffer=1.1, color_list=colors):
            ax.bar(x_pos, data, color=color_list)
            ax.set_title(title, fontsize=16)
            ax.set_ylabel(ylabel, fontsize=14)
            if len(data) > 0:
                ax.set_ylim(0, max(data) * ylim_buffer)
            ax.tick_params(axis="y", labelsize=12)
            for i, v in enumerate(data):
                ax.text(x_pos[i], v * 1.01, f"{v:.1f}", ha="center", va="bottom", fontsize=12)

        # 1) Operation Count
        plot_bar(axes[0], ops, "Operation Count", "M Ops")

        # 2) Average Power
        plot_bar(axes[1], power, "Average Power", "mW")

        # 3) Energy per Inference
        plot_bar(axes[2], energy, "Energy per Inference", "mJ")

        # 4) Latency
        plot_bar(axes[3], latency, "Measured Latency", "ms")

        # 5) Accuracy
        plot_bar(axes[4], accuracy, acc_label, "%", ylim_buffer=1.2)
        axes[4].set_ylim(0, 100) # Force 0-100 for accuracy

        # X Axis
        axes[4].set_xticks(x_pos)
        axes[4].set_xticklabels(names, rotation=30, ha="right", fontsize=14)

        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved JackNet plot to {filename}")
            plt.close(fig)
        else:
            plt.show()

        return subset


## Main class ##
class ModelStatsPlotting:
    """
    Class containing the primary function to generate plots for each model type
    Currently image classification, object detection, segmentation (WIP)
    """
    def __init__(self,xlsx,plotdir):
        self.sheet = path.abspath(xlsx)
        self.plotdir = path.abspath(plotdir)

    def img_class_plt(self):
        """Qtriple Stacked Bar Chart of Model Stats for Image Classification Models"""

        model_dir = REPO_ROOT / "data/models/Image_Classification"
        pc = ParamCounts(model_dir)
        inf_ms = inference_from_csvs("IMG_CLASS_NEW") # as measured from Saleae
        param_counts = [x/1e6 for x in pc.scan_models()]  # scale to millions

        ic_df = pd.read_excel(
            self.sheet,
            sheet_name="Img_Class",
            header=0,
            usecols=["Model name","Latency (ms)", "Top-1 Accuracy (measured)", "Top-5 Accuracy"]
        )

        model_names = ic_df["Model name"].tolist()
        latency_ms = ic_df["Latency (ms)"].tolist() # from coral docs
        top1accuracy = ic_df["Top-1 Accuracy (measured)"].tolist()
        top5accuracy = ic_df["Top-5 Accuracy"].tolist()

        print(len(inf_ms), len(latency_ms), len(model_names), len(param_counts), len(top1accuracy))


        combined = list(zip(inf_ms, latency_ms,model_names, param_counts, top1accuracy))
        combined.sort(key=lambda x: x[0]) # sort by latency - fastest to slowest
        trimmed = combined[:-2]

        inf_ms, latency_ms, model_names, param_counts, top1accuracy = zip(*combined) # overwrite with sorted lists

        updated_df = pd.DataFrame({
            "Model name": model_names,
            "Latency (ms)": latency_ms,  # quoted
            "Top-1 Accuracy": top1accuracy,
            "Top-5 Accuracy": top5accuracy,
            "Measured Inference Time (ms)": inf_ms,  # measured
            "Parameter Count (M)": param_counts
        })

        # Save back into Excel
        with pd.ExcelWriter(self.sheet, mode="a", if_sheet_exists="replace", engine="openpyxl") as writer:
            updated_df.to_excel(writer, sheet_name="Img_Class_outs", index=False)

        titles = ["Parameter Count","Inference Time","Quoted Latency", "Top-1 Accuracy"]
        values = [param_counts, inf_ms,latency_ms, top1accuracy]
        units = ["# Params (M)", "ms", "ms", "%"]

        make_figure(titles,model_names,values,units,self.plotdir+"/img_class_plot.png")
        make_combined_latency_figure(
            model_names,
            param_counts,
            top1accuracy,
            inf_ms,         # measured
            latency_ms,     # quoted
            ["# Params (M)", "%", "ms"],
            self.plotdir+"/img_class_combined.png"
        )
        param_latency_scatter(model_names,param_counts,latency_ms,self.plotdir+"/img_class_sctplot.png")
        param_latency_scatter(model_names,param_counts,inf_ms,self.plotdir+"/img_class_sctplot_inf.png")

        inf_ms, latency_ms, model_names, param_counts, top1accuracy = zip(*trimmed)
        titles = ["Parameter Count","Inference Time","Quoted Latency", "Top-1 Accuracy"]
        values = [param_counts, inf_ms,latency_ms, top1accuracy]
        units = ["# Params (M)", "ms", "ms", "%"]

        make_figure(titles,model_names,values,units,self.plotdir+"/img_class_plot_trimmed.png")
        make_combined_latency_figure(
            model_names,
            param_counts,
            top1accuracy,
            inf_ms,         # measured
            latency_ms,     # quoted
            ["# Params (M)", "%", "ms"],
            self.plotdir+"/img_class_combined_trimmed.png"
        )

    def power_inf_runs(self, df, results_dir: pathlib.Path,
                    model_category=None, run_names=None, filename=None):
        """
        Generates two figures:
        
        Figure 1 (4-row): Standard metrics (Power, Energy, Latency, Accuracy).
        Figure 2 (1-row, 2-col): Efficiency metrics side-by-side.
                                - Left: Sorted by Correct Inf/Sec
                                - Right: Sorted by Correct Inf/Joule

        Returns the sorted dataframe used for plotting.
        """

        valid_categories = ["Img_Class", "Obj_Det", "Segmentation", "Audio_Classification"]
        if model_category is None:
            sheet_name = "Img_Class"
        elif model_category in valid_categories:
            sheet_name = model_category
        else:
            raise ValueError(
                f"Model Category {model_category} does not exist. "
                f"Choose from: {', '.join(valid_categories)}"
            )

        if run_names is None:
            run_names = [
                "EfficientNet-EdgeTpu (L)",
                "EfficientNet-EdgeTpu (M)",
                "EfficientNet-EdgeTpu (S)",
                "Inception V1",
                "Inception V2",
                "MobileNet V1 (0.25)",
                "MobileNet V1 (0.50)",
                "MobileNet V1 (.75)",
                "MobileNet V1 (1.0)",
                "MobileNet V1 (TF_ver_2.0)",
                "MobileNet V2",
                "MobileNet V2 (TF_ver_2.0)",
                "MobileNet V3",
            ]

        # Collect Saleae results and Excel metadata
        print(f"[DEBUG] results dir: {results_dir}")
        results_dict = collect_results(results_dir)
        df = pd.read_excel(self.sheet, sheet_name)

        # Restrict to selected models
        subset = df[df["Model name"].isin(run_names)].copy()
        if subset.empty:
            raise ValueError(f"No matching models found for run_names={run_names}")

        # Map Saleae results
        subset["Measured Inference Time (ms)"] = subset["Model name"].map(
            lambda m: results_dict.get(m, {}).get("inference_time_ms")
        )
        subset["Energy per Inference (mJ)"] = subset["Model name"].map(
            lambda m: results_dict.get(m, {}).get("energy_mJ")
        )

        # Derive Average Power: E [mJ] / T [ms] * 1000 = mW
        subset["Average Power (mW)"] = subset["Model name"].map(
            lambda m: results_dict.get(m, {}).get("avg_power_mW")
        )

        # Coerce numeric
        numeric_cols = [
            "Energy per Inference (mJ)",
            "Measured Inference Time (ms)",
            "Latency (ms)",
            "Top-1 Accuracy (measured)",
            "Top-1 Accuracy",
        ]
        for col in numeric_cols:
            if col in subset.columns:
                subset[col] = pd.to_numeric(subset[col], errors="coerce")
            else:
                raise KeyError(f"Required column missing: {col}")

        # Drop rows missing required values
        valid_mask = subset[numeric_cols].notnull().all(axis=1)
        if not valid_mask.all():
            dropped = subset.loc[~valid_mask, "Model name"].tolist()
            print(f"[DEBUG] Dropped runs due to missing data: {dropped}")
            subset = subset.loc[valid_mask].copy()
        if subset.empty:
            raise ValueError("No valid runs left after dropping rows with missing data.")

        # -------------------------------------------------------------------------
        # NEW CALCULATIONS: Correct Inferences per Joule / Second
        # -------------------------------------------------------------------------
        subset["Inf_per_Sec"] = 1000.0 / subset["Measured Inference Time (ms)"]
        subset["Inf_per_Joule"] = 1000.0 / subset["Energy per Inference (mJ)"]

        # Calculate "Correct" metrics
        acc_ratio = subset["Top-1 Accuracy (measured)"] / 100.0
        subset["Correct_Inf_per_Sec"] = subset["Inf_per_Sec"] * acc_ratio
        subset["Correct_Inf_per_Joule"] = subset["Inf_per_Joule"] * acc_ratio

        # -------------------------------------------------------------------------
        # FIGURE 1 GENERATION (Sorted by Energy for the original 4-row plot)
        # -------------------------------------------------------------------------
        # Sort for Figure 1
        subset_energy_sort = subset.sort_values("Energy per Inference (mJ)", ascending=True).copy()
        
        names_e = subset_energy_sort["Model name"].tolist()
        power = subset_energy_sort["Average Power (mW)"].to_numpy()
        energy = subset_energy_sort["Energy per Inference (mJ)"].to_numpy()
        measured_lat = subset_energy_sort["Measured Inference Time (ms)"].to_numpy()
        quoted_lat = subset_energy_sort["Latency (ms)"].to_numpy()
        acc_measured = subset_energy_sort["Top-1 Accuracy (measured)"].to_numpy()
        acc_quoted = subset_energy_sort["Top-1 Accuracy"].to_numpy()

        cmap = matplotlib.colormaps["tab10"]
        colors_e = [cmap(i % 10) for i in range(len(names_e))]
        x_pos_e = np.arange(len(names_e))

        fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(14, 18))

        # 1) Average Power
        axes[0].bar(x_pos_e, power, color=colors_e)
        axes[0].set_title("Average Power", fontsize=22)
        axes[0].set_ylabel("mW", fontsize=20)
        axes[0].set_ylim(0, 1200)
        axes[0].tick_params(axis="y", labelsize=20)
        for i, p in enumerate(power):
            axes[0].text(x_pos_e[i], p * 1.01, f"{p:.1f}", ha="center", va="bottom", fontsize=16)

        # 2) Energy per Inference
        axes[1].bar(x_pos_e, energy, color=colors_e)
        axes[1].set_title("Energy per Inference", fontsize=22)
        axes[1].set_ylabel("mJ", fontsize=20)
        axes[1].set_ylim(0, 60)
        axes[1].tick_params(axis="y", labelsize=20)
        for i, e in enumerate(energy):
            axes[1].text(x_pos_e[i], e * 1.01, f"{e:.1f}", ha="center", va="bottom", fontsize=16)

        # 3) Latency
        for i, (m, q) in enumerate(zip(measured_lat, quoted_lat)):
            base_color = colors_e[i]
            light = lighten_color(base_color, 0.5)
            if m >= q:
                axes[2].bar(x_pos_e[i], q, color=base_color)
                axes[2].bar(x_pos_e[i], m - q, bottom=q, color=light)
            else:
                axes[2].bar(x_pos_e[i], m, color=light)
                axes[2].bar(x_pos_e[i], q - m, bottom=m, color=base_color)
            axes[2].text(x_pos_e[i], m * 1.01, f"{m:.1f}", ha="center", va="bottom", fontsize=18)
            axes[2].text(x_pos_e[i], q * 1.01, f"{q:.1f}", ha="center", va="bottom", fontsize=18)

        axes[2].set_title("Latency (Quoted vs Measured)", fontsize=22)
        axes[2].set_ylabel("ms", fontsize=20)
        axes[2].set_ylim(0, 60)
        axes[2].tick_params(axis="y", labelsize=20)

        # 4) Accuracy
        width = 0.4
        axes[3].bar(x_pos_e - width/2, acc_quoted, width=width, color=colors_e, label="Quoted")
        axes[3].bar(x_pos_e + width/2, acc_measured, width=width, color=[lighten_color(c, 0.5) for c in colors_e], label="Measured")
        axes[3].set_title("Top-1 Accuracy (Quoted vs Measured)", fontsize=22)
        axes[3].set_ylabel("%", fontsize=20)
        axes[3].set_ylim(0, 100)
        axes[3].tick_params(axis="y", labelsize=20)
        for i, (a_m, a_q) in enumerate(zip(acc_measured, acc_quoted)):
            axes[3].text(x_pos_e[i] + width/2, a_m * 1.01, f"{a_m:.0f}", ha="center", va="bottom", fontsize=18)
            axes[3].text(x_pos_e[i] - width/2, a_q * 1.01, f"{a_q:.0f}", ha="center", va="bottom", fontsize=18)
        axes[3].legend(fontsize=18)

        axes[3].set_xticks(x_pos_e)
        axes[3].set_xticklabels(names_e, rotation=30, ha="right", fontsize=20)

        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved Figure 1 to {filename}")
            plt.close(fig)
        else:
            plt.show()

        # -------------------------------------------------------------------------
        # FIGURE 2 GENERATION (Independently Sorted Columns)
        # -------------------------------------------------------------------------
        
        # 1. Dataset sorted by Speed (Inferences per Second)
        df_speed = subset.sort_values("Correct_Inf_per_Sec", ascending=True).copy()
        
        # 2. Dataset sorted by Efficiency (Inferences per Joule)
        df_efficiency = subset.sort_values("Correct_Inf_per_Joule", ascending=True).copy()
        
        fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

        # Helper function to plot one column with specific sorting
        def plot_sorted_col(ax, sorted_df, total_col, correct_col, title, ylabel):
            
            names = sorted_df["Model name"].tolist()
            total_vals = sorted_df[total_col].to_numpy()
            correct_vals = sorted_df[correct_col].to_numpy()
            
            local_x = np.arange(len(names))
            local_colors = [cmap(i % 10) for i in range(len(names))]
            
            max_y = 0
            
            for i, (total, correct) in enumerate(zip(total_vals, correct_vals)):
                base_color = local_colors[i]
                light = lighten_color(base_color, 0.5)

                # Stack: Correct (solid) at bottom, Wasted (Total - Correct) (light) on top
                ax.bar(local_x[i], correct, color=base_color)
                ax.bar(local_x[i], total - correct, bottom=correct, color=light)
                
                # Label Total: Above the bar
                ax.text(local_x[i], total * 1.02, f"{total:.1f}", 
                        ha="center", va="bottom", fontsize=16, fontweight='bold')
                
                # Label Correct: Inside the bar, just below the top of the solid section
                if correct > 0:
                    ax.text(local_x[i], correct - (correct * 0.05), f"{correct:.1f}", 
                            ha="center", va="top", fontsize=16, color="white", fontweight='bold')
                
                if total > max_y: max_y = total

            ax.set_title(title, fontsize=22)
            ax.set_ylabel(ylabel, fontsize=20)
            ax.set_ylim(0, max_y * 1.10) # 25% headroom
            ax.tick_params(axis="y", labelsize=20)
            ax.set_xticks(local_x)
            ax.set_xticklabels(names, rotation=30, ha="right", fontsize=20)
            
            # Custom Legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='gray', label='Correct (Solid)'),
                               Patch(facecolor='lightgray', label='Total (Faded)')]
            ax.legend(handles=legend_elements, fontsize=16, loc='upper left')

        # Plot Left: Sorted by Speed
        plot_sorted_col(axes2[0], df_speed, 
                        "Inf_per_Sec", "Correct_Inf_per_Sec", 
                        "Inferences per Second", "Inf / Sec")

        # Plot Right: Sorted by Efficiency
        plot_sorted_col(axes2[1], df_efficiency, 
                        "Inf_per_Joule", "Correct_Inf_per_Joule", 
                        "Inferences per Joule", "Inf / Joule")

        plt.tight_layout()
        if filename:
            import os
            path, ext = os.path.splitext(filename)
            file2 = f"{path}_efficiency_metrics{ext}"
            plt.savefig(file2, dpi=300, bbox_inches="tight")
            print(f"Saved Figure 2 to {file2}")
            plt.close(fig2)
        else:
            plt.show()

        return subset
    def budgeted_correct_inferences(self, df, buffer_energy, frame_time,results_dir: pathlib.Path,
                    model_category=None, run_names=None, filename=None):
        
        # Prepare arrays
        names = df["Model name"].tolist()
        power = df["Average Power (mW)"].to_numpy()
        energy = df["Energy per Inference (mJ)"].to_numpy()*1e-3  # convert to joules
        measured_lat = df["Measured Inference Time (ms)"].to_numpy()*1e-3 # convert to seconds
        print(measured_lat)
        quoted_lat = df["Latency (ms)"].to_numpy()
        acc_measured = df["Top-1 Accuracy (measured)"].to_numpy()
        acc_quoted = df["Top-1 Accuracy"].to_numpy()

        ## Buffers
        energy_budget = buffer_energy


        ## Time Calculation
        time_budget = frame_time


        ## THE METRIC!!!
        
        acc_frac = acc_measured / 100.0 # revisit with recall

        time_correct = np.floor(time_budget / measured_lat) * acc_frac
        energy_correct = np.floor(energy_budget / energy) * acc_frac
        overall_correct = np.minimum(time_correct, energy_correct)

        df["Correct Inferences (Time budget)"] = time_correct
        df["Correct Inferences (Energy budget)"] = energy_correct
        df["Correct Inferences (Overall)"] = overall_correct

        # Sort by overall feasible inferences
        df.sort_values("Correct Inferences (Overall)", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)

        names = df["Model name"].tolist()
        x_pos = np.arange(len(names))
        cmap = matplotlib.colormaps["tab10"]
        colors = [cmap(i) for i in range(len(names))]

        fig2, axes2 = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(14, 12))


        ## Plotting  for Metric (Buffers vs Time)
        # correct inferences within energy budget
        axes2[0].bar(x_pos, df["Correct Inferences (Energy budget)"], color=colors)
        axes2[0].set_title("Correct Inferences within Energy Budget", fontsize=22)
        axes2[0].set_ylabel("# Inferences", fontsize=20)
        axes2[0].tick_params(axis="y", labelsize=20)

        energy_vals = df["Correct Inferences (Energy budget)"].to_numpy()
        for i, val in enumerate(energy_vals):
            axes2[0].text(x_pos[i], val * 1.02, f"{int(val)}",
                        ha="center", va="bottom", fontsize=18)
        #axes2[0].set_ylim(0, max(energy_vals) * 1.25)

        # correct inferences within time budget
        axes2[1].bar(x_pos, df["Correct Inferences (Time budget)"], color=colors)
        axes2[1].set_title("Correct Inferences within Time Budget", fontsize=22)
        axes2[1].set_ylabel("# Inferences", fontsize=20)
        axes2[1].tick_params(axis="y", labelsize=20)

        time_vals = df["Correct Inferences (Time budget)"].to_numpy()
        for i, val in enumerate(time_vals):
            axes2[1].text(x_pos[i], val * 1.02, f"{int(val)}",
                        ha="center", va="bottom", fontsize=18)
        axes2[1].set_ylim(0, max(time_vals) * 1.25)
        
        # overall correct inferences (feasible)
        axes2[2].bar(x_pos, df["Correct Inferences (Overall)"], color=colors)
        axes2[2].set_title("Overall Correct Inferences (min of Time/Energy)", fontsize=22)
        axes2[2].set_ylabel("# Inferences", fontsize=20)
        axes2[2].tick_params(axis="y", labelsize=20)
        axes2[2].set_xticks(x_pos)
        axes2[2].set_xticklabels(names, rotation=30, ha="right", fontsize=20)

        time_vals = df["Correct Inferences (Time budget)"].to_numpy()
        energy_vals = df["Correct Inferences (Energy budget)"].to_numpy()
        overall_vals = df["Correct Inferences (Overall)"].to_numpy()

        x_positions = np.arange(len(names))

        # Decide which budget limited each model
        limiting = ["Time" if t < e else "Energy" for t, e in zip(time_vals, energy_vals)]

        for x, y, lim in zip(x_positions, overall_vals, limiting):
            axes2[2].text(x, y * 1.05, lim,
                        ha="center", va="bottom",
                        fontsize=14, fontweight="bold")

        # Expand ylim to leave room for labels
        axes2[2].set_ylim(0, max(overall_vals) * 1.25)
    
        plt.tight_layout()
        if filename:
            outpath = str(pathlib.Path(filename).with_name("budgeted_correct_inferences.png"))
            plt.savefig(outpath, dpi=300, bbox_inches="tight")
            plt.close(fig2)
        else:
            plt.show()

        # Determine overall winner
        best_idx = df["Correct Inferences (Overall)"].idxmax()
        best_row = df.iloc[best_idx]

        # Determine what limited the winner
        best_limiting = (
            "Time" if best_row["Correct Inferences (Time budget)"] 
                    < best_row["Correct Inferences (Energy budget)"] 
            else "Energy"
        )

        # Build return string
        winner_string = f"{best_row['Model name']} ({max(overall_vals)} correct inferences) limited by {best_limiting}"

        return winner_string
    
    def budget_correct_loop(self, df: pd.DataFrame, buffers: list, frame_times: list, results_dir: pathlib.Path, filename="all_grids.xlsx"): 
        """
        Loop through all models, create individual grids, and generate a comparison plot 
        specifically for MobileNet V1 0.50 vs 0.75.
        """
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / filename
        plot_dir = results_dir / "plots" / "budget_traces"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Store data for the comparison plot
        comparison_data = []
        target_models = ["MobileNet V1 (0.50)", "MobileNet V1 (.75)"] 

        with pd.ExcelWriter(output_path) as writer:
            
            for _, row in df.iterrows():
                name = row["Model name"]
                print(f"Processing model: {name}")

                energy_j = row["Energy per Inference (mJ)"] * 1e-3
                latency_s = row["Measured Inference Time (ms)"] * 1e-3
                acc_frac = row["Top-1 Accuracy (measured)"] / 100.0

                # Check if this model is one of the target models for comparison
                if any(t in name for t in target_models):
                                comparison_data.append({
                                    "name": name,
                                    "acc": acc_frac,
                                    "energy_j": energy_j,
                                    "latency_s": latency_s,
                                    "power_w": energy_j / latency_s,
                                    "color": "tab:blue" if "0.50" in name else "tab:orange"
                                })

                # build grid of model performance for different budgets
                grid_out = np.empty((len(buffers), len(frame_times)), dtype=object)
                for i, energy_buffer in enumerate(buffers):
                    for j, frame_time in enumerate(frame_times):
                        # inferences within budget (float)
                        count_time = frame_time / latency_s
                        count_energy = energy_buffer / energy_j
                        #mul by acc and floor
                        correct_time = int(count_time * acc_frac)
                        correct_energy = int(count_energy * acc_frac)
                        # track limiting factor
                        if correct_time < correct_energy:
                            final = correct_time
                            lim = "Time"
                        else:
                            final = correct_energy
                            lim = "Energy"
                        grid_out[i, j] = f"{final} correct ({lim})"

                df_grid = pd.DataFrame(grid_out, 
                                    index=[f"Energy: {np.round(b,2)}J" for b in buffers],
                                    columns=[f"Time: {np.round(t,2)}s" for t in frame_times])
                # write to excel sheet
                safe_sheet_name = name[:31].replace(":", "-").replace("/", "-")
                df_grid.to_excel(writer, sheet_name=safe_sheet_name)

        # Comparison Plot: 0.50 vs 0.75

        # if len(comparison_data) == 2:
        #     print("Generating Decision Frontier (Time vs. Energy Strategy)...")
            
        #     # 1. Identify "Sprinter" (Efficient) vs "Powerhouse" (High Rate)
        #     # We assume one is more efficient (higher slope) and one has higher max rate.
        #     # If one model is better at BOTH, there is no crossover (it always wins).
            
        #     # Sort by Max Rate (Correct Inferences / Sec)
        #     sorted_by_rate = sorted(comparison_data, key=lambda x: x['acc']/x['latency_s'])
        #     model_low_rate = sorted_by_rate[0]  # MobileNet 0.50 (saturates early)
        #     model_high_rate = sorted_by_rate[1] # MobileNet 0.75 (higher ceiling)
            
        #     rate_low = model_low_rate['acc'] / model_low_rate['latency_s']
        #     slope_high = model_high_rate['acc'] / model_high_rate['energy_j']
        #     slope_low = model_low_rate['acc'] / model_low_rate['energy_j']

        #     # Only proceed if a trade-off actually exists
        #     if slope_low > slope_high:
                
        #         plt.figure(figsize=(10, 6))
                
        #         # Create Time Range (0 to 100s, or slightly more than max frame time)
        #         max_t = 100
        #         time_range = np.linspace(0, max_t, 200)
                
        #         # Calculate the Breakeven Energy Line: E = T * (Rate_Low / Slope_High)
        #         # This is the energy required for the High-Power model to catch up 
        #         # to the Low-Power model's saturation point.
        #         k = rate_low / slope_high
        #         breakeven_energy = time_range * k
                
        #         # Plot the dividing line
        #         plt.plot(time_range, breakeven_energy, 
        #                 color='black', linewidth=2, linestyle='--', 
        #                 label=f"Breakeven Frontier (k={k:.2f} J/s)")
                
        #         # Shade the Regions
        #         # Region Below: Battery is too small -> Use Efficient Model
        #         plt.fill_between(time_range, 0, breakeven_energy, 
        #                         color=model_low_rate['color'], alpha=0.15)
        #         plt.text(max_t * 0.75, max(breakeven_energy) * 0.25, 
        #                 f"ZONE: {model_low_rate['name']}\n(Battery Limited)", 
        #                 ha='center', va='center', fontweight='bold', color=model_low_rate['color'],fontsize = 20)
                
        #         # Region Above: Battery is sufficient -> Use High-Accuracy Model
        #         # We set an arbitrary top y-limit for shading (e.g. 1.5x max breakeven)
        #         y_top = max(breakeven_energy) * 1.5
        #         plt.fill_between(time_range, breakeven_energy, y_top, 
        #                         color=model_high_rate['color'], alpha=0.15)
        #         plt.text(max_t * 0.25, max(breakeven_energy) * 1.25, 
        #                 f"ZONE: {model_high_rate['name']}\n(Time Limited)", 
        #                 ha='center', va='center', fontweight='bold', color=model_high_rate['color'],fontsize = 20)

        #         plt.title("Mission Strategy: Which Model Should I Run?",fontsize = 20)
        #         plt.xlabel("Pass Duration (Seconds)",fontsize = 20)
        #         plt.ylabel("Required Energy Buffer (Joules)",fontsize = 20)
        #         plt.xlim(0, max_t)
        #         plt.ylim(0, y_top)
        #         plt.xticks(fontsize=14)
        #         plt.yticks(fontsize=14)
        #         plt.grid(True, linestyle=':', alpha=0.5)
        #         # plt.legend(loc="upper left")
                
        #         plt.tight_layout()
        #         plt.savefig(plot_dir / "MobileNet_Decision_Frontier.png")
        #         plt.close()

        if len(comparison_data) > 0:
                print("Generating Comparison Plot (Correct Inferences)...")
                
                plt.figure(figsize=(10, 6))
                
                smooth_buffers = np.linspace(min(buffers), max(buffers), 1000)
                max_pass_time = 10.0

                # Metrics storage for calculating crossover
                metrics = {}

                for item in comparison_data:
                    # 1. Calculate fundamental rates
                    slope_correct_per_joule = item['acc'] / item['energy_j']
                    ceiling_correct_per_sec = item['acc'] / item['latency_s']
                    
                    # Store for comparison logic
                    metrics[item['name']] = {
                        "slope": slope_correct_per_joule,
                        "rate": ceiling_correct_per_sec,
                        "ceiling_val": ceiling_correct_per_sec * max_pass_time,
                        "color": item['color']
                    }
                    
                    # 2. Calculate the trace
                    correct_energy_bound = smooth_buffers * slope_correct_per_joule
                    max_possible_inferences = ceiling_correct_per_sec * max_pass_time
                    correct_time_bound = np.full_like(smooth_buffers, max_possible_inferences)
                    
                    correct_trace = np.minimum(correct_energy_bound, correct_time_bound)
                    
                    # 3. Plot
                    plt.plot(smooth_buffers, correct_trace, 
                            linewidth=3, 
                            color=item["color"], 
                            label=f"{item['name']}\n(Slope: {slope_correct_per_joule:.1f}/J | Rate: {ceiling_correct_per_sec:.1f}/s)")

                # -----------------------------------------------------
                # Calculate and Plot the Exact Crossover Point
                # -----------------------------------------------------
                # We need to find where the "Powerhouse" (Higher Ceiling) crosses the "Sprinter" (Steeper Slope)
                # Sort by Ceiling (Rate) to find the 'High Performer' and 'Efficient Performer'
                sorted_models = sorted(metrics.items(), key=lambda x: x[1]['rate'])
                
                # Assumption: The one with the higher rate (0.75) has the lower slope (less efficient). 
                # If not, it simply wins everywhere and they never cross.
                low_ceiling_model = sorted_models[0]  # MobileNet 0.50
                high_ceiling_model = sorted_models[1] # MobileNet 0.75
                
                name_lo, data_lo = low_ceiling_model
                name_hi, data_hi = high_ceiling_model
                
                # Check if a crossover is physically possible (High Ceiling must have Lower Slope)
                if data_hi['slope'] < data_lo['slope']:
                    # The Intersection Equation:
                    # High_Slope * Buffer = Low_Ceiling_Value
                    # Buffer = Low_Ceiling_Value / High_Slope
                    
                    breakeven_joules = data_lo['ceiling_val'] / data_hi['slope']
                    breakeven_inferences = data_lo['ceiling_val']
                    
                    print(f"\n*** BREAKEVEN ANALYSIS ({max_pass_time}s Pass) ***")
                    print(f"Model {name_lo} maxes out at {int(data_lo['ceiling_val'])} inferences.")
                    print(f"Model {name_hi} needs {breakeven_joules:.2f} Joules to match that.")
                    print(f"VERDICT: For batteries > {breakeven_joules:.2f}J, use {name_hi}. Otherwise use {name_lo}.\n")
                    
                    # Plot the Crossover Marker
                    if min(buffers) < breakeven_joules < max(buffers):
                        plt.axvline(breakeven_joules, color='red', linestyle=':', alpha=0.8)
                        plt.scatter([breakeven_joules], [breakeven_inferences], color='red', zorder=10)
                        plt.text(breakeven_joules, breakeven_inferences * 1.05, 
                                f" Breakeven: {breakeven_joules:.1f}J", 
                                color='red', fontsize=12, fontweight='bold')
                # -----------------------------------------------------

                plt.title(f"Correct Inferences vs. Battery Size\n(Pass Duration: {max_pass_time:.1f}s)", fontsize=20)
                plt.xlabel("Energy Buffer (Joules)", fontsize=20)
                plt.ylabel("Total Correct Inferences", fontsize=20)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.ylim(0,600)
                plt.xlim(0,10)
                plt.grid(True, linestyle=':', alpha=0.5)
                plt.legend(loc='upper left', fontsize = 16)
                plt.tight_layout()
                
                plt.savefig(plot_dir / "MobileNet_Comparison_Value.png")
                plt.close()
        else:
            print("Warning: Target MobileNet models not found in dataframe for comparison plot.")

        print(f"File saved successfully to {output_path}")
    
    # def budget_correct_loop(self, df: pd.DataFrame, buffers: list, frame_times: list, results_dir: pathlib.Path, filename="all_grids.xlsx"): 
    #     """
    #     Loop through all models and create a sweep of performance against budget combos.
    #     Saves each model's grid as a separate sheet in the Excel file.
    #     """
    #     # Ensure directory exists
    #     results_dir.mkdir(parents=True, exist_ok=True)
    #     output_path = results_dir / filename

    #     # Use ExcelWriter to save multiple sheets to the same file
    #     with pd.ExcelWriter(output_path) as writer:
            
    #         # iterate through each model individually
    #         for _, row in df.iterrows():
    #             name = row["Model name"]
    #             print(f"Processing model: {name}")

    #             # extract metrics for this specific model
    #             energy_j = row["Energy per Inference (mJ)"] * 1e-3
    #             latency_s = row["Measured Inference Time (ms)"] * 1e-3
    #             acc_frac = row["Top-1 Accuracy (measured)"] / 100.0

    #             # create a fresh grid for this model (Rows: Buffers, Cols: Times)
    #             grid_out = np.empty((len(buffers), len(frame_times)), dtype=object)

    #             for i, energy_buffer in enumerate(buffers):
    #                 for j, frame_time in enumerate(frame_times):
                        
    #                     # Calculate max possible inferences 
    #                     count_time_limited = frame_time / latency_s
    #                     count_energy_limited = energy_buffer / energy_j

    #                     # Calculate correct 
    #                     correct_time = int(count_time_limited * acc_frac)
    #                     correct_energy = int(count_energy_limited * acc_frac)

    #                     # Determine which budget limits inference
    #                     if correct_time < correct_energy:
    #                         final_correct = correct_time
    #                         limiter = "Time"
    #                     else:
    #                         final_correct = correct_energy
    #                         limiter = "Energy"

    #                     # Format string for the cell
    #                     grid_out[i, j] = f"{final_correct} correct ({limiter})"

    #             # Create DataFrame for this model's grid
    #             df_grid = pd.DataFrame(
    #                 grid_out,
    #                 index=[f"Energy: {np.round(b,2)}J" for b in buffers],
    #                 columns=[f"Time: {np.round(t,2)}s" for t in frame_times]
    #             )

    #             # Write to a sheet named after the model
    #             safe_sheet_name = name[:31].replace(":", "-").replace("/", "-")
    #             df_grid.to_excel(writer, sheet_name=safe_sheet_name)

    #     print(f"File saved successfully to {output_path}")


        def obj_det_plt(self):
            """Triple Stacked Bar Chart of Model Stats for Object Detection"""
            model_dir = REPO_ROOT / "data/models/Object_Detection"

            pc = ParamCounts(model_dir)
            inf_ms = inference_from_csvs("OBJ_DET_NEW") # as measured from Saleae

            param_counts = [x/1e6 for x in pc.scan_models()]  # scale to millions
            ic_df = pd.read_excel(
                self.sheet,
                sheet_name="Obj_Det",
                header=0,
                usecols=["Model Name","Latency (ms)", "mAP (measured)"]
            )

            model_names = ic_df["Model Name"].tolist()
            latency_ms = ic_df["Latency (ms)"].tolist()
            mAP = ic_df["mAP (measured)"].tolist()

            print(len(inf_ms), len(latency_ms), len(model_names), len(param_counts), len(mAP))

            combined = list(zip(inf_ms,latency_ms, model_names, param_counts, mAP))
            combined.sort(key=lambda x: x[0]) # sort by measured latency

            inf_ms, latency_ms, model_names, param_counts, mAP = zip(*combined) # overwrite with sorted lists
            updated_df = pd.DataFrame({
                "Model Name": model_names,
                "Latency (ms)": latency_ms,  # quoted
                "Measured Inference Time (ms)": inf_ms,  # measured
                "mAP (measured)": mAP,
                "Parameter Count (M)": param_counts
            })

            # Save back into Excel
            with pd.ExcelWriter(self.sheet, mode="a", if_sheet_exists="replace", engine="openpyxl") as writer:
                updated_df.to_excel(writer, sheet_name="Obj_Det_outs", index=False)

            titles = ["Parameter Count","Inference Time (ms)","Latency (ms)", "Mean Average Precision"]
            values = [param_counts,inf_ms, latency_ms, mAP]
            units = ["# Params (M)", "ms","ms", "%"]

            make_figure(titles,model_names,values,units,self.plotdir+"/obj_det_plot.png")
            make_combined_latency_figure(
                model_names,
                param_counts,
                mAP,
                inf_ms,         # measured
                latency_ms,     # quoted
                ["# Params (M)", "%", "ms"],
                self.plotdir+"/obj_det_combined.png"
            )

        def segmentation_plt(self):
            """Double Stacked Bar Chart of Model Stats for Segmentation"""
            model_dir = "~/Coral-TPU-Characterization/data/models/Segmentation"

            pc = ParamCounts(model_dir)
            param_counts = [x/1e6 for x in pc.scan_models()]  # scale to millions

            ic_df = pd.read_excel(
                self.sheet,
                sheet_name="Segmentation",
                header=0,
                usecols=["Model Name","Latency (ms)"]
            )

            model_names = ic_df["Model Name"].tolist()
            latency_ms = ic_df["Latency (ms)"].tolist()

            titles = ["Parameter Count","Latency (ms)"]
            values = [param_counts, latency_ms]
            units = ["# Params (M)", "ms"]

            make_figure(titles,model_names,values,units,self.plotdir+"/segmentation_plot.png")

    def power_analog_trace(self):
        pass

if __name__ == "__main__":

    REPO_ROOT = get_repo_root()

    plots = ModelStatsPlotting(
    REPO_ROOT / "data/Model_Stats.xlsx",
    REPO_ROOT / "results/plots/"
    )

    ## Baseline Model Parameter Plotting
    #plots.img_class_plt()
    #plots.obj_det_plt()
    # plots.segmentation_plt()


    ## Experimental Power + Inference Plotting
    # Select the runs you want to visualize
    run_names=[
        "EfficientNet-EdgeTpu (M)",
        "EfficientNet-EdgeTpu (S)",
        #"Inception V1",
        "MobileNet V1 (0.25)",
        "MobileNet V1 (0.50)",
        "MobileNet V1 (.75)",
        "MobileNet V1 (1.0)",
        #"MobileNet V1 (TF_ver_2.0)",
        "MobileNet V2",
        #"MobileNet V2 (TF_ver_2.0)",
        #"MobileNet V3"
    ]
    # Call the method that already merges Excel + Saleae results
    print("[INFO] Generating power and inference plots...")
    subset = plots.power_inf_runs(
        df=None,
        results_dir= (REPO_ROOT / "results/captures/IMG_CLASS02"),
        model_category="Img_Class",
        run_names=run_names,
        filename=REPO_ROOT / "results/plots/img_class_power_runs.png"
    )

    ## Budgeted Correct Inferences Plotting
    print("[INFO] Generating budgeted correct inferences plots...")

    def Energy_in_Cap(C,V):
        return 0.5 * C * V * V  # joules
    def Frame_Time(H_m, FOV_deg):
        R_E = 6371e3  # m
        mu_E = 3.986e14  # m^3/s^2
        R = H_m + R_E
        V_SAT = (mu_E / R) ** 0.5  # m/s
        FOV_rad = np.deg2rad(FOV_deg)
        Res= 2 * H_m * np.tan(FOV_rad / 2)
        GT = Res / V_SAT  # sec
        return GT
    
    # Somewhat arbitrary combinations of buffer sizes and frame times
    capcitances = [10e-3, 1e-1, 1, 5.6, 8]; # F
    H = [600e3,600e3, 600e3, 600e3, 600e3]; # m
    FOV = [45, 30, 15, 5, 2.5]; # degrees

    # Generate Budgets
    buffers  = [Energy_in_Cap(C,5) for C in capcitances]  # joules
    frame_times = [Frame_Time(H[i], FOV[i]) for i in range(len(H))]  # sec

    # sort ascending
    buffers.sort()
    frame_times.sort()

    plots.budget_correct_loop(subset, buffers,frame_times ,REPO_ROOT/ "results")

    # matrix_winner = np.zeros((len(buffers), len(frame_times)), dtype=object)

    # # Test all combinations
    # for i, buffer in enumerate(buffers):
    #     for j, frame_time in enumerate(frame_times):

    #         matrix_winner[i, j] = plots.budgeted_correct_inferences(
    #             df=subset,
    #             buffer_energy=buffer,     # use actual buffer
    #             frame_time=frame_time,    # use actual frame time
    #             results_dir=(REPO_ROOT / "results/captures/IMG_CLASS02"),
    #             model_category="Img_Class",
    #             run_names=run_names,
    #             filename=REPO_ROOT / "results/plots/img_class_power_runs.png"
    #         )
    
    # df_winner = pd.DataFrame(
    #     matrix_winner,
    #     index=buffers,
    #     columns=frame_times
    # )

    # df_winner.to_excel("winner_matrix.xlsx")

# run_names=["EfficientNet-EdgeTpu (L)",
#     "EfficientNet-EdgeTpu (M)",
#     "EfficientNet-EdgeTpu (S)",
#     "Inception V1",
#     "Inception V2",
#     "MobileNet V1 (0.25)",
#     "MobileNet V1 (0.50)",
#     "MobileNet V1 (.75)",
#     "MobileNet V1 (1.0)",
#     "MobileNet V1 (TF_ver_2.0)",
#     "MobileNet V2",
#     "MobileNet V2 (TF_ver_2.0)",
#     "MobileNet V3"]

    jacknet_sweep_plot(REPO_ROOT / "results/captures/jacknet_sweep", REPO_ROOT / "results/plots/jacknet_sweep.png")