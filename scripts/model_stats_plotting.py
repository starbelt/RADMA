import sys

import matplotlib, pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path
from adjustText import adjust_text
from ParamCounts import ParamCounts
from saleae_parsing import SaleaeOutputParsing


# Data Collection and Sorting Functions ## 
def inference_from_csvs(rundir):
    """
    If you only need the inference time and no power metrics (e.g. if you only have a digital.csv file)
    """
    top_dir = pathlib.Path("captures") / rundir
    inference_time_ms = []
    for root, dirs, files in top_dir.walk():
        for d in sorted(dirs):
            parsed = SaleaeOutputParsing(root / d)
            if parsed.avg_inference_time() is not None:
                inference_time_ms.append(parsed.avg_inference_time() * 1e3)
            else:
                inference_time_ms.append(None)
    return inference_time_ms

def collect_results(run_dir, psu_dc_volts=5.0, r_shunt=1.0):
    """
    Collects primary results of digital and analog Saleae results into a dict for ease of use.
    Point to directory that directly parents the .CSVs
    """
    run_dir = pathlib.Path(run_dir).expanduser()
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
            mean_pwr, all_pwr, mean_energy, all_energy = parsed.avg_power_measurement(
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

    axes[2].set_title("Measured vs Quoted Latency", fontsize=20)
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

        model_dir = "~/Coral-TPU-Characterization/models/Image_Classification"
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

    def power_inf_runs(self, df, results_dir : str, model_category = None, run_names = None,  filename=None):
        """
        4-row figure ordered by ascending Energy per Inference (mJ):
        1) Energy per inference (mJ)
        2) Top-1 Accuracy (%)
        3) Stacked latency (measured vs quoted) (ms)
        4) Top-1 Accuracy per Energy (% / mJ)

        Returns the sorted dataframe used for plotting.
        """

        if model_category is None:
            sheet_name = "Img_Class"
        elif model_category not in ["Img_Class","Obj_Det","Segmentation","Audio_Classification"]:
            sheet_name = model_category
        else:
            raise ValueError(
            f"Model Category {model_category} does not exist. "
            "Choose from: Img_Class, Obj_Det, Segmentation, Audio_Classification"
            )
        if run_names is None:
            run_names=[ # all image classification models
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
                "MobileNet V3"
                ]
        ## Read model parameters from excel sheet and test results from csvs in the results_dir
        results_dict = collect_results(results_dir)
        df = pd.read_excel(self.sheet, sheet_name)

        ## Map those two into a unified dataframe, only including the specified models
        if not run_names:
            subset = df
        subset = df[df["Model name"].isin(run_names)].copy()
        if subset.empty:
            raise ValueError(f"No matching models found for run_names={run_names}")

        # map measured/power/energy from results_dict
        subset["Measured Inference Time (ms)"] = subset["Model name"].map(
            lambda m: results_dict.get(m, {}).get("inference_time_ms")
        )
        subset["Average Power (mW)"] = subset["Model name"].map(
            lambda m: results_dict.get(m, {}).get("avg_power_mW")
        )
        subset["Energy per Inference (mJ)"] = subset["Model name"].map(
            lambda m: results_dict.get(m, {}).get("energy_mJ")
        )

        # coerce numeric and required columns
        for col in ["Energy per Inference (mJ)", "Measured Inference Time (ms)", "Latency (ms)", "Top-1 Accuracy (measured)"]:
            if col in subset.columns:
                subset[col] = pd.to_numeric(subset[col], errors="coerce")
            else:
                raise KeyError(f"Required column missing in dataframe: {col}")

        # drop rows missing any required value
        required = ["Energy per Inference (mJ)", "Measured Inference Time (ms)", "Latency (ms)", "Top-1 Accuracy (measured)"]
        valid_mask = subset[required].notnull().all(axis=1)
        if not valid_mask.all():
            dropped = subset.loc[~valid_mask, "Model name"].tolist()
            print(f"Dropped runs due to missing data: {dropped}")
            subset = subset.loc[valid_mask].copy()

        if subset.empty:
            raise ValueError("No valid runs left after dropping rows with missing data.")

        # sort by energy ascending and reset index
        subset.sort_values("Energy per Inference (mJ)", ascending=True, inplace=True)
        subset.reset_index(drop=True, inplace=True)

        # prepare plotting arrays AFTER sorting
        names = subset["Model name"].tolist()
        energy = subset["Energy per Inference (mJ)"].to_numpy()
        accuracy = subset["Top-1 Accuracy (measured)"].to_numpy()
        measured = subset["Measured Inference Time (ms)"].to_numpy()
        quoted = subset["Latency (ms)"].to_numpy()
        acc_per_energy = accuracy / energy

        cmap = matplotlib.colormaps["tab20"]
        colors = [cmap(i) for i in range(len(names))]
        x_pos = np.arange(len(names))

        fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(14, 16))

        # 1) Energy per inference
        axes[0].bar(x_pos, energy, color=colors)
        axes[0].set_title("Energy per Inference", fontsize=18)
        axes[0].set_ylabel("mJ", fontsize=14)
        for i, e in enumerate(energy):
            axes[0].text(x_pos[i], e * 1.01, f"{e:.1f}", ha="center", va="bottom", fontsize=10)

        # 2) Top-1 Accuracy
        axes[1].bar(x_pos, accuracy, color=colors)
        axes[1].set_title("Top-1 Accuracy", fontsize=18)
        axes[1].set_ylabel("%", fontsize=14)
        axes[1].set_ylim(0, 100)
        for i, a in enumerate(accuracy):
            axes[1].text(x_pos[i], a * 1.01, f"{a:.1f}", ha="center", va="bottom", fontsize=10)

        # 3) Stacked latency (measured vs quoted)
        for i, (m, q) in enumerate(zip(measured, quoted)):
            if m >= q:
                axes[2].bar(x_pos[i], q, color=colors[i])
                axes[2].bar(x_pos[i], m - q, bottom=q, color=lighten_color(colors[i], 0.5))
                axes[2].text(x_pos[i], m * 1.01, f"{m:.1f}", ha="center", va="bottom", fontsize=10)
            else:
                axes[2].bar(x_pos[i], m, color=colors[i])
                axes[2].bar(x_pos[i], q - m, bottom=m, color=lighten_color(colors[i], 0.5))
                axes[2].text(x_pos[i], q * 1.01, f"{q:.1f}", ha="center", va="bottom", fontsize=10)

        axes[2].set_title("Measured vs Quoted Latency", fontsize=18)
        axes[2].set_ylabel("ms", fontsize=14)
        y_max = max(measured.max(initial=0), quoted.max(initial=0))
        axes[2].set_ylim(0, y_max * 1.2)

        # 4) Accuracy per Energy
        axes[3].bar(x_pos, acc_per_energy, color=colors)
        axes[3].set_title("Top-1 Accuracy per Energy (% / mJ)", fontsize=18)
        axes[3].set_ylabel("% / mJ", fontsize=14)
        for i, v in enumerate(acc_per_energy):
            axes[3].text(x_pos[i], v * 1.01, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

        # shared x labels
        axes[3].set_xticks(x_pos)
        axes[3].set_xticklabels(names, rotation=45, ha="right", fontsize=12)

        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

        return subset

    def obj_det_plt(self):
        """Triple Stacked Bar Chart of Model Stats for Object Detection"""
        model_dir = "~/Coral-TPU-Characterization/models/Object_Detection"
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
        model_dir = "~/Coral-TPU-Characterization/models/Segmentation"

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


if __name__ == "__main__":
    plots = ModelStatsPlotting("scripts/Model_Stats.xlsx", "plots/")
    plots.img_class_plt()
    plots.obj_det_plt()
    # plots.segmentation_plt()

    # Select the runs you want to visualize
    run_names = ["EfficientNet-EdgeTpu (M)", "EfficientNet-EdgeTpu (S)", "Inception V3", "MobileNet V1 (0.25)"]

    # Call the method that already merges Excel + Saleae results
    subset = plots.power_inf_runs(
        df=None,  # will get loaded inside the method
        results_dir="captures/IMG_CLASS_NEW",
        model_category="Img_Class",
        run_names=run_names,
        filename="plots/img_class_power_runs.png"
    )

    print(subset["Model name"].tolist())


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