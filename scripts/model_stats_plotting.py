#!/usr/bin/env python3
import sys
import pathlib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from os import path
from adjustText import adjust_text

from ParamCounts import ParamCounts
from saleae_parsing import SaleaeOutputParsing


def inference_from_csvs(rundir):
    """
    Returns measured inference times (ms) for all subdirs in rundir.
    Works even if only digital.csv is available.
    """
    top_dir = pathlib.Path("captures") / rundir
    inference_time_ms = []
    if not top_dir.exists():
        return inference_time_ms

    for root, dirs, _ in top_dir.walk():
        for d in sorted(dirs):
            try:
                parsed = SaleaeOutputParsing(pathlib.Path(root) / d)
                inference_time_ms.append(parsed.avg_inference_time() * 1e3)
            except FileNotFoundError:
                continue
    return inference_time_ms


def collect_results(run_dir, psu_dc_volts=5.0, r_shunt=1.0):
    """
    Collect digital+analog Saleae results into dict.
    """
    run_dir = pathlib.Path(run_dir).expanduser()
    results = {}
    if not run_dir.exists():
        return results

    for subdir in sorted(run_dir.iterdir()):
        if not subdir.is_dir():
            continue
        try:
            parsed = SaleaeOutputParsing(subdir)
        except FileNotFoundError:
            continue
        avg_time = parsed.avg_inference_time() * 1e3  # ms
        mean_pwr, _, mean_energy, _ = parsed.avg_power_measurement(psu_dc_volts, r_shunt)
        if mean_pwr is not None:
            results[subdir.name] = {
                "inference_time_ms": avg_time,
                "avg_power_mW": mean_pwr * 1e3,
                "energy_mJ": mean_energy * 1e3,
            }
    return results



def lighten_color(color, factor=0.5):
    """
    Lightens the given color.
    factor=0 -> white, factor=1 -> original color.
    """
    r, g, b = matplotlib.colors.to_rgb(color)
    return (1 - factor) + factor * r, (1 - factor) + factor * g, (1 - factor) + factor * b


def make_figure(titles, names, values, units, filename):
    cmap = matplotlib.colormaps["tab20"]
    colors = [cmap(i) for i in range(len(names))]
    fig, ax = plt.subplots(nrows=len(titles), ncols=1, sharex=True, figsize=(16, 12))

    for i, (title, unit, y) in enumerate(zip(titles, units, values)):
        x_pos = range(len(names))
        ax[i].bar(x_pos, y, color=colors)
        ax[i].set_title(title, fontsize=20)
        ax[i].set_ylabel(unit, fontsize=18)
        ax[i].set_xticks(x_pos)
        ax[i].set_xticklabels(names, rotation=45, fontsize=18, ha="right")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")


def make_combined_latency_figure(names, param_counts, accuracy, measured_acc, measured, quoted, units, filename):
    """
    Create a 3-row figure:
    1. Parameter count
    2. Accuracy (Quoted vs Measured)
    3. Stacked latency: measured vs quoted
    """
    cmap = matplotlib.colormaps["tab20"]
    colors = [cmap(i) for i in range(len(names))]
    x_pos = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(16, 12))

    # 1) Param count
    axes[0].bar(x_pos, param_counts, color=colors)
    axes[0].set_title("Parameter Count", fontsize=20)
    axes[0].set_ylabel(units[0], fontsize=18)

    # 2) Accuracy
    axes[1].bar(x_pos - width / 2, accuracy, width=width, color=colors, label="Quoted")
    axes[1].bar(
        x_pos + width / 2,
        measured_acc,
        width=width,
        color=[lighten_color(c, 0.5) for c in colors],
        label="Measured",
    )
    axes[1].set_title("Accuracy", fontsize=20)
    axes[1].set_ylabel(units[1], fontsize=18)
    axes[1].set_ylim(0, 100)
    axes[1].legend()

    # 3) Latency stacked
    measured = np.array(measured)
    quoted = np.array(quoted)
    for i, (m, q) in enumerate(zip(measured, quoted)):
        if m >= q:
            axes[2].bar(x_pos[i], q, color=colors[i], label="Quoted" if i == 0 else "")
            axes[2].bar(
                x_pos[i], m - q, bottom=q, color=lighten_color(colors[i], 0.5), label="Measured" if i == 0 else ""
            )
            axes[2].text(x_pos[i], m, f"{m:.1f}", ha="center", va="bottom", fontsize=12)
        else:
            axes[2].bar(x_pos[i], m, color=colors[i], label="Measured" if i == 0 else "")
            axes[2].bar(
                x_pos[i], q - m, bottom=m, color=lighten_color(colors[i], 0.5), label="Quoted" if i == 0 else ""
            )
            axes[2].text(x_pos[i], q, f"{q:.1f}", ha="center", va="bottom", fontsize=12)

    axes[2].set_title("Measured vs Quoted Latency", fontsize=20)
    axes[2].set_ylabel(units[2], fontsize=18)
    axes[2].set_ylim(0, max(measured.max(), quoted.max()) * 1.15)
    axes[2].legend()

    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(names, rotation=45, ha="right", fontsize=16)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def param_latency_scatter(names, paramcount, latency, filename):
    cmap = matplotlib.colormaps["tab20"]
    colors = [cmap(i) for i in range(len(names))]
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(paramcount, latency, s=100, color=colors)
    ax.set_xlabel("Parameter count (Millions)", fontsize=18)
    ax.set_ylabel("Latency (ms)", fontsize=18)

    texts = []
    for i, name in enumerate(names):
        texts.append(ax.text(paramcount[i], latency[i], name, fontsize=12))

    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="->", color="gray"),
        expand=(1.2, 1.4),
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")


class ModelStatsPlotting:
    def __init__(self, xlsx, plotdir):
        self.sheet = path.abspath(xlsx)
        self.plotdir = path.abspath(plotdir)

    def img_class_plt(self):
        model_dir = "~/Coral-TPU-Characterization/models/Image_Classification"
        pc = ParamCounts(model_dir)
        inf_ms = inference_from_csvs("IMG_CLASS_10s_doublearena")
        param_counts = [x / 1e6 for x in pc.scan_models()]

        ic_df = pd.read_excel(
            self.sheet,
            sheet_name="Img_Class",
            header=0,
            usecols=["Model name", "Latency (ms)", "Top-1 Accuracy", "Top-1 Accuracy (measured)", "Top-5 Accuracy"],
        )

        model_names = ic_df["Model name"].tolist()
        latency_ms = ic_df["Latency (ms)"].tolist()
        top1accuracy = ic_df["Top-1 Accuracy"].tolist()
        measured_acc_list = ic_df["Top-1 Accuracy (measured)"].tolist()

        # Build combined tuples (ensure consistent length)
        combined = list(zip(inf_ms, latency_ms, model_names, param_counts, top1accuracy, measured_acc_list))
        if not combined:
            print("Warning: no models found in combined list.")
            return

        combined.sort(key=lambda x: x[0])
        inf_ms, latency_ms, model_names, param_counts, top1accuracy, measured_acc_list = zip(*combined)

        updated_df = pd.DataFrame(
            {
                "Model name": model_names,
                "Latency (ms)": latency_ms,
                "Top-1 Accuracy": top1accuracy,
                "Top-1 Accuracy (measured)": measured_acc_list,
                "Measured Inference Time (ms)": inf_ms,
                "Parameter Count (M)": param_counts,
            }
        )
        with pd.ExcelWriter(self.sheet, mode="a", if_sheet_exists="replace", engine="openpyxl") as writer:
            updated_df.to_excel(writer, sheet_name="Img_Class_outs", index=False)

        make_combined_latency_figure(
            model_names,
            param_counts,
            top1accuracy,
            measured_acc_list,
            inf_ms,
            latency_ms,
            ["# Params (M)", "%", "ms"],
            self.plotdir + "/img_class_combined.png",
        )

        param_latency_scatter(model_names, param_counts, latency_ms, self.plotdir + "/img_class_sctplot.png")
        param_latency_scatter(model_names, param_counts, inf_ms, self.plotdir + "/img_class_sctplot_inf.png")


if __name__ == "__main__":
    plots = ModelStatsPlotting("scripts/Model_Stats.xlsx", "plots/")
    plots.img_class_plt()
