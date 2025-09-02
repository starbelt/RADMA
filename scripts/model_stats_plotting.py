import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from ParamCounts import ParamCounts

def make_figure(titles,names,values,units,filename):
    cmap = matplotlib.colormaps["tab20"]
    colors = [cmap(i) for i in range(len(names))]

    fig, ax = plt.subplots(nrows=len(titles), ncols=1, sharex=True, figsize=(16, 12))

    for i, (title, unit, y) in enumerate(zip(titles, units, values)):
        x_pos = range(len(names))
        ax[i].bar(x_pos, y, color=colors)
        ax[i].set_title(title)
        ax[i].set_ylabel(unit)
        ax[i].set_xticks(x_pos)
        ax[i].set_xticklabels(names, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")

def img_class_plt(sheet):
    """Quadruple Stacked Bar Chart of Model Stats for Image Classification Models"""

    model_dir = "~/Coral-TPU-Characterization/models/Image_Classification"
    pc = ParamCounts(model_dir)

    param_counts = [x/1e6 for x in pc.scan_models()]  # scale to millions

    ic_df = pd.read_excel(
        sheet,
        sheet_name="Img_Class",
        header=0,
        usecols=["Model name","Latency (ms)", "Top-1 Accuracy", "Top-5 Accuracy"]
    )

    model_names = ic_df["Model name"].tolist()
    latency_ms = ic_df["Latency (ms)"].tolist()
    top1accuracy = ic_df["Top-1 Accuracy"].tolist()
    top5accuracy = ic_df["Top-5 Accuracy"].tolist()

    titles = ["Parameter Count","Latency ", "Top-1 Accuracy", "Top-5 Accuracy"]
    values = [param_counts, latency_ms, top1accuracy, top5accuracy]
    units = ["# Params (M)", "ms", "%", "%"]

    make_figure(titles,model_names,values,units,"img_class_plot.png")

def obj_det_plt(sheet):
    """Triple Stacked Bar Chart of Model Stats for Object Detection"""
    model_dir = "~/Coral-TPU-Characterization/models/Object_Detection"
    pc = ParamCounts(model_dir)

    param_counts = [x/1e6 for x in pc.scan_models()]  # scale to millions

    ic_df = pd.read_excel(
        sheet,
        sheet_name="Obj_Det",
        header=0,
        usecols=["Model name","Latency (ms)", "mAP"]
    )

    model_names = ic_df["Model name"].tolist()
    latency_ms = ic_df["Latency (ms)"].tolist()
    mAP = ic_df["mAP"].tolist()

    titles = ["Parameter Count","Latency (ms)", "Mean Average Precision"]
    values = [param_counts, latency_ms, mAP]
    units = ["# Params (M)", "ms", "%"]

    make_figure(titles,model_names,values,units,"obj_det_plot.png")

def segmentation_plt(sheet):
    """Double Stacked Bar Chart of Model Stats for Segmentation"""
    model_dir = "~/Coral-TPU-Characterization/models/Segmentation"

    pc = ParamCounts(model_dir)
    param_counts = [x/1e6 for x in pc.scan_models()]  # scale to millions

    ic_df = pd.read_excel(
        sheet,
        sheet_name="Segmentation",
        header=0,
        usecols=["Model name","Latency (ms)"]
    )

    model_names = ic_df["Model name"].tolist()
    latency_ms = ic_df["Latency (ms)"].tolist()

    titles = ["Parameter Count","Latency (ms)"]
    values = [param_counts, latency_ms]
    units = ["# Params (M)", "ms"]

    make_figure(titles,model_names,values,units,"segmentation_plot.png")


if __name__ == "__main__":
    img_class_plt("scripts/Model_Stats.xlsx")
