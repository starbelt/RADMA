import sys

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from adjustText import adjust_text
from ParamCounts import ParamCounts

def read_inf_times(file):
    # convert csv to data frame
    df =pd.read_csv(file)
    model = df["model"].astype(str).to_numpy() # model file name
    time = df["avg_logic_ms"].astype(float).to_numpy() # average inference time output from logic analyzer
    path = df["model_dir"].astype(str).to_numpy()


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
class ModelStatsPlotting:
    def __init__(self,xlsx,plotdir):
        self.sheet = path.abspath(xlsx)
        self.plotdir = path.abspath(plotdir)

    def img_class_plt(self):
        """Quadruple Stacked Bar Chart of Model Stats for Image Classification Models"""

        model_dir = "~/Coral-TPU-Characterization/models/Image_Classification"
        pc = ParamCounts(model_dir)
        inf_real = "Image_Classufication_inference_results.csv"
        param_counts = [x/1e6 for x in pc.scan_models()]  # scale to millions

        ic_df = pd.read_excel(
            self.sheet,
            sheet_name="Img_Class",
            header=0,
            usecols=["Model name","Latency (ms)", "Top-1 Accuracy", "Top-5 Accuracy"]
        )

        model_names = ic_df["Model name"].tolist()
        latency_ms = ic_df["Latency (ms)"].tolist()
        top1accuracy = ic_df["Top-1 Accuracy"].tolist()
        # top5accuracy = ic_df["Top-5 Accuracy"].tolist()

        combined = list(zip(latency_ms, model_names, param_counts, top1accuracy))
        combined.sort(key=lambda x: x[0]) # sort by latency - fastest to slowest

        latency_ms, model_names, param_counts, top1accuracy = zip(*combined) # overwrite with sorted lists

        titles = ["Parameter Count","Latency ", "Top-1 Accuracy"]
        values = [param_counts, latency_ms, top1accuracy]
        units = ["# Params (M)", "ms", "%"]

        make_figure(titles,model_names,values,units,self.plotdir+"/img_class_plot.png")
        param_latency_scatter(model_names,param_counts,latency_ms,self.plotdir+"/img_class_sctplot.png")

    def obj_det_plt(self):
        """Triple Stacked Bar Chart of Model Stats for Object Detection"""
        model_dir = "~/Coral-TPU-Characterization/models/Object_Detection"
        pc = ParamCounts(model_dir)

        param_counts = [x/1e6 for x in pc.scan_models()]  # scale to millions
        ic_df = pd.read_excel(
            self.sheet,
            sheet_name="Obj_Det",
            header=0,
            usecols=["Model Name","Latency (ms)", "mAP"]
        )

        model_names = ic_df["Model Name"].tolist()
        latency_ms = ic_df["Latency (ms)"].tolist()
        mAP = ic_df["mAP"].tolist()

        combined = list(zip(latency_ms, model_names, param_counts, mAP))
        combined.sort(key=lambda x: x[0]) # sort by latency - fastest to slowest

        latency_ms, model_names, param_counts, mAP = zip(*combined) # overwrite with sorted lists

        titles = ["Parameter Count","Latency (ms)", "Mean Average Precision"]
        values = [param_counts, latency_ms, mAP]
        units = ["# Params (M)", "ms", "%"]

        make_figure(titles,model_names,values,units,self.plotdir+"/obj_det_plot.png")

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
    plots = ModelStatsPlotting("scripts/Model_Stats.xlsx","plots/")
    plots.img_class_plt()
    plots.obj_det_plt()
    plots.segmentation_plt()
