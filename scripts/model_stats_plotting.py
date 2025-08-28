import pandas as pd
import matplotlib.pyplot as plt
from Param_Counts import Param_Counts

def img_class_plt(sheet):
    """Triple Stacked Bar Chart of Model Stats for Image Classification Models"""
    pc = Param_Counts("/home/jackr/Coral-TPU-Characterization/models/Image_Classification")
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

    fig, ax = plt.subplots(nrows=len(titles), ncols=1, sharex=True, figsize=(16, 12))

    for i, (title, unit, y) in enumerate(zip(titles, units, values)):
        x_pos = range(len(model_names))
        ax[i].bar(x_pos, y)
        ax[i].set_title(title)
        ax[i].set_ylabel(unit)
        ax[i].set_xticks(x_pos)
        ax[i].set_xticklabels(model_names, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("img_class_plot.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    img_class_plt("scripts/Model_Stats.xlsx")
