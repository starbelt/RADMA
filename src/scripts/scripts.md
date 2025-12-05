# Scripts

## [model_stats_plotting.py](model_stats_plotting.py)

Collects data utilizing saleae parsing and paramcounts to plot parameters regarding model performance. Plottable metrics include:

* Parameter Count
* Inference Time
* Top-1 Accuracy (Image Classification), mAP (Object Detection)
* Energy per Inference
* Correct inferences within power/time budgets

plots written to ~/results/plots

## [bench_mark_sweep.py](bench_mark_sweep.py)

Performs automated building, flashing, and saleae data collection on Coral Micro Dev devices. Calls the Logic2 API to interface with the Saleae device. Writes run data for power usage to ~/results/captures. Experimental setup information can be found at ENTER PATH TO EXPERIMENTS MD HERE. Build files written to ~/out

## [ParamCounts.py](ParamCounts.py)

Utilizes flatbuffers to read tensor sizes of tensorflow files for CPU usage (NOT COMPILED FOR EDGETPU) to calculate the parameter counts for a model

## [path_utils.py](path_utils.py)

A simple to use utility for finding the working root so that pathing may be written entirely relative to the repo root, allows for ease of use on both native linux and WSL.
