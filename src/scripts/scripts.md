# Scripts

This document describes the scripts contained in `src/scripts` and its subfolders. For each script we list the relative path, a verbose description of behavior, inputs/outputs, and notes on where results are written.

---

## Summary

- Location root: `libs/coral_tpu_characterization/src/scripts`
- Subfolders: `dynamic_selection_sims/`, `hardware_characterization/` (with `plotting/`), and `utils/`
- Common outputs:
  - Saleae captures and raw CSVs (digital/analog) → `results/captures/*/saleae_raw`
  - Processed plots → `results/plots` or script-specific `output_dir`
  - Measurement CSVs (e.g., inference results) → `inference_results.csv` or user-specified file

---

## hardware_characterization/

### `hardware_characterization/bench_mark_sweep.py`

- Purpose: Full automation of firmware build, board flash, and capture collection for EdgeTPU-compiled models.
- Behavior: For each compiled `_edgetpu.tflite` model it can (optionally) build the coralmicro firmware (via CMake + Make), flash the target board using the coralmicro flashtool, and then call the Saleae Logic 2 automation API to perform a timed capture. After capture it uses the `SaleaeOutputParsing` helper to compute average inference time and energy metrics.
- Inputs: model directory (defaults to `~/Coral-TPU-Characterization/data/models/edgeTPU_acc`), capture directory, serial port, shunt resistance and supply voltage, build/test flags.
- Outputs: One capture folder per run (default `results/captures/<model_stem>_<timestamp>/saleae_raw`) and an `inference_results.csv` with rows: model, avg inference time (ms), average energy per inference (J or mJ), category.
- Notes: Relies on the `saleae.automation` package and a local Saleae Manager; requires correct serial port and coralmicro flashtool available on PATH.

### `hardware_characterization/plotting/model_stats_plotting.py`

- Purpose: Generate comparative plots across model sweeps (image classification, object detection, etc.) using Saleae-derived power/latency metrics and spreadsheet metadata.
- Behavior: Collects measured latency/power using `SaleaeOutputParsing`, combines these with Excel metadata (accuracy, quoted latency), computes derived metrics (Inf/s, Inf/J, Correct Inf metrics) and produces multi-panel figures (parameter count, latency, accuracy, combined quoted vs measured plots) written to a user-specified `plotdir` or `results/plots`.
- Inputs: Excel metadata workbook + run directories under `results/captures` or an explicit directory pointer.
- Outputs: PNG plots (e.g., `img_class_plot.png`, `img_class_combined.png`) and optionally appended Excel sheets with collected/derived values.
- Notes: Exposes `ModelStatsPlotting` class for programmatic use and helper functions for quick analyses (e.g., `jacknet_sweep_plot`).

### `hardware_characterization/plotting/tpunet_plotting.py`

- Purpose: Aggregate grid sweep JSON metrics and Saleae measurements for the custom Grid/TpuNet model family and produce rich plotting tools and champion-finding helpers.
- Behavior: `GridStatsPlotting` loads JSON evaluation files (e.g., `Grid_A0.25_D02_quant_eval.json`), finds corresponding Saleae captures (searching `saleae_raw` locations), uses `ParamCounts` to attach parameter counts, derives efficiency/throughput metrics and provides several plotting methods (standard metrics, grouped metrics by alpha/depth, efficiency overviews, 3D surfaces).
- Inputs: Directory of JSON metric files, `saleae_root` pointing to the base captures directory, and an `output_dir` for plots.
- Outputs: Multiple PNGs (grid metrics, grouped metrics, 3D surfaces) saved under the provided `output_dir`.
- Notes: Designed to be robust to missing JSONs or missing Saleae runs (will warn and skip missing entries).

---

## dynamic_selection_sims/

### `dynamic_selection_sims/dynamic_execution.py`

- Purpose: Simulate satellite onboard inference scheduling and dynamic model selection to maximize correct inferences given energy and time constraints.
- Behavior: Implements `SatelliteInferenceSim` which (1) loads orbit telemetry and solar intensity CSVs, (2) ingests model performance data (via `GridStatsPlotting` aggregation), (3) calculates geometry-derived per-step demand (dwell time, pixels-per-target, required inferences), and (4) runs multiple policies: sequential dynamic optimization, static baselines, naive baseline, and produces battery traces, success/failure tallies and per-step annotations.
- Inputs: STK-derived CSVs (orbit/velocity/solar tables), path to JSON metrics and Saleae measurements for models, simulation config (battery capacity, solar generation, baseload, thresholds).
- Outputs: A DataFrame with per-step decisions (selected model, number of inferences, energy consumed, status) and summary reports printed to stdout. The class also supports saving result traces to disk if the caller arranges it.
- Notes: Useful for research experiments that combine on-orbit energy constraints with model tradeoffs; extendable to include switching penalties and more complex schedules.

### `dynamic_selection_sims/decision_plots.py`

- Purpose: Visualization helpers to plot the outputs of dynamic selection simulations.
- Behavior: Functions to draw battery traces, decision timelines, histograms of statuses (Success/Partial/Fail/Blind), and overlays of solar intensity and demand. Accepts the DataFrame produced by `SatelliteInferenceSim`.
- Inputs: Simulation DataFrames and plotting configuration (save paths, figure sizes).
- Outputs: PNG figures summarizing simulation performance.

### `dynamic_selection_sims/ground_track_stk.py`

- Purpose: Helpers to load and process STK-provided ground track CSVs used by the satellite simulations.
- Behavior: Loads `HEO_Sat_Fixed_Position_Velocity.csv`, `HEO_Sat_Classical_Orbit_Elements.csv`, and optional solar intensity CSVs; performs light cleaning, wrapping, and provides geometric utilities used by `SatelliteInferenceSim`.
- Inputs: Directory containing STK CSV exports.
- Outputs: Cleaned pandas DataFrame(s) ready for simulation.

### `dynamic_selection_sims/static_model_dynamic.py`

- Purpose: Baseline simulation modules that run static scheduling policies (e.g., always run a particular model) for performance comparison against the dynamic strategy.
- Behavior: Implements naive/static baseline strategies (throughput-focused, efficiency-focused, worst-case, etc.) and returns performance metrics and battery traces.

---

## utils/

### `utils/saleae_parsing.py`

- Purpose: Core parser for Saleae Logic 2 raw CSV outputs (digital and analog channels).
- Behavior: `SaleaeOutputParsing` locates `digital.csv` and `analog.csv` (in `saleae_raw` folders), extracts rising/falling edge times, computes average inference time, average power and energy per inference (with optional idle subtraction), and provides plotting helpers such as `plot_saleae_trace()` for diagnostics.
- Inputs: Directory containing `digital.csv` and `analog.csv`; parameters for PSU voltage and shunt resistance.
- Outputs: Programmatic metrics (avg inference time [s], avg power [W], energy per inference [J]) and optional diagnostic PNGs.
- Notes: Saves/loads computed idle power at `results/captures/idle_power/idle.csv` to allow subtracting idle consumption across runs.

### `utils/ParamCounts.py`

- Purpose: Count model parameters from `.tflite` files using FlatBuffers and the TFLite schema.
- Behavior: Walks a provided directory, loads each `.tflite` model, inspects buffers and tensor types, and calculates total parameter count (raw element counts) while handling different tensor data types (float32, int8, etc.).
- Inputs: Directory root to scan for `.tflite` files.
- Outputs: A tuple `(list_of_counts, dict_name_to_count)` and optionally a `param_counts.json` when run directly in a script context.
- Notes: Intended for CPU `.tflite` models — when using EdgeTPU compiled models, parameter arrays may differ or be fused; interpret with care.

### `utils/model_data_manager.py`

- Purpose: Produce a clean, merged DataFrame combining Excel metadata (accuracies, quoted latencies) with Saleae-collected measurements for downstream plotting or simulations.
- Behavior: Scans a `results/captures`-like directory for saleae runs, loads the referenced Excel sheet (`Img_Class` by default), aligns measured runs with metadata, coalesces measured vs quoted accuracy, drops invalid rows, and computes derived metrics (`Inf_per_Sec`, `Inf_per_Joule`, `Correct_Inf_per_Sec`, `Correct_Inf_per_Joule`).
- Inputs: Path to Excel workbook and results directory containing Saleae captures.
- Outputs: A tidy pandas DataFrame ready for plotting or simulation.

### `utils/path_utils.py`

- Purpose: Locate the repository root and provide a single canonical method to build absolute paths within scripts.
- Behavior: `get_repo_root()` climbs parent directories from the calling file location looking for repo markers such as `.git` or the presence of `libs` + `results`. It also respects an environment override `CORAL_REPO`.
- Inputs: None (auto-detects from file location); optional env var `CORAL_REPO` can set path explicitly.
- Outputs: `pathlib.Path` pointing to the repository root.
- Notes: This function is used widely in scripts to build paths independent of the current working directory.
