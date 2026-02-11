import pathlib
import pandas as pd
import numpy as np
from libs.coral_tpu_characterization.src.scripts.utils.saleae_parsing import SaleaeOutputParsing
from libs.coral_tpu_characterization.src.scripts.utils.path_utils import get_repo_root

class ModelDataManager:
    """
    Handles the ingestion of Saleae power/latency data and merges it with 
    Excel metadata (Accuracy, Model Names) to produce a unified DataFrame
    for the Dynamic Execution Model.
    """
    def __init__(self, excel_path, results_dir):
        self.excel_path = pathlib.Path(excel_path).resolve()
        self.results_dir = pathlib.Path(results_dir).resolve()

    def _collect_saleae_results(self, psu_dc_volts=5.0, r_shunt=0.2):
        """
        Scans the results directory for valid Saleae captures.
        Returns a dict keyed by folder name.
        """
        results = {}
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {self.results_dir}")

        for subdir in sorted(self.results_dir.iterdir()):
            if not subdir.is_dir():
                continue
            try:
                # Assuming SaleaeOutputParsing is available in your environment
                parsed = SaleaeOutputParsing(subdir)
            except (FileNotFoundError, ValueError):
                continue

            if parsed.avg_inference_time() is not None:
                avg_time = parsed.avg_inference_time() * 1e3  # convert s to ms
                mean_pwr, _, mean_energy, _ = parsed.avg_power_measurement(
                    psu_dc_volts, r_shunt
                )
                
                # Only add if we have valid power data
                if mean_pwr is not None:
                    results[subdir.name] = {
                        "inference_time_ms": avg_time,
                        "avg_power_mW": mean_pwr * 1e3,
                        "energy_mJ": mean_energy * 1e3 if mean_energy else 0
                    }
        return results

    def get_compiled_dataframe(self, sheet_name="Img_Class", run_names=None):
        """
        Generates the master dataframe containing both custom and pre-built models.
        
        Args:
            sheet_name: The Excel sheet to read metadata from.
            run_names: (Optional) List of specific model names to filter by. 
                       If None, tries to match all folders found in results_dir.
        """
        # 1. Collect Raw Data
        saleae_data = self._collect_saleae_results()
        
        # 2. Load Metadata
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.excel_path}")
        
        df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
        
        # 3. Filter Models
        # If run_names is provided, filter by it. 
        # Otherwise, filter by what we actually have measured data for.
        if run_names:
            subset = df[df["Model name"].isin(run_names)].copy()
        else:
            # Only keep models we have measurements for
            measured_models = list(saleae_data.keys())
            subset = df[df["Model name"].isin(measured_models)].copy()

        if subset.empty:
            raise ValueError(f"No matching models found. Checked against {len(saleae_data)} measured runs.")

        # 4. Map Saleae Data to DataFrame
        # We use map() to ensure alignment with the "Model name" column
        subset["Measured Inference Time (ms)"] = subset["Model name"].map(
            lambda m: saleae_data.get(m, {}).get("inference_time_ms")
        )
        subset["Energy per Inference (mJ)"] = subset["Model name"].map(
            lambda m: saleae_data.get(m, {}).get("energy_mJ")
        )
        subset["Average Power (mW)"] = subset["Model name"].map(
            lambda m: saleae_data.get(m, {}).get("avg_power_mW")
        )

        # 5. Data Cleaning
        # Ensure numeric types
        cols_to_numeric = [
            "Measured Inference Time (ms)", 
            "Energy per Inference (mJ)", 
            "Top-1 Accuracy", 
            "Top-1 Accuracy (measured)"
        ]
        
        for col in cols_to_numeric:
            if col in subset.columns:
                subset[col] = pd.to_numeric(subset[col], errors="coerce")
        
        # Handle 'Top-1 Accuracy (measured)' vs 'Top-1 Accuracy' fallback
        if "Top-1 Accuracy (measured)" not in subset.columns:
            subset["Top-1 Accuracy (measured)"] = subset["Top-1 Accuracy"]
        
        # Fill NaNs where appropriate or drop broken rows
        subset.dropna(subset=cols_to_numeric, inplace=True)

        # 6. Calculate Derived Metrics (The "Correct" Metrics)
        subset["Inf_per_Sec"] = 1000.0 / subset["Measured Inference Time (ms)"]
        subset["Inf_per_Joule"] = 1000.0 / subset["Energy per Inference (mJ)"]

        # Use measured accuracy if available, else standard
        acc_ratio = subset["Top-1 Accuracy (measured)"] / 100.0
        
        subset["Correct_Inf_per_Sec"] = subset["Inf_per_Sec"] * acc_ratio
        subset["Correct_Inf_per_Joule"] = subset["Inf_per_Joule"] * acc_ratio

        return subset.reset_index(drop=True)