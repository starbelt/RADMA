import pandas as pd
import numpy as np
import pathlib
import json
import sys


current_file = pathlib.Path(__file__).resolve()
project_root = current_file.parents[5] 
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from libs.coral_tpu_characterization.src.scripts.utils.path_utils import get_repo_root
from libs.coral_tpu_characterization.src.scripts.hardware_characterization.plotting.tpunet_plotting import GridStatsPlotting
from libs.coral_tpu_characterization.src.scripts.utils.saleae_parsing import SaleaeOutputParsing


class ModelDataManager:
    """
    Handles the ingestion of Saleae power/latency data and merges it with 
    Excel metadata (Accuracy, Model Names) to produce a unified DataFrame.
    """
    def __init__(self, excel_path, results_dir):
        self.excel_path = pathlib.Path(excel_path).resolve()
        self.results_dir = pathlib.Path(results_dir).resolve()

    def _collect_saleae_results(self, psu_dc_volts=5.0, r_shunt=0.2):
        results = {}
        if not self.results_dir.exists():
            print(f"[WARN] Results directory not found: {self.results_dir}")
            return {}

        for subdir in sorted(self.results_dir.iterdir()):
            if not subdir.is_dir():
                continue
            try:
                parsed = SaleaeOutputParsing(subdir)
            except (FileNotFoundError, ValueError):
                continue

            if parsed.avg_inference_time() is not None:
                avg_time = parsed.avg_inference_time() * 1e3  # s to ms
                mean_pwr, _, mean_energy, _ = parsed.avg_power_measurement(
                    psu_dc_volts, r_shunt
                )
                if mean_pwr is not None:
                    results[subdir.name] = {
                        "inference_time_ms": avg_time,
                        "avg_power_mW": mean_pwr * 1e3,
                        "energy_mJ": mean_energy * 1e3 if mean_energy else 0
                    }
        return results

    def get_compiled_dataframe(self, sheet_name="Img_Class"):
        saleae_data = self._collect_saleae_results()
        
        if not self.excel_path.exists():
            print(f"[WARN] Excel file not found: {self.excel_path}")
            return pd.DataFrame() # Return empty on failure
        
        try:
            df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
        except Exception as e:
            print(f"[WARN] Failed to read Excel: {e}")
            return pd.DataFrame()

        # Filter to measured models
        measured_models = list(saleae_data.keys())
        subset = df[df["Model name"].isin(measured_models)].copy()

        if subset.empty:
            print(f"[WARN] No matching pre-trained models found in Excel matching Saleae captures.")
            return pd.DataFrame()

        # Map Data
        subset["Measured Inference Time (ms)"] = subset["Model name"].map(
            lambda m: saleae_data.get(m, {}).get("inference_time_ms")
        )
        subset["Energy per Inference (mJ)"] = subset["Model name"].map(
            lambda m: saleae_data.get(m, {}).get("energy_mJ")
        )
        subset["Average Power (mW)"] = subset["Model name"].map(
            lambda m: saleae_data.get(m, {}).get("avg_power_mW")
        )
        
        # Determine Measured Accuracy if available
        if "Top-1 Accuracy (measured)" not in subset.columns:
            subset["Top-1 Accuracy (measured)"] = np.nan
            
        return subset


def generate_unified_dataset(
    excel_path,
    pretrained_results_dir,
    custom_json_dir,
    custom_saleae_dir,
    output_path
):
    print(f"--- Starting Data Aggregation ---")
    
    #  Process Pre-trained 
    print(f"[1/3] Processing Pre-trained Models...")
    try:
        mdm = ModelDataManager(excel_path=excel_path, results_dir=pretrained_results_dir)
        df_pretrained = mdm.get_compiled_dataframe(sheet_name="Img_Class")
        
        if not df_pretrained.empty:
            # Logic: Use Measured accuracy if available, else claimed
            df_pretrained["Final_Accuracy"] = df_pretrained["Top-1 Accuracy (measured)"].fillna(
                df_pretrained["Top-1 Accuracy"]
            )
            df_pretrained["Source"] = "PreTrained"
        else:
            print("Warning: Pre-trained dataframe is empty.")
            
    except Exception as e:
        print(f"Error processing pre-trained models: {e}")
        df_pretrained = pd.DataFrame()

    #  Process Custom
    print(f"[2/3] Processing Custom Models...")
    try:
        # Use a temp dir for plot outputs if needed, or just the parent of output_path
        plotter = GridStatsPlotting(custom_json_dir, custom_saleae_dir, pathlib.Path(output_path).parent)
        df_custom = plotter.load_and_aggregate_data()
        
        if df_custom is not None and not df_custom.empty:
            df_custom["Final_Accuracy"] = df_custom["Top-1 Accuracy"]
            df_custom["Source"] = "Custom"
        else:
            print("Warning: No custom model data found.")
            df_custom = pd.DataFrame()
            
    except Exception as e:
        print(f"Error processing custom models: {e}")
        import traceback
        traceback.print_exc()
        df_custom = pd.DataFrame()

    # Unification 
    print(f"[3/3] Unifying Datasets...")

    # construct new DataFrames explicitly to avoid rename collisions (e.g. duplicate Top-1 Accuracy columns)
    schema_map = {
        "Model name": "Model name",
        "Measured Inference Time (ms)": "Measured Inference Time (ms)",
        "Energy per Inference (mJ)": "Energy per Inference (mJ)",
        "Final_Accuracy": "Top-1 Accuracy", 
        "Source": "Source"
    }

    def extract_clean_subset(df_in, schema):
        if df_in.empty: return pd.DataFrame()
        
        data_dict = {}
        for src_col, target_col in schema.items():
            if src_col in df_in.columns:
                data_dict[target_col] = df_in[src_col].values
            else:
                # Fill missing columns with NaN if they don't exist
                data_dict[target_col] = [np.nan] * len(df_in)
        return pd.DataFrame(data_dict)

    df_p_clean = extract_clean_subset(df_pretrained, schema_map)
    df_c_clean = extract_clean_subset(df_custom, schema_map)

    unified_df = pd.concat([df_p_clean, df_c_clean], ignore_index=True)

    numeric_cols = [
        "Measured Inference Time (ms)", 
        "Energy per Inference (mJ)", 
        "Top-1 Accuracy"
    ]
    
    # Force numeric and drop bad rows
    for col in numeric_cols:
        if col in unified_df.columns:
            if isinstance(unified_df[col], pd.DataFrame):
                print(f"[ERROR] Duplicate column detected for {col}. Collapsing...")
                # If duplicate, take the first one
                unified_df = unified_df.loc[:, ~unified_df.columns.duplicated()]
            
            unified_df[col] = pd.to_numeric(unified_df[col], errors='coerce')

    initial_len = len(unified_df)
    unified_df.dropna(subset=numeric_cols, inplace=True)
    
    # Remove duplicates (prefer custom over pretrained if names clash, or just last)
    unified_df.drop_duplicates(subset=["Model name"], keep='last', inplace=True)
    
    final_len = len(unified_df)
    print(f"Dropped {initial_len - final_len} rows due to missing data.")

    # Exports
    out_path = pathlib.Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    unified_df.to_json(out_path, orient='records', indent=4)
    
    print(f"\nSUCCESS: Unified characterization data saved to:")
    print(f"-> {out_path}")
    print(f"Total Models: {len(unified_df)}")
    if not unified_df.empty:
        print(unified_df[["Model name", "Source", "Top-1 Accuracy"]].head())


if __name__ == "__main__":
    REPO_ROOT = get_repo_root()


    EXCEL_METADATA_PATH = "/home/jackr/CoralGUI/libs/coral_tpu_characterization/data/Model_Stats.xlsx"
    
    # Pre-trained Inputs
    PRETRAINED_SALEAE_DIR = "/home/jackr/CoralGUI/libs/coral_tpu_characterization/results/captures/IMG_CLASS02"
    
    # Custom Inputs
    CUSTOM_JSON_DIR = REPO_ROOT / "data" / "tpunet_acc"
    CUSTOM_SALEAE_DIR = "/home/jackr/CoralGUI/results/captures"
    
    # Output
    OUTPUT_JSON_PATH = REPO_ROOT / "data" / "compiled_characterization.json"

    generate_unified_dataset(
        excel_path=EXCEL_METADATA_PATH,
        pretrained_results_dir=PRETRAINED_SALEAE_DIR,
        custom_json_dir=CUSTOM_JSON_DIR,
        custom_saleae_dir=CUSTOM_SALEAE_DIR,
        output_path=OUTPUT_JSON_PATH
    )