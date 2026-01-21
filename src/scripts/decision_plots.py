import pathlib
import ipympl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path

from path_utils import get_repo_root
from model_stats_plotting import collect_results

def power_inference_dataframe(results_dir: pathlib.Path,
                model_category=None, run_names=None, sheet = "data/Model_Stats.xlsx"):
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
    df = pd.read_excel(path.abspath(sheet), sheet_name)

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

   
    subset["Inf_per_Sec"] = 1000.0 / subset["Measured Inference Time (ms)"]
    subset["Inf_per_Joule"] = 1000.0 / subset["Energy per Inference (mJ)"]

    # Calculate "Correct" metrics
    acc_ratio = subset["Top-1 Accuracy (measured)"] / 100.0
    subset["Correct_Inf_per_Sec"] = subset["Inf_per_Sec"] * acc_ratio
    subset["Correct_Inf_per_Joule"] = subset["Inf_per_Joule"] * acc_ratio

    return subset

def budget_correct_loop(df: pd.DataFrame, buffers: list, frame_times: list, results_dir: pathlib.Path, filename="all_grids.xlsx", mode = "all"): 
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


    if len(comparison_data) > 0 and (mode == "frontier" or mode == 'all'):
        print("Generating Decision Frontier (Time vs. Energy Strategy)...")
        
        # 1. Identify "Sprinter" (Efficient) vs "Powerhouse" (High Rate)
        # We assume one is more efficient (higher slope) and one has higher max rate.
        # If one model is better at BOTH, there is no crossover (it always wins).
        
        # Sort by Max Rate (Correct Inferences / Sec)
        sorted_by_rate = sorted(comparison_data, key=lambda x: x['acc']/x['latency_s'])
        model_low_rate = sorted_by_rate[0]  # MobileNet 0.50 (saturates early)
        model_high_rate = sorted_by_rate[1] # MobileNet 0.75 (higher ceiling)
        
        rate_low = model_low_rate['acc'] / model_low_rate['latency_s']
        slope_high = model_high_rate['acc'] / model_high_rate['energy_j']
        slope_low = model_low_rate['acc'] / model_low_rate['energy_j']

        # Only proceed if a trade-off actually exists
        if slope_low > slope_high:
            
            plt.figure(figsize=(10, 6))
            
            # Create Time Range (0 to 100s, or slightly more than max frame time)
            max_t = 100
            time_range = np.linspace(0, max_t, 200)
            
            # Calculate the Breakeven Energy Line: E = T * (Rate_Low / Slope_High)
            # This is the energy required for the High-Power model to catch up 
            # to the Low-Power model's saturation point.
            k = rate_low / slope_high
            breakeven_energy = time_range * k
            
            # Plot the dividing line
            plt.plot(time_range, breakeven_energy, 
                    color='black', linewidth=2, linestyle='--', 
                    label=f"Breakeven Frontier (k={k:.2f} J/s)")
            
            # Shade the Regions
            # Region Below: Battery is too small -> Use Efficient Model
            plt.fill_between(time_range, 0, breakeven_energy, 
                            color=model_low_rate['color'], alpha=0.15)
            plt.text(max_t * 0.75, max(breakeven_energy) * 0.25, 
                    f"ZONE: {model_low_rate['name']}\n(Battery Limited)", 
                    ha='center', va='center', fontweight='bold', color=model_low_rate['color'],fontsize = 20)
            
            # Region Above: Battery is sufficient -> Use High-Accuracy Model
            # We set an arbitrary top y-limit for shading (e.g. 1.5x max breakeven)
            y_top = max(breakeven_energy) * 1.5
            plt.fill_between(time_range, breakeven_energy, y_top, 
                            color=model_high_rate['color'], alpha=0.15)
            plt.text(max_t * 0.25, max(breakeven_energy) * 1.25, 
                    f"ZONE: {model_high_rate['name']}\n(Time Limited)", 
                    ha='center', va='center', fontweight='bold', color=model_high_rate['color'],fontsize = 20)

            plt.title("Mission Strategy: Which Model Should I Run?",fontsize = 20)
            plt.xlabel("Pass Duration (Seconds)",fontsize = 20)
            plt.ylabel("Required Energy Buffer (Joules)",fontsize = 20)
            plt.xlim(0, max_t)
            plt.ylim(0, y_top)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.grid(True, linestyle=':', alpha=0.5)
            # plt.legend(loc="upper left")
            
            plt.tight_layout()
            plt.savefig(plot_dir / "MobileNet_Decision_Frontier.png")
            plt.close()

    if len(comparison_data) > 0 and (mode == "crossover" or mode == 'all'):
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

    # 3D plot

    if len(comparison_data) > 0 and (mode == '3D' or mode == 'all'):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d import proj3d

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        for item in comparison_data:
            # 1. Calculate fundamental rates
            slope_correct_per_joule = item['acc'] / item['energy_j']
            ceiling_correct_per_sec = item['acc'] / item['latency_s']
            
            # 2. Create grid for surface
            buffer_range = np.linspace(min(buffers), max(buffers), 50)
            time_range = np.linspace(0.1, max(frame_times), 50)
            Buffer_grid, Time_grid = np.meshgrid(buffer_range, time_range)
            
            # Calculate Correct Inferences Surface
            Correct_energy_bound = Buffer_grid * slope_correct_per_joule
            Correct_time_bound = Time_grid * ceiling_correct_per_sec
            Correct_surface = np.minimum(Correct_energy_bound, Correct_time_bound)
            
            # Plot Surface
            ax.plot_surface(Buffer_grid, Time_grid, Correct_surface, 
                            alpha=0.5, color=item['color'], label=item['name'])

        ax.set_title("3D Surface: Correct Inferences vs. Battery & Time", fontsize=16)
        ax.set_xlabel("Energy Buffer (Joules)", fontsize=14)
        ax.set_ylabel("Pass Duration (Seconds)", fontsize=14)
        ax.set_zlabel("Total Correct Inferences", fontsize=14)
        plt.tight_layout()
        # Rotating the view
        ax.view_init(elev=30, azim=120)

        plt.savefig(plot_dir / "MobileNet_3D_Surface.png")
        plt.close()


if __name__ == "__main__":

    REPO_ROOT = get_repo_root()

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
    subset = power_inference_dataframe(
        results_dir= (REPO_ROOT / "results/captures/IMG_CLASS02"),
        model_category="Img_Class",
        run_names=run_names
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

    budget_correct_loop(subset, buffers,frame_times ,REPO_ROOT/ "results")