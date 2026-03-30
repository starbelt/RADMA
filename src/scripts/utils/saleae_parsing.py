"""
Class and methods to parse through the digital and analog output .CSVs from the Logic 2 API. 
Requires a target directory in which to find the CSVs of note, and returns metrics including:
__
average inference time
energy per inference
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import pathlib, os
from libs.coral_tpu_characterization.src.scripts.utils.path_utils import get_repo_root
# from path_utils import get_repo_root
import matplotlib.pyplot as plt

# helpers and diagnostic plots
def plot_methodology_trace(directory, psu_voltage=5.0, r_shunt=0.2, output_filename="methodology_trace.pdf"):
    """
    Creates a narrow, paper-ready vertical stack showing the digital trigger 
    and the resulting analog power trace. Isolates ~3 inferences and trims the first.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Enforce global serif font layout for native LaTeX integration
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix"
    })

    parser = SaleaeOutputParsing(directory)
    
    if parser.rising is None or len(parser.rising) < 4:
        print("[WARN] Not enough inferences found in trace to plot 3 clean ones.")
        return

    # Skip the 1st inference (index 0), grab the next 3 (indices 1, 2, 3)
    pad_s = 0.01  # 10ms padding on either side
    start_time = parser.rising[1] - pad_s
    end_time = parser.falling[3] + pad_s

    # --- Analog Data Extraction & Power Calculation ---
    ana_mask = (parser.t_analog >= start_time) & (parser.t_analog <= end_time)
    t_ana = parser.t_analog[ana_mask]
    v1 = parser.v1[ana_mask]
    v2 = parser.v2[ana_mask]

    current = (v1 - v2) / r_shunt       # Amps
    power_mw = current * psu_voltage * 1000  # Milliwatts

    # Normalize time to start at 0 ms for the plot
    t_ana_ms = (t_ana - start_time) * 1000

    # --- Digital Data Extraction ---
    dig_mask = (parser.t_digital >= start_time) & (parser.t_digital <= end_time)
    t_dig = parser.t_digital[dig_mask]
    d = parser.d[dig_mask]
    t_dig_ms = (t_dig - start_time) * 1000

    # --- Plotting ---
    # Very narrow width (3.0 inches) to fit nicely next to the hardware photo
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(3.0, 4.0), sharex=True, gridspec_kw={'hspace': 0.15})

    # Top Pane: Digital Trace
    ax1.step(t_dig_ms, d, where="post", color='#333333', linewidth=1.5)
    ax1.set_ylabel("Logic State", fontsize=11)
    
    # Clean up digital y-axis to just show High/Low
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Idle', 'Active'], fontsize=10)
    
    # Bottom Pane: Power Trace
    # Using a nice bold color (like Hokie Maroon) to contrast with the photo
    ax2.plot(t_ana_ms, power_mw, color='#861F41', linewidth=1.2)
    ax2.set_ylabel("Power (mW)", fontsize=11)
    ax2.set_xlabel("Time (ms)", fontsize=11)
    ax2.tick_params(axis="both", labelsize=10)
    
    # Optional: Lightly shade the active power regions to visually link the two panes
    for r, f in zip(parser.rising[1:4], parser.falling[1:4]):
        r_ms = (r - start_time) * 1000
        f_ms = (f - start_time) * 1000
        ax2.axvspan(r_ms, f_ms, color='gray', alpha=0.15, lw=0)

    # Paper-ready styling
    for ax in [ax1, ax2]:
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    # Save as PDF
    pdf_filename = Path(output_filename).with_suffix('.pdf')
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"[PLOT] Saved methodology trace to {pdf_filename}")

def plot_saleae_trace(directory, psu_voltage=5.0, r_shunt=0.2, plot_energy_bars=True):
    parser = SaleaeOutputParsing(directory)
    zoom_s = 0.2

    # ---- Analog current + power
    mask = parser.t_analog <= zoom_s
    t = parser.t_analog[mask]
    v1 = parser.v1[mask]
    v2 = parser.v2[mask]

    current = (v1 - v2) / r_shunt   # A
    power = current * psu_voltage   # W

    # Energy increments + cumulative
    dt = np.diff(t, prepend=t[0])
    incr_energy = power * dt   # J
    cum_energy = np.cumsum(incr_energy)   # J

    # ---- Digital
    d_mask = parser.t_digital <= zoom_s
    t_d = parser.t_digital[d_mask]
    d = parser.d[d_mask]

    # Scale digital to overlay on current trace
    digital_scaled = d * (current.max() * 0.5)

    # ---- Plot
    fig, ax = plt.subplots(figsize=(20, 8))

    # Current trace
    ax.plot(t * 1e3, current, color="C0", label="Current [A]")

    # Incremental energy shading (optional visual)
    if plot_energy_bars:
        ax.bar(t * 1e3,
               incr_energy / incr_energy.max() * current.max(),
               width=dt * 1e3, alpha=0.3, color="C1", label="Energy")

    # Digital overlay
    ax.step(t_d * 1e3, digital_scaled, where="post",
            color="C2", linewidth=2, label="Digital signal")

    ax.set_xlabel("Time [ms]", fontsize=20)
    ax.set_ylabel("Current [A]", fontsize=20)
    ax.tick_params(axis="both", labelsize=18)
    ax.set_title(f"Saleae Trace: {directory.name}", fontsize=24)
    ax.grid(True)

    # ---- Secondary axis for cumulative energy
    ax2 = ax.twinx()
    ax2.plot(t * 1e3, cum_energy * 1e3, color="C3", linewidth=2, label="Cumulative Energy")
    ax2.set_ylabel("Energy [mJ]", fontsize=20)
    ax2.tick_params(axis="y", labelsize=18)

    # Combined legend
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels + labels2, handles + handles2))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=22)

    plt.tight_layout()
    plt.savefig("Diagnostic_Trace_with_EnergyAxis.png", dpi=300, bbox_inches="tight")
    plt.show()
# def plot_saleae_trace(directory, psu_voltage=5.0, r_shunt=0.1, plot_energy_bars=True):
#     parser = SaleaeOutputParsing(directory)
    
#     # Zoom to first 0.12 seconds
#     mask = parser.t_analog <= 0.12
#     t = parser.t_analog[mask]
#     v1 = parser.v1[mask]
#     v2 = parser.v2[mask]
    
#     current = (v1 - v2) / r_shunt  # Amps
#     power = current * psu_voltage  # Watts
    
#     # Incremental energy per sample
#     dt = np.diff(t, prepend=t[0])
#     incremental_energy = current * psu_voltage * dt  # Joules per slice
    
#     fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 8), sharex=True)
    
#     # --- Voltage traces ---
#     ax1.plot(t, v1, label="V1 (before shunt)")
#     ax1.plot(t, v2, label="V2 (after shunt)")
#     ax1.set_ylabel("Voltage [V]",fontsize=18)
#     ax1.set_title("Voltage Traces",fontsize=20)
#     ax1.grid(True)
#     ax1.legend(loc="upper right")
    
#     # --- Current trace with incremental energy ---
#     ax2.plot(t, current, color="C0", label="Current [A]")
#     ax2.bar(t, incremental_energy / incremental_energy.max() * current.max(),
#             width=dt, alpha=0.3, color="C1", label="Incremental Energy (scaled)")
#     ax2.set_xlabel("Time [s]",fontsize=18)
#     ax2.set_ylabel("Current [A]",fontsize=18)
#     ax2.set_title("Current Trace with Incremental Energy Overlay",fontsize=20)
#     ax2.grid(True)
#     # Clean up legend duplicates
#     handles, labels = ax2.get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     ax2.legend(by_label.values(), by_label.keys(), loc="upper right")
#     fig.suptitle(directory.name,fontsize=24)
#     plt.tight_layout()
#     plt.savefig("Diagnostic_Trace_.png", dpi=300, bbox_inches="tight")

class SaleaeOutputParsing:
    def __init__(self, data_directory=None):
        """Initialize parser, find and load digital/analog CSVs."""
        if data_directory is None:
            data_directory = pathlib.Path.cwd()
        else:
            data_directory = pathlib.Path(data_directory).expanduser()

        # find files
        digital_files = list(data_directory.rglob("digital.csv"))
        analog_files = list(data_directory.rglob("analog.csv"))

        if not digital_files:
            raise FileNotFoundError("No digital.csv file found")
        self.idle_power = None
        self.digital_file = digital_files[0]
        self.analog_file = analog_files[0] if analog_files else None

        # load
        self.t_digital, self.d, self.t_analog, self.v1, self.v2 = self.load_data()
        # find valid GPIO high times
        self.rising, self.falling = self.find_edges(self.t_digital, self.d)

        if self.rising is None or self.falling is None:
            self.inf_times = None
        else:
            self.inf_times = self.falling - self.rising

    def load_data(self):
        """Load digital and analog Saleae CSVs"""
        
        # Digital: only rising/falling stored
        digital_df = pd.read_csv(self.digital_file)
        t_digital = digital_df["Time [s]"].astype(float).to_numpy()[1:-2]
        d = digital_df["Channel 0"].astype(float).to_numpy()[1:-2]

        # Analog: continuous traces
        if not self.analog_file:
            return t_digital, d, None, None, None
        else:
            analog_df = pd.read_csv(self.analog_file)
            t_analog = analog_df["Time [s]"].astype(float).to_numpy()[1:-2]
            v1 = analog_df["Channel 1"].astype(float).to_numpy()[1:-2]  # before shunt
            v2 = analog_df["Channel 2"].astype(float).to_numpy()[1:-2]  # after shunt
            
        return t_digital, d, t_analog, v1, v2

    def find_edges(self, t : np.ndarray, d : np.ndarray, min_pulse_width=1e-3):
        # find raw transitions
        rising_idx = np.where((d[1:] == 1) & (d[:-1] == 0))[0] + 1
        falling_idx = np.where((d[1:] == 0) & (d[:-1] == 1))[0] + 1
        rising = t[rising_idx]
        falling = t[falling_idx]

        # pair them, then filter short pulses
        if len(rising) == 0 or len(falling) == 0:
            return None, None

        # ensure rising starts first
        if falling[0] < rising[0]:
            falling = falling[1:]
        n = min(len(rising), len(falling))
        rising = rising[:n]; falling = falling[:n]

        long_mask = (falling - rising) >= min_pulse_width
        return rising[long_mask], falling[long_mask]
    
    def find_idle_power(self):
        from libs.coral_tpu_characterization.src.scripts.utils.path_utils import get_repo_root
        import os

        idle_csv = get_repo_root() / "results/captures/idle_power/idle.csv"

        # If the idle CSV does not exist, compute and save it (in Watts)
        if not idle_csv.exists() or os.stat(idle_csv).st_size == 0:
            # compute mean_power in Watts (do NOT ask avg_power_measurement to subtract idle)
            mean_power_w, _, _, _ = self.avg_power_measurement(subtract_idle=False, r_shunt=0.1)
            if mean_power_w is None:
                raise RuntimeError("Could not compute idle power from measurement.")
            # store in Watts explicitly
            df = pd.DataFrame([{"Average Idle Power (W)": float(mean_power_w)}])
            idle_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(idle_csv, index=False)
            self.idle_power = float(mean_power_w)
        else:
            idle_df = pd.read_csv(idle_csv)
            # Find a sensible column: prefer names containing 'idle'
            col = None
            for c in idle_df.columns:
                if "idle" in c.lower():
                    col = c
                    break
            # fallback: take first numeric column
            if col is None:
                # choose first column that contains a numeric-looking value
                for c in idle_df.columns:
                    try:
                        float(idle_df[c].iloc[0])
                        col = c
                        break
                    except Exception:
                        continue
            if col is None:
                raise RuntimeError(f"Couldn't find idle power value in CSV: {idle_csv}")

            raw_val = float(idle_df[col].iloc[0])
            col_lower = col.lower()

            # If the column name says mW -> convert
            if "mw" in col_lower:
                idle_w = raw_val / 1000.0
            elif "w" in col_lower:
                idle_w = raw_val
            else:
                # No unit in header: heuristic check. Typical idle for these devices is << 50 W.
                # If value is extremely large (>50), assume it was recorded in mW and convert.
                if raw_val > 50.0:
                    # very likely this was stored in mW accidentally
                    idle_w = raw_val / 1000.0
                else:
                    idle_w = raw_val

            self.idle_power = float(idle_w)

        # Print both for human clarity
        print(f"[INFO] Idle power loaded: {self.idle_power:.6f} W ({self.idle_power*1e3:.3f} mW)")
        return self.idle_power

    def avg_inference_time(self):
        """Return average inference time in seconds"""
        if self.inf_times is None or len(self.inf_times)==0:
            raise Exception("There are no valid inferences")
        else:
            return np.mean(self.inf_times)
        
    def avg_power_measurement(self, psu_dc_volts: float = 5.0, r_shunt: float = 0.2, subtract_idle: bool = False):
            """Compute average power and energy during inference windows"""

            if self.v1 is None or self.v2 is None or self.t_analog is None:
                raise Exception("No analog file output")
            if len(self.v1) == 0 or len(self.v2) == 0 or len(self.t_analog) == 0:
                raise Exception("Analog arrays empty")

            v_shunt = self.v1 - self.v2
            v_device = psu_dc_volts - v_shunt
            I = v_shunt / r_shunt

            # If subtract_idle=True, try to subtract cached idle
            idle = 0.0
            if subtract_idle:
                if self.idle_power is None:
                    # Load from CSV, but don't recurse into avg_power_measurement()
                    self.find_idle_power()
                idle = self.idle_power if self.idle_power is not None else 0.0

            p_inst = v_device * I - 0*idle

            if self.rising is None or self.falling is None:
                mean_power = np.mean(p_inst)
                return mean_power, None, None, None

            powers = []
            energies = []
            durations = []

            start_indices = np.searchsorted(self.t_analog, self.rising)
            end_indices = np.searchsorted(self.t_analog, self.falling)

            for s_idx, e_idx in zip(start_indices, end_indices):
                # Safety check: ensure indices are within bounds and valid
                if s_idx >= len(self.t_analog) or s_idx >= e_idx:
                    continue

                # Direct slicing instead of masking
                t_window = self.t_analog[s_idx:e_idx]
                p_window = p_inst[s_idx:e_idx]
                
                if len(t_window) < 2: 
                    continue

                energy = np.trapezoid(p_window, t_window)   # Joules
                duration = t_window[-1] - t_window[0]

                if duration > 0:
                    powers.append(energy / duration)
                    durations.append(duration)
                    energies.append(energy)

            mean_power = np.sum(energies)/np.sum(durations) if durations else None
            mean_energy = np.mean(energies) if energies else None

            return mean_power, np.array(powers), mean_energy, np.array(energies)

if __name__ == "__main__":

    REPO_ROOT = get_repo_root()
    target_dir = REPO_ROOT / "results/captures/IMG_CLASS02/MobileNet V1 (1.0)"
    
    # Generate the methodology plot
    plot_methodology_trace(
        directory=target_dir, 
        psu_voltage=5.0, 
        r_shunt=0.2, 
        output_filename="methodology_trace_figure.pdf"
    )

    # plot_saleae_trace(REPO_ROOT/"results/captures/IMG_CLASS02/EfficientNet-EdgeTpu (S)", psu_voltage=5.0, r_shunt=0.2, plot_energy_bars=True)

    # r_shunt = 0.1  # ohms
    # parser = SaleaeOutputParsing()  # defaults to cwd

    # mean_pwr, all_pwr, mean_energy, all_energy = parser.avg_power_measurement(5, r_shunt)

    # inf_time = parser.avg_inference_time()
    # if inf_time is not None:
    #     print(f"Average inference time: {inf_time*1e3:.2f} ms")
    # else:
    #     print("Average inference time: N/A")

    # if mean_pwr is not None:
    #     print(f"Average power: {mean_pwr*1e3:.2f} mW")
    # else:
    #     print("Average power: N/A")

    # if mean_energy is not None:
    #     print(f"Average energy per inference: {mean_energy*1e3:.2f} mJ")
    # else:
    #     print("Average energy per inference: N/A")

    # if parser.inf_times is not None:
    #     print(f"Number of inferences: {len(parser.inf_times)}")
    # else:
    #     print("Number of inferences: 0")

    