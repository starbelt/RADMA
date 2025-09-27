"""
Class and methods to parse through the digital and analog output .CSVs from the Logic 2 API. 
Requires a target directory in which to find the CSVs of note, and returns metrics including:
__
average inference time
energy per inference
"""

import numpy as np
import pandas as pd
import pathlib, os
from path_utils import get_repo_root
import matplotlib.pyplot as plt

# helpers and diagnostic plots
def plot_saleae_trace(directory, psu_voltage=5.0, r_shunt=0.1, plot_energy_bars=True):
    parser = SaleaeOutputParsing(directory)
    
    # Zoom to first 0.12 seconds
    mask = parser.t_analog <= 0.12
    t = parser.t_analog[mask]
    v1 = parser.v1[mask]
    v2 = parser.v2[mask]
    
    current = (v1 - v2) / r_shunt  # Amps
    power = current * psu_voltage  # Watts
    
    # Incremental energy per sample
    dt = np.diff(t, prepend=t[0])
    incremental_energy = current * psu_voltage * dt  # Joules per slice
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 8), sharex=True)
    
    # --- Voltage traces ---
    ax1.plot(t, v1, label="V1 (before shunt)")
    ax1.plot(t, v2, label="V2 (after shunt)")
    ax1.set_ylabel("Voltage [V]")
    ax1.set_title("Voltage Traces")
    ax1.grid(True)
    ax1.legend(loc="upper right")
    
    # --- Current trace with incremental energy ---
    ax2.plot(t, current, color="C0", label="Current [A]")
    ax2.bar(t, incremental_energy / incremental_energy.max() * current.max(),
            width=dt, alpha=0.3, color="C1", label="Incremental Energy (scaled)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Current [A]")
    ax2.set_title("Current Trace with Incremental Energy Overlay")
    ax2.grid(True)
    # Clean up legend duplicates
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc="upper right")
    fig.suptitle(directory.name)
    plt.tight_layout()
    plt.savefig("Diagnostic_Trace_.png", dpi=300, bbox_inches="tight")



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
        """Cache or load persistent idle power from CSV"""

        from path_utils import get_repo_root
        idle_csv = get_repo_root() / "results/captures/idle_power/idle.csv"

        # if csv with stored value doesn't exist, find the value and store it for future use
        if not idle_csv.exists() or os.stat(idle_csv).st_size == 0:
            mean_power, _, _, _ = self.avg_power_measurement(subtract_idle=False)
            df = pd.DataFrame([{"Average Idle Power W": mean_power}])
            idle_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(idle_csv, index=False)
            self.idle_power = mean_power
        else:
            idle_df = pd.read_csv(idle_csv)
            self.idle_power = idle_df["Average Idle Power W"].iloc[0]
        
        print(f"Idle Power = {self.idle_power} mW")
        return self.idle_power

    def avg_inference_time(self):
        """Return average inference time in seconds"""
        if self.inf_times is None or len(self.inf_times)==0:
            raise Exception("There are no valid inferences")
        else:
            return np.mean(self.inf_times)

    def avg_power_measurement(self, psu_dc_volts: float = 5.0, r_shunt: float = 1.0, subtract_idle: bool = True):
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
        
        print(f"v_device * I = {v_device * I}, idle = {idle}")

        p_inst = v_device * I - idle

        if self.rising is None or self.falling is None:
            mean_power = np.mean(p_inst)
            return mean_power, None, None, None

        avg_powers = []
        energies = []

        for t_start, t_end in zip(self.rising, self.falling):
            mask = (self.t_analog >= t_start) & (self.t_analog <= t_end)
            if not mask.any():
                continue

            t_window = self.t_analog[mask]
            p_window = p_inst[mask]

            energy = np.trapezoid(p_window, t_window)   # Joules
            duration = t_window[-1] - t_window[0]

            if duration > 0:
                avg_powers.append(energy / duration)
                energies.append(energy)

        mean_power = np.mean(avg_powers) if avg_powers else None
        mean_energy = np.mean(energies) if energies else None

        return mean_power, np.array(avg_powers), mean_energy, np.array(energies)

if __name__ == "__main__":
    REPO_ROOT = get_repo_root()
    
    plot_saleae_trace(REPO_ROOT/"results/captures/obj_det_runs/SSDMobileNetV2TF1", psu_voltage=5.0, r_shunt=0.1, plot_energy_bars=True)

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

    