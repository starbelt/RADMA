"""
Class and methods to parse through the digital and analog output .CSVs from the Logic 2 API. 
Requires a target directory in which to find the CSVs of note, and returns metrics including:
__
average inference time
energy per inference
"""

import numpy as np
import pandas as pd
import pathlib

class SaleaeOutputParsing:
    def __init__(self, data_directory=None):
        """Initialize parser, find and load digital/analog CSVs."""
        if data_directory is None:
            data_directory = pathlib.Path.cwd()
        else:
            data_directory = pathlib.Path(data_directory)

        # find files
        digital_files = list(data_directory.rglob("digital.csv"))
        analog_files = list(data_directory.rglob("analog.csv"))

        if not digital_files:
            raise FileNotFoundError("No digital.csv file found")

        self.digital_file = digital_files[0]
        self.analog_file = analog_files[0] if analog_files else None

        # load
        self.t_digital, self.d, self.t_analog, self.v1, self.v2 = self.load_data()
        self.rising, self.falling = self.find_edges(self.t_digital, self.d)
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

    def find_edges(self, t_digital, d):
        """Find rising and falling edges in digital trace"""
        rising_idx = np.where((d[1:] == 1) & (d[:-1] == 0))[0] + 1
        falling_idx = np.where((d[1:] == 0) & (d[:-1] == 1))[0] + 1

        rising = t_digital[rising_idx]
        falling = t_digital[falling_idx]

        if len(rising) == 0 or len(falling) == 0:
            raise RuntimeError("No edges found in digital trace")

        # Ensure first edge is rising
        if falling[0] < rising[0]:
            falling = falling[1:]

        # Trim to equal length
        n = min(len(rising), len(falling))
        return rising[:n], falling[:n]

    def avg_inference_time(self):
        """Return average inference time in seconds"""
        return np.mean(self.inf_times)

    def avg_power_measurement(self, psu_dc_volts, r_shunt):
        """Compute average power and energy during inference windows"""
        if not self.v1:
            raise Exception("No analog file output")
        v_shunt = self.v1 - self.v2   # voltage across resistor
        v_device = psu_dc_volts - v_shunt
        I = v_shunt / r_shunt
        p_inst = v_device * I  # instantaneous power

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

        if not avg_powers:
            raise RuntimeError("No valid analog windows found")

        mean_power = np.mean(avg_powers)
        mean_energy = np.mean(energies)

        return mean_power, np.array(avg_powers), mean_energy, np.array(energies)


if __name__ == "__main__":
    r_shunt = 1.0  # ohms
    parser = SaleaeOutputParsing()  # defaults to cwd

    mean_pwr, all_pwr, mean_energy, all_energy = parser.avg_power_measurement(5, r_shunt)

    print(f"Average inference time: {parser.avg_inference_time()*1e3:.2f} ms")
    print(f"Average power: {mean_pwr*1e3:.2f} mW")
    print(f"Average energy per inference: {mean_energy*1e3:.2f} mJ")
    print(f"Number of inferences: {len(parser.inf_times)}")
