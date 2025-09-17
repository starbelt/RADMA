import numpy as np
import pandas as pd

def load_data(digital_csvfile, analog_csvfile):
    """Load digital and analog Saleae CSVs"""
    digital_df = pd.read_csv(digital_csvfile)
    analog_df = pd.read_csv(analog_csvfile)

    # Digital: only rising/falling stored
    t_digital = digital_df["Time [s]"].astype(float).to_numpy()[1:-2]
    d = digital_df["Channel 0"].astype(float).to_numpy()[1:-2]

    # Analog: continuous traces
    t_analog = analog_df["Time [s]"].astype(float).to_numpy()[1:-2]
    v1 = analog_df["Channel 1"].astype(float).to_numpy()[1:-2]  # before shunt
    v2 = analog_df["Channel 2"].astype(float).to_numpy()[1:-2]  # after shunt

    return t_digital, d, t_analog, v1, v2


def find_edges(t_digital, d):
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


def avg_inference_time(rising, falling):
    """Compute average inference time"""
    inf_times = falling - rising
    return inf_times.mean(), inf_times

def avg_power_measurement(rising, falling, t_analog, v1, v2, r_shunt):
    """Compute average power and energy during inference windows"""
    currents = (v1 - v2) / r_shunt  # low-side shunt: Vbefore - Vafter
    p_inst = 5 * currents          # instantaneous power (const 5V DC)

    avg_powers = []
    energies = []

    for t_start, t_end in zip(rising, falling):
        mask = (t_analog >= t_start) & (t_analog <= t_end)
        if not mask.any():
            continue

        t_window = t_analog[mask]
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
    digital_csv = "digital.csv"
    analog_csv = "analog.csv"
    r_shunt = 1.0  # ohms 

    t_digital, d, t_analog, v1, v2 = load_data(digital_csv, analog_csv)
    rising, falling = find_edges(t_digital, d)

    mean_inf, all_inf = avg_inference_time(rising, falling)
    mean_pwr, all_pwr, mean_energy, all_energy = avg_power_measurement(
    rising, falling, t_analog, v1, v2, r_shunt
)
    print(f"Average inference time: {mean_inf*1e3:.2f} ms")
    print(f"Average power: {mean_pwr*1e3:.2f} mW")
    print(f"Average energy per inference: {mean_energy*1e3:.2f} mJ")
        
        
    print(f"Per-inference energies: {all_energy}")
