import csv, pathlib
import numpy as np
import pandas as pd


def measure_pulses(csv_file):
    """Parse csv for model inf time data and return average high pulse width in ms"""
    df =pd.read_csv(csv_file)
    t = df["Time [s]"].astype(float).to_numpy()[1:-2] #time (seconds)
    d = df["Channel 0"].astype(float).to_numpy() [1:-2]# digital output (0 or 1)


    if len(d) == 0:
        raise Exception("No data found")

    rising_idx = np.where((d[1:] == 1) & (d[:-1] == 0))[0] + 1
    falling_idx = np.where((d[1:] == 0) & (d[:-1] == 1))[0] + 1

    rising = t[rising_idx]
    falling = t[falling_idx]

    # Align edges
    if len(rising) == 0 or len(falling) == 0:
        print("No valid edges found")
        return None
    if falling[0] < rising[0]:
        falling = falling[1:]

    n = min(len(rising), len(falling))
    inf_time = falling[:n] - rising[:n]

    if len(inf_time) == 0:
        print("You did it wrong :)")
        return None

    return inf_time.mean() * 1000  # mean inference time in ms

if __name__ == "__main__":
    Top_Dir =  pathlib.Path("captures/IMG_CLASS_10s_doublearena").expanduser()
    files = sorted(Top_Dir.rglob("*.csv"))
    output_path = "captures/inferences.csv"


    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model", "avg_logic_ms", "category"]) # header
        for file in files:
            model = file.parent.parent.name
            inference_time_ms = measure_pulses(file)
            print(model)
            print(inference_time_ms)