import os, subprocess, time, serial, csv, pathlib

import numpy as np
import pandas as pd
from datetime import datetime
from saleae import automation

MODELS_DIR = pathlib.Path("~/Coral-TPU-Characterization/models/Image_Classification/EfficientNet/S/").expanduser()
RESULTS_FILE = "inference_results.csv"

# serial settings
SERIAL_PORT = "/dev/ttyACM0"
BAUDRATE = 115200

def wait_for_serial(port=SERIAL_PORT, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        if pathlib.Path(port).exists():
            print("Serial Connection Established to Coral\n")
            return True
        time.sleep(0.5)
    raise TimeoutError(f"Serial port {port} not found within {timeout}s")

def measure_pulses(csv_file):
    """Parse csv for model inf time data and return average high pulse width in ms"""
    df =pd.read_csv(csv_file)
    t = df["Time [s]"].astype(float).to_numpy() #time (seconds)
    d = df["Channel 0"].astype(float).to_numpy() # digital output (0 or 1)


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

with open(RESULTS_FILE, "w", newline="") as csvfile:
    with automation.Manager.connect(port=10430) as manager:
        writer = csv.writer(csvfile)
        writer.writerow(["model", "serial_invoke_ms", "avg_logic_ms", "saleae_dir"]) # header
        # find all files ending with "_edgetpu.tflite" (recursively)
        models = sorted(MODELS_DIR.rglob("*_edgetpu.tflite"))

        for model in models:
            print(f"Testing {model.name}\n")
            rel_path = model.relative_to(MODELS_DIR)
            device_path = f"/models/Image_Classification/EfficientNet/S/{rel_path.as_posix()}"  # for header
            host_path = str(model.resolve())  # absolute host path for cmake

            # Patch header file
            with open("libs/inference_model_config.h", "w") as f:
                f.write(f'#define MODEL_PATH "{device_path}"\n')
            #
            # # Build & flash
            # subprocess.run([
            #     "cmake",
            #     "-B", "out",
            #     "-S", ".",
            #     f"-DMODEL_PATH={host_path}"
            # ], check=True)
            #
            # subprocess.run(["make", "-C",
            #     "out", "-j12"], check=True)
            #
            # subprocess.run([
            #     "python3", "coralmicro/scripts/flashtool.py",
            #     "--build_dir", "out",
            #     "--elf_path", "out/coralmicro-app" ,"--nodata"
            # ], check=True)


            # Give board time to boot
            wait_for_serial(SERIAL_PORT, timeout=30)
            time.sleep(2)  # extra settle time

            # Prepare Saleae capture configs
            print("Preparing Logic Analyzer\n")
            device_configuration = automation.LogicDeviceConfiguration(
                enabled_digital_channels=[0],  # CTS GPIO on channel 0
                digital_sample_rate=100_000_000,
                digital_threshold_volts=3.3
            )
            capture_configuration = automation.CaptureConfiguration(
                capture_mode=automation.TimedCaptureMode(duration_seconds=2.0)
            )

            # Create output directory per model (or perhaps not)
            print("Preparing Output Directories\n")
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_dir = pathlib.Path(f"captures/{model.stem}_{ts}")
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"Directory Created: {out_dir}")

            # Folder for Saleae (grr)
            saleae_dir = out_dir / "saleae_raw"
            saleae_dir.mkdir(parents=True, exist_ok=True)
            assert saleae_dir.exists(), f"Saleae directory {saleae_dir} was not created successfully."


            # Run Saleae capture

            print("Running Capture\n")
            with manager.start_capture(
                    device_id='C7495E5575CEA129',
                    device_configuration=device_configuration,
                    capture_configuration=capture_configuration) as capture:
                # wait for timed capture (2 seconds)
                capture.wait()
                print("Creating digital.csv")
                capture.export_raw_data_csv(
                    directory=str(saleae_dir.resolve()),
                    digital_channels=[0]
                )

            # Capture serial output
            # print("\nCapturing Serial\n")
            # ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=10)
            # lines = []
            # while True:
            #     line = ser.readline().decode(errors="ignore").strip()
            #     if not line:
            #         break
            #     print(line)
            #     lines.append(line)
            # ser.close()
            #
            # # Parse result (sanity check from serial)
            # inf_time = None
            # for l in lines:
            #     if "invoke_ms:" in l:
            #         inf_time = l.split(":")[1]
            #         break

            # Measure pulses
            print("Measuring Inference Time\n")
            raw_csv = saleae_dir / "digital.csv"  # Saleae names file "digital.csv"
            avg_logic = None
            if raw_csv.exists():
                avg_logic = measure_pulses(raw_csv)

            # Write results
            writer.writerow([model.name, avg_logic, str(out_dir)])
            print(f"\nMeasurements written to {out_dir}\n")
            print("\n------Parsing to new Model------\n")

