import sys, subprocess, time, csv, pathlib

import numpy as np
import pandas as pd
from saleae_parsing import SaleaeOutputParsing
from datetime import datetime
from saleae import automation

def wait_for_serial(port: str, timeout=30):
    """Waits until the serial port is open, or raises a timeout"""
    if not port:
        raise ValueError("Serial port is not provided or is None.")

    start = time.time()
    while time.time() - start < timeout:
        if pathlib.Path(port).exists():
            print("Serial Connection Established to Coral\n")
            return True
        time.sleep(0.5)
    raise TimeoutError(f"Serial port {port} not found within {timeout}s")


def test_model(model: pathlib.PosixPath, model_dir: pathlib.PosixPath, capture_dir : str, serial_port : str ,dc_volts_in : float, r_shunt :float):
    """
    Builds, flashes, and tests one model compiled for the edge TPU
    model: posix path of compiled model (should end in "_edgetpu.tflite")
    model_dir: the top level directory through which you were searching for models
    serial port: the name of the port (on the host PC) through which you are flashing
    """

    print(f"Testing {model.name}\n")

    # Resolve relative paths from model to upper directory
    rel_path = model.relative_to(pathlib.Path.cwd()) 
    device_path = rel_path # for header -> kModelPath
    print(f"Running {device_path}\n")
    host_path = str(model.resolve())  # absolute host path for cmake
    
    # Patch header file
    with open("libs/inference_model_config.h", "w") as f:
        f.write(f'#define MODEL_PATH "{device_path}"\n')
    category = model.relative_to(model_dir).parts[0]

    # # Build & flash (cmake, make, then flash)
    subprocess.run([
        "cmake",
        "-B", "out",
        "-S", ".",
        f"-DMODEL_PATH={host_path}"
    ], check=True)

    subprocess.run(["make", "-C",
        "out", "-j12"], check=True)

    subprocess.run([
        "python3", "coralmicro/scripts/flashtool.py",
        "--build_dir", "out",
        "--elf_path", "out/coralmicro-app" #,"--nodata"
    ], check=True)

    # Give board time to boot
    wait_for_serial(serial_port, timeout=30)
    # time.sleep(30)  # extra settle time for TPU context

    # Prepare Saleae capture configs
    print("Preparing Logic Analyzer\n")
    device_configuration = automation.LogicDeviceConfiguration(
        enabled_digital_channels=[0],  # CTS GPIO on channel 0
        enabled_analog_channels=[1,2],
        digital_sample_rate=500_000_000,
        digital_threshold_volts=1.8,
        analog_sample_rate=31250
    )
    capture_configuration = automation.CaptureConfiguration(
        capture_mode=automation.TimedCaptureMode(duration_seconds=30.0)
    )

    # Create output directory per model (or perhaps not)
    # print("Preparing Output Directories\n")100_000_000
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if capture_dir:
        out_dir = pathlib.Path(f"captures/{capture_dir}/{model.stem}_{ts}")
    else:
        out_dir = pathlib.Path(f"captures/{model.stem}_{ts}")
    # # Folder for Saleae
    saleae_dir = out_dir / "saleae_raw"

    # Run Saleae capture
    with automation.Manager.connect(port=10430) as manager:
        print("Running Capture\n")
        with manager.start_capture(
                device_id='C7495E5575CEA129',
                device_configuration=device_configuration,
                capture_configuration=capture_configuration) as capture:
            # wait for timed capture (2 seconds)
            capture.wait()
            print("Creating digital.csv and analog.csv")
            capture.export_raw_data_csv(
                directory=str(saleae_dir.resolve()),
                digital_channels=[0],
                analog_channels=[1,2]
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
        parser = SaleaeOutputParsing(saleae_dir)
        mean_inf_ms = parser.avg_inference_time()*1e3
        mean_pwr, all_pwr, mean_energy, all_energy = parser.avg_power_measurement(dc_volts_in, r_shunt)
        
    return mean_inf_ms, mean_energy, category, out_dir


if __name__ == "__main__":
    
    if len(sys.argv)<=1:
        MODELS_DIR = pathlib.Path("~/Coral-TPU-Characterization/models/Image_Classification").expanduser()
        CAPTURE_DIR = None
        RESULTS_FILE = "inference_results.csv"
    elif len(sys.argv)>1:
        MODELS_DIR = pathlib.Path("~/Coral-TPU-Characterization/"+sys.argv[1]).expanduser()
        RESULTS_FILE = "inference_results.csv"
        CAPTURE_DIR = None
    elif len(sys.argv)>=2:
        CAPTURE_DIR = sys.argv[2]
    # serial settings
    SERIAL_PORT = "/dev/ttyACM0"
    BAUDRATE = 115200
    
    with open(RESULTS_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model", "avg inference time ms", "average energy per inference" ,"category"]) # header
        # find all files ending with "_edgetpu.tflite" (recursively)
        models = sorted(MODELS_DIR.rglob("*_edgetpu.tflite"))
    
        for model in models:
            mean_inf_ms, mean_energy, category, out_dir = test_model(model, MODELS_DIR, CAPTURE_DIR, SERIAL_PORT,dc_volts_in=5,r_shunt=0.1)
            # Write results
            writer.writerow([model.name, mean_inf_ms, mean_energy, category])
            print(f"\nMeasurements written to {out_dir}\n")