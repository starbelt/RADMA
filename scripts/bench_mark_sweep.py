import os, subprocess, time, serial, csv, pathlib
from datetime import datetime
from saleae import automation

MODELS_DIR = pathlib.Path("~/Coral-TPU-Characterization/models").expanduser()
RESULTS_FILE = "inference_results.csv"

# serial settings
SERIAL_PORT = "/dev/ttyACM0"
BAUDRATE = 115200

# connect to Saleae Logic (Logic 2 must already be running)
manager = automation.Manager.connect(port=10430)

with open(RESULTS_FILE, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["model", "inference_time_ms", "saleae_dir"])  # header

    # find all files ending with "_edgetpu.tflite" (recursively)
    models = sorted(MODELS_DIR.rglob("*_edgetpu.tflite"))

    for model in models:
        print(f"=== Testing {model.name} ===")

        # Patch header file
        with open("inference_config.h", "w") as f:
            f.write(f'#define MODEL_PATH "{model}"\n')

        # Build & flash
        subprocess.run(["make","-C", "out", "-j12"], check=True)
        subprocess.run([
            "python3", "coralmicro/scripts/flashtool.py",
            "--build_dir", "out",
            "--elf_path", "out/coralmicro-app"
        ], check=True)

        # Give board time to boot
        time.sleep(20)

        # Prepare Saleae capture configs
        device_config = automation.LogicDeviceConfiguration(
            enabled_digital_channels=[0],  # CTS GPIO on channel 0
            digital_sample_rate=50_000_000,
            digital_threshold_volts=3.3
        )
        capture_config = automation.CaptureConfiguration(
            capture_mode=automation.TimedCaptureMode(duration_seconds=2.0)
        )

        # Create output directory per model
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = pathlib.Path(f"captures/{model.stem}_{ts}")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Run Saleae capture
        with manager.start_capture(
                device_configuration=device_config,
                capture_configuration=capture_config
        ) as capture:
            capture.wait()
            capture.save_capture(str(out_dir / "capture.sal"))
            capture.export_raw_data_csv(
                directory=str(out_dir),
                digital_channels=[0]
            )

        # Capture serial output
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=10)
        lines = []
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                break
            print(line)
            lines.append(line)
        ser.close()

        # Parse result (sanity check from serial)
        inf_time = None
        for l in lines:
            if "invoke_ms:" in l:
                inf_time = l.split(":")[1].strip().replace("ms", "")
                break

        # Write results
        writer.writerow([model.name, inf_time, str(out_dir)])

manager.close()
