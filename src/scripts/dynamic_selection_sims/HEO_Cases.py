import sys
from pathlib import Path

try:
    from libs.coral_tpu_characterization.src.scripts.utils.path_utils import get_repo_root
    ROOT_DIR = get_repo_root()
except ImportError:
    ROOT_DIR = Path(".").resolve()

from selection_case_studies import ContinuousSatSim

if __name__ == "__main__":
    model_json = ROOT_DIR / "data/compiled_characterization.json" 
    orbit_path = ROOT_DIR / "data/stk"
    out_dir = ROOT_DIR / "results/case_studies/heo"

    # just 1 orbit for heo since the period is so long
    sim_heo = ContinuousSatSim(orbit_path, model_json, out_dir, sat_prefix='HEO', num_orbits=1)
    heo_cfg = ContinuousSatSim.get_heo_config()
    
    # case 1: baseline heo
    sim_heo.run_case_study("HEO_01_Baseline", config_overrides=heo_cfg, events=None)

    # case 2: simulate massive data collection at perigee
    # you'll want to adjust the start time to match the actual altitude dip in the plots
    perigee_burst = [
        {'start': 2000, 'duration': 1500, 'extra_demand_ips': 100.0},
    ]
    sim_heo.run_case_study("HEO_02_Perigee_Deluge", config_overrides=heo_cfg, events=perigee_burst)

    # case 3: radar or comms active during the perigee pass
    perigee_drain = [
        {'start': 2000, 'duration': 1500, 'power_w': 0.8},
    ]
    sim_heo.run_case_study("HEO_03_Power_Starved_Perigee", config_overrides=heo_cfg, events=perigee_drain)