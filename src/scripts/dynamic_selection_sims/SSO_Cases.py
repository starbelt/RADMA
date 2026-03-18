import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = None
for parent in current_file.parents:
    if parent.name == "CoralGUI":
        project_root = parent
        break

if project_root:
    sys.path.insert(0, str(project_root))
else:
    
    sys.path.insert(0, str(current_file.parents[5]))
try:
    from libs.coral_tpu_characterization.src.scripts.utils.path_utils import get_repo_root
    ROOT_DIR = get_repo_root()
except ImportError:
    ROOT_DIR = Path(".").resolve()

from selection_case_studies import ContinuousSatSim

if __name__ == "__main__":
    model_json = ROOT_DIR / "data/compiled_characterization.json" 
    orbit_path = ROOT_DIR / "data/stk"
    out_dir = ROOT_DIR / "results/case_studies/sso"

    sim_sso = ContinuousSatSim(orbit_path, model_json, out_dir, sat_prefix='SSO', num_orbits=5)
    sso_cfg = ContinuousSatSim.get_sso_config()
    

    sim_sso.run_case_study("SSO_01_Baseline", config_overrides=sso_cfg, events=None)


    burst_events = [
        {'start': 500, 'duration': 1000, 'extra_demand_ips': 200.0},
        {'start': 3000, 'duration': 1000, 'extra_demand_ips': 200.0},
    ]
    sim_sso.run_case_study("SSO_02_Time_Crunch", config_overrides=sso_cfg, events=burst_events)

    drain_events = [
        {'start': 1000, 'duration': 1500, 'power_w': 0.5},
    ]
    sim_sso.run_case_study("SSO_03_Power_Starved", config_overrides=sso_cfg, events=drain_events)

    strict_cfg = sso_cfg.copy()
    strict_cfg['hard_min_infs'] = 70.0 
    sim_sso.run_case_study("SSO_04_Strict_Limits", config_overrides=strict_cfg, events=None)

    events = [
        {'start': 2000, 'duration': 1000, 'power_w': 0.4},                     
        {'start': 2500, 'duration': 200, 'extra_demand_ips': 150.0},           
        {'start': 2600, 'duration': 50, 'power_w': 0.0, 'blocked': True},     
    ]
    sim_sso.run_case_study("SSO_05_Perfect_Storm", config_overrides=strict_cfg, events=events)