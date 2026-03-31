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



    # degraded_events = [
    #     {'start': 15000, 'duration': 50000, 'solar_scale': 0.6},
    # ]
    # sim_sso.run_case_study("SSO_03_Degraded_Arrays", config_overrides=sso_cfg, events=degraded_events)

    degraded_cfg = sso_cfg.copy()
    degraded_cfg['solar_generation_mw'] = 1300 
    solar_failure_events = [
        {'start': 10000, 'duration': 50000, 'solar_scale': 0.65} 
    ]
    
    # sim_sso.run_case_study("SSO_02_Panel_Failure", config_overrides=degraded_cfg, events=solar_failure_events)
