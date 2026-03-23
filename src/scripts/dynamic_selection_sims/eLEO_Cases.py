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
    out_dir = ROOT_DIR / "results/case_studies/eleo"

    sim_eleo = ContinuousSatSim(
        orbit_data_path=orbit_path, 
        model_json_path=model_json, 
        output_dir=out_dir, 
        sat_prefix='eLEO', 
        num_orbits=3,
        model_source='Custom' 
    )
    

    eleo_cfg = ContinuousSatSim.get_sso_config()
    eleo_cfg.update({
        # optics and inference
        'focal_length_mm': 85.0,
        'pixel_pitch_um': 1.55,       
        'sensor_res': 2048,           
        'tpu_dim': 170,
        
        # altitudes
        'alt_threshold_km': 600.0,    
        'low_alt_target_km': 200.0,   
        'high_alt_target_km': 600.0,

        # power
        'battery_capacity_wh': 1.5,
        'solar_generation_mw': 1800.0, 

    })

    sim_eleo.run_case_study("eLEO_01_Baseline", config_overrides=eleo_cfg, events=None)

    burst_events = [
        {'start': 2700, 'duration': 800, 'extra_demand_ips': 150.0},
        {'start': 8245, 'duration': 800, 'extra_demand_ips': 150.0}, # Second orbit perigee
    ]
    sim_eleo.run_case_study("eLEO_02_Time_Crunch", config_overrides=eleo_cfg, events=burst_events)

    drain_events = [
        {'start': 1000, 'duration': 2000, 'power_w': 1.0},
    ]
    sim_eleo.run_case_study("eLEO_03_Power_Starved", config_overrides=eleo_cfg, events=drain_events)

    strict_cfg = eleo_cfg.copy()
    strict_cfg['hard_min_infs'] = 75.0 
    sim_eleo.run_case_study("eLEO_04_Strict_Limits", config_overrides=strict_cfg, events=None)

    storm_events = [
        {'start': 2000, 'duration': 1000, 'power_w': 0.8},                     
        {'start': 2500, 'duration': 200, 'extra_demand_ips': 120.0},           
        {'start': 2600, 'duration': 50, 'power_w': 0.0, 'blocked': True},     
    ]
    sim_eleo.run_case_study("eLEO_05_Perfect_Storm", config_overrides=strict_cfg, events=storm_events)

    """
    A pretty solid case config!
    
        eleo_cfg.update({
        # optics and inference
        'focal_length_mm': 85.0,
        'pixel_pitch_um': 1.55,       
        'sensor_res': 2048,           
        'tpu_dim': 170,
        # altitudes
        'alt_threshold_km': 600.0,    
        'low_alt_target_km': 200.0,   
        'high_alt_target_km': 600.0,
        # power
        'battery_capacity_wh': 1.5,
        'solar_generation_mw': 1800.0, 
        })
    
    """