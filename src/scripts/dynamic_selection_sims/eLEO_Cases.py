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
        num_orbits=10,
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
        'solar_generation_mw': 1600.0, 

    })

    sim_eleo.run_case_study("eLEO_01_Baseline", config_overrides=eleo_cfg, events=None)

    # Crank up the initial generation so the TPU starts "energy unlimited"
    degraded_cfg = eleo_cfg.copy()
    degraded_cfg['solar_generation_mw'] = 3600.0 
    
    # An eLEO orbit is roughly 90 mins (5400s). 
    # For a 3-orbit sim (~16200s), halfway is around t=8100s.
    # We simulate a 50% loss in solar generation by adding a massive constant 
    # baseload (1.8W or 1800mW) from t=8100s until the end of the simulation.
    # solar_failure_events = [
    #     # At t=8100s, permanently drop solar generation to 50% capacity
    #     {'start': 8100, 'duration': 15000, 'solar_scale': 0.5} 
    # ]
    
    # sim_eleo.run_case_study("eLEO_02_Panel_Failure", config_overrides=degraded_cfg, events=solar_failure_events)

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