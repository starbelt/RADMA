import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from enum import Enum
import sys
import json
import math
import copy

# Try to import repo root, else fallback
try:
    from libs.coral_tpu_characterization.src.scripts.utils.path_utils import get_repo_root
    ROOT_DIR = get_repo_root()
except ImportError:
    ROOT_DIR = Path(".").resolve()

# --- Data Structures ---

class FlightRegime(Enum):
    SUNLIGHT = 1
    ECLIPSE = 2

@dataclass
class OrbitalEvent:
    time_start: float
    time_end: float
    regime: FlightRegime

class FrameJob:
    """
    Represents a single frame (or tile-set) to be processed.
    """
    def __init__(self, job_id, total_inferences, timestamp):
        self.id = job_id
        self.total_inferences = total_inferences
        self.remaining_inferences = total_inferences
        self.timestamp = timestamp
        self.assigned_model = None 

class PowerManager:
    """
    Predictive Energy Budget Controller.
    Calculates the maximum safe energy expenditure per frame to ensure
    the satellite meets specific State of Charge (SoC) waypoints 
    (e.g., entering eclipse full, exiting eclipse alive).
    """
    def __init__(self, battery_capacity_j, hysteresis_conf, events):
        """
        hysteresis_conf: dict with 'compute_disable_pct'
        events: list of OrbitalEvent (sorted)
        """
        self.capacity_j = battery_capacity_j
        self.events = events 
        self.current_event_idx = 0
        
        # --- CONSTANTS ---
        # Target: Enter eclipse fully charged (95%) to maximize survival time
        self.TARGET_ECLIPSE_ENTRY_SOC = 0.95 
        
        # Target: Exit eclipse slightly ABOVE the disable limit. 
        # If disable is 45%, we target 50% so we can work immediately upon sunrise.
        self.TARGET_ECLIPSE_EXIT_SOC = hysteresis_conf['compute_disable_pct'] + 0.05

    def _get_current_event(self, t_now):
        # Fast-forward to correct event if time has passed
        while (self.current_event_idx < len(self.events) and 
               t_now > self.events[self.current_event_idx].time_end):
            self.current_event_idx += 1
            
        if self.current_event_idx >= len(self.events):
            return None # End of simulation schedule
        return self.events[self.current_event_idx]

    def get_allowed_budget(self, t_now, current_battery_j, solar_input_w, dt):
        """
        Returns the max Joules we can spend this frame (dt) while strictly
        adhering to the glide path for the next event.
        """
        event = self._get_current_event(t_now)
        # If no schedule exists (e.g. simulation ran past orbit data), default to safe mode (0 budget)
        if not event: return 0.0 

        time_remaining = event.time_end - t_now
        if time_remaining <= 1e-3: time_remaining = dt # Prevent div/0

        # --- DETERMINE TARGET STATE ---
        if event.regime == FlightRegime.SUNLIGHT:
            # We are in sun, preparing for eclipse
            target_j = self.capacity_j * self.TARGET_ECLIPSE_ENTRY_SOC
        else:
            # We are in eclipse, preparing for sunrise
            target_j = self.capacity_j * self.TARGET_ECLIPSE_EXIT_SOC

        # --- CALCULATE GLIDE PATH ---
        # Current Deviation from Target
        energy_diff_j = current_battery_j - target_j
        
        # Slope: How many Watts (J/s) do we need to burn/save to hit 0 diff at t_end?
        # If energy_diff is positive (we have extra), this yields positive power to spend.
        # If energy_diff is negative (we are behind), this yields negative power (must charge).
        correction_power_w = energy_diff_j / time_remaining

        # Total Allowed Spend = Solar Input + Battery Correction
        # In Eclipse: Solar is 0, so we just drain the battery at the calculated safe rate.
        # In Sun: We spend solar + whatever excess battery we have (or minus what we need to save).
        total_allowed_w = solar_input_w + correction_power_w

        # Clamp results
        # Never return < 0 (can't un-spend energy)
        allowed_joules = max(0.0, total_allowed_w * dt)
        
        return allowed_joules

class ContinuousSatSim:
    """
    Simulates satellite operation over an orbit, managing power, data buffers, 
    and compute workloads using both a dynamic model selection strategy and 
    a naive baseline for comparison.
    """
    
    BASE_SYSTEM = {
        'pixel_pitch_um': 3.45,
        'sensor_res': 4096,
        'tpu_dim': 224,
        'min_pixels': 10,
        'system_baseload_mw': 300.0,
        'sim_dt_s': 1.0,
        
        # Power Management Logic
        'initial_charge_pct': 0.85,
        'compute_enable_pct': 0.70,   # Used for plotting reference only now
        'compute_disable_pct': 0.45,  # Used as hard floor
    }

    @staticmethod
    def get_heo_config():
        """
        HEO Configuration: Optimized for long-range observation.
        """
        cfg = ContinuousSatSim.BASE_SYSTEM.copy()
        cfg.update({
            'focal_length_mm': 300.0,
            'target_tile_km': 50.0,
            'battery_capacity_wh': 1.0,
            'solar_generation_mw': 500.0,
            'buffer_max_frames': 200,
        })
        return cfg

    @staticmethod
    def get_sso_config():
        """
        SSO Configuration: Optimized for high-speed mapping.
        """
        cfg = ContinuousSatSim.BASE_SYSTEM.copy()
        cfg.update({
            'focal_length_mm': 85.0,
            'target_tile_km': 20.0,
            'battery_capacity_wh': 1.5,  # less than a 18650 cell, more than the supercap
            'solar_generation_mw': 500.0,
            'buffer_max_frames': 500,
        })
        return cfg

    def __init__(self, orbit_data_path, model_json_path, output_dir, sat_prefix='HEO', num_orbits=1, model_source='Custom', naive_model_name=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sat_prefix = sat_prefix
        self.naive_model_name = naive_model_name
        
        print(f"--- Loading Models (Source={model_source}) & Orbit Data for {sat_prefix} ---")
        self.models = self._load_models(model_json_path, model_source)
        self.raw_orbit, self.sunlight_intervals = self._load_orbit_data(orbit_data_path, num_orbits)
        
    def _load_models(self, json_path, source_filter):
        if not Path(json_path).exists():
            raise FileNotFoundError(f"Model JSON not found at: {json_path}")
        df = pd.read_json(json_path)
        if df.empty: raise ValueError("Model data JSON is empty.")

        if 'Source' in df.columns and source_filter is not None:
            df = df[df['Source'] == source_filter].copy()

        df['acc_decimal'] = df['Top-1 Accuracy'] / 100.0
        df['lat_s'] = df['Measured Inference Time (ms)'] / 1000.0
        df['eng_j'] = df['Energy per Inference (mJ)'] / 1000.0
        
        # Precompute Correct Inferences Per Second/Joule for report
        df['correct_infs_per_sec'] = (1.0 / df['lat_s']) * df['acc_decimal']
        df['correct_infs_per_joule'] = (1.0 / df['eng_j']) * df['acc_decimal']
        
        return df

    def _load_orbit_data(self, data_path, num_orbits):
        root = Path(data_path)
        p_path = root / f'{self.sat_prefix}_Sat_Fixed_Position_Velocity.csv'
        c_path = root / f'{self.sat_prefix}_Sat_Classical_Orbit_Elements.csv'
        l_path = root / f'{self.sat_prefix}_Sat_LLA_Position.csv'
        
        if not p_path.exists() or not c_path.exists() or not l_path.exists():
            print(f"[WARN] Missing STK Files for {self.sat_prefix}. Simulation will be skipped.")
            return pd.DataFrame(), []

        v = pd.read_csv(p_path)
        k = pd.read_csv(c_path)
        l = pd.read_csv(l_path)
        
        for d in [v, k, l]: 
            d.columns = d.columns.str.strip()
            if "Time (EpSec)" in d.columns:
                d["Time (EpSec)"] = d["Time (EpSec)"].round(4)

        try:
            df = v.merge(k, on="Time (EpSec)").merge(l, on="Time (EpSec)")
        except Exception as e:
            print(f"[ERROR] Merge failed for {self.sat_prefix}: {e}")
            return pd.DataFrame(), []

        if df.empty:
            print(f"[WARN] Orbit data for {self.sat_prefix} resulted in empty DataFrame.")
            return pd.DataFrame(), []
        
        if 'True Anomaly (deg)' in df.columns:
            ta = df['True Anomaly (deg)'].values
            diffs = np.diff(ta) 
            wrap_indices = np.where(diffs < -300)[0]
            if len(wrap_indices) > 0 and len(wrap_indices) >= num_orbits:
                cutoff_idx = wrap_indices[num_orbits - 1]
                df = df.iloc[:cutoff_idx+1].copy()

        light_path = root / f'{self.sat_prefix}_Sat_Lighting_Times.csv'
        sunlight_intervals = []
        if light_path.exists():
            sunlight_intervals = self._parse_lighting_schedule(light_path)
        else:
            t_min, t_max = df['Time (EpSec)'].min(), df['Time (EpSec)'].max()
            sunlight_intervals = [(t_min, t_max)]

        print(f"[INFO] Loaded {len(df)} rows of orbit data for {self.sat_prefix}.")
        return df, sunlight_intervals

    def _parse_lighting_schedule(self, file_path):
        """
        Parses the specific text format of STK Lighting Times reports.
        """
        intervals = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        capture = False
        header_found_count = 0
        for line in lines:
            if "Start Time (EpSec)" in line:
                header_found_count += 1
                if header_found_count > 1: break
                capture = True
                continue
            if capture:
                if "Statistics" in line or line.strip() == "": break 
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        start = float(parts[0].replace('"', ''))
                        stop = float(parts[1].replace('"', ''))
                        intervals.append((start, stop))
                    except ValueError: continue
        return intervals

    def _build_event_schedule(self, t_start_abs, t_end_abs, sunlight_intervals):
        """
        Converts absolute sunlight intervals into a relative-time event schedule 
        (Sunlight vs Eclipse segments) for the PowerManager.
        """
        events = []
        current_t = t_start_abs
        
        # Sort intervals just in case
        sorted_intervals = sorted(sunlight_intervals, key=lambda x: x[0])

        for (sun_start, sun_end) in sorted_intervals:
            # Check if there is an eclipse gap before this sun period
            if sun_start > current_t:
                # ECLIPSE SEGMENT
                # Clamp start to simulation start
                start_seg = max(current_t, t_start_abs)
                end_seg = min(sun_start, t_end_abs)
                if end_seg > start_seg:
                    events.append(OrbitalEvent(
                        start_seg - t_start_abs, 
                        end_seg - t_start_abs, 
                        FlightRegime.ECLIPSE
                    ))
            
            # SUNLIGHT SEGMENT
            start_seg = max(sun_start, t_start_abs)
            end_seg = min(sun_end, t_end_abs)
            
            # Break if we've passed the simulation end
            if start_seg >= t_end_abs: break
            
            if end_seg > start_seg:
                events.append(OrbitalEvent(
                    start_seg - t_start_abs, 
                    end_seg - t_start_abs, 
                    FlightRegime.SUNLIGHT
                ))
            
            current_t = max(current_t, end_seg)
            
        # Check if there is trailing eclipse after last sun
        if current_t < t_end_abs:
             events.append(OrbitalEvent(
                current_t - t_start_abs, 
                t_end_abs - t_start_abs, 
                FlightRegime.ECLIPSE
            ))
             
        return events

    def _select_model(self, energy_budget_j, time_budget_s, workload_infs):
        """
        Selects the optimal model for the current time step.
        """
        if energy_budget_j <= 0: return None, "Energy_Depleted"
        
        max_inf_time = time_budget_s / self.models['lat_s']
        max_inf_energy = energy_budget_j / self.models['eng_j']
        capacity_infs = np.minimum(max_inf_time, max_inf_energy)
        actual_processing_infs = np.minimum(capacity_infs, workload_infs)
        expected_correct = actual_processing_infs * self.models['acc_decimal']
        
        best_idx = expected_correct.idxmax()
        best_model = self.models.loc[best_idx]
        
        # Determine limiting factor for logging
        lim_time = max_inf_time[best_idx]
        lim_eng = max_inf_energy[best_idx]
        if lim_eng < workload_infs and lim_eng < lim_time: reason = "Energy_Limited"
        elif lim_time < workload_infs and lim_time < lim_eng: reason = "Time_Limited"
        elif workload_infs < lim_time and workload_infs < lim_eng: reason = "Workload_Clearing" 
        else: reason = "Balanced"
        return best_model, reason

    def run_case_study(self, case_name, config_overrides=None, events=None):
        """
        Executes the main simulation loop.
        """
        if self.raw_orbit.empty:
            print(f"[SKIP] Skipping {case_name} (Orbit data is empty or failed to load)")
            return

        print(f"\n>>> Running Case Study: {case_name}")
        cfg = self.BASE_SYSTEM.copy()
        if config_overrides: cfg.update(config_overrides)
        
        sim_data = self._interpolate_orbit(self.raw_orbit, self.sunlight_intervals, cfg['sim_dt_s'], cfg)
        if sim_data.empty: return

        # --- SETUP POWER MANAGER ---
        flight_schedule = self._build_event_schedule(
            sim_data['Time (EpSec)'].min(), 
            sim_data['Time (EpSec)'].max(), 
            self.sunlight_intervals
        )
        
        power_manager = PowerManager(
            cfg['battery_capacity_wh'] * 3600.0, 
            cfg, 
            flight_schedule
        )

        # Setup Naive Model
        naive_model = None
        naive_power_w = 0.0
        naive_throughput_ips = 0.0
        if self.naive_model_name:
            if self.naive_model_name in self.models['Model name'].values:
                naive_model = self.models[self.models['Model name'] == self.naive_model_name].iloc[0]
                naive_power_w = naive_model['eng_j'] / naive_model['lat_s']
                naive_throughput_ips = 1.0 / naive_model['lat_s']
            else:
                print(f"[WARN] Naive model '{self.naive_model_name}' not found. Comparison disabled.")

        # Simulation State Initialization
        BATTERY_CAPACITY_J = cfg['battery_capacity_wh'] * 3600.0
        limit_disable_j = BATTERY_CAPACITY_J * cfg['compute_disable_pct']
        
        # Dynamic State
        current_battery_j = BATTERY_CAPACITY_J * cfg['initial_charge_pct']
        frame_buffer = deque() 
        current_job = None 
        
        # Naive State
        n_battery_j = current_battery_j
        n_recharging = False
        n_buffer = deque()
        n_total_correct = 0

        # Reporting Stats
        report_stats = {
            'time_limited_count': 0,
            'energy_limited_count': 0,
            'buffered_count': 0,
            'unprocessed_count': 0,
            'total_steps': len(sim_data)
        }

        logs = {
            'time_rel': [], 'battery_wh': [], 'buffer_count': [], 
            'throughput_infs': [], 'backlog_infs': [], 'model_name': [],
            'avg_accuracy': [], 'active_power_w': [], 'is_lit': [], 
            'demand_infs': [], 'alt_km': [], 'speed_km_s': [],
            'cum_correct': [], 'naive_battery_wh': [], 'naive_cum_correct': []
        }
        
        t_start = sim_data['Time (EpSec)'].iloc[0]
        total_infs_correct = 0

        for i, row in sim_data.iterrows():
            t_rel = row['Time (EpSec)'] - t_start
            dt = cfg['sim_dt_s']
            
            # --- Environment & Ingestion ---
            disturb_power_w = 0.0
            cpu_blocked = False
            if events:
                for e in events:
                    if e['start'] <= t_rel < (e['start'] + e['duration']):
                        disturb_power_w += e.get('power_w', 0)
                        if e.get('blocked', False): cpu_blocked = True

            solar_w = (cfg['solar_generation_mw'] / 1000.0) * row['is_lit']
            base_w = (cfg['system_baseload_mw'] / 1000.0) + disturb_power_w
            env_energy_j = (solar_w - base_w) * dt 
            infs_for_this_second = row['demand_infs_per_sec'] * dt
            
            # Populate Buffers & Track Frame Stats
            buffered_this_step = False
            if row['px_per_object'] < cfg['min_pixels']:
                active_model_name = "Blind" 
                report_stats['unprocessed_count'] += 1 
            else:
                active_model_name = "Idle"
                if infs_for_this_second > 1.0: 
                    if len(frame_buffer) < cfg['buffer_max_frames']:
                        frame_buffer.append(FrameJob(i, infs_for_this_second, t_rel))
                        buffered_this_step = True
                        if naive_model is not None:
                            n_buffer.append(FrameJob(i, infs_for_this_second, t_rel))
                    else:
                        report_stats['unprocessed_count'] += 1 # Buffer overflow drop
            
            if buffered_this_step and len(frame_buffer) > 1:
                 report_stats['buffered_count'] += 1

            # --- Dynamic Simulation Logic (Predictive Manager) ---
            
            # 1. Ask Manager for Allowed Budget
            safe_budget_j = power_manager.get_allowed_budget(t_rel, current_battery_j, solar_w, dt)
            
            # 2. Hard Hardware Constraint
            # We must also respect the hardware shutoff limit (e.g. 45% battery)
            # If current battery is below disable limit, hard_budget becomes negative -> Energy Depleted
            hard_budget_j = current_battery_j - limit_disable_j
            
            # 3. Effective Budget is the stricter of the two
            energy_budget_j = min(safe_budget_j, hard_budget_j)

            time_available_s = dt
            processed_infs_step = 0
            processing_energy_j = 0
            current_accuracy = 0.0
            step_limitation = None 

            while time_available_s > 0:
                if cpu_blocked: active_model_name = "BLOCKED"; break
                
                # If budget is effectively zero or negative, we must Idle/Recharge
                if energy_budget_j <= 1e-6:
                    active_model_name = "RECHARGE" # Effectively saving energy
                    break
                
                if current_job is None:
                    if len(frame_buffer) > 0:
                        current_job = frame_buffer.popleft()
                        
                        # calculate the total true backlog across all queued frames
                        total_workload = current_job.remaining_inferences + sum(j.remaining_inferences for j in frame_buffer)
                        
                        # pass the total workload so efficient models can score higher
                        model, reason = self._select_model(energy_budget_j, time_available_s, total_workload)
                        
                        if reason == "Time_Limited": step_limitation = "Time"
                        elif reason == "Energy_Limited": step_limitation = "Energy"
                        
                        if model is None: 
                            # fallback safety
                            frame_buffer.appendleft(current_job)
                            current_job = None
                            active_model_name = "RECHARGE"
                            break
                        current_job.assigned_model = model
                    else: break
                
                # Execute Job
                if current_job:
                    model = current_job.assigned_model
                    active_model_name = model['Model name']
                    current_accuracy = model['acc_decimal']
                    
                    infs_possible = min(current_job.remaining_inferences, 
                                        time_available_s / model['lat_s'], 
                                        energy_budget_j / model['eng_j'])
                    
                    if infs_possible <= 0: active_model_name = "RECHARGE"; break

                    current_job.remaining_inferences -= infs_possible
                    e_spent = infs_possible * model['eng_j']
                    t_spent = infs_possible * model['lat_s']
                    
                    current_battery_j -= e_spent
                    energy_budget_j -= e_spent # Decrement frame budget
                    time_available_s -= t_spent
                    processed_infs_step += infs_possible
                    processing_energy_j += e_spent
                    
                    if current_job.remaining_inferences <= 1e-6: current_job = None

            # Stats Accumulation
            if step_limitation == "Time": report_stats['time_limited_count'] += 1
            elif step_limitation == "Energy": report_stats['energy_limited_count'] += 1
            
            total_load_w = base_w + (processing_energy_j / dt)
            current_battery_j = np.clip(current_battery_j + env_energy_j, 0, BATTERY_CAPACITY_J)
            total_infs_correct += processed_infs_step * current_accuracy

            # --- Naive Model Logic (Legacy Hysteresis) ---
            if naive_model is not None:
                # Naive keeps the old "dumb" hysteresis logic
                limit_enable_j = BATTERY_CAPACITY_J * cfg['compute_enable_pct'] # Needed for naive only
                
                if n_battery_j < limit_disable_j:
                    n_recharging = True
                elif n_recharging and n_battery_j > limit_enable_j:
                    n_recharging = False
                
                if cpu_blocked or n_recharging:
                    n_battery_j = np.clip(n_battery_j + env_energy_j, 0, BATTERY_CAPACITY_J)
                else:
                    available_energy_j = max(0, n_battery_j - limit_disable_j)
                    step_energy_demand_j = dt * naive_power_w
                    
                    time_runnable_s = dt
                    if available_energy_j < step_energy_demand_j:
                        time_runnable_s = available_energy_j / naive_power_w
                        n_recharging = True 
                    
                    n_battery_j -= (time_runnable_s * naive_power_w)
                    potential_infs = time_runnable_s * naive_throughput_ips
                    
                    processed_naive_infs = 0
                    while potential_infs > 0 and len(n_buffer) > 0:
                        job = n_buffer[0]
                        taken = min(potential_infs, job.remaining_inferences)
                        job.remaining_inferences -= taken
                        potential_infs -= taken
                        processed_naive_infs += taken
                        if job.remaining_inferences <= 1e-6: n_buffer.popleft()
                    
                    n_battery_j = np.clip(n_battery_j + env_energy_j, 0, BATTERY_CAPACITY_J)
                    n_total_correct += processed_naive_infs * naive_model['acc_decimal']
            
            # --- Logging ---
            buffer_backlog = sum(j.remaining_inferences for j in frame_buffer)
            if current_job: buffer_backlog += current_job.remaining_inferences

            logs['time_rel'].append(t_rel)
            logs['battery_wh'].append(current_battery_j / 3600.0)
            logs['buffer_count'].append(len(frame_buffer))
            logs['backlog_infs'].append(buffer_backlog)
            logs['throughput_infs'].append(processed_infs_step)
            logs['model_name'].append(active_model_name)
            logs['avg_accuracy'].append(current_accuracy if processed_infs_step > 0 else np.nan)
            logs['active_power_w'].append(total_load_w)
            logs['is_lit'].append(row['is_lit'])
            logs['demand_infs'].append(infs_for_this_second)
            logs['alt_km'].append(row['Alt (km)'])
            logs['speed_km_s'].append(row['v_ground_km_s'])
            logs['cum_correct'].append(total_infs_correct)
            logs['naive_battery_wh'].append(n_battery_j / 3600.0)
            logs['naive_cum_correct'].append(n_total_correct)

        self._plot_telemetry(logs, case_name, cfg)
        self._print_verbose_report(case_name, report_stats, logs, sim_data, cfg)
        return logs

    def _print_verbose_report(self, case_name, stats, logs, sim_data, cfg):
        # print case report header
        print(f"\n{'='*60}")
        print(f"case report: {case_name}")
        print(f"{'='*60}")
        
        # find best models
        best_throughput = self.models.loc[self.models['correct_infs_per_sec'].idxmax()]
        best_efficiency = self.models.loc[self.models['correct_infs_per_joule'].idxmax()]
        best_acc = self.models.loc[self.models['acc_decimal'].idxmax()]
        
        print("model landscape:")
        print(f"  * highest throughput: {best_throughput['Model name']:<20} ({best_throughput['correct_infs_per_sec']:.1f} correct inf/s)")
        print(f"  * best efficiency:    {best_efficiency['Model name']:<20} ({best_efficiency['correct_infs_per_joule']:.1f} correct inf/j)")
        print(f"  * max accuracy:       {best_acc['Model name']:<20} ({best_acc['acc_decimal']*100:.1f}%)")
        print("-" * 60)
        
        # calc tiles per frame
        tiles_per_frame = (sim_data['swath_km'] / cfg['target_tile_km'])**2

        # print new figures of merit (min / max)
        print("figures of merit (min / max):")
        print(f"  * swath (km):             {sim_data['swath_km'].min():.2f} / {sim_data['swath_km'].max():.2f}")
        print(f"  * gsd (m):                {sim_data['gsd_m'].min():.2f} / {sim_data['gsd_m'].max():.2f}")
        print(f"  * ground velocity (km/s): {sim_data['v_ground_km_s'].min():.2f} / {sim_data['v_ground_km_s'].max():.2f}")
        print(f"  * tiles per frame:        {tiles_per_frame.min():.2f} / {tiles_per_frame.max():.2f}")
        print("-" * 60)
        
        # sim stats
        total_frames = stats['total_steps']
        print("simulation statistics:")
        if total_frames > 0:
            print(f"  * time limited frames:   {stats['time_limited_count']:5d} ({stats['time_limited_count']/total_frames*100:5.1f}%)")
            print(f"  * energy limited frames: {stats['energy_limited_count']:5d} ({stats['energy_limited_count']/total_frames*100:5.1f}%)")
            print(f"  * frames buffered:       {stats['buffered_count']:5d} ({stats['buffered_count']/total_frames*100:5.1f}%)")
            print(f"  * frames dropped/missed: {stats['unprocessed_count']:5d} ({stats['unprocessed_count']/total_frames*100:5.1f}%)")
        print("-" * 60)

        # util breakdown
        print("model utilization (% of active processing time):")
        df_log = pd.DataFrame({'model': logs['model_name']})
        
        # filter out non-compute states to see just compute usage
        compute_models = df_log[~df_log['model'].isin(['Idle', 'RECHARGE', 'BLOCKED', 'Blind'])]
        
        if not compute_models.empty:
            breakdown = compute_models['model'].value_counts(normalize=True) * 100
            for name, pct in breakdown.items():
                print(f"  * {name:<30}: {pct:5.1f}%")
        else:
            print("  (no models executed)")
        print(f"{'='*60}\n")

    def _interpolate_orbit(self, df, sunlight_intervals, dt, cfg):
        """
        Interpolates the variable-step orbit data to a fixed time step (dt).
        """
        if df.empty: return pd.DataFrame()
        
        t_start = df['Time (EpSec)'].min()
        t_end = df['Time (EpSec)'].max()
        new_times = np.arange(t_start, t_end, dt)
        
        cols = ['x (km)', 'y (km)', 'z (km)', 'vx (km/sec)', 'vy (km/sec)', 'vz (km/sec)', 'Alt (km)']
        new_data = {'Time (EpSec)': new_times}
        for c in cols:
            new_data[c] = np.interp(new_times, df['Time (EpSec)'], df[c])
        
        is_lit = np.zeros_like(new_times)
        for s, e in sunlight_intervals:
            mask = (new_times >= s) & (new_times <= e)
            is_lit[mask] = 1.0
        new_data['is_lit'] = is_lit
        new_df = pd.DataFrame(new_data)

        new_df['gsd_m'] = (new_df['Alt (km)'] * cfg['pixel_pitch_um']) / cfg['focal_length_mm']
        new_df['swath_km'] = (new_df['gsd_m'] * cfg['sensor_res']) / 1000.0
        
        r = new_df[['x (km)', 'y (km)', 'z (km)']].values
        v = new_df[['vx (km/sec)', 'vy (km/sec)', 'vz (km/sec)']].values
        r_norm = np.linalg.norm(r, axis=1, keepdims=True)
        v_vert = np.sum(v * (r/r_norm), axis=1, keepdims=True) * (r/r_norm)
        v_ground = np.linalg.norm(v - v_vert, axis=1)
        
        new_df['v_ground_km_s'] = v_ground
        new_df['dwell_time_s'] = new_df['swath_km'] / (v_ground + 1e-9)

        target_size_m = cfg['target_tile_km'] * 1000.0
        new_df['px_per_object'] = target_size_m / new_df['gsd_m']
        
        tpu_dim = cfg['tpu_dim']
        infs_per_object = np.where(new_df['px_per_object'] < cfg['min_pixels'], 0.0, (new_df['px_per_object'] / tpu_dim)**2)
        
        objects_in_fov = (new_df['swath_km'] / cfg['target_tile_km'])**2
        total_infs_per_frame = objects_in_fov * infs_per_object
        new_df['demand_infs_per_sec'] = total_infs_per_frame / new_df['dwell_time_s']
        
        return new_df

    def _plot_telemetry(self, logs, case_name, cfg):
        """
        Generates a 4-panel dashboard visualizing satellite telemetry.
        """
        t_plot = np.array(logs['time_rel'])
        
        fig = plt.figure(figsize=(12, 16))
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1.5, 1.5, 1])
        fig.suptitle(f"Case Study: {case_name}", fontsize=16)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax4 = fig.add_subplot(gs[3])

        # Orbit
        ax1.plot(t_plot, logs['alt_km'], color='gray', label='Altitude')
        ax1.set_ylabel('Altitude (km)')
        ax1_t = ax1.twinx()
        ax1_t.plot(t_plot, logs['speed_km_s'], 'r--', label='Speed')
        ax1_t.set_ylabel('Speed (km/s)', color='r')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.get_xticklabels(), visible=False)

        # Energy
        ax2.plot(t_plot, logs['battery_wh'], 'g', label='Dynamic Battery')
        if any(logs['naive_battery_wh']):
            ax2.plot(t_plot, logs['naive_battery_wh'], color='tab:red', linestyle='--', alpha=0.7, label=f'Naive ({self.naive_model_name})')
        
        ax2.axhline(cfg['battery_capacity_wh']*cfg['compute_disable_pct'], color='r', linestyle=':', label='Hard Min')
        # Note: Resume line is less relevant for Dynamic model now, but kept for reference
        ax2.axhline(cfg['battery_capacity_wh']*cfg['compute_enable_pct'], color='g', linestyle=':', label='Ref Resume')
        
        lit = np.array(logs['is_lit'])
        ax2.fill_between(t_plot, 0, 1, where=(lit > 0.5), transform=ax2.get_xaxis_transform(), 
                         color='gold', alpha=0.2, label='Sunlight')
        ax2.set_ylabel('Battery (Wh)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.get_xticklabels(), visible=False)

        # Performance & Backlog
        ax3.plot(t_plot, logs['cum_correct'], 'k', label='Dynamic Correct')
        if any(logs['naive_cum_correct']):
            ax3.plot(t_plot, logs['naive_cum_correct'], color='tab:red', linestyle='--', label='Naive Correct')
        ax3.set_ylabel('Total Correct Inferences')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        ax3_t = ax3.twinx()
        ax3_t.fill_between(t_plot, logs['backlog_infs'], color='tab:orange', alpha=0.15, label='Backlog')
        ax3_t.plot(t_plot, logs['backlog_infs'], color='tab:orange', linewidth=0.5, alpha=0.4)
        ax3_t.set_ylabel('Dynamic Backlog Size', color='tab:orange')
        ax3_t.tick_params(axis='y', labelcolor='tab:orange')
        ax3.set_xlabel('Time (s)')

        # Utilization
        df_log = pd.DataFrame({'model': logs['model_name'], 'infs': logs['throughput_infs']})
        stats = df_log[df_log['infs'] > 0].groupby('model')['infs'].sum().sort_values()
        
        if not stats.empty:
            bars = ax4.barh(stats.index, stats.values, color='tab:purple')
            ax4.bar_label(bars, fmt='{:,.0f}', padding=3)
        else:
            ax4.text(0.5, 0.5, "No Inferences Performed", ha='center', va='center')
        
        ax4.set_xlabel('Total Inferences Processed')
        ax4.set_title('Total Model Utilization')
        ax4.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        filename = f"{case_name}.png"
        save_path = self.output_dir / filename
        plt.savefig(save_path)
        print(f"Saved plot to: {save_path}")
        plt.close()

def run_all_case_studies():
    model_json = ROOT_DIR / "data/compiled_characterization.json" 
    orbit_path = ROOT_DIR / "data/stk"
    out_dir = ROOT_DIR / "results/case_studies"

    sim_heo = ContinuousSatSim(orbit_path, model_json, out_dir, sat_prefix='HEO', num_orbits=1, naive_model_name='Grid A1.0 D06')
    
    # HEO Cases
    sim_heo.run_case_study("HEO_01_Standard", config_overrides=ContinuousSatSim.get_heo_config())
    
    heo_crisis = ContinuousSatSim.get_heo_config()
    heo_crisis['initial_charge_pct'] = 0.25
    sim_heo.run_case_study("HEO_02_PowerCrisis", config_overrides=heo_crisis)
    
    isr_events = [{'start': 20000, 'duration': 600, 'power_w': 1.5, 'blocked': True}] # 1.5W transmission
    sim_heo.run_case_study("HEO_03_ISR_Interruption", config_overrides=ContinuousSatSim.get_heo_config(), events=isr_events)

    # SSO Cases 
    sim_sso = ContinuousSatSim(orbit_path, model_json, out_dir, sat_prefix='SSO', num_orbits=20, naive_model_name='Grid A1.0 D06')
    
    sso_cfg = ContinuousSatSim.get_sso_config()
    sim_sso.run_case_study("SSO_01_Standard", config_overrides=sso_cfg)
    
    sso_vol = sso_cfg.copy()
    sso_vol['target_tile_km'] = 40.0 
    sim_sso.run_case_study("SSO_02_High_Data_Volume", config_overrides=sso_vol)
    
    sso_deg = sso_cfg.copy()
    sso_deg['solar_generation_mw'] = 400.0  # Solar panels damaged (drop to 400mW)
    sim_sso.run_case_study("SSO_03_Degraded_Power", config_overrides=sso_deg)

if __name__ == "__main__":
    run_all_case_studies()