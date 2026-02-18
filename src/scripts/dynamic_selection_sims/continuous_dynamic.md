# Satellite Edge Computing: Continuous Simulation Logic

This document outlines the mathematical models and control logic used in the `ContinuousSatSim` Python class. The simulation uses a continuous time-integration model ($\Delta t$) to track orbital mechanics, energy balance, and queue-based workload processing.

---

## 1. System Configuration & Constants

These parameters define the satellite's physical hardware and operational constraints.

| Parameter | Symbol | Script Variable | Unit | Description |
| :--- | :---: | :--- | :--- | :--- |
| **Sim Time Step** | $\Delta t$ | `sim_dt_s` | s | Integration step size (default 1.0s). |
| **Buffer Capacity** | $B_{max}$ | `buffer_max_frames` | Frames | Max queue size before new data is dropped. |
| **Solar Gen** | $P_{solar}$ | `solar_generation_mw` | mW | Max power from solar panels in sunlight. |
| **Baseload** | $P_{base}$ | `system_baseload_mw` | mW | Idle power (OBC + Radio RX). |
| **Battery Cap** | $E_{cap}$ | `battery_capacity_wh` | Wh | Total energy storage. |
| **Hard Min Limit**| $\gamma_{min}$ | `compute_disable_pct` | % | Hardware shutoff threshold (e.g., 45%). |
| **Target Tile** | $L_{target}$ | `target_tile_km` | km | Size of the ground ROI (e.g., 20x20km). |

---

## 2. Workload Physics: Geometry & Tiling

The workload ($\lambda$) is dynamic, derived from the satellite's instantaneous altitude and speed.

### A. Orbital Geometry
* **Ground Velocity ($v_{g}$):** Speed of the ground track relative to the sensor.
* **Ground Sample Distance ($GSD$):** Physical size of one pixel.
* **Swath Width ($W$):** Total width of the sensor's footprint.

$$GSD(t) = \frac{h(t) \cdot P_{pitch}}{f_{len}}, \quad W(t) = \frac{GSD(t) \cdot R_{sensor}}{1000}$$

### B. The "Tiling" Logic
1.  **Dwell Time:** Time available to capture the current scene.
    $$T_{dwell} = W(t) / v_{g}(t)$$

2.  **Tiles per Frame:** Number of target tiles in the current FOV.
    $$N_{tiles} = (W(t) / L_{target})^2$$

3.  **Inferences per Tile:** Patches ($224 \times 224$) needed per Target Tile.
    $$I_{tile} = \left( \frac{L_{target} \text{ (in meters)}}{GSD(t) \cdot D_{tpu}} \right)^2$$

### C. Arrival Rate ($\lambda$)
The instantaneous demand (Inferences per Second).

$$\lambda(t) = \frac{N_{tiles} \cdot I_{tile}}{T_{dwell}}$$

---

## 3. Buffer Dynamics (FIFO Queue)

* **Ingestion:** If $\lambda(t) > 1.0$ and $Buffer < B_{max}$, a new `FrameJob` is pushed.
* **Overflow:** If $Buffer \ge B_{max}$, the new frame is dropped.
* **Processing:** Jobs are popped from the queue based on the **Predictive Control Policy** (Section 5).

---

## 4. Energy Management: Predictive Waypoint Budgeting

Unlike standard hysteresis (on/off) controllers, this system uses a **Predictive Glide Path**. It calculates a dynamic energy budget for every frame to ensure the satellite meets specific State of Charge (SoC) targets at future orbital events (Eclipse Entry/Exit).

### A. Flight Regimes & Targets
The orbit is divided into segments based on lighting.

| Regime | Condition | Target ($E_{target}$) | Goal |
| :--- | :--- | :--- | :--- |
| **Sunlight** | $P_{in} > 0$ | $0.95 \cdot E_{cap}$ | Enter eclipse fully charged to maximize survival time. |
| **Eclipse** | $P_{in} = 0$ | $(\gamma_{min} + 0.05) \cdot E_{cap}$ | Survive darkness without hitting the hard cutoff. |

### B. Dynamic Budget Calculation
At any time $t$, we calculate the max safe energy expenditure ($E_{safe}$) to reach the next target $E_{target}$ at time $t_{event}$.

1.  **Time Remaining:** $\Delta t_{rem} = t_{event} - t$
2.  **Required Correction Power:** (Positive = Spend Surplus, Negative = Charge Deficit)
    $$P_{corr} = \frac{E_{batt}(t) - E_{target}}{\Delta t_{rem}}$$
3.  **Allowed Power:**
    $$P_{allowed} = P_{solar}(t) + P_{corr}$$

### C. Power Integration
$$E_{batt}(t+\Delta t) = \text{CLIP} \left( E_{batt}(t) + (P_{in}(t) - P_{load}(t)) \cdot \Delta t, \quad 0, \quad E_{cap} \right)$$

---

## 5. Control Policy (Budget-Constrained Optimization)

The system selects an AI model using a modified greedy approach. It maximizes throughput *within* the predictive budget.

At time $t$, with integration step $\Delta t$:

1.  **Define Constraints:**
    * **Predictive Budget:** $E_{budget} = \max(0, P_{allowed} \cdot \Delta t)$
    * **Hard Floor:** $E_{avail} = E_{batt} - (\gamma_{min} \cdot E_{cap})$
    * **Final Limit:** $E_{limit} = \min(E_{budget}, E_{avail})$

2.  **Evaluate Models:** For every model $M_i$:
    * $N_{time} = \Delta t / \text{Latency}_i$
    * $N_{energy} = E_{limit} / \text{EnergyPerInf}_i$
    * $N_{possible} = \min(N_{time}, N_{energy}, N_{job})$

3.  **Score & Select:**
    $$\text{Score}_i = N_{possible} \times \text{Accuracy}_i$$
    The model maximizing this score is executed. If $E_{limit} \le 0$, the system enters **Recharge/Idle** mode.

---

## 6. Naive Baseline Logic (Legacy Hysteresis)

A parallel simulation runs a "dumb" control strategy for comparison.

1.  **Fixed Model:** Always uses a single pre-selected model (e.g., 'EfficientNet-M').
2.  **Reactive Control:** Uses a simple Hysteresis Latch.
    * **STOP:** If $E_{batt} < \gamma_{min}$.
    * **START:** If $E_{batt} > \gamma_{resume}$ (where $\gamma_{resume} \approx 70\%$).
3.  **No Smoothing:** It runs at full power until it hits the limit, often resulting in "sawtooth" energy profiles and long dead times during eclipses.