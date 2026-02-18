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
| **Target Tile** | $L_{target}$ | `target_tile_km` | km | Size of the ground ROI we want to analyze (e.g., 20x20km). |
| **TPU Input** | $D_{tpu}$ | `tpu_dim` | px | Dimension of the AI model input (e.g., 224x224). |

---

## 2. Workload Physics: Geometry & Tiling

The workload ($\lambda$) is not static; it is derived dynamically from the satellite's instantaneous altitude and speed. The system calculates how many "Target Tiles" are visible and how many inference patches are required to process them at the current resolution.

### A. Orbital Geometry
First, we determine the physical ground coverage.
* **Ground Velocity ($v_{g}$):** Speed of the ground track relative to the sensor.
* **Ground Sample Distance ($GSD$):** Physical size of one pixel.
* **Swath Width ($W$):** Total width of the sensor's footprint.

$$GSD(t) = \frac{h(t) \cdot P_{pitch}}{f_{len}}, \quad W(t) = \frac{GSD(t) \cdot R_{sensor}}{1000}$$

### B. The "Tiling" Logic
We assume the mission goal is to process specific ROIs (Target Tiles) within the sensor's field of view.

1.  **Dwell Time:** The time available to capture the current scene before the satellite moves one full frame height.
    $$T_{dwell} = W(t) / v_{g}(t)$$

2.  **Tiles per Frame:** How many $L_{target}$ tiles fit into the current FOV?
    $$N_{tiles} = (W(t) / L_{target})^2$$

3.  **Inferences per Tile:** How many $224 \times 224$ patches are needed to cover one Target Tile at current resolution?
    $$I_{tile} = \left( \frac{L_{target} \text{ (in meters)}}{GSD(t) \cdot D_{tpu}} \right)^2$$

### C. Arrival Rate ($\lambda$)
The instantaneous demand (Inferences per Second) is the total work per frame divided by the time it takes to pass that frame.

$$\lambda(t) = \frac{N_{tiles} \cdot I_{tile}}{T_{dwell}}$$

---

## 3. Buffer Dynamics (FIFO Queue)

The system uses a **First-In-First-Out (FIFO)** queue of `FrameJob` objects to decouple ingestion from processing.

* **Ingestion:** If $\lambda(t) > 1.0$ and $Buffer < B_{max}$, a new `FrameJob` containing $\lambda(t) \cdot \Delta t$ inferences is pushed to the queue.
* **Overflow:** If $Buffer \ge B_{max}$, the new frame is dropped (simulating data loss).
* **Processing:** Jobs are popped from the queue and processed by the active model logic.

---

## 4. Energy Management & Battery Integration

The power system is modeled continuously using Euler integration, governed by a **Hysteresis Latch** to prevent rapid on/off cycling ("chattering") at the threshold.

### A. The Hysteresis Latch
Let $S_{charge}$ be the system state ($True$ = Recharging/Idle, $False$ = Active). The state only changes when specific thresholds are crossed:

$$
S_{charge}(t+\Delta t) = 
\begin{cases} 
\text{True (Stop)} & \text{if } E_{batt} < \gamma_{off} \cdot E_{cap} \\
\text{False (Start)} & \text{if } E_{batt} > \gamma_{on} \cdot E_{cap} \\
S_{charge}(t) & \text{otherwise (Maintain State)}
\end{cases}
$$

**Constraint:** Computer processing is strictly forbidden when $S_{charge}$ is True.

### B. Power Integration
The battery state $E_{batt}$ is updated at the end of every time step based on the net power flow.

1.  **Solar Input:** $P_{in}(t) = P_{solar} \cdot \mathbb{I}_{sunlight}(t)$
2.  **Load:** $P_{load}(t) = P_{base} + P_{compute}(t) + P_{events}(t)$
3.  **Integration:**

$$E_{batt}(t+\Delta t) = \text{CLIP} \left( E_{batt}(t) + (P_{in}(t) - P_{load}(t)) \cdot \Delta t, \quad 0, \quad E_{cap} \right)$$

---

## 5. Dynamic Control Policy (Greedy Optimization)

When the system is Active ($S_{charge} = False$), it selects an AI model using **Resource-Constrained Greedy Optimization**.

At time $t$, with available duration $\Delta t$ and available energy $E_{avail} = E_{batt} - E_{cutoff}$:

1.  **Fetch Job:** Peak at the oldest job in the buffer.
2.  **Evaluate Models:** For every available AI model $M_i$, calculate the maximum work possible ($N_{possible}$) given three strict constraints:
    * **Time Limit:** $N_{time} = \Delta t / \text{Latency}_i$
    * **Energy Limit:** $N_{energy} = E_{avail} / \text{EnergyPerInf}_i$
    * **Job Size:** $N_{job} = \text{RemainingInfs}_{job}$
    
    $$N_{possible} = \min(N_{time}, N_{energy}, N_{job})$$

3.  **Score:** Calculate the **Expected Correct Inferences**:
    $$\text{Score}_i = N_{possible} \times \text{Accuracy}_i$$

4.  **Select:** Execute the model $M_{best}$ that maximizes the Score.

---

## 6. Naive Baseline Logic

A parallel simulation runs a "dumb" control strategy for comparison.

1.  **Fixed Model:** Always uses a single, pre-selected model (e.g., 'EfficientNet-M').
2.  **No Lookahead:** It attempts to process for the full duration of $\Delta t$, regardless of queue size (it burns power even if the buffer empties mid-step).
3.  **Strict Cutoff:** It runs at full power until it hits the $\gamma_{off}$ limit. It does not throttle to save energy; it simply runs until it dies, then waits for the hysteresis reset.