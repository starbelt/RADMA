# Satellite Edge Computing: Simulation Logic & Mathematics

This document outlines the mathematical models and control logic used in the `SatelliteInferenceSim` Python class. The simulation evaluates an orbital edge computing payload, balancing energy harvesting against computational demands to maximize total accurate inferences.

---

## 1. System Configuration & Constants

These parameters define the sensor capabilities, target mission profile, and operational thresholds used to generate the workload and govern system behavior.

| Parameter | Symbol | Value (Default) | Unit | Description |
| :--- | :---: | :--- | :--- | :--- |
| **Field of View** | $\theta$ | `2.0` | deg | The angular extent of the optical sensor. | 
| **Sensor Resolution** | $R_{sensor}$ | `4096` | px | Width of the square camera sensor. |
| **TPU Input Dim** | $dim_{tpu}$ | `224` | px | Required input size for the edge accelerator. |
| **Target Feature Size** | $L_{target}$ | `10.0` | km | Physical size of the ground feature to be detected. |
| **Blindness Threshold** | $\gamma_{blind}$ | `10` | px | Minimum pixel density required to attempt inference. |
| **Restart Threshold** | $\gamma_{restart}$ | `0.65` | % | Battery level required to exit "Recovery Mode" (Naive baseline). |

---
The satellite is modeled as a constrained system with finite battery capacity, solar generation capabilities, and a constant baseload power draw.

| Parameter | Symbol | Script Variable | Unit | Description |
| :--- | :---: | :--- | :--- | :--- |
| **Battery Capacity** | $E_{cap}$ | `battery_capacity_wh` | Joules (J) | Total energy storage (converted from Wh). |
| **Solar Generation** | $P_{solar}$ | `solar_generation_mw` | mW | Max power from solar panels in full sun. |
| **Baseload Power** | $P_{base}$ | `system_baseload_mw` | mW | Idle power (MCU + Radio RX) required to stay alive. |
| **Safety Threshold** | $\gamma_{safe}$ | `min_safe_battery_pct` | % | Minimum battery % before forcing a recharge state. |

---

## 2. Orbital Geometry & Workload

Before running the energy simulation, the script pre-calculates the physical environment and computational demand for every time step $t$ in the orbit.

### A. Ground Velocity & Swath

First, we determine how fast the ground is moving beneath the sensor and how wide the camera's view is.

$$v_{ground} = \sqrt{v_x^2 + v_y^2 + v_z^2}$$

where $v_x$, $v_y$, $v_z$, are all values taken from STK, measured w.r.t the ground-track.

The **Swath Width** ($W$) is calculated using the altitude ($h$) and the sensor Field of View ($\theta$):

$$W = 2 \cdot h \cdot \tan\left(\frac{\theta}{2}\right)$$

### B. Dwell Time

The **Dwell Time** ($T_{dwell}$) represents the time window available to process a specific patch of ground before it leaves the sensor's view.

$$T_{dwell} = \frac{W}{v_{ground}}$$

### C. Computational Demand

The workload is dynamic. The script calculates how many inferences ($N_{req}$) are required based on the ground resolution and target feature size.

1. **Pixels per Patch:** $px_{patch} = \left( \frac{R_{sensor}}{W} \right) \cdot L_{target}$
2. **Inferences per Patch:** If the patch resolution is too low (< `min_pixels`), the demand is 0. Otherwise:
    $$I_{patch} = \left( \frac{px_{patch}}{dim_{tpu}} \right)^2$$
3. **Total Demand:**
    $$N_{req} = \lceil (\text{Patches in View}) \times I_{patch} \rceil$$

---

## 3. Energy Budgeting

For every time step, we calculate an updated energy budget.

### A. Energy Harvesting

Energy gain depends on the solar intensity factor ($\beta_{solar}$), where $\beta \in [0, 1]$. If the satellite is in eclipse, $\beta \approx 0$.
Right now this just 1, I can pull more detail in from STK if desired.

$$E_{harvest} = P_{solar} \cdot \beta_{solar} \cdot T_{dwell}$$

### B. Baseload Consumption

The cost of simply existing during the time step, a :

$$E_{base} = P_{base} \cdot T_{dwell}$$

### C. Available Budget

The energy available for computing ($E_{avail}$) in a specific step is the current battery state plus harvesting, minus the survival cost and switching penalties.

$$E_{avail} = E_{battery}(t) + E_{harvest} - E_{base} - E_{switch}$$

Right now the switching costs are zero, but this is something we can make more accurate after testing the real system.

---

## 4. Sequential Inference Logic

The function `_evaluate_step_sequential` determines how many inferences a specific Deep Learning (DL) model can perform given the constraints.

Given a specific model $M$ with:

* **Latency:** $L_M$ (sec)
* **Energy Cost:** $J_M$ (Joules per inference)
* **Accuracy:** $A_M$ (%)

We calculate two "caps" (limits):

1. **Energy Cap:** Maximum runs before draining $E_{avail}$.
    $$N_{cap}^{E} = \lfloor \frac{E_{avail}}{J_M} \rfloor$$
2. **Time Cap:** Maximum runs that fit in the dwell time.
    $$N_{cap}^{T} = \lfloor \frac{T_{dwell} - T_{switch}}{L_M} \rfloor$$

**Actual Inferences Performed ($N_{actual}$):** Which limit would be hit
$$N_{actual} = \min( N_{req}, N_{cap}^{E}, N_{cap}^{T} )$$

**Step Score:** the number of correct inferences
$$Score = N_{actual} \times A_M$$

---

## 5. Decision Strategies

The simulation compares two primary ways of selecting which model to run.

### A. Dynamic Optimization (Ours)

This uses a greedy algorithm at the moment. At every time step $t$, the satellite:

1. Iterates through *all* available models ($M_1, M_2, ... M_n$).
2. Simulates the step for each model to calculate potential scores.
3. **Recharge Logic:** If $E_{battery}(t) < \gamma_{safe} \cdot E_{cap}$, force the model to "None" (Recharging).
4. **Selection:** Otherwise, select the model $M_{best}$ that maximizes $Score$.

$$M_{best} = \underset{M}{\text{argmax}} ( N_{actual}^{(M)} \cdot A_{M} )$$

The next step here is to create a profile for any given orbit that can be pre-calculated during downtime, or uplinked as necessary, to avoid calculation at every step, as switching only occurs in relatively predictable areas.

### B. Naive Baseline

This represents a standard "dumb" system:

* It selects one fixed model for the entire orbit.
* It assumes maximum throughput is always needed ($N_{req}$ is ignored; it tries to fill $T_{dwell}$).
* **Failure Mode:** If $E_{needed} > E_{avail}$, the battery crashes to 0, and the system enters a "Recovery Mode" where it must charge to 65% before restarting.

### C. Static Baselines (with Constraint knowledge)

* Selects one fixed model.
* Respects the $N_{req}$ (demand) and battery constraints.
* Will simply stop inferencing if energy runs out, rather than crashing the system.

---

## 6. Battery State Propagation

After a decision is made and work is performed, the battery state is updated for the next time step $t+1$.

$$E_{consumed} = E_{base} + E_{switch} + (N_{actual} \cdot J_{M})$$

$$E_{battery}(t+1) = \text{CLIP} \left( E_{battery}(t) + E_{harvest} - E_{consumed}, \quad 0, \quad E_{cap} \right)$$

The battery state can't go above 100% or below 0%

This loop continues until the end of the orbit data.

## Note on Myolnia Velocity

orbit params used:

| Parameter | Symbol |
| :--- | :---: |
| **$\mu** | 398,600 km$^3$/$s^2$ |
| **a** | 26,600 km |
| **$r_p$** | 7271 |

took a quick back of the envelope calc with vis viva

$$v_p = \sqrt{\mu (\frac{2}{r_p} - \frac{1}{a})} \approx 9.7 \text{km/s}$$

This doesn't include the rotation of the earth, but that would only account for some fraction of a km/s
