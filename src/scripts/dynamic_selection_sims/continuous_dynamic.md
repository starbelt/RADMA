# Satellite Edge Computing: Continuous Simulation Logic

This document outlines the mathematical models and control logic used in the updated `ContinuousSatSim` Python class. The simulation transitions from discrete "dwell-based" steps to a continuous time-integration model ($\Delta t$), introducing a **Workload Queue (Buffer)** to decouple data acquisition from processing.

---

## 1. System Configuration & Constants

These parameters define the sensor capabilities, system constraints, and the simulation time step.

| Parameter | Symbol | Script Variable | Unit | Description |
| :--- | :---: | :--- | :--- | :--- |
| **Sim Time Step** | $\Delta t$ | `sim_dt_s` | s | Integration step size (default 1.0s). |
| **Buffer Capacity** | $B_{max}$ | `buffer_max_images` | Frames | Maximum size of the input queue before data is dropped. |
| **Solar Generation** | $P_{solar}$ | `solar_generation_mw` | mW | Max power from solar panels in sunlight. |
| **Baseload Power** | $P_{base}$ | `system_baseload_mw` | mW | Idle power (MCU + Radio RX) required to stay alive. |
| **Battery Capacity** | $E_{cap}$ | `battery_capacity_wh` | Wh | Total energy storage. |
| **Critical Threshold** | $\gamma_{crit}$ | `0.05` | % | Battery level triggering emergency power saving. |

---

## 2. Orbital Geometry & Workload Arrival

Unlike the previous logic which calculated total demand per "ground patch," this model calculates an instantaneous **Arrival Rate** ($\lambda$) of work.

### A. Ground Velocity & Swath
We determine the instantaneous velocity of the ground track relative to the sensor and the sensor's physical coverage width.

$$v_{ground}(t) = ||\vec{v}_{sat} - \text{proj}_{\vec{r}}(\vec{v}_{sat})||$$
*(The magnitude of the velocity vector projected onto the plane perpendicular to the nadir vector)*

The **Swath Width** ($W$) at altitude $h(t)$:

$$W(t) = \frac{h(t) \cdot P_{pitch}}{f_{len}} \cdot R_{sensor}$$

### B. Workload Arrival Rate ($\lambda$)
The rate at which new imagery (and thus inference demand) enters the system depends on how fast we sweep new area.

1. **Area Rate:** $\dot{A}(t) = W(t) \cdot v_{ground}(t)$
2. **Tiles per Second:** $\dot{T}(t) = \frac{\dot{A}(t)}{L_{target}^2}$
3. **Inferences per Tile:** $I_{tile}$ (Function of Ground Sample Distance vs TPU Input Dim).

The **Arrival Rate** (Inferences per second) is:
$$\lambda(t) = \dot{T}(t) \cdot I_{tile}(t)$$

---

## 3. Buffer Dynamics (Queue Theory)

The system now implements a First-In-First-Out (FIFO) queue. Work arrives at rate $\lambda(t)$ and is processed at service rate $\mu(t)$.

### A. Queue Evolution
For a time step $\Delta t$:

$$Work_{in} = \lambda(t) \cdot \Delta t$$

The buffer state $B$ (in frames/inferences) evolves as:

$$B(t+\Delta t) = \min \left( B(t) + Work_{in} - Work_{processed}, \quad B_{max} \right)$$

*Note: If $B(t) + Work_{in} > B_{max}$, the excess work is dropped (Overflow).*

---

## 4. Energy Model & Events

Energy is integrated continuously. We now handle binary lighting (Sunlight/Eclipse) and external power events.

### A. Solar Input (Binary)
Using STK lighting intervals, the solar factor $S(t)$ is binary:
$$
S(t) = 
\begin{cases} 
1 & \text{if } t \in \text{Sunlight Intervals} \\
0 & \text{otherwise (Eclipse/Umbra)}
\end{cases}
$$

$$P_{in}(t) = P_{solar} \cdot S(t)$$

### B. Power Consumption & Disturbances
The total load includes baseload, dynamic processing power, and external events (e.g., Radio TX).

$$P_{load}(t) = P_{base} + P_{processing}(t) + P_{event}(t)$$

**Event Logic:**
If an external event $E$ is active at time $t$:
* $P_{event}(t) = E_{power}$
* **CPU Block:** If $E_{blocked} = \text{True}$, then processing is halted ($\mu(t) = 0$).

---

## 5. Control Policy (State Machine)

Instead of optimizing every step, the satellite uses a robust **Priority-Based State Machine** to select the active AI Model.

**Available Models:**
* **Fast Model:** High Throughput ($\mu_{fast}$), High Power.
* **Eco Model:** High Efficiency ($\eta_{eco}$), Lower Power.

### State Selection Logic

At each time step $t$, the system evaluates conditions in this specific order:

1.  **Hardware Interlock:**
    * IF `CPU_Blocked` (Event active): $\rightarrow$ **State: BLOCKED** (No processing).

2.  **Safety Limits:**
    * IF $E_{batt} \le 0$: $\rightarrow$ **State: DEAD** (System crash).
    * IF $E_{batt} < \gamma_{crit}$:
        * IF $B(t) > 0.9 \cdot B_{max}$ (Buffer nearly overflowing): $\rightarrow$ **State: CRIT_DRAIN** (Use Eco Model to clear space).
        * ELSE: $\rightarrow$ **State: RECHARGE** (Idle to recover energy).

3.  **Buffer Management (Nominal Ops):**
    * IF $B(t) > 0.4 \cdot B_{max}$: $\rightarrow$ **State: FAST** (Use Fast Model to drain queue).
    * IF $B(t) > 0$: $\rightarrow$ **State: ECO** (Use Efficient Model to process steady state).
    * ELSE: $\rightarrow$ **State: IDLE** (Buffer empty, save power).

### Processing & Physics Update
Once a model $M$ is selected, the **Service Rate** ($\mu$) is:
$$\mu(t) = \frac{1}{\text{Latency}_M}$$

**Work Done:**
$$Work_{processed} = \min \left( B(t), \quad \mu(t) \cdot \Delta t \right)$$

**Energy Consumed:**
$$E_{consumed} = Work_{processed} \cdot \text{EnergyPerInf}_M$$
$$P_{processing} = \frac{E_{consumed}}{\Delta t}$$

---

## 6. Battery Integration

The battery state is updated using Euler integration:

$$E_{batt}(t+\Delta t) = \text{CLIP} \left( E_{batt}(t) + (P_{in}(t) - P_{load}(t)) \cdot \Delta t, \quad 0, \quad E_{cap} \right)$$