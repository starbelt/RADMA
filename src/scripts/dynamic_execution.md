# Logic Explanation: Dynamic Orbit Optimization

The core goal is to maximize **Total Accurate Inferences** ($N_{acc}$) across an orbit by dynamically switching CNN models based on changing physical constraints.

### 1. Workload Calculation (The Regime Switch)
Unlike static video processing, satellite ground sampling can change drastically with altitude. We use a **Constant Ground Feature Size** assumption (e.g., we always want to detect 5km objects).

The number of inferences required ($N_{req}$) is derived from the Pixel Density ($P_d$):

$$P_d = \frac{\text{Sensor Res}}{\text{Swath Width}} \times \text{Target Size}_{km}$$

This creates two distinct operating regimes:

1.  **Tiling Regime ($P_d > \text{TPU}_{dim}$):** At low altitudes (Perigee), the target patch is larger than the TPU input (224px). We must chop the patch into multiple tiles.
    * Workload per patch $> 1$.
2.  **Aggregation Regime ($P_d < \text{TPU}_{dim}$):** At high altitudes (Apogee), the target patch is tiny. We can fit multiple ground patches into a single TPU input.
    * Workload per patch $< 1$.

### 2. Constraints (The Budgets)
For every time step $t$, we calculate two hard limits:

* **Time Budget ($L_t$):** The dwell time before the ground moves out of the Field of View.
    $$L_t = \frac{\text{Swath Width}}{V_{ground}}$$
* **Energy Budget ($L_e$):** The total energy available for processing during that dwell time.
    $$L_e = P_{avail} \times L_t$$
    * *Note:* $P_{avail}$ drops significantly during Eclipse (180°-360° True Anomaly).

### 3. Model Selection (The Optimization)
For each frame, we test every available model $M_i$. We calculate the maximum **integer** number of inferences ($N_{cap}$) the hardware can perform given the model's Latency ($Lat_i$) and Energy Cost ($Eng_i$):

$$N_{cap} = \min \left( \left\lfloor \frac{L_t - C_{switch}}{Lat_i} \right\rfloor, \left\lfloor \frac{L_e - C_{switch}}{Eng_i} \right\rfloor \right)$$

* *Where $C_{switch}$ is the hysteresis penalty applied if switching models. Right now this is zero but I'd like to get a read on this on the hardwre once I get around to the actual RTOS app to follow this offboard python logic.*

The actual inferences performed ($N_{actual}$) is the lesser of the hardware capacity and the job requirement:

$$N_{actual} = \min(N_{cap}, N_{req})$$

Finally, the **Score** for the model is:

$$\text{Score}_i = N_{actual} \times \text{Accuracy}_i$$

The system simply selects $\arg \max (\text{Score}_i)$ for every step.
