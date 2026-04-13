// radma_scheduler.cc
#include "radma_scheduler.h"
#include "radma_data.h"
#include "sensor_configs.h"

#include <cmath>
#include <cfloat>

/*
SelectOptimalModel
------------------
Picks the model that maximises *total correct inferences* within a
single frame, subject to both a time and an energy cap.

Objective:
    score(m) = n(m) * accuracy(m)

where n(m) is the number of complete inferences the model can
execute before either budget (time or energy) is exhausted,
further capped at kNumTiles (the number of non-overlapping
sensor tiles per frame):

    n(m) = min( floor(time_budget  / time_ms),
                floor(energy_budget / energy_mj),
                kNumTiles )

Returns the index of the winning model, or -1 if no model fits
within the supplied budget.
 */

int RadmaScheduler::SelectOptimalModel(const FrameBudget &budget,
                                       const ModelProfile *models,
                                        int num_models)
{
    int   best_idx   = -1;
    float best_score = -1.0f;

    for (int i = 0; i < num_models; ++i)
    {
        const ModelProfile &m = models[i];
        if (m.time_ms <= 0.0f || m.energy_mj <= 0.0f)
            continue;

        // Maximum inferences possible under each constraint
        const int n_time   = static_cast<int>(budget.time_budget_ms   / m.time_ms);
        const int n_energy = static_cast<int>(budget.energy_budget_mj / m.energy_mj);

        // The lesser of the two (also capped by the number of useful sensor tiles)
        int n = n_time < n_energy ? n_time : n_energy;
        if (n > kNumTiles) n = kNumTiles;

        // If no complete inference fits within this budget, skip
        if (n <= 0)
            continue;

        const float score = static_cast<float>(n) * m.accuracy;

        if (score > best_score)
        {
            best_score = score;
            best_idx   = i;
        }
    }

    return best_idx; // -1 signals "no model fits"
}
