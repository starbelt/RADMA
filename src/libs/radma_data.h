#pragma once
#include <cmath>
#include "radma_types.h"
#include "sensor_configs.h"
#include "inference_model_config.h"


constexpr ModelProfile kAvailableModels[] = {
    // name                 file_path              time_ms     energy_mj   accuracy
    {"Grid A0.50 D02", MODEL_PATH_A050_D02, 10.5569f, 8.3220f, 0.44036f},
    {"Grid A0.50 D04", MODEL_PATH_A050_D04, 11.1740f, 8.8723f, 0.46773f},
    {"Grid A0.75 D02", MODEL_PATH_A075_D02, 11.3206f, 9.7816f, 0.48786f},
    {"Grid A0.75 D04", MODEL_PATH_A075_D04, 12.0460f, 10.3036f, 0.49958f},
    {"Grid A1.00 D02", MODEL_PATH_A100_D02, 12.2322f, 10.8907f, 0.50646f},
    {"Grid A1.00 D04", MODEL_PATH_A100_D04, 13.0620f, 12.1607f, 0.52785f},
    {"Grid A1.25 D02", MODEL_PATH_A125_D02, 12.6660f, 12.3334f, 0.51916f},
    {"Grid A1.25 D06", MODEL_PATH_A125_D06, 14.6222f, 15.2992f, 0.54724f},
};


/* // Model Swap Cost Full Profile
constexpr ModelProfile kAvailableModels[] = {
    name                 file_path              time_ms     energy_mj   accuracy
    {"Grid A0.25 D02", MODEL_PATH_A025_D02, 9.8668f, 7.9753f, 0.34183f},
    {"Grid A0.25 D04", MODEL_PATH_A025_D04, 10.3984f, 8.1933f, 0.32378f},
    {"Grid A0.25 D06", MODEL_PATH_A025_D06, 10.8927f, 8.7264f, 0.36517f},
    {"Grid A0.25 D08", MODEL_PATH_A025_D08, 11.3700f, 9.0669f, 0.36060f},
    {"Grid A0.25 D10", MODEL_PATH_A025_D10, 11.8857f, 9.0575f, 0.36468f},
    {"Grid A0.25 D12", MODEL_PATH_A025_D12, 12.3791f, 10.4075f, 0.35735f},

    {"Grid A0.50 D02", MODEL_PATH_A050_D02, 10.5569f, 8.3220f, 0.44036f},
    {"Grid A0.50 D04", MODEL_PATH_A050_D04, 11.1740f, 8.8723f, 0.46773f},
    {"Grid A0.50 D06", MODEL_PATH_A050_D06, 11.7427f, 9.5461f, 0.45425f},
    {"Grid A0.50 D08", MODEL_PATH_A050_D08, 12.3658f, 10.3291f, 0.47133f},
    {"Grid A0.50 D10", MODEL_PATH_A050_D10, 12.9490f, 10.7017f, 0.46459f},
    {"Grid A0.50 D12", MODEL_PATH_A050_D12, 13.5096f, 11.9914f, 0.47004f},

    {"Grid A0.75 D02", MODEL_PATH_A075_D02, 11.3206f, 9.7816f, 0.48786f},
    {"Grid A0.75 D04", MODEL_PATH_A075_D04, 12.0460f, 10.3036f, 0.49958f},
    {"Grid A0.75 D06", MODEL_PATH_A075_D06, 12.7000f, 11.2285f, 0.50325f},
    {"Grid A0.75 D08", MODEL_PATH_A075_D08, 13.4150f, 12.3815f, 0.51503f},
    {"Grid A0.75 D10", MODEL_PATH_A075_D10, 14.1001f, 12.8386f, 0.50482f},
    {"Grid A0.75 D12", MODEL_PATH_A075_D12, 14.7811f, 13.8573f, 0.50906f},

    {"Grid A1.00 D02", MODEL_PATH_A100_D02, 12.2322f, 10.8907f, 0.50646f},
    {"Grid A1.00 D04", MODEL_PATH_A100_D04, 13.0620f, 12.1607f, 0.52785f},
    {"Grid A1.00 D06", MODEL_PATH_A100_D06, 13.9099f, 13.4152f, 0.53717f},
    {"Grid A1.00 D08", MODEL_PATH_A100_D08, 14.7406f, 14.1624f, 0.53387f},
    {"Grid A1.00 D10", MODEL_PATH_A100_D10, 15.5851f, 15.0864f, 0.52352f},
    {"Grid A1.00 D12", MODEL_PATH_A100_D12, 16.4503f, 16.8669f, 0.53120f},

    {"Grid A1.25 D02", MODEL_PATH_A125_D02, 12.6660f, 12.3334f, 0.51916f},
    {"Grid A1.25 D04", MODEL_PATH_A125_D04, 13.6628f, 13.7663f, 0.53506f},
    {"Grid A1.25 D06", MODEL_PATH_A125_D06, 14.6222f, 15.2992f, 0.54724f},
    {"Grid A1.25 D08", MODEL_PATH_A125_D08, 15.6588f, 16.3142f, 0.54451f},
    {"Grid A1.25 D10", MODEL_PATH_A125_D10, 43.2228f, 38.6851f, 0.54310f},
    {"Grid A1.25 D12", MODEL_PATH_A125_D12, 75.2953f, 66.3131f, 0.53944f},

    {"Grid A1.50 D02", MODEL_PATH_A150_D02, 13.4214f, 13.1372f, 0.53763f},
    {"Grid A1.50 D04", MODEL_PATH_A150_D04, 30.9005f, 28.4696f, 0.54505f},
    {"Grid A1.50 D06", MODEL_PATH_A150_D06, 75.7001f, 59.4048f, 0.55421f},
    {"Grid A1.50 D08", MODEL_PATH_A150_D08, 120.8858f, 92.8045f, 0.55563f},
    {"Grid A1.50 D10", MODEL_PATH_A150_D10, 166.6874f, 134.1166f, 0.55704f},
    {"Grid A1.50 D12", MODEL_PATH_A150_D12, 212.4300f, 169.3063f, 0.55257f}
};
*/

constexpr int kNumModels = sizeof(kAvailableModels) / sizeof(kAvailableModels[0]);

constexpr FrameBudget kSimulationScenario[] = {
    // msec      mJ
    // scheduler should pick Grid A1.25 D06 (highest accuracy that fits)
    {10000.0f, 10000.0f}, // 20 of each per tile for 4k/224 res/dim

    // scheduler should pick Grid A0.50 D02
    {3674.0f, 3006.0f}, // 11 per, 9 per
};
constexpr int kNumFrames = sizeof(kSimulationScenario) / sizeof(kSimulationScenario[0]);

const int kNumTiles = floor(kSensorWidth * kSensorWidth / (kModelInputWidth * kModelInputWidth));
