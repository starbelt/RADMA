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
constexpr int kNumModels = sizeof(kAvailableModels) / sizeof(kAvailableModels[0]);

constexpr FrameBudget kSimulationScenario[] = {
    // msec      mJ
    // scheduler should pick Grid A1.25 D06 (highest accuracy that fits)
    {10000.0f, 10000.0f}, // 20 of each per tile for 4k/224 res/dim

    // scheduler should pick Grid A0.50 D02
    {3674.0f, 3006.0f}, // 11 per, 9 per
};
constexpr int kNumFrames = sizeof(kSimulationScenario) / sizeof(kSimulationScenario[0]);

const int kNumTiles = floor(kSensorWidth * kSensorWidth / kModelInputWidth * kModelInputWidth);
