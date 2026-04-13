#pragma once
#include <cmath>
#include "radma_types.h"
#include "sensor_configs.h"

// To be generated from models.json
constexpr ModelProfile kAvailableModels[] = {
    {"MobileNet V3", "/models/mobilenet.tflite", 19.017f, 15.228f, 0.737f},
    {"Grid A0.25", "/models/grid.tflite", 9.866f, 7.975f, 0.341f}};
constexpr int kNumModels = sizeof(kAvailableModels) / sizeof(kAvailableModels[0]);

// To be generated from budgets.csv
constexpr FrameBudget kSimulationScenario[] = {
    {100.0f, 50.0f},
    {50.0f, 20.0f},
    // ...
};
constexpr int kNumFrames = sizeof(kSimulationScenario) / sizeof(kSimulationScenario[0]);
const int kNumTiles = floor(kSensorWidth / kModelInputWidth);

