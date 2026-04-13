#pragma once
#include <cstdint>

struct ModelProfile
{
    const char *name;
    const char *file_path;
    float time_ms;
    float energy_mj;
    float accuracy; // 0.0 to 1.0
};

struct FrameBudget
{
    float time_budget_ms;
    float energy_budget_mj;
};

