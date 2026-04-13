// radma_scheduler.h
#pragma once
#include "radma_types.h"

class RadmaScheduler
{
public:
    static int SelectOptimalModel(const FrameBudget &budget,
                                    const ModelProfile *models,
                                    int num_models);
};