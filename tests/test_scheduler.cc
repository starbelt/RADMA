#include <gtest/gtest.h>
#include "radma_scheduler.h"
#include "radma_types.h"
#include "sensor_configs.h"

class SchedulerTest : public ::testing::Test
{
protected:
    ModelProfile test_models[3] = {
        {"Fast_LowAcc", "path1", 10.0f, 5.0f, 0.40f},
        {"Med_MedAcc", "path2", 20.0f, 15.0f, 0.60f},
        {"Slow_HighAcc", "path3", 50.0f, 40.0f, 0.90f}
    };
    int num_test_models = 3;
};

TEST_F(SchedulerTest, SelectHighestYieldUnderTimeConstraint)
{
    //  expected
    // 5 Fast
    // 2 med
    // 1 slow
    FrameBudget timeconstrained = {50.0f, 1000.0f};

    int best_idx = RadmaScheduler::SelectOptimalModel(
        timeconstrained, test_models, num_test_models);

    EXPECT_EQ(best_idx, 0);
}

TEST_F(SchedulerTest, SelectHighestYieldUnderTimeConstraint)
{
    //  expected
    // 5 Fast
    // 2 med
    // 1 slow
    FrameBudget timeconstrained = {5.0f, 2.0f};

    int best_idx = RadmaScheduler::SelectOptimalModel(
        timeconstrained, test_models, num_test_models);

    EXPECT_EQ(best_idx, -1);
}
