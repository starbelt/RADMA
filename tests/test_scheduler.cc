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

        {"Grid A0.50 D02", "path1", 10.5569f, 8.3220f, 0.44036f},
        {"Grid A0.50 D04", "path2", 11.1740f, 8.8723f, 0.46773f},
        {"Grid A0.75 D02", "path3", 11.3206f, 9.7816f, 0.48786f},
        {"Grid A0.75 D04", "path4", 12.0460f, 10.3036f, 0.49958f},
        {"Grid A1.00 D02", "path5", 12.2322f, 10.8907f, 0.50646f},
        {"Grid A1.00 D04", "path6", 13.0620f, 12.1607f, 0.52785f},
        {"Grid A1.25 D02", "path7", 12.6660f, 12.3334f, 0.51916f},
        {"Grid A1.25 D06", "path8", 14.6222f, 15.2992f, 0.54724f}};
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

TEST_F(SchedulerTest, NoModelShouldFit)
{
    //  expected
    // 5 Fast
    // 2 med
    // 1 slow
    FrameBudget allconstrained = {5.0f, 2.0f};

    int best_idx = RadmaScheduler::SelectOptimalModel(
        allconstrained, test_models, num_test_models);

    EXPECT_EQ(best_idx, -1);
}
