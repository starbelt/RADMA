#include <vector>
#include <cstdio>

#include "inference_model_config.h"
#include "radma_data.h"
#include "radma_scheduler.h"
#include "coralmicro/libs/base/filesystem.h"
#include "coralmicro/libs/base/led.h"
#include "coralmicro/libs/tpu/edgetpu_manager.h"
#include "coralmicro/libs/tpu/edgetpu_op.h"
#include "coralmicro/libs/base/gpio.h"
#include "coralmicro/libs/base/console_m7.h"

#include "coralmicro/third_party/freertos_kernel/include/FreeRTOS.h"
#include "coralmicro/third_party/freertos_kernel/include/task.h"

#include "coralmicro/libs/tensorflow/utils.h"
#include "coralmicro/third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "coralmicro/third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "coralmicro/third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace coralmicro
{
  namespace
  {
    // Tensor arena (preallocated in SDRAM)
    constexpr int kTensorArenaSize = 8 * 1024 * 1024;
    STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

    TaskHandle_t h = nullptr;

    // ------------------------------------------------------------------
    // RunFrame
    // Loads the cached model data into the interpreter, runs exactly
    // `n_inferences` invocations, and prints per-invoke timing.
    // Returns the total accumulated inference time.
    // ------------------------------------------------------------------
    float RunFrame(const ModelProfile &profile,
                  const std::vector<uint8_t> &model_data,
                  int n_inferences,
                  tflite::MicroErrorReporter &error_reporter,
                  tflite::MicroMutableOpResolver<3> &resolver)
    {
      printf("\r\n--- Frame: model=%s  n=%d ---\r\n",
            profile.name, n_inferences);

      // Scoped interpreter — freed on exit so tensor arena can be cleanly reused
      {
        tflite::MicroInterpreter interpreter(
            tflite::GetModel(model_data.data()), resolver,
            tensor_arena, kTensorArenaSize, &error_reporter);

        if (interpreter.AllocateTensors() != kTfLiteOk)
        {
          printf("ERROR: AllocateTensors() failed\r\n");
          return 0.0f;
        }

        // Static grey input (127) — identical to the old harness
        auto *input_tensor = interpreter.input_tensor(0);
        std::vector<uint8_t> static_image(input_tensor->bytes, 127);
        memcpy(input_tensor->data.uint8, static_image.data(), input_tensor->bytes);

        float total_ms = 0.0f;

#if defined(SystemCoreClock)
        const double cpu_hz = static_cast<double>(SystemCoreClock);
#else
        const double cpu_hz = 800e6;
#endif

        for (int tile = 0; tile < n_inferences; ++tile)
        {
          GpioSet(kUartCts, true);
          __DSB();
          __ISB();
          const uint32_t t0 = DWT->CYCCNT;

          if (interpreter.Invoke() != kTfLiteOk)
          {
            printf("ERROR: Invoke() failed on tile %d\r\n", tile);
            break;
          }

          const uint32_t t1 = DWT->CYCCNT;
          GpioSet(kUartRts, false);
          __DSB();
          __ISB();

          const double ms = static_cast<double>(t1 - t0) / (cpu_hz / 1000.0);
          total_ms += static_cast<float>(ms);
          printf("  tile %2d / %2d  invoke_ms=%.3f\r\n", tile + 1, n_inferences, ms);
        }

        printf("  frame_total_ms=%.3f  expected_correct=%.2f\r\n",
               total_ms, static_cast<float>(n_inferences) * profile.accuracy);

        return total_ms;

      } // interpreter destroyed here — tensor arena freed
    }

    // ------------------------------------------------------------------
    // InferenceTask
    // Iterates through every scenario in kSimulationScenario, asks the
    // scheduler which model to use, handles flash I/O caching, and runs.
    // ------------------------------------------------------------------
    [[noreturn]] void InferenceTask(void *pvParameters)
    {
      LedSet(Led::kStatus, true);
      printf("Starting RADMA Inference Task\r\n");
      printf("  %d models available,  %d tiles/frame\r\n",
            kNumModels, kNumTiles);

      // GPIO for external timing probe
      GpioSetMode(kUartCts, GpioMode::kOutput);
      GpioSet(kUartCts, false);
      GpioSetMode(kUartRts, GpioMode::kOutput);
      GpioSet(kUartRts, false);
      __DSB();
      __ISB();

      // DWT cycle counter for high-resolution timing
      CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
      DWT->CYCCNT = 0;
      DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

      // Open TPU context once — kept alive for the lifetime of the task
      auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
      if (!tpu_context)
      {
        printf("ERROR: Failed to open EdgeTpu context\r\n");
        LedSet(Led::kStatus, false);
        vTaskSuspend(nullptr);
      }

      tflite::MicroErrorReporter error_reporter;
      tflite::MicroMutableOpResolver<3> resolver;
      resolver.AddDequantize();
      resolver.AddDetectionPostprocess();
      resolver.AddCustom(kCustomOp, RegisterCustomOp());

      // --- State tracking for model caching ---
      std::vector<uint8_t> cached_model_data;
      int current_model_idx = -1;


      for (;;)
      {
        for (int frame = 0; frame < kNumFrames; ++frame)
        {
          const FrameBudget &budget = kSimulationScenario[frame];

          printf("\r\n========================================\r\n");
          printf("Frame %d  budget: time=%.1f ms  energy=%.1f mJ\r\n",
                frame, budget.time_budget_ms, budget.energy_budget_mj);

          GpioSet(kUartCts, true);
          // The SCHEDULER!!
          const int idx = RadmaScheduler::SelectOptimalModel(
              budget, kAvailableModels, kNumModels);

          if (idx < 0)
          {
            printf("WARNING: No model fits this budget — skipping frame.\r\n");
            continue;
          }

          const ModelProfile &chosen = kAvailableModels[idx];

          // Compute how many inferences the scheduler would allocate
          const int n_time = static_cast<int>(budget.time_budget_ms / chosen.time_ms);
          const int n_energy = static_cast<int>(budget.energy_budget_mj / chosen.energy_mj);
          int n = (n_time < n_energy) ? n_time : n_energy;
          if (n > kNumTiles)
            n = kNumTiles;

          printf("Scheduler selected: [%d] %s\r\n", idx, chosen.name);
          printf("  time/inf=%.2f ms  energy/inf=%.2f mJ  acc=%.1f%%\r\n",
                 chosen.time_ms, chosen.energy_mj, chosen.accuracy * 100.0f);
          printf("  -> running %d inference(s) this frame\r\n", n);

          // --- Flash I/O Caching Logic ---
          if (idx != current_model_idx)
          {
            printf("  Switching models: Loading %s from LittleFS...\r\n", chosen.file_path);

            // Record start time of the flash read if you want to benchmark the swap cost later
            // const uint32_t load_t0 = DWT->CYCCNT;

            if (!LfsReadFile(chosen.file_path, &cached_model_data))
            {
              printf("ERROR: Failed to load %s\r\n", chosen.file_path);
              continue; // Skip frame if load fails
            }
            current_model_idx = idx;
          }
          else
          {
            printf("  Model %s already active in memory. Skipping LittleFS read.\r\n", chosen.name);
          }
          GpioSet(kUartRts, false);

          // Pass the dynamically cached buffer into the frame runner
          RunFrame(chosen, cached_model_data, n, error_reporter, resolver);

          // Brief pause between frames so the serial log is readable
          vTaskDelay(pdMS_TO_TICKS(500));
        }
      }
    }

    void Main()
    {
      printf("Main() entered\r\n");
      LedSet(Led::kStatus, true);
      vTaskDelay(pdMS_TO_TICKS(200));
      LedSet(Led::kStatus, false);
      vTaskDelay(pdMS_TO_TICKS(200));
      printf("About to create InferenceTask\r\n");

      BaseType_t rc = xTaskCreate(InferenceTask, "InferenceTask", 16384, nullptr,
                                  configMAX_PRIORITIES - 1, &h);
      printf("xTaskCreate returned %ld, handle %p\r\n", (long)rc, h);

      vTaskSuspend(nullptr);
    }

  } // namespace
} // namespace coralmicro

extern "C" void app_main(void *param)
{
  (void)param;
  coralmicro::Main();
}