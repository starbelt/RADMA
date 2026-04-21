#include <vector>
#include <cstdio>

#include "inference_model_config.h"
#include "radma_data.h"
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
    // InferenceTask
    // Iterates through every model in kAvailableModels, loads the model,
    // and loops 30 times: initialize interpreter, run inference, destroy.
    // ------------------------------------------------------------------
    [[noreturn]] void InferenceTask(void *pvParameters)
    {
      LedSet(Led::kStatus, true);
      printf("Starting RADMA Model Swaps Test\r\n");
      printf("  %d models available\r\n", kNumModels);

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

      std::vector<uint8_t> cached_model_data;

      for (;;)
      {
        for (int idx = 0; idx < kNumModels; ++idx)
        {
          const ModelProfile &chosen = kAvailableModels[idx];

          printf("\r\n========================================\r\n");
          printf("Testing model [%d]: %s\r\n", idx, chosen.name);

          // Mark RTS high while loading the file from flash (model swap)
          GpioSet(kUartRts, true);
          printf("  Loading %s from LittleFS...\r\n", chosen.file_path);
          if (!LfsReadFile(chosen.file_path, &cached_model_data))
          {
            printf("  ERROR: Failed to load %s\r\n", chosen.file_path);
            GpioSet(kUartRts, false);
            continue; // Skip if load fails
          }
          GpioSet(kUartRts, false);

          for (int i = 0; i < 30; ++i)
          {
            // Scoped interpreter — freed on exit so tensor arena can be cleanly reused
            tflite::MicroInterpreter interpreter(
                tflite::GetModel(cached_model_data.data()), resolver,
                tensor_arena, kTensorArenaSize, &error_reporter);

            if (interpreter.AllocateTensors() != kTfLiteOk)
            {
              printf("ERROR: AllocateTensors() failed on iteration %d\r\n", i);
              break;
            }

            // Static grey input (127)
            auto *input_tensor = interpreter.input_tensor(0);
            std::vector<uint8_t> static_image(input_tensor->bytes, 127);
            memcpy(input_tensor->data.uint8, static_image.data(), input_tensor->bytes);

            // Mark CTS high during inference
            GpioSet(kUartCts, true);
            __DSB();
            __ISB();

            if (interpreter.Invoke() != kTfLiteOk)
            {
              printf("ERROR: Invoke() failed on iteration %d\r\n", i);
            }

            // Mark CTS low after inference
            GpioSet(kUartCts, false);
            __DSB();
            __ISB();
          }

          printf("  Completed 30 inferences for %s. Asserting RTS for 10s...\r\n", chosen.name);
          // Mark RTS high for 10 seconds (super long "done" signal)
          GpioSet(kUartRts, true);
          vTaskDelay(pdMS_TO_TICKS(10000));
          GpioSet(kUartRts, false);
          printf("  Done asserting RTS.\r\n");
        }
        
        printf("\r\n--- All models tested. Repeating in 5 seconds. ---\r\n");
        vTaskDelay(pdMS_TO_TICKS(5000));
      }
    }

    void Main()
    {
      printf("Main() entered (model_swaps)\r\n");
      LedSet(Led::kStatus, true);
      vTaskDelay(pdMS_TO_TICKS(200));
      LedSet(Led::kStatus, false);
      vTaskDelay(pdMS_TO_TICKS(200));
      printf("About to create InferenceTask for model swaps test\r\n");

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
