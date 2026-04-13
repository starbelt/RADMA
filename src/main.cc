#include <vector>
#include <cstdio>

#include "inference_model_config.h"
#include "coralmicro/libs/base/filesystem.h"
#include "coralmicro/libs/base/led.h"
#include "coralmicro/libs/tpu/edgetpu_manager.h"
#include "coralmicro/libs/tpu/edgetpu_op.h"
#include "coralmicro/libs/base/gpio.h"
#include "coralmicro/libs/base/console_m7.h" // Added Console M7 header

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

    constexpr char kModelPath1[] = MODEL_PATH_A075_D02;
    constexpr char kModelPath2[] = MODEL_PATH_A075_D04;

    // Tensor arena (preallocated in SDRAM)
    constexpr int kTensorArenaSize = 8 * 1024 * 1024;
    STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

    TaskHandle_t h = nullptr;

    // Inference task
    [[noreturn]] void InferenceTask(void *pvParameters)
    {
      LedSet(Led::kStatus, true);
      printf("Starting Inference Task\r\n");

      // Setup GPIO pin for timing
      GpioSetMode(kUartCts, GpioMode::kOutput);
      GpioSet(kUartCts, false);
      __DSB();
      __ISB();

      // Enable DWT cycle counter (do this once)
      CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
      DWT->CYCCNT = 0;
      DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

      // Open TPU Context once. It remains open across model swaps.
      auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
      if (!tpu_context)
      {
        printf("ERROR: Failed to get EdgeTpu context\r\n");
        LedSet(Led::kStatus, false);
        vTaskSuspend(nullptr);
      }

      // Setup error reporter and resolver once
      tflite::MicroErrorReporter error_reporter;
      tflite::MicroMutableOpResolver<3> resolver;
      resolver.AddDequantize();
      resolver.AddDetectionPostprocess();
      resolver.AddCustom(kCustomOp, RegisterCustomOp());

      char active_model_char = '1';

      // --- OUTER LOOP: Handles Model Loading & Interpreter Setup ---
      for (;;)
      {
        const char *active_model_path = (active_model_char == '1') ? kModelPath1 : kModelPath2;
        printf("\r\nLoading model: %s\r\n", active_model_path);

        // Load model into vector
        std::vector<uint8_t> model;
        if (!LfsReadFile(active_model_path, &model))
        {
          printf("ERROR: Failed to load %s\r\n", active_model_path);
          vTaskSuspend(nullptr);
        }

        // --- SCOPED BLOCK: Interpreter Lifetime ---
        // The interpreter will be destroyed automatically when we break out of this block
        {
          tflite::MicroInterpreter interpreter(
              tflite::GetModel(model.data()), resolver, tensor_arena, kTensorArenaSize,
              &error_reporter);

          if (interpreter.AllocateTensors() != kTfLiteOk)
          {
            printf("ERROR: AllocateTensors() failed\r\n");
            LedSet(Led::kStatus, false);
            vTaskSuspend(nullptr);
          }

          // Fill a static input
          auto *input_tensor = interpreter.input_tensor(0);
          std::vector<uint8_t> static_image(input_tensor->bytes, 127);
          memcpy(input_tensor->data.uint8, static_image.data(), input_tensor->bytes);

          bool switch_model = false;

          // --- INNER LOOP: Inference Execution & UART Polling ---
          while (!switch_model)
          {

            // 1. Check UART for input (non-blocking)
            char ch;
            int bytes = coralmicro::ConsoleM7::GetSingleton()->Read(&ch, 1);
            if (bytes == 1)
            {
              coralmicro::ConsoleM7::GetSingleton()->Write(&ch, 1); // Echo character

              if (ch == '1' || ch == '2')
              {
                if (ch != active_model_char)
                {
                  active_model_char = ch;
                  switch_model = true; // Trigger the break
                  printf("\r\nSwitching to model %c...\r\n", active_model_char);
                }
                else
                {
                  printf("\r\nModel %c is already running.\r\n", active_model_char);
                }
              }
            }

            // Break inner loop to destroy current interpreter and load the new one
            if (switch_model)
            {
              break;
            }

            // Run Inference & timing GPIO
            const uint32_t t_start = DWT->CYCCNT;
            GpioSet(kUartCts, true);
            __DSB();
            __ISB();

            if (interpreter.Invoke() != kTfLiteOk)
            {
              printf("ERROR: InferenceTask() failed\r\n");
              vTaskSuspend(nullptr);
            }

            GpioSet(kUartCts, false);
            __DSB();
            __ISB();
            const uint32_t t_end = DWT->CYCCNT;

            // Sleep briefly to yield to other tasks
            vTaskDelay(pdMS_TO_TICKS(10));

#if defined(SystemCoreClock)
            const double cpu_hz = static_cast<double>(SystemCoreClock);
#else
            const double cpu_hz = 800e6;
#endif
            const double ms = static_cast<double>(t_end - t_start) / (cpu_hz / 1000.0);
            printf("invoke_ms=%.3f\r\n", ms);

          } // End of Inner Loop
        } // End of Scoped Block (interpreter is de-initialized here)
      } // End of Outer Loop
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