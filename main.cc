// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include "coralmicro/libs/base/filesystem.h"
#include "coralmicro/libs/base/led.h"
// #include "coralmicro/libs/camera/camera.h"
// #include "coralmicro/libs/tensorflow/detection.h"
#include "coralmicro/libs/tensorflow/utils.h"
#include "coralmicro/libs/tpu/edgetpu_manager.h"
#include "coralmicro/libs/tpu/edgetpu_op.h"
#include "coralmicro/third_party/freertos_kernel/include/FreeRTOS.h"
#include "coralmicro/third_party/freertos_kernel/include/task.h"
#include "coralmicro/third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "coralmicro/third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "coralmicro/third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "coralmicro/libs/base/gpio.h"

enum GpioSignal {GPIO_START, GPIO_END};

static QueueHandle_t gpioQueue;

namespace coralmicro {
namespace {
constexpr char kModelPath[] =
    "coralmicro/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite";
constexpr int kTopK = 5;
constexpr float kThreshold = 0.5;

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 8 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

  // High-priority GPIO task
  void GpioTask(void* param) {
    GpioSignal sig;
    while (true) {
      if (xQueueReceive(gpioQueue, &sig, portMAX_DELAY) == pdPASS) {
        if (sig == GPIO_START) {
          coralmicro::GpioSet(coralmicro::kUartCts, true);
        } else if (sig == GPIO_END) {
          coralmicro::GpioSet(coralmicro::kUartCts, false);
        }
        // Data sync barrier to make sure it’s really visible
        __DSB(); __ISB();
      }
    }
  }


[[noreturn]] void InferenceTask() {
  //printf("Model Latency Tracing\r\n");
  // Turn on Status LED to show the board is on.
  LedSet(Led::kStatus, true);

  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath, &model)) {
    printf("ERROR: Failed to load %s\r\n", kModelPath);
    vTaskSuspend(nullptr);
  }

  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice(); // initialize TPU
  if (!tpu_context) {
    printf("ERROR: Failed to get EdgeTpu context\r\n");
    vTaskSuspend(nullptr);
  }

  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<3> resolver;
  resolver.AddDequantize();
  resolver.AddDetectionPostprocess();
  resolver.AddCustom(kCustomOp, RegisterCustomOp());

  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       tensor_arena, kTensorArenaSize,
                                       &error_reporter);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors() failed\r\n");
    vTaskSuspend(nullptr);
  }

  // Create a dummy image to feed into the model
  auto* input_tensor = interpreter.input_tensor(0);
  std::vector<uint8_t> static_image(input_tensor->bytes, 127);  // fill with gray
  memcpy(input_tensor->data.uint8, static_image.data(), input_tensor->bytes);

  // Initialize UART-CTS pin as GPIO pin to be traced by Logic Analyzer
  coralmicro::GpioSetMode(coralmicro::kUartCts,coralmicro::GpioMode::kOutput);

  // DWT cycle counter
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  while (true) {
    // Test Salaea with RTS pin - results in ~1.3 microseconds total delay
    // I don't think this would account for the ~20ms delay observed.

    // Start time
    GpioSignal start = GPIO_START;
    xQueueSend(gpioQueue, &start, 0);

    const uint32_t t_start = DWT->CYCCNT; // Check Cycles before Invoke
    // Call for inference
    if (interpreter.Invoke() != kTfLiteOk) {
      printf("Failed to invoke\r\n");
      vTaskSuspend(nullptr);
    }
    const uint32_t t_end = DWT->CYCCNT; // Check Cycles after Invoke

    // End time
    GpioSignal end = GPIO_END;
    xQueueSend(gpioQueue, &end,0);

    printf("invoke_ms=%f\r\n",
      (t_end - t_start) / (800000000.0 / 1000.0) // calculate time from cycles
      ); // MCU = 800 MHz as per coral datasheet

  }
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main() {
  gpioQueue = xQueueCreate(4,sizeof(GpioSignal));
  xTaskCreate(GpioTask, "GpioTask", 2048, nullptr, configMAX_PRIORITIES-1, nullptr);
  xTaskCreate(InferenceTask, "InferenceTask", 8192, nullptr, configMAX_PRIORITIES-2, nullptr);
}