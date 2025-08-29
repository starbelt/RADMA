/*#include <cstdio>

#include "../coralmicro/libs/base/led.h"
#include "../coralmicro/third_party/freertos_kernel/include/FreeRTOS.h"
#include "../coralmicro/third_party/freertos_kernel/include/task.h"

extern "C" [[noreturn]] void app_main(void *param) {
  (void)param;
  // Turn on Status LED to show the board is on.
  LedSet(coralmicro::Led::kStatus, true);

  printf("Hello out-of-tree world!\r\n");
  vTaskSuspend(nullptr);
} // hmm
*/

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
#include "coralmicro/libs/camera/camera.h"
#include "coralmicro/libs/tensorflow/detection.h"
#include "coralmicro/libs/tensorflow/utils.h"
#include "coralmicro/libs/tpu/edgetpu_manager.h"
#include "coralmicro/libs/tpu/edgetpu_op.h"
#include "coralmicro/third_party/freertos_kernel/include/FreeRTOS.h"
#include "coralmicro/third_party/freertos_kernel/include/task.h"
#include "coralmicro/third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "coralmicro/third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "coralmicro/third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "coralmicro/libs/base/gpio.h"

// Runs face detection on the Edge TPU, using the on-board camera, printing
//  results to the serial console and turning on the User LED when a face
// is detected.
//
// To build and flash from coralmicro root:
//    bash build.sh
//    python3 scripts/flashtool.py -e face_detection

namespace coralmicro {
namespace {
constexpr char kModelPath[] =
    "coralmicro/models/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite";
constexpr int kTopK = 5;
constexpr float kThreshold = 0.5;

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 8 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

[[noreturn]] void Main() {
  printf("Model Latency Tracing\r\n");
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

  if (interpreter.inputs().size() != 1) {
    printf("ERROR: Model must have only one input tensor\r\n");
    vTaskSuspend(nullptr);
  }

  // Starting Camera.
  CameraTask::GetSingleton()->SetPower(true);
  CameraTask::GetSingleton()->Enable(CameraMode::kStreaming);

  auto* input_tensor = interpreter.input_tensor(0);
  int model_height = input_tensor->dims->data[1];
  int model_width = input_tensor->dims->data[2];

  // Initialize UART-CTS pin as GPIO pin to be traced by Logic Analyzer
  coralmicro::GpioSetMode(coralmicro::kUartCts,coralmicro::GpioMode::kOutput);

  // DWT cycle counter
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  while (true) {

    CameraFrameFormat fmt{CameraFormat::kRgb,
                          CameraFilterMethod::kBilinear,
                          CameraRotation::k270,
                          model_width,
                          model_height,
                          false,
                          tflite::GetTensorData<uint8_t>(input_tensor)};

    if (!CameraTask::GetSingleton()->GetFrame({fmt})) {
      printf("Failed to capture image\r\n");
      vTaskSuspend(nullptr);
    }

    // Start time
    uint32_t t_before_set = DWT->CYCCNT;
    coralmicro::GpioSet(coralmicro::kUartCts, true); // Throw HIGH
    // TickType_t start_tick = xTaskGetTickCount(); // Check Time Since Start

    __DSB(); __ISB();  // ensure visibility
    uint32_t t_after_set = DWT->CYCCNT;


    uint32_t t_start = DWT->CYCCNT;
    if (interpreter.Invoke() != kTfLiteOk) {
      printf("Failed to invoke\r\n");
      vTaskSuspend(nullptr);
    }
    uint32_t t_end = DWT->CYCCNT;

    // end time
    coralmicro::GpioSet(coralmicro::kUartCts, false); // Throw LOW
    uint32_t t_after_clear = DWT->CYCCNT;
    // TickType_t diff = xTaskGetTickCount() - start_tick; // Time since last check

    printf("set_us=%f, invoke_us=%f, total_us=%f\r\n",
      (t_after_set - t_before_set) / (600000000.0 / 1000.0),
      (t_end - t_start) / (600000000.0 / 1000.0),
      (t_after_clear - t_before_set) / (600000000.0 / 1000.0)
      ); // MCU = 600 MHz


    // print to serial
    printf("Inference Latency (RTOS): %lu ms\r\n", diff);
    printf("Inference Latency (CLOCK): %.3f ms\r\n", ms);

    if (auto results =
            tensorflow::GetDetectionResults(&interpreter, kThreshold, kTopK);
        !results.empty()) {
      printf("Found %d result:\r\n%s\r\n", results.size(),
             tensorflow::FormatDetectionOutput(results).c_str());
      LedSet(Led::kUser, true);
    } else {
      LedSet(Led::kUser, false);
    }
  }
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}