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
  //printf("Model Latency Tracing\r\n");
  // Turn on Status LED to show the board is on.
  // LedSet(Led::kStatus, true);

  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath, &model)) {
    printf("ERROR: Failed to load %s\r\n", kModelPath);
    vTaskSuspend(nullptr);
  }
  // Make dummy gray image
  uint8_t static_image[320*320*3] = {127};

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
  memcpy(input->data.uint8, static_image, input->bytes);
  int model_height = input_tensor->dims->data[1];
  int model_width = input_tensor->dims->data[2];

  // Initialize UART-CTS pin as GPIO pin to be traced by Logic Analyzer
  coralmicro::GpioSetMode(coralmicro::kUartCts,coralmicro::GpioMode::kOutput);
  // coralmicro::GpioSetMode(coralmicro::Gpio::kUartRts, coralmicro::GpioMode::kOutput);

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

    // Test Salaea with RTS pin - results in ~1.3 microseconds total delay
    coralmicro::GpioSet(kUartRts, true);
    uint32_t t_rts0 = DWT->CYCCNT;
    __DSB(); __ISB();
    volatile bool rbtest = coralmicro::GpioGet(coralmicro::kUartRts); (void)rbtest;
    coralmicro::GpioSet(coralmicro::kUartRts, false);
    uint32_t t_rts1 = DWT->CYCCNT;
    printf("Salaea Test Comparison: %f ms",(t_rts1-t_rts0)/((800000000.0 / 1000.0)));


    // Start time
    coralmicro::GpioSet(coralmicro::kUartCts, true); // Throw HIGH for Saleae
    __DSB(); __ISB();  // ensure visibility
    volatile bool rb = coralmicro::GpioGet(coralmicro::kUartCts);
    (void)rb;

    const uint32_t t_start = DWT->CYCCNT; // Check Cycles before Invoke
    if (interpreter.Invoke() != kTfLiteOk) {
      printf("Failed to invoke\r\n");
      vTaskSuspend(nullptr);
    }
    const uint32_t t_end = DWT->CYCCNT; // Check Cycles after Invoke

    // End time
    coralmicro::GpioSet(coralmicro::kUartCts, false); // Throw LOW for Saleae
    __DSB(); __ISB();
    volatile bool rb2 = coralmicro::GpioGet(coralmicro::kUartCts); (void)rb2;

    printf("invoke_ms=%f\r\n",
      (t_end - t_start) / (800000000.0 / 1000.0) // Calculate time from Cycles
      ); // MCU = 800 MHz as per coral datasheet


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