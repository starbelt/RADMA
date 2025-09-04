#include <vector>

#include "../coralmicro/libs/base/filesystem.h"
#include "../coralmicro/libs/base/led.h"
#include "../coralmicro/libs/tpu/edgetpu_manager.h"
#include "../coralmicro/libs/tpu/edgetpu_op.h"
#include "../coralmicro/libs/base/gpio.h"

#include "../coralmicro/third_party/freertos_kernel/include/FreeRTOS.h"
#include "../coralmicro/third_party/freertos_kernel/include/task.h"

#include "../coralmicro/libs/tensorflow/utils.h"
#include "../coralmicro/third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "../coralmicro/third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "../coralmicro/third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace coralmicro {
namespace {
// Path to model inside the image
constexpr char kModelPath[] =
    "models/Image_Classification/EfficientNet/S/efficientnet-edgetpu-S_quant_edgetpu.tflite";

// Tensor arena (preallocated in SDRAM)
constexpr int kTensorArenaSize = 12 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

// task handle for GPIO task
TaskHandle_t gpioTaskHandle = nullptr;
TaskHandle_t inferenceTaskHandle = nullptr;

// GPIO task (highest priority)
[[noreturn]] void GpioTask(void* pvParameters) {
  GpioSetMode(kUartCts, GpioMode::kOutput); // uses CTS pin (left of camera)
  GpioSet(kUartCts, false);
  __DSB(); __ISB();

  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    GpioSet(kUartCts, true);
    __DSB(); __ISB();
    xTaskNotifyGive(inferenceTaskHandle); // notify the inference task

    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    GpioSet(kUartCts, false);
    __DSB(); __ISB();
    xTaskNotifyGive(inferenceTaskHandle); // notify the inference task
  }
}

// Inference task (lower priority)
[[noreturn]] void InferenceTask(void* pvParameters) {
  LedSet(Led::kStatus, true); // easy "on" check

  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath, &model)) {
    printf("ERROR: Failed to load %s\r\n", kModelPath);
    vTaskSuspend(nullptr);
  }

  // TPU setup
  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!tpu_context) {
    printf("ERROR: Failed to get EdgeTpu context\r\n");
    vTaskSuspend(nullptr);
  }
 // tflite setup
  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<3> resolver;
  resolver.AddDequantize();
  resolver.AddDetectionPostprocess();
  resolver.AddCustom(kCustomOp, RegisterCustomOp());

  tflite::MicroInterpreter interpreter(
      tflite::GetModel(model.data()), resolver, tensor_arena, kTensorArenaSize,
      &error_reporter);

  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors() failed\r\n");
    vTaskSuspend(nullptr);
  }
  // generate and fill a static grey image
  auto* input_tensor = interpreter.input_tensor(0);
  std::vector<uint8_t> static_image(input_tensor->bytes, 127);
  memcpy(input_tensor->data.uint8, static_image.data(), input_tensor->bytes);

  // Enable DWT cycle counter
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  for (;;) {
    const uint32_t t_start = DWT->CYCCNT;
    xTaskNotifyGive(gpioTaskHandle); // tell GpioTask to go HIGH
    taskYIELD();
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY); // wait until HIGH has been set
    if (interpreter.Invoke()!=kTfLiteOk) {
      printf("ERROR: InferenceTask() failed\r\n");
      vTaskSuspend(nullptr);
    };
    xTaskNotifyGive(gpioTaskHandle); // tell GpioTask to go LOW
    taskYIELD();
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY); // wait until LOW has been set
    const uint32_t t_end = DWT->CYCCNT;
#if defined(SystemCoreClock) // uses SystemCoreClock if its available, or default to value in coral datasheet
    const double cpu_hz = static_cast<double>(SystemCoreClock);
#else
    const double cpu_hz = 800e6;
#endif
    const double ms = static_cast<double>(t_end - t_start) / (cpu_hz / 1000.0);
    printf("invoke_ms=%.3f\r\n", ms);

    vTaskDelay(pdMS_TO_TICKS(200));
  }
}

//  Create FreeRTOS tasks
void Main() {
  xTaskCreate(GpioTask, "GpioTask", 1024, nullptr, configMAX_PRIORITIES - 1,
              &gpioTaskHandle); // max priority for minimal gpio delay

  xTaskCreate(InferenceTask, "InferenceTask", 8192, nullptr,
              configMAX_PRIORITIES - 2,
              &inferenceTaskHandle); // right after the gpio, but above all other on-board housekeeping
  vTaskSuspend(nullptr);
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}
