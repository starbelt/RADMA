cmake_minimum_required(VERSION 3.13)

# Toolchain must be set before project() call.
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/coralmicro/cmake/toolchain-arm-none-eabi-gcc.cmake)
endif()

project(Coral-TPU-Characterization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(coralmicro)
add_subdirectory(coralmicro)

add_executable_m7(coralmicro-app
        inference_time_combined.cc
        DATA
        ${PROJECT_SOURCE_DIR}/models/Image_Classification/EfficientNet/S/efficientnet-edgetpu-S_quant_edgetpu.tflite)

target_link_libraries(coralmicro-app
        libs_base-m7_freertos
)

