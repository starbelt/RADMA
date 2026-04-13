cmake_minimum_required(VERSION 3.13)

# Toolchain must be set before project() call.
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/../coralmicro/cmake/toolchain-arm-none-eabi-gcc.cmake)
endif()

project(Coral-TPU-Characterization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(..)
include_directories(../coralmicro)
include_directories(libs)

add_subdirectory(../coralmicro coralmicro_build)

add_executable_m7(coralmicro-app
        main.cc
        DATA
        # Alpha = 0.25
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.25_D02_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.25_D04_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.25_D06_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.25_D08_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.25_D10_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.25_D12_quant_edgetpu.tflite"
        # Alpha = 0.5
        "${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.5_D02_quant_edgetpu.tflite"
        "${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.5_D04_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.5_D06_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.5_D08_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.5_D10_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.5_D12_quant_edgetpu.tflite"
        # Alpha = 0.75
        "${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.75_D02_quant_edgetpu.tflite"
        "${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.75_D04_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.75_D06_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.75_D08_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.75_D10_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.75_D12_quant_edgetpu.tflite"
        # Alpha = 1.0
        "${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A1.0_D02_quant_edgetpu.tflite"
        "${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A1.0_D04_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A1.0_D06_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A1.0_D08_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A1.0_D10_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A1.0_D12_quant_edgetpu.tflite"
        # Alpha = 1.25
        "${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A1.25_D02_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A1.25_D04_quant_edgetpu.tflite"
        "${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A1.25_D06_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A1.25_D08_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A1.25_D10_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A1.25_D12_quant_edgetpu.tflite"
        # Alpha = 1.5
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A1.5_D02_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A1.5_D04_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A1.5_D06_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A1.5_D08_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A1.5_D10_quant_edgetpu.tflite"
        #"${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A1.5_D12_quant_edgetpu.tflite"
)

target_link_libraries(coralmicro-app
        libs_base-m7_freertos
)