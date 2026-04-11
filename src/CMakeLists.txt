cmake_minimum_required(VERSION 3.13)

# Toolchain must be set before project() call.
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/../coralmicro/cmake/toolchain-arm-none-eabi-gcc.cmake)
endif()

if (NOT DEFINED MODEL_PATH_1 OR NOT DEFINED MODEL_PATH_2)
    message(FATAL_ERROR "Model paths not set! Pass -DMODEL_PATH_1=... and -DMODEL_PATH_2=...")
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
        "${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.25_D02_quant_edgetpu.tflite"
        "${CMAKE_CURRENT_LIST_DIR}/../data/models/coral_target/Grid_A0.25_D04_quant_edgetpu.tflite"
)

target_link_libraries(coralmicro-app
        libs_base-m7_freertos
)