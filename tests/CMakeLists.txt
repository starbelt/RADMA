cmake_minimum_required(VERSION 3.14)
project(RADMA_Host_Tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# 
# Adjust these paths based on where your tests/ folder is relative to the code
include_directories(..)
include_directories(../coralmicro)
include_directories(../src/libs)
include_directories(../src)

#test executable
# We compile the test file AND the specific embedded source file we want to test
add_executable(test_scheduler
    test_scheduler.cc
    ../src/libs/radma_scheduler.cc
)

# Link GTest
target_link_libraries(test_scheduler GTest::gtest_main)

# Enable CTest discovery
include(GoogleTest)
gtest_discover_tests(test_scheduler)