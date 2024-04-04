cmake_minimum_required(VERSION 3.10)
project(HOTDOGNOTHOTDOG)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)

add_executable(HOTDOGNOTHOTDOG src/main.cpp src/DataLoader.cpp)
target_link_libraries(HOTDOGNOTHOTDOG ${CUDA_LIBRARIES} cudnn opencv_core opencv_imgproc opencv_imgcodecs)
