# cmake needs this line
cmake_minimum_required(VERSION 3.1)

# Define project name
project(opencl_demo_project)

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



find_library(OPENCL_LIBRARY OpenCL PATHS /System/Library/Frameworks)
# Declare the executable target built from your sources
add_executable(demo 00-图像灰度化.cpp)
target_include_directories(demo PRIVATE /Users/runge.liu/Documents/code/)

target_link_libraries(demo PRIVATE ${OPENCL_LIBRARY})
set_target_properties(demo PROPERTIES LINK_FLAGS "-framework OpenCL")
