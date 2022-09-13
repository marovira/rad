#================================
# Project setup.
#================================
cmake_minimum_required(VERSION 3.24)
project(rad LANGUAGES CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#================================
# Option variables.
#================================
set(ZEUS_BUILD_TESTS OFF CACHE INTERNAL "")

#================================
# Directory variables.
#================================
option(RAD_USE_ONNX "Use ONNX" OFF)

set(RAD_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(RAD_SOURCE_ROOT ${RAD_SOURCE_DIR}/src)

# If you wish to use ONNX, it is your responsibility to fill out the
# ONNXRUNTIME_ROOT variable with the location of ONNX.
if (RAD_USE_ONNX)
    if (NOT MSVC)
        message(FATAL_ERROR "ONNX builds are not supported on Linux")
    endif()
    set(RAD_ONNXRUNTIME_INCLUDE "${ONNXRUNTIME_ROOT}/include")
    set(RAD_ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib")
endif()

find_package(zeus REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TBB REQUIRED)

#================================
# Compiler flags.
#================================
set(RAD_COMPILER_FLAGS /W4 /WX /MP)
set(RAD_DEBUG_FLAGS "$<$<CONFIG:Debug>:/ZI>")
set(RAD_COMPILE_DEFINITIONS -DNOMINMAX)

if (RAD_USE_ONNX)
    list(APPEND RAD_COMPILE_DEFINITIONS -DRAD_ONNX_ENABLED)
endif()

##================================
## Add subdirectories.
##================================
add_subdirectory(${RAD_SOURCE_ROOT})

