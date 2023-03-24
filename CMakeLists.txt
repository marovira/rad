#================================
# Project setup.
#================================
cmake_minimum_required(VERSION 3.24)
project(coeus VERSION 1.0.0 LANGUAGES CXX)

if (APPLE)
    message(ERROR "Coeus does not support Apple platforms.")
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Define version variables.
set(COEUS_VERSION_MAJOR "1")
set(COEUS_VERSION_MINOR "0")
set(COEUS_VERSION_PATCH "0")
set(COEUS_VERSION_EXTRA "")
set(COEUS_VERSION "${COEUS_VERSION_MAJOR}.${COEUS_VERSION_MINOR}")
set(COEUS_VERSION_FULL
    "${COEUS_VERSION}.${COEUS_VERSION_PATCH}${COEUS_VERSION_EXTRA}")

#================================
# Option variables.
#================================
set(ZEUS_BUILD_TESTS OFF CACHE INTERNAL "")
option(COEUS_USE_ONNX "Enable features that rely on ONNXRuntime" OFF)
option(COEUS_BUILD_TESTS "Build Coeus unit tests" OFF)
option(COEUS_INSTALL_TARGET "Create install target" OFF)

#================================
# Directory variables.
#================================
set(COEUS_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(COEUS_SOURCE_ROOT ${COEUS_SOURCE_DIR}/src)
set(COEUS_CMAKE_ROOT ${COEUS_SOURCE_DIR}/cmake)


#================================
# Find packages.
#================================
find_package(zeus 1.0.1 REQUIRED)
find_package(OpenCV 4.7.0 REQUIRED)
find_package(TBB 2021.8.0 REQUIRED)

if (COEUS_USE_ONNX)
    list(APPEND CMAKE_MODULE_PATH ${COEUS_CMAKE_ROOT})
    find_package(onnxruntime 1.14.1 REQUIRED)
endif()

#================================
# Add subdirectories.
#================================
add_subdirectory(${COEUS_SOURCE_ROOT}/coeus)

# TODO: Add tests here.

#================================
# Source groups.
#================================
source_group("include" FILES)
source_group("include\\coeus" FILES ${COEUS_INCLUDE_LIST})
source_group("source" FILES)
source_group("source\\coeus" FILES ${COEUS_SOURCE_LIST})

#================================
# Add library.
#================================
add_library(coeus ${COEUS_INCLUDE_LIST} ${COEUS_SOURCE_LIST})
target_include_directories(coeus PUBLIC 
    $<BUILD_INTERFACE:${COEUS_SOURCE_ROOT}>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(coeus PUBLIC zeus::zeus TBB::tbb ${OpenCV_LIBS})

if(COEUS_USE_ONNX)
    target_link_libraries(coeus PUBLIC onnxruntime::onnxruntime)
endif()

target_compile_features(coeus PUBLIC cxx_std_20)
add_library(coeus::coeus ALIAS coeus)
