#================================
# Project setup.
#================================
cmake_minimum_required(VERSION 3.30)
project(rad VERSION 2.5.0 LANGUAGES CXX)

# Don't allow builds for Apple.
if (APPLE)
    message(ERROR "RAD does not support Apple platforms.")
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CONFIGURATION_TYPES "Debug;Release")

# Generate compilation database on all Unix systems
if (NOT MSVC)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

#================================
# Option variables.
#================================
option(RAD_USE_ONNX "Enable features that rely on ONNXRuntime" OFF)
option(RAD_USE_ONNX_DML "Enable usage of ONNXRuntime with DML provider" OFF)
option(RAD_ENABLE_THROW_ON_FAILED_ORT_INIT
    "Enable throwing an exception when ONNXRuntime fails to initialise" OFF)
option(RAD_BUILD_TESTS "Build RAD unit tests" OFF)
option(RAD_INSTALL_TARGET "Create install target" OFF)

if (NOT WIN32 AND RAD_USE_ONNX_DML)
    message(FATAL_ERROR "DML provider for ONNXRuntime is only available in Windows")
endif()

if (NOT RAD_USE_ONNX AND RAD_USE_ONNX_DML)
    message(FATAL_ERROR "ONNX usage is disabled but DML is requested")
endif()

if (NOT RAD_USE_ONNX AND RAD_ENABLE_THROW_ON_FAILED_ORT_INIT)
    message(WARNING
        "ONNX usage is disabled, so RAD_ENABLE_THROW_ON_FAILED_ORT_INIT will have no effect")
endif()

#================================
# Directory variables.
#================================
set(RAD_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(RAD_INCLUDE_ROOT ${RAD_SOURCE_DIR}/include)
set(RAD_SOURCE_ROOT ${RAD_SOURCE_DIR}/src)
set(RAD_CMAKE_ROOT ${RAD_SOURCE_DIR}/cmake)
set(RAD_TEST_ROOT ${RAD_SOURCE_DIR}/test)

#================================
# SDK Versions
#================================
set(RAD_ZEUS_VERSION 1.6.0)
set(RAD_TBB_VERSION 2022.0.0)
set(RAD_OPENCV_VERSION 4.11.0)
set(RAD_ONNX_VERSION 1.21.0)
set(RAD_CATCH_VERSION 3.8.0) # This must match the version that Zeus uses.

#================================
# Find packages.
#================================
find_package(zeus ${RAD_ZEUS_VERSION} QUIET)
if (NOT zeus_FOUND)
    include(FetchContent)
    FetchContent_Declare(zeus
        GIT_REPOSITORY https://github.com/marovira/zeus.git
        GIT_TAG a442ea14cb490df8dc34bbc561fe2910ae6e515d
        )
    FetchContent_MakeAvailable(zeus)
    list(APPEND CMAKE_MODULE_PATH ${zeus_SOURCE_DIR}/cmake)
endif()

find_package(OpenCV ${RAD_OPENCV_VERSION} REQUIRED)
find_package(TBB ${RAD_TBB_VERSION} REQUIRED)

if (RAD_USE_ONNX)
    find_package(onnxruntime ${RAD_ONNX_VERSION} REQUIRED)

    if (RAD_USE_ONNX_DML)
        # The directory where we "find" onnxruntime lives under
        # <root>/lib/cmake/onnxruntime. The DLLs are stored under <root>/bin (this changed
        # in version 1.17, previously they were stored under <root>/lib). Since we need
        # the path where DirectML is stored, we need to retrieve the parent directory
        # twice and then append "bin/DirectML.dll" to the path. If it exists, we can then
        # assign the variable so CopySharedLibs knows what to copy.
        cmake_path(GET onnxruntime_DIR PARENT_PATH RAD_ORT_DIRECTML_PATH)
        cmake_path(GET RAD_ORT_DIRECTML_PATH PARENT_PATH RAD_ORT_DIRECTML_PATH)
        cmake_path(GET RAD_ORT_DIRECTML_PATH PARENT_PATH RAD_ORT_DIRECTML_PATH)
        cmake_path(APPEND RAD_ORT_DIRECTML_PATH "bin" "DirectML.dll")
        if (EXISTS "${RAD_ORT_DIRECTML_PATH}")
            set(RAD_DIRECTML_PATH "${RAD_ORT_DIRECTML_PATH}")
        else()
            message(WARNING
                "warning: DirectML.dll was not found in the installation path for \
                onnxruntime. RAD will use the default version of DirectML that is \
                shipped with Windows which may not be compatible with ONNXRuntime \
                version ${RAD_ONNX_VERSION}. To avoid possible errors, please copy \
                the correct version of DirectML.dll into the build directory")
        endif()
    endif()
endif()

# Generate the CopySharedLibs script. Make sure we only replace @...@ so CMake variables
# are left alone.
configure_file(${RAD_CMAKE_ROOT}/CopySharedLibs.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/CopySharedLibs.cmake
    @ONLY
    )

#================================
# Add subdirectories.
#================================
add_subdirectory(${RAD_INCLUDE_ROOT}/rad)
add_subdirectory(${RAD_SOURCE_ROOT})

source_group(TREE ${RAD_INCLUDE_ROOT} PREFIX "include" FILES ${RAD_INCLUDE_LIST})
source_group(TREE ${RAD_SOURCE_ROOT} PREFIX "source" FILES ${RAD_SOURCE_LIST})

#================================
# RAD library.
#================================
add_library(rad ${RAD_INCLUDE_LIST} ${RAD_SOURCE_LIST})
target_include_directories(rad PUBLIC
    $<BUILD_INTERFACE:${RAD_INCLUDE_ROOT}>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(rad PUBLIC zeus::zeus TBB::tbb ${OpenCV_LIBS})

if(RAD_USE_ONNX)
    target_link_libraries(rad PUBLIC onnxruntime::onnxruntime)

    if (RAD_ENABLE_THROW_ON_FAILED_ORT_INIT)
        target_compile_definitions(rad PUBLIC RAD_THROW_ON_FAILED_ORT_INIT)
    endif()
endif()

if (RAD_USE_ONNX_DML)
    target_compile_definitions(rad PUBLIC RAD_ONNX_DML_ENABLED)
endif()

target_compile_features(rad PUBLIC cxx_std_20)
set_target_properties(rad PROPERTIES DEBUG_POSTFIX "d")
add_library(rad::rad ALIAS rad)

#================================
# Install targets.
#================================
if (RAD_INSTALL_TARGET)
    install(TARGETS rad
        EXPORT rad
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION lib
        INCLUDES DESTINATION include
        )

    include(CMakePackageConfigHelpers)
    include(GNUInstallDirs)
    write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/radConfigVersion.cmake"
        VERSION ${PACKAGE_VERSION}
        COMPATIBILITY SameMajorVersion
        )

    install(EXPORT rad
        FILE radTargets.cmake
        NAMESPACE rad::
        DESTINATION lib/cmake/rad
        )

    if (RAD_USE_ONNX)
        set(CONFIG_FILE "${RAD_CMAKE_ROOT}/radConfig_onnx.cmake")
    else()
        set(CONFIG_FILE "${RAD_CMAKE_ROOT}/radConfig.cmake")
    endif()

    # Install the config file based on whether we're using ONNX. Either way, rename to
    # radConfig.
    install(FILES
        ${CONFIG_FILE}
        DESTINATION lib/cmake/rad
        RENAME radConfig.cmake
        )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/cmake/CopySharedLibs.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/radConfigVersion.cmake"
        DESTINATION lib/cmake/rad
        )

    if (RAD_USE_ONNX)
        install(DIRECTORY
            ${RAD_INCLUDE_ROOT}/rad
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING
            PATTERN "*.hpp"
            )
    else()
        install(DIRECTORY
            ${RAD_INCLUDE_ROOT}/rad
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING
            PATTERN "*.hpp"
            PATTERN "onnx" EXCLUDE
            )
    endif()

    if (MSVC)
        set(PDB_DIR_SUFFIX
            "$<$<CONFIG:Debug>:Debug>$<$<CONFIG:RelWithDebInfo>:RelWithDebInfo>")
        set(LIB_SUFFIX "$<$<CONFIG:Debug>:d>")

        set(PDB_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${PDB_DIR_SUFFIX}")
        set(PDB_NAME "rad${LIB_SUFFIX}.pdb")
        install(FILES
            "${PDB_OUTPUT_DIRECTORY}/${PDB_NAME}"
            DESTINATION lib OPTIONAL)
    endif()
endif()

#================================
# Build the tests.
#================================
if (RAD_BUILD_TESTS)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR}/cmake)
    include(FetchLibrary)
    include(CopySharedLibs)

    find_package(Catch2 ${RAD_CATCH_VERSION} QUIET)

    if (NOT Catch2_FOUND)
        fetch_library(
            NAME Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG f981c9cbcac07a2690e5a86767eba490b5465463
            )
        list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
    endif()

    include(CTest)
    include(Catch)

    add_subdirectory(${RAD_TEST_ROOT})
endif()
