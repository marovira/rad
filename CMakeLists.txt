#================================
# Project setup.
#================================
cmake_minimum_required(VERSION 3.24)
project(rad VERSION 1.3.0 LANGUAGES CXX)

if (APPLE)
    message(ERROR "RAD does not support Apple platforms.")
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CONFIGURATION_TYPES "Debug;Release")

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

list(APPEND CMAKE_MODULE_PATH ${RAD_CMAKE_ROOT})

#================================
# Find packages.
#================================
# Include Zeus first so we have access to fetch_library.
find_package(zeus 1.2.0 QUIET)
if (NOT zeus_FOUND)
    include(FetchContent)
    FetchContent_Declare(zeus
        GIT_REPOSITORY https://github.com/marovira/zeus.git
        GIT_TAG 1809ceff4be28f11325b1e2f6cc71137cb31cc05
        )
    FetchContent_MakeAvailable(zeus)
    list(APPEND CMAKE_MODULE_PATH ${zeus_SOURCE_DIR}/cmake)
endif()

set(RAD_OPENCV_VERSION 4.8.0)
set(RAD_TBB_VERSION 2021.8.0)
set(RAD_ONNX_VERSION 1.15.0)

find_package(OpenCV ${RAD_OPENCV_VERSION} REQUIRED)
find_package(TBB ${RAD_TBB_VERSION} REQUIRED)

if (RAD_USE_ONNX)
    if (RAD_USE_ONNX_DML)
        set(RAD_ONNX_PROVIDERS DML)
    endif()
    find_package(onnxruntime ${RAD_ONNX_VERSION} REQUIRED ${RAD_ONNX_PROVIDERS})
endif()

#================================
# Add subdirectories.
#================================
add_subdirectory(${RAD_INCLUDE_ROOT}/rad)
add_subdirectory(${RAD_SOURCE_ROOT})

source_group(TREE ${RAD_INCLUDE_ROOT} PREFIX "include" FILES ${RAD_INCLUDE_LIST})
source_group(TREE ${RAD_SOURCE_ROOT} PREFIX "source" FILES ${RAD_SOURCE_LIST})

if (RAD_BUILD_TESTS)
    add_subdirectory(${RAD_TEST_ROOT})
endif()

# Generate the CopySharedLibs script. Make sure we only replace @...@ so CMake variables
# are left alone.
configure_file(${RAD_CMAKE_ROOT}/CopySharedLibs.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/CopySharedLibs.cmake
    @ONLY
    )

#================================
# Add library.
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
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Findonnxruntime.cmake"
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

    find_package(Catch2 3.3.2 QUIET)

    if (NOT Catch2_FOUND)
        fetch_library(
            NAME Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG 97c48e0c343d26d50764fafdc90b1e630fbd10ce
            )
        list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
    endif()

    source_group("source" FILES ${RAD_TEST_LIST})
    add_executable(rad_test ${RAD_TEST_LIST})
    target_link_libraries(rad_test PRIVATE
        rad
        Catch2::Catch2WithMain)
    add_custom_command(TARGET rad_test
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
        ${RAD_TEST_ROOT}/data
        $<TARGET_FILE_DIR:rad_test>/data
        )

    if (WIN32)
        # Force VS to use the root where the executable is placed as the working
        # directory.
        set_target_properties(rad_test PROPERTIES
            VS_DEBUGGER_WORKING_DIRECTORY
            "$<TARGET_FILE_DIR:rad_test>"
            )

        if (${RAD_USE_ONNX} AND ${RAD_USE_ONNX_DML})
            copy_shared_libs(TARGET rad_test COPY_ONNX COPY_DML)
        elseif(${RAD_USE_ONNX})
            copy_shared_libs(TARGET rad_test COPY_ONNX)
        else()
            copy_shared_libs(TARGET rad_test)
        endif()
    endif()

    include(CTest)
    include(Catch)
    catch_discover_tests(rad_test)
endif()
