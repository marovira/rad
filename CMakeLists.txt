#================================
# Project setup.
#================================
cmake_minimum_required(VERSION 3.24)
project(rad VERSION 1.0.0 LANGUAGES CXX)

if (APPLE)
    message(ERROR "RAD does not support Apple platforms.")
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#================================
# Option variables.
#================================
option(RAD_USE_ONNX "Enable features that rely on ONNXRuntime" OFF)
option(RAD_BUILD_TESTS "Build RAD unit tests" OFF)
option(RAD_INSTALL_TARGET "Create install target" OFF)

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
find_package(zeus 1.1.0 QUIET)
if (NOT zeus_FOUND)
    include(FetchContent)
    FetchContent_Declare(zeus
        GIT_REPOSITORY https://github.com/marovira/zeus.git
        GIT_TAG aec7c460444e1e6edfe175b892c90b3c87614209
        )
    FetchContent_MakeAvailable(zeus)
    list(APPEND CMAKE_MODULE_PATH ${zeus_SOURCE_DIR}/cmake)
endif()

set(RAD_OPENCV_VERSION 4.7.0)
set(RAD_TBB_VERSION 2021.8.0)
set(RAD_ONNX_VERSION 1.14.1)

find_package(OpenCV ${RAD_OPENCV_VERSION} REQUIRED)
find_package(TBB ${RAD_TBB_VERSION} REQUIRED)

if (RAD_USE_ONNX)
    find_package(onnxruntime ${RAD_ONNX_VERSION} REQUIRED)
endif()

#================================
# Add subdirectories.
#================================
add_subdirectory(${RAD_INCLUDE_ROOT}/rad)
add_subdirectory(${RAD_SOURCE_ROOT})

if (RAD_BUILD_TESTS)
    add_subdirectory(${RAD_TEST_ROOT})
endif()


#================================
# Source groups.
#================================
source_group("include" FILES)
source_group("include\\rad" FILES ${RAD_INCLUDE_LIST})
source_group("source" FILES ${RAD_SOURCE_LIST})

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
endif()

target_compile_features(rad PUBLIC cxx_std_20)
set_target_properties(rad PROPERTIES DEBUG_POSTFIX "d")
add_library(rad::rad ALIAS rad)

#================================
# Install targets.
#================================
if (RAD_INSTALL_TARGET)
    if (MSVC)
        set(INCLUDE_INSTALL_PATH "include")
    else()
        set(INCLUDE_INSTALL_PATH "include/librad")
    endif()

    install(TARGETS rad
        EXPORT rad
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION lib
        INCLUDES DESTINATION ${INCLUDE_INSTALL_PATH}
        )

    include(CMakePackageConfigHelpers)
    include(GNUInstallDirs)
    write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/radConfigVersion.cmake"
        VERSION ${PACKAGE_VERSION}
        COMPATIBILITY ExactVersion
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
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CopySharedLibs.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Findonnxruntime.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/radConfigVersion.cmake"
        DESTINATION lib/cmake/rad
        )

    if (MSVC)
        set(INCLUDE_INSTALL_DEST "${CMAKE_INSTALL_INCLUDEDIR}/rad")
    else()
        set(INCLUDE_INSTALL_DEST "${CMAKE_INSTALL_INCLUDEDIR}/librad")
    endif()
    install(FILES
        ${RAD_INCLUDE_LIST}
        DESTINATION ${INCLUDE_INSTALL_DEST}
        )

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

    if (WIN32)
        if (${RAD_USE_ONNX})
            copy_shared_libs(TARGET rad_test
                TBB_VERSION ${RAD_TBB_VERSION}
                OPENCV_VERSION ${RAD_OPENCV_VERSION}
                ONNX_VERSION ${RAD_ONNX_VERSION}
                COPY_ONNX
                )
        else()
            copy_shared_libs(TARGET rad_test
                TBB_VERSION ${RAD_TBB_VERSION}
                OPENCV_VERSION ${RAD_OPENCV_VERSION}
                ONNX_VERSION ${RAD_ONNX_VERSION}
                )
        endif()
    endif()


    include(CTest)
    include(Catch)
    catch_discover_tests(rad_test)
endif()
