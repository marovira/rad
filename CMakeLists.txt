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
set(COEUS_TEST_ROOT ${COEUS_SOURCE_DIR}/test)


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

if (COEUS_BUILD_TESTS)
    add_subdirectory(${COEUS_TEST_ROOT})
endif()


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
set_target_properties(coeus PROPERTIES DEBUG_POSTFIX "d")
add_library(coeus::coeus ALIAS coeus)

#================================
# Install targets.
#================================
if (COEUS_INSTALL_TARGET)
    if (MSVC)
        set(INCLUDE_INSTALL_PATH "include")
    else()
        set(INCLUDE_INSTALL_PATH "include/libcoeus")
    endif()

    install(TARGETS coeus
        EXPORT coeus
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION lib
        INCLUDES DESTINATION ${INCLUDE_INSTALL_PATH}
        )

    include(CMakePackageConfigHelpers)
    include(GNUInstallDirs)
    write_basic_package_version_file(
        "${PROJECT_BINARY_DIR}/coeusConfigVersion.cmake"
        VERSION ${PACKAGE_VERSION}
        COMPATIBILITY AnyNewerVersion
        )

    install(EXPORT coeus
        FILE coeusTargets.cmake
        NAMESPACE coeus::
        DESTINATION lib/cmake/coeus
        )

    if (COEUS_USE_ONNX)
        set(CONFIG_FILE "${COEUS_CMAKE_ROOT}/coeusConfig_onnx.cmake")
    else()
        set(CONFIG_FILE "${COEUS_CMAKE_ROOT}/coeusConfig.cmake")
    endif()

    install(FILES
        ${CONFIG_FILE}
        DESTINATION lib/cmake/coeus
        RENAME coeusConfig.cmake
        )

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/coeusConfigVersion.cmake"
        DESTINATION lib/cmake/coeus
        )

    if (MSVC)
        set(INCLUDE_INSTALL_DEST "${CMAKE_INSTALL_INCLUDEDIR}/coeus")
    else()
        set(INCLUDE_INSTALL_DEST "${CMAKE_INSTALL_INCLUDEDIR}/libzeus/coeus")
    endif()
    install(FILES
        ${COEUS_INCLUDE_LIST}
        DESTINATION ${INCLUDE_INSTALL_DEST}
        )

    if (MSVC)
        set(PDB_DIR_SUFFIX 
            "$<$<CONFIG:Debug>:Debug>$<$<CONFIG:RelWithDebInfo>:RelWithDebInfo>")
        set(LIB_SUFFIX "$<$<CONFIG:Debug>:d>")

        set(PDB_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${PDB_DIR_SUFFIX}")
        set(PDB_NAME "coeus${LIB_SUFFIX}.pdb")
        install(FILES
            "${PDB_OUTPUT_DIRECTORY}/${PDB_NAME}"
            DESTINATION lib OPTIONAL)
    endif()
endif()

#================================
# Build the tests.
#================================
if (COEUS_BUILD_TESTS)
    find_package(Catch2 3.3.2 QUIET)

    if (NOT Catch2_FOUND)
        include(FetchContent)
        FetchContent_Declare(Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG 97c48e0c343d26d50764fafdc90b1e630fbd10ce
            )
        list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
    endif()

    source_group("source" FILES ${COEUS_TEST_LIST})
    add_executable(coeus_test ${COEUS_TEST_LIST})
    target_link_libraries(coeus_test PRIVATE
        coeus
        Catch2::Catch2WithMain)

    include(CTest)
    include(Catch)
    catch_discover_tests(coeus_test)
endif()
