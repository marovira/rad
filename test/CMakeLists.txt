set(TEST_INCLUDE
    )

set(TEST_SOURCE
    ${RAD_TEST_ROOT}/image_utils_test.cpp
    ${RAD_TEST_ROOT}/processing_util_test.cpp
    ${RAD_TEST_ROOT}/processing_test.cpp
    )

if (RAD_USE_ONNX)
    list(APPEND TEST_INCLUDE
        ${RAD_TEST_ROOT}/model_file_manager.hpp
        ${RAD_TEST_ROOT}/onnx_test_helpers.hpp
        )

    list(APPEND TEST_SOURCE
        ${RAD_TEST_ROOT}/onnx_reporters.cpp
        ${RAD_TEST_ROOT}/onnx_common_test.cpp
        ${RAD_TEST_ROOT}/onnx_env_test.cpp
        ${RAD_TEST_ROOT}/onnx_session_test.cpp
        ${RAD_TEST_ROOT}/onnx_memory_backed_tensor_set_test.cpp
        ${RAD_TEST_ROOT}/onnx_tensor_conversion_test.cpp
        )
endif()

source_group("include" FILES ${TEST_INCLUDE})
source_group("source" FILES ${TEST_SOURCE})
add_executable(rad_test ${TEST_INCLUDE} ${TEST_SOURCE})
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

catch_discover_tests(rad_test)
