set(TEST_INCLUDE
    )

set(TEST_SOURCE
    ${RAD_TEST_ROOT}/image_utils_test.cpp
    ${RAD_TEST_ROOT}/processing_util_test.cpp
    ${RAD_TEST_ROOT}/processing_test.cpp
    )

if (RAD_USE_ONNX)
    set(RAD_TEST_DATA_ROOT ${RAD_TEST_ROOT}/data)
    set(RAD_TEST_DATA_HEADER
        ${RAD_TEST_ROOT}/test_data_root.hpp)
    configure_file(${RAD_TEST_ROOT}/test_data_root.hpp.in ${RAD_TEST_DATA_HEADER})

    list(APPEND TEST_INCLUDE
        ${RAD_TEST_ROOT}/model_file_manager.hpp
        ${RAD_TEST_ROOT}/onnx_test_helpers.hpp
        ${RAD_TEST_DATA_HEADER}
        )

    list(APPEND TEST_SOURCE
        ${RAD_TEST_ROOT}/onnx_reporters.cpp
        ${RAD_TEST_ROOT}/onnx_perform_safe_op_test.cpp
        ${RAD_TEST_ROOT}/onnx_env_test.cpp
        ${RAD_TEST_ROOT}/onnx_session_test.cpp
        ${RAD_TEST_ROOT}/onnx_memory_backed_tensor_set_test.cpp
        ${RAD_TEST_ROOT}/onnx_tensor_conversion_test.cpp
        ${RAD_TEST_ROOT}/onnx_concepts_test.cpp
        ${RAD_TEST_ROOT}/onnx_tensor_data_types_test.cpp
        )
endif()

source_group("include" FILES ${TEST_INCLUDE})
source_group("source" FILES ${TEST_SOURCE})
add_executable(rad_test ${TEST_INCLUDE} ${TEST_SOURCE})
target_link_libraries(rad_test PRIVATE
    rad
    Catch2::Catch2WithMain)

if (WIN32)
    if (${RAD_USE_ONNX} AND ${RAD_USE_ONNX_DML})
        copy_shared_libs(TARGET rad_test COPY_ONNX COPY_DML)
    elseif(${RAD_USE_ONNX})
        copy_shared_libs(TARGET rad_test COPY_ONNX)
    else()
        copy_shared_libs(TARGET rad_test)
    endif()
endif()

catch_discover_tests(rad_test)
