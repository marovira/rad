set(TEST_LIST
    ${RAD_TEST_ROOT}/image_utils_test.cpp
    ${RAD_TEST_ROOT}/processing_util_test.cpp
    ${RAD_TEST_ROOT}/processing_test.cpp

    ${RAD_TEST_ROOT}/test_file_manager.hpp
    )

if (RAD_USE_ONNX)
    list(APPEND TEST_LIST
        ${RAD_TEST_ROOT}/onnx_test.cpp
        )
endif()

set(RAD_TEST_LIST
    ${TEST_LIST}
    PARENT_SCOPE)
