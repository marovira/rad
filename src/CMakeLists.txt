set(RAD_DIR ${RAD_SOURCE_ROOT}/rad)

set(SOURCE_LIST
    ${RAD_DIR}/assert.cpp
    ${RAD_DIR}/image_utils.cpp
    ${RAD_DIR}/onnx.cpp
    ${RAD_DIR}/processing_util.cpp
    )

set(INCLUDE_LIST
    ${RAD_DIR}/image_utils.hpp
    ${RAD_DIR}/onnx.hpp
    ${RAD_DIR}/onnxruntime.hpp
    ${RAD_DIR}/opencv.hpp
    ${RAD_DIR}/parallel_for.hpp
    ${RAD_DIR}/processing.hpp
    ${RAD_DIR}/processing_util.hpp
    ${RAD_DIR}/split_string.hpp
    ${RAD_DIR}/batch_processing.hpp
    )

source_group("source" FILES ${SOURCE_LIST})
source_group("include" FILES ${INCLUDE_LIST})

add_library(rad ${SOURCE_LIST} ${INCLUDE_LIST})
target_include_directories(rad PUBLIC ${RAD_SOURCE_ROOT})
target_link_libraries(rad PUBLIC
    ${OpenCV_LIBS}
    TBB::tbb
    zeus::zeus
    )
target_compile_features(rad PUBLIC cxx_std_20)
target_compile_options(rad PUBLIC ${RAD_COMPILER_FLAGS})
target_compile_options(rad PUBLIC ${RAD_DEBUG_FLAGS})
target_compile_definitions(rad PUBLIC ${RAD_COMPILE_DEFINITIONS})

if (RAD_USE_ONNX)
    target_include_directories(rad PUBLIC ${RAD_ONNXRUNTIME_INCLUDE})
    target_link_libraries(rad PUBLIC "${RAD_ONNXRUNTIME_LIB}/onnxruntime.lib")
endif()
