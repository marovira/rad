set(ONNX_ROOT ${CMAKE_CURRENT_LIST_DIR})

set(ONNX_INCLUDE
    ${ONNX_ROOT}/onnxruntime.hpp
    ${ONNX_ROOT}/perform_safe_op.hpp
    ${ONNX_ROOT}/env.hpp
    ${ONNX_ROOT}/session.hpp
    ${ONNX_ROOT}/concepts.hpp
    ${ONNX_ROOT}/tensor_conversion.hpp
    ${ONNX_ROOT}/tensor_data_types.hpp
    ${ONNX_ROOT}/tensor_set.hpp
    PARENT_SCOPE)
