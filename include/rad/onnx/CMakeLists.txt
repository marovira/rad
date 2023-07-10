set(ONNX_ROOT ${CMAKE_CURRENT_LIST_DIR})

set(ONNX_INCLUDE
    ${ONNX_ROOT}/onnxruntime.hpp
    ${ONNX_ROOT}/common.hpp
    ${ONNX_ROOT}/onnx.hpp
    ${ONNX_ROOT}/env.hpp
    ${ONNX_ROOT}/tensor.hpp
    ${ONNX_ROOT}/session.hpp
    ${ONNX_ROOT}/inference.hpp
    PARENT_SCOPE)
