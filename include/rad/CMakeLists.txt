set(INCLUDE_ROOT ${CMAKE_CURRENT_LIST_DIR})

set(RAD_VERSION_HEADER
    ${INCLUDE_ROOT}/rad.hpp
    )
configure_file(${INCLUDE_ROOT}/rad.hpp.in ${RAD_VERSION_HEADER})

set(INCLUDE_LIST
    ${RAD_VERSION_HEADER}
    ${INCLUDE_ROOT}/image_utils.hpp
    ${INCLUDE_ROOT}/image_utils.hpp
    ${INCLUDE_ROOT}/opencv.hpp
    ${INCLUDE_ROOT}/processing.hpp
    ${INCLUDE_ROOT}/processing_util.hpp
    )

if (RAD_USE_ONNX)
    list(APPEND INCLUDE_LIST
        ${INCLUDE_ROOT}/onnx.hpp
        ${INCLUDE_ROOT}/onnxruntime.hpp
        )
endif()

set(RAD_INCLUDE_LIST ${INCLUDE_LIST} PARENT_SCOPE)
