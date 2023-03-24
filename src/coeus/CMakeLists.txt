set(SRC_ROOT ${COEUS_SOURCE_ROOT}/coeus)

set(COEUS_VERSION_HEADER
    ${SRC_ROOT}/coeus.hpp
    )
configure_file(${SRC_ROOT}/coeus.hpp.in ${COEUS_VERSION_HEADER})

set(INCLUDE_LIST
    ${COEUS_VERSION_HEADER}
    ${SRC_ROOT}/image_utils.hpp
    ${SRC_ROOT}/opencv.hpp
    ${SRC_ROOT}/processing.hpp
    ${SRC_ROOT}/processing_util.hpp
    )

set(SOURCE_LIST
    ${SRC_ROOT}/assert.cpp
    ${SRC_ROOT}/image_utils.cpp
    ${SRC_ROOT}/processing_util.cpp
    )

if (COEUS_USE_ONNX)
    list(APPEND INCLUDE_LIST 
        ${SRC_ROOT}/onnx.hpp
        ${SRC_ROOT}/onnxruntime.hpp
        )
    list(APPEND SOURCE_LIST
        ${SRC_ROOT}/onnx.cpp
        )
endif()

set(COEUS_INCLUDE_LIST
    ${INCLUDE_LIST}
    PARENT_SCOPE)

set(COEUS_SOURCE_LIST
    ${SOURCE_LIST}
    PARENT_SCOPE)

