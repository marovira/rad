set(SRC_ROOT ${CMAKE_CURRENT_LIST_DIR})

set(SOURCE_LIST
    ${SRC_ROOT}/assert.cpp
    ${SRC_ROOT}/image_utils.cpp
    ${SRC_ROOT}/processing_util.cpp
    )

if (RAD_USE_ONNX)
    add_subdirectory(${SRC_ROOT}/onnx)
    list(APPEND SOURCE_LIST
        ${ONNX_SOURCE}
        )
endif()


set(RAD_SOURCE_LIST ${SOURCE_LIST} PARENT_SCOPE)
