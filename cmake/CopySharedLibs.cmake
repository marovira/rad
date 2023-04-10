function(copy_shared_libs)
    set(options COPY_ONNX)
    set(oneValueArgs TARGET TBB_VERSION OPENCV_VERSION ONNX_VERSION)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT MSVC)
        message(FATAL_ERROR "Shared libraries can only be copied in Windows builds")
    endif()

    find_package(OpenCV ${ARG_OPENCV_VERSION})
    find_package(TBB ${ARG_TBB_VERSION})

    file(GLOB OPENCV_DLLS "${_OpenCV_LIB_PATH}/*.dll")
    get_target_property(TBB_SHARED_LIB_DEBUG TBB::tbb IMPORTED_LOCATION_DEBUG)
    get_target_propertY(TBB_SHARED_LIB_RELEASE TBB::tbb IMPORTED_LOCATION_RELEASE)

    set(DLL_LIST
        ${OPENCV_DLLS}
        ${TBB_SHARED_LIB_DEBUG}
        ${TBB_SHARED_LIB_RELEASE}
        )

    if (COPY_ONNX)
        find_package(onnxruntime ${ARG_ONNX_VERSION})
        list(APPEND DLL_LIST ${ONNXRUNTIME_SHARED_LIBS})
    endif()

    add_custom_command(TARGET ${ARG_TARGET}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${DLL_LIST}
        $<TARGET_FILE_DIR:${ARG_TARGET}>
        )
endfunction()
