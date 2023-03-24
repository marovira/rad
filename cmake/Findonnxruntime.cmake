# Locate ONNXRUNTIME library
# This module defines
#  ONNXRUNTIME_INCLUDE_DIR, where to find nvtt.h
#  ONNXRUNTIME_SHARED_LIBS
#  ONNXRUNTIME_LIBRARIES, defined only in Windows builds
#=============================================================================
# The required version of ONNXruntime can be specified using the 
# standard syntax, e.g. FIND_PACKAGE(onnxruntime 14). Note that due to the
# way ONNXRuntime is versioned, this module only supports a major version which will be
# matched to the API version in the header. Otherwise the module will search for
# any available implementation.
# the module will search for any available implementation.
# Note: it is your responsibility to ensure that the shared library
# defined in ONNXRUNTIME_SHARED_LIBS is copied to the final executable.

set(_POSSIBLE_ORT_INCLUDE include include/onnxruntime)
set(_POSSIBLE_ORT_SHARED_LIBS onnxruntime.dll onnxruntime.so)
set(_POSSIBLE_ORT_LIB onnxruntime.lib)
if (onnxruntime_FIND_VERSION_MAJOR AND
        onnxruntime_FIND_VERSION_MINOR AND
        onnxruntime_FIND_VERSION_PATCH)
    set(_POSSIBLE_SUFFIXES
        "${onnxruntime_FIND_VERSION_MAJOR}.${onnxruntime_FIND_VERSION_MINOR}.${onnxruntime_FIND_VERSION_PATCH}"
        )
endif()


if (WIN32)
    set(_POSSIBLE_PATHS
        "C:/Program Files/onnxruntime"
        "C:/Program Files (x86)/onnxruntime"
        "C:/onnxruntime"
        )
else()
    set(_POSSIBLE_PATHS
        "~/Library/Frameworks"
        "/Library/Frameworks"
        "/usr/local"
        "/usr"
        "/sw"
        "/opt/local"
        "/opt/csw"
        "/opt"
        "~/.local"
        )
endif()

# Versioned suffixes are only used in Linux installs.
foreach(_SUFFIX ${_POSSIBLE_SUFFIXES})
    list(APPEND _POSSIBLE_ORT_SHARED_LIBS "libonnxruntime.so.${_SUFFIX}")
endforeach()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_c_api.h
    PATH_SUFFIXES ${_POSSIBLE_ORT_INCLUDE}
    PATHS ${_POSSIBLE_PATHS}
    )

find_file(ONNXRUNTIME_SHARED_LIBS 
    NAMES ${_POSSIBLE_ORT_SHARED_LIBS}
    PATH_SUFFIXES lib lib64
    PATHS ${_POSSIBLE_PATHS}
    )

# Find the version file. For Windows, assume it is located at the installation root. For
# Linux, assume it is located next to the header file.
find_file(ONNXRUNTIME_VERSION_FILE
    NAMES VERSION_NUMBER
    PATHS ${_POSSIBLE_PATHS}
    PATH_SUFFIXES ${_POSSIBLE_ORT_INCLUDE}
    )

if (WIN32)
    find_file(ONNXRUNTIME_LIBRARIES 
        NAMES ${_POSSIBLE_ORT_LIB}
        PATH_SUFFIXES lib lib64
        PATHS ${_POSSIBLE_PATHS}
        ) 
else()
    set(ONNXRUNTIME_LIBRARIES)
endif()

if (EXISTS ${ONNXRUNTIME_VERSION_FILE})
    file(STRINGS ${ONNXRUNTIME_VERSION_FILE} ort_version_str
        REGEX "[0-9]+.[0-9]+.[0-9]+"
        )
    string(
        REGEX REPLACE "([0-9]+.[0-9]+.[0-9]+)" "\\1"
        ONNXRUNTIME_VERSION_STRING "${ort_version_str}")
    unset(ort_version_str)
endif()

include(FindPackageHandleStandardArgs)

if (WIN32)
    find_package_handle_standard_args(onnxruntime
        REQUIRED_VARS ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_SHARED_LIBS ONNXRUNTIME_LIBRARIES
        VERSION_VAR ONNXRUNTIME_VERSION_STRING
        )
else()
    find_package_handle_standard_args(onnxruntime
        REQUIRED_VARS ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_SHARED_LIBS
        VERSION_VAR ONNXRUNTIME_VERSION_STRING
        )
endif()

mark_as_advanced(ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_SHARED_LIBS ONNXRUNTIME_LIBRARIES)

if (ONNXRUNTIME_FOUND AND NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime STATIC IMPORTED)
    if (WIN32)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION ${ONNXRUNTIME_LIBRARIES}
            INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_INCLUDE_DIR}
            )
    else()
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION ${ONNXRUNTIME_SHARED_LIBS}
            INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_INCLUDE_DIR}
            )
    endif()
endif()
