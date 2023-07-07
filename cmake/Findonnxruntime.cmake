# Locate ONNXRUNTIME library
# This module defines
#  ONNXRUNTIME_INCLUDE_DIR, where to find onnxruntime_c_api.h
#  ONNXRUNTIME_SHARED_LIBS
#  ONNXRUNTIME_LIBRARIES, defined only in Windows builds
#=============================================================================
# The required version of ONNXRuntime can be specified using the standard syntax, e.g.
# FIND_PACKAGE(onnxruntime 15.0.0). If no version is provided, the module will search for
# any available implementation.
#
# Note (1): if you're using versions, then it is recommended that you use the build script
# bundled with RAD, since the required file isn't guaranteed to be there. Please see the
# README for details.
#
# Note (2): it is your responsibility to ensure that the shared library defined in
# ONNXRUNTIME_SHARED_LIBS is copied to the final executable (particularly for Windows). If
# you're using RAD, you may use CopySharedLibs. See the README for more details.

# ONNXRuntime has several different ways in which the headers may be placed, so we need to
# make sure we cover all of them.
set(_POSSIBLE_ORT_INCLUDE
    include                             # Pre-built package
    include/onnxruntime                 # Linux local install to /usr/local
    include/onnxruntime/core/session    # Windows local install
    )
set(_POSSIBLE_ORT_SHARED_LIBS onnxruntime.dll onnxruntime.so)
set(_POSSIBLE_ORT_LIB onnxruntime.lib)

# DirectML is for Windows only!
if (WIN32)
    set(_POSSIBLE_ORT_DML_SHARED_LIB DirectML.dll)
endif()

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

foreach(provider ${onnxruntime_FIND_COMPONENTS})
    if (${provider} STREQUAL "DML")
        find_path(ONNXRUNTIME_DML_INCLUDE dml_provider_factory.h
            PATH_SUFFIXES ${_POSSIBLE_ORT_INCLUDE}
            PATHS ${_POSSIBLE_PATHS}
            )

        # DirectML is installed by default on newer versions of Windows 10/11, but that
        # version tends to be very old. It is likely that while searching it will get
        # picked up (especially if System is in the path), so make sure we exclude it from
        # the search options.
        # NOTE: I'm assuming that the correct version of DirectML is placed next to the
        # DLL of onxxruntime itself, since otherwise we'd have no way of finding it.
        find_file(ONNXRUNTIME_DML_SHARED_LIB
            NAMES ${_POSSIBLE_ORT_DML_SHARED_LIB}
            PATHS ${_POSSIBLE_PATHS}
            PATH_SUFFIXES lib lib64
            NO_SYSTEM_ENVIRONMENT_PATH
            )

        # Mark these as advanced so the user doesn't see them.
        mark_as_advanced(ONNXRUNTIME_DML_SHARED_LIB ONNXRUNTIME_DML_INCLUDE)

        # If both the header and the DLL are found, then set DML_FOUND to true.
        if (ONNXRUNTIME_DML_INCLUDE AND ONNXRUNTIME_DML_SHARED_LIB)
            set(onnxruntime_DML_FOUND TRUE)
        endif()
    endif()
endforeach()

# Windows 11 ships with a version of onnxruntime that is very old, and it will likely be
# hit before the options that we give (especially if System is in the path somewhere). To
# solve this, exclude the system path from the search for Windows.
if (WIN32)
    find_file(ONNXRUNTIME_SHARED_LIBS
        NAMES ${_POSSIBLE_ORT_SHARED_LIBS}
        PATH_SUFFIXES lib lib64
        PATHS ${_POSSIBLE_PATHS}
        NO_SYSTEM_ENVIRONMENT_PATH
        )
else()
    find_file(ONNXRUNTIME_SHARED_LIBS
        NAMES ${_POSSIBLE_ORT_SHARED_LIBS}
        PATH_SUFFIXES lib lib64
        PATHS ${_POSSIBLE_PATHS}
        )
endif()

# Find the version file. For Windows, assume it is located at the installation root. For
# Linux, assume it is located next to the header file. Make sure we mark it as advanced so
# it's not shown to the user.
find_file(ONNXRUNTIME_VERSION_FILE
    NAMES VERSION_NUMBER
    PATHS ${_POSSIBLE_PATHS}
    PATH_SUFFIXES ${_POSSIBLE_ORT_INCLUDE}
    )
mark_as_advanced(ONNXRUNTIME_VERSION_FILE)

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
        HANDLE_COMPONENTS
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
