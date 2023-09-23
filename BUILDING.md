# Building RAD

Due to the complexity of the libraries that RAD relies on, they are not bundled in the
library itself, nor are they kept submodules or fetched through CMake. As a result, it is
expected that you build the libraries yourself and have them installed in your system
prior to using RAD. In order to streamline this, we present a quick guide for how to setup
the different libraries that RAD relies on. Please note that Zeus is the only library that
is fetched by RAD if it hasn't been found, so we'll skip it from this guide.

RAD only supports Debug and Release builds, so it is recommended that all SDKs that are
built from source are built for both of these configurations.

## Installing OneAPI TBB

RAD uses TBB internally for multi-threading operations, and it is also recommended that
OpenCV be built with TBB to avoid clashes between different threading systems. As a
result, TBB should be installed first. To do so, clone the code from
[here](https://github.com/oneapi-src/oneTBB) and checkout the version that RAD supports
(as noted in the README). When building, it is recommended that you set `TBB_TEST` to
false in order to speed up compilation. Once it is built, install it using the install
target.

## Installing OpenCV

OpenCV can be cloned from these two repositories:

* [OpenCV](https://github.com/opencv/opencv)
* [OpenCV Contrib](https://github.com/opencv/opencv_contrib) (optional)

Once the appropriate version of OpenCV Is checked out, it is time to build OpenCV. If the
contrib module was cloned, please set `OPENCV_EXTRA_MODULES_PATH` to the corresponding
directory. The following variables must also be set:

1. `WITH_TBB=ON`: builds OpenCV with TBB as the threading back-end
2. `BUILD_TESTS=OFF`: disable unit tests
3. `BUILD_opencv_world`: build OpenCV as a single DLL (optional).

Once cmake runs, build and install OpenCV as usual.

## Installing ONNXRuntime

ONNXRuntime is an optional dependecy for RAD that is only used for inferencing neural
networks. Currently, the only supported way of linking with ONNXRuntime is by building the
library from source directly. This is an unfortunate limitation as the pre-built packages
do not contain the necessary CMake scripts to allow linking with the library, but they do
appear when building from source. Below we provide steps for building ONNXRuntime from
source.

Before continuing, please ensure you have all the required dependencies as listed
[here](https://onnxruntime.ai/docs/build/inferencing.html). Once you do, clone the code:

```bash
git clone --recursive https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
git checkout <latest-supported-tag>
```

The latest supported version of ONNXRuntime can be found in the project's README. Once the
code is checked out, we can build the library depending on which backend is required. Note
that we must build both Debug and Release configurations of ONNXRuntime (for Windows
only).

> **Note:**
> The following instructions have been tested on Windows only. Linux should be able to
> follow similar steps, but this has not been tested.

### Building for CPU

Open an administrator terminal and navigate to the directory where the code is checked
out. Then enter:

```bat
.\build.bat --config Debug --build_shared_lib --skip_tests --parallel --cmake_generator "Visual Studio 17 2022" --cmake_extra_defines CMAKE_DEBUG_POSTFIX=d
```

After the code builds, it can be installed like this:

```bat
.\build.bat --config Debug --target install --build_shared_lib --skip_tests --parallel --cmake_generator "Visual Studio 17 2022" --cmake_extra_defines CMAKE_DEBUG_POSTFIX=d
```

For release builds, the commands are as follows:

```bat
.\build.bat --config Release --build_shared_lib --skip_tests --parallel --cmake_generator "Visual Studio 17 2022"
.\build.bat --config Release --target install --build_shared_lib --skip_tests --parallel --cmake_generator "Visual Studio 17 2022"
```

### Building for DirectML (Windows only)

The process for building with DML is identical to the CPU process, with the exception that
for all build commands, you must add the `--use_dml` flag. Once both versions of
ONNXRuntiem have been built and installed, navigate to
`onnxruntime/build/Windows/Release/Release` and copy `DirectML.dll` to the `lib` folder in
the install location. This will ensure that RAD can automatically copy the DLL when
building through the provided `CopySharedLibs` script.

### Troubleshooting

Building ONNXRuntime can fail due to incongruence's in the code. Below are some
suggestions for solving common problems:

* Compiler errors appear regarding undefined symbols for `STRSAFE_LPSTR` and similar
  aliases. To solve this, navigate to
  `onnxruntime/build/Windows/<Debug/Release>/_deps/microsoft-wil-src/include/wil` and open
  `result_macros.h`. Immediately after the include block that contains `#include
  <strsafe.h>`, copy the following:

  ```c++
// Deprecated, use the non STRSAFE_ prefixed types instead (e.g. LPSTR or PSTR) as they are the same as these.
typedef _Null_terminated_ char* STRSAFE_LPSTR;
typedef _Null_terminated_ const char* STRSAFE_LPCSTR;
typedef _Null_terminated_ wchar_t* STRSAFE_LPWSTR;
typedef _Null_terminated_ const wchar_t* STRSAFE_LPCWSTR;
typedef _Null_terminated_ const wchar_t UNALIGNED* STRSAFE_LPCUWSTR;

// Deprecated, use the base types instead.
// Strings where the string is NOT guaranteed to be null terminated (does not have _Null_terminated_).
typedef const char* STRSAFE_PCNZCH;
typedef const wchar_t* STRSAFE_PCNZWCH;
typedef const wchar_t UNALIGNED* STRSAFE_PCUNZWCH;
  ```

* Linker errors appear regarding: `IID_ID3D12Device`, `IID_ID3D12RootSignature`, and
    `IID_ID3D12PipelineState`. Navigate to
    `onnxruntime/onnxruntime/core/providers/dml/DMLExecutionProvider/src/Operators` and
    open `DmlDFT.h`. Replace any instances of those types with
    `__uuidof(<name-without-IID>)`. Also open `DmlGridSample.h` and do the same.
