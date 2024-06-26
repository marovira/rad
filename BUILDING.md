# Building RAD

## Using Pre-built Binaries

For convenience, we provide the dependencies for each release of RAD in a separate
repository called [rad_deps](https://github.com/marovira/rad_deps). You are strongly
recommended to build the libraries RAD depends on from source, but you may also want to
use the pre-built binaries to avoid conflicts between versions. To use the binaries,
please see the [usage
instructions](https://github.com/marovira/rad_deps/blob/master/README.md).

## Building from Source

Below is a quick guide on the configurations used to build the libraries RAD depends on.
Note that while RAD does depend on Zeus it (and its dependencies) can be pulled directly
through CMake if you don't wish to build everything yourself.

RAD only supports Debug and Release builds, so it is recommended that all SDKs that are
built from source are built for both of these configurations.

### Installing OneAPI TBB

RAD uses TBB internally for multi-threading operations, and it is also recommended that
OpenCV be built with TBB to avoid clashes between different threading systems. As a
result, TBB should be installed first. To do so, clone the code from
[here](https://github.com/oneapi-src/oneTBB) and checkout the version that RAD supports
(as noted in the README). When building, it is recommended that you set `TBB_TEST` to
false in order to speed up compilation. Once it is built, install it using the install
target.

### Installing OpenCV

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

### Installing ONNXRuntime

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

### Building for CPU

#### Windows

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

#### Linux

Open a terminal and navigate tot he code where the code is checked out. Then enter:

```sh
./build.sh --config Release --build_shared_lib --parallel --skip_tests
./build.sh --config Release --build_shared_lib --parallel --skip_tests --target install
```

### Building for DirectML (Windows only)

The process for building with DML is identical to the CPU process, with the exception that
for all build commands, you must add the `--use_dml` flag. Once both versions of
ONNXRuntiem have been built and installed, navigate to
`onnxruntime/build/Windows/Release/Release` and copy `DirectML.dll` to the `lib` folder in
the install location. This will ensure that RAD can automatically copy the DLL when
building through the provided `CopySharedLibs` script.

### Troubleshooting

Building ONNXRuntime can fail due to incongruences in the code. Below are some
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

* Linker errors appear regarding: `DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS` and
  `DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE`. Navigate to
  `onnxruntime/onnxruntime/core/providers/dml` and open `dml_provider_factory.cc`. At the
  top of the file, after the final include, copy the following:

  ```c++
#if !defined(INITGUID)
#define INITGUID
#include <guiddef.h>
#endif
DEFINE_GUID(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS, 0x0c9ece4d, 0x2f6e, 0x4f01, 0x8c, 0x96, 0xe8, 0x9e, 0x33, 0x1b, 0x47, 0xb1);
DEFINE_GUID(DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE, 0x248e2800, 0xa793, 0x4724, 0xab, 0xaa, 0x23, 0xa6, 0xde, 0x1b, 0xe0, 0x90);
  ```
