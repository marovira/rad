# Building RAD

There are two ways to build the dependencies for RAD:

1. Use the built-in script, or
2. Build the dependencies manually.

It is recommended that you build the dependencies yourself if you require specific
configurations for OpenCV and ONNXRuntime or if you are going to be sharing these
libraries with other projects. If you need to build a specific version of RAD, then you're
encouraged to use the built-in script.

## Using the Script

Starting with RAD 2.3.0, a Python script is provided to automatically build all
dependencies. In RAD 2.5.0, the use of Astral's [uv](https://docs.astral.sh/uv/) was
introduced and is now required to run the script:

```sh
uv sync

uv run ./tools/build_dependencies.py <install-root>
```

The script will automatically clone, build, and install the correct versions for all
dependencies in the `<install-root>` directory. Once the script finishes, you can build
RAD by appending `-DCMAKE_PREFIX_PATH=<install-root>` to the list of CMake options when
configuring the build system.

## Manually Building

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
networks. It is important to note that CMake **cannot** link with the pre-built libraries
as they don't contain the correct setup to allow CMake to properly include them.

Before continuing, please ensure you have all the required dependencies as listed
[here](https://onnxruntime.ai/docs/build/inferencing.html). Once you do, clone the code:

```bash
git clone https://github.com/Microsoft/onnxruntime.git --branch <version> --depth 1 --recurse-submodules
cd onnxruntime
```

The latest supported version of ONNXRuntime can be found in the project's README. Once the
code is checked out, we can build the library depending on which backend is required. Note
that we must build both Debug and Release configurations of ONNXRuntime (for Windows
only).

### Building for CPU

#### Windows

Open a terminal and navigate to the directory where the code is checked out. Then enter:

```bat
.\build.bat --config <config> --target install --build_shared_lib --parallel --skip_submodule_sync --skip_tests --cmake_extra_defines CMAKE_INSTALL_PREFIX=<install-path> onnxruntime_BUILD_UNIT_TESTS=OFF
```

Where `<config>` is the config you wish to build and `<install-path>` is the path where
the library should be installed to.

> **Note:**
> If you wish to install to the system directories, you **must** use an administrator
> prompt to do so.


#### Linux

Open a terminal and navigate tot he code where the code is checked out. Then enter:

```sh
./build.sh --config <config> --target install --build_shared_lib --parallel --skip_submodule_sync --skip_tests --cmake_extra_defines CMAKE_INSTALL_PREFIX=<install-path> onnxruntime_BUILD_UNIT_TESTS=OFF
```

### Building for DirectML (Windows only)

The process for building with DML is identical to the CPU process, with the exception that
for all build commands, you must add the `--use_dml` flag. Once both versions of
ONNXRuntime have been built and installed, navigate to
`onnxruntime/build/Windows/Release/Release` and copy `DirectML.dll` to the `bin` folder in
the install location. This will ensure that RAD can automatically copy the DLL when
building through the provided `CopySharedLibs` script.

### Troubleshooting

If the latest version of the Windows SDK is not used, then ONNXRuntime may fail to
compile. For reference, see [this
issue](https://onnxruntime.ai/docs/build/inferencing.html). Below are some workarounds to
the compiler errors that may arise

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
