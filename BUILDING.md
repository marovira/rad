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

ONNXRuntime is an optional dependency for RAD that is only used for inferencing of neural
networks. Unfortunately ONNXRuntime does not have a standard way in which it is installed
for C++, so setting it up requires more manual work depending on your use case. Once it is
installed, RAD will be able to find it automatically through CMake provided it is
installed in any of the following locations:

* For Windows:
    * C:/Program Files/onnxruntime
    * C:/Program Files (x86)/onnxruntime
    * C:/onnxruntime
* For Linux:
    * Default CMake install locations

ONNXRuntime offers a variety of execution providers, of which RAD currently supports:

* CPU: enabled by default. Supported on both Linux and Windows.
* DirectML: can be enabled through `RAD_USE_ONNX_DML`. Only supported on Windows.

### Building for CPU Only

The recommended way is to download the latest release supported by RAD from
[here](https://github.com/microsoft/onnxruntime/releases). The files can then be extracted
and placed in the appropriate location depending on the OS. Note that the layout provided
by the released bundles is supported by RAD out of the box, so no further work needs to be
done.

ONNXRuntime can alternatively be built from source, which we will discuss in the next
section.

### Building for DirectML (Windows Only)

ONNXRuntime does not provide bundled binaries for C++, so they need to built from source.
To do so, clone the repository from [here](https://github.com/microsoft/onnxruntime) and
checkout the corresponding tag. Please ensure that you have all the required dependencies
to build ONNXRuntime as mentioned in their
[documentation](https://onnxruntime.ai/docs/build/inferencing.html).

Once everything is ready, open a command line and navigate to the directory where the code
was cloned. From there, enter the following:

```bat
.\build.bat --config Release --build_shared_lib --parallel --use_dml --cmake_generator "Visual Studio 17 2022" --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF CMAKE_INSTALL_PREFIX=install
```

This will do the following:

* Build ONNXRuntime as a DLL,
* Skip all unit tests,
* Set the install directory to `build/RelWithDebInfo/install`

Once the process is done building, enter the following:

```bat
.\build.bat --config Release --target install --build_shared_lib --parallel --use_dml --cmake_generator "Visual Studio 17 2022" --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF CMAKE_INSTALL_PREFIX=install
```

This will install the files to the directory we specified. Once that is done, make a new
folder in any of the accepted locations for system-wide installation. In that folder,
create two new folders: `include` and `lib`. Now open the folder where the code was cloned
and copy:

* The following files will be copied to the root of the system-wide installation
  directory:
    * `VERSION_NUMBER`
* The following files will be copied to the `include` folder you made earlier:
    * Navigate to `include/onnxruntime/core/session` and copy:
        * `onnxruntime_c_api.h`
        * `onnxruntime_cxx_api.h`
        * `onnxruntime_cxx_inline.h`
        * `onnxruntime_run_options_config_keys.h`
        * `onnxruntime_session_options_config_keys.h`
        * `provider_options.h`
    * Navigate to `build/Windows/Release/install/include/onnxruntime/providers` and copy:
        * `cpu/cpu_provider_factory.h`
        * `dml/dml_provider_factory.h`
* The following files will be copied to the `lib` folder you made earlier:
    * Navigate to `build/Windows/Release/install/bin` and copy:
        * `onnxruntime.dll`
        * `onnxruntime_providers_shared.dll`
    * Navigate to `build/Windows/Release/install/lib` and copy:
        * `onnxruntime.lib`
        * `onnxruntime_providers_shared.lib`
    * Navigate to `build/Windows/packages/Microsoft.Ai.DirectMLX.XX.X/bin/x64-win` where
      `X.XX.X` is the version of DirectML used, and copy:
        * `DirectML.dll`

Once all the files are copied, RAD will be able to automatically detect ONNXRuntime and
everything will work normally.

### Explanation

For some reason, ONNXRuntime does not build into the same configuration as the bundles you
can download from their releases page. Moreover, the folder structure for the headers in
particular is completely different from what the bundles have and what their examples
assume that you should have. Furthermore, ONNXRuntime does not provide a native
integration to CMake by default. In order to solve these issues, we decided to go with the
following approach:

1. Assume that the folder structure will be the same as the bundles, as they provide the
   same general interface that the examples use.
2. Ship RAD with a `Findonnxruntime.cmake` file to automatically find and link it using
   modern CMake.
3. Ship with a custom function called `CopySharedLibs` to automatically copy DLLs into the
   executable directory.

This then led to another issue: ideally we would want to be able to do this:

```cmake
find_package(onnxruntime 1.14.0 REQUIRED)
```

In order to ensure that we use the correct version of ONNXRuntime. This presented an issue
because the main header (`onnxruntime_c_api.h`) only provides an API version which seems
to match the minor version of ONNXRuntime itself. As a workaround, we decided to use the
`VERSION_NUMBER` file that comes with the bundles in order to determine which version of
ONNXRuntime we found.

This works so long as we only used CPU providers, but it broke when we tried to use
DirectML, which unfortunately **has** to be built from source as ONNXRuntime does not
provide pre-built binaries. This introduced further issues. As mentioned previously, the
folder structure does not match what the bundles had, and the headers are separated in
strange ways. As a workaround, it was decided to just manually copy everything into the
correct locations, since trying to come up with a unified build script would've been too
cumbersome. The final issue came from `DirectML.dll`. Windows ships with a version of this
library which is much older than what ONNXRuntime uses, so we had to manually copy that as
well and re-configure `Findonnxruntime.cmake` to automatically find this DLL so it could
be copied.

TL;DR: ONNXRuntime does not have a standard way of installing, so we had to come up with
one. This results in files needing to be copied if the code is built from source, but
there's nothing we can really do to fix this.
