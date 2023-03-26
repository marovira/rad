# Coeus

> A C++ research and development platform for computer vision.

[![Generic badge](https://img.shields.io/badge/License-BSD3-blue)](LICENSE)
[![Generic badge](https://img.shields.io/badge/Language-C++20-red.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![CodeFactor](https://www.codefactor.io/repository/github/marovira/coeus/badge)](https://www.codefactor.io/repository/github/marovira/coeus)

## What is Coeus?

Named after one of the Titans born to Uranus and Gaia, Coeus is a C++ library that serves
as a platform for rapid development of R&D applications in computer vision. It supports
the following workflows:

* Image processing through OpenCV,
* Parallel development through oneAPI's TBB,
* Inference of neural networks through ONNXRuntime.

## Dependencies

The following are **core** requirements of Coeus:

* CMake 3.24+

Coeus supports the following platforms and compilers:

| Platform (Compiler) | Version |
|---------------------|---------|
| Windows (MSVC) | 19.34 |

> **Note:** Support for Linux is untested at this point, but will be added later on.

Please note that macOS is **not** supported.

In addition, Coeus depends on the following libraries:

| Library | Version |
|---------------------|---------|
| [Zeus](https://github.com/marovira/zeus) | 1.0.1 |
| [OpenCV](https://github.com/opencv/opencv) |4.7.0 |
| [TBB](https://github.com/oneapi-src/oneTBB) |2021.8.0 |
| [ONNXRuntime](https://github.com/microsoft/onnxruntime) | 1.14.1 |

Note that ONNXRuntime is an optional dependency that may be removed when the library is
built.

## Contributing

There are three ways in which you can contribute to Coeus:

* If you find a bug, please open an issue. Similarly, if you have a question
  about how to use it, or if something is unclear, please post an issue so it
  can be addressed.
* If you have a fix for a bug, or a code enhancement, please open a pull
  request. Before you submit it though, make sure to abide by the rules written
  below.
* If you have a feature proposal, you can either open an issue or create a pull
  request. If you are submitting a pull request, it must abide by the rules
  written below. Note that any new features need to be approved by me.

If you are submitting a pull request, the guidelines are the following:

1. Ensure that your code follows the standards and formatting of the framework.
   The coding standards can be seen throughout the code, and the formatting is
   handled through the `.clang-format` file located at the root of the
   directory. Any changes that do not follow the style and format will be
   rejected.
2. Ensure that *all* unit tests are working prior to submitting the pull
   request. If you are adding a new feature that has been approved, it is your
   responsibility to provide the corresponding unit tests (if applicable). 

## License

Coeus is published under the BSD-3 license and can be viewed
[here](https://github.com/marovira/coeus/blob/master/LICENSE).
