<a id="top"></a>
![RAD logo](data/logo/logo-transparent.png)

[![Build](https://github.com/marovira/rad/actions/workflows/build.yml/badge.svg)](https://github.com/marovira/rad/actions/workflows/build.yml)
[![CodeFactor](https://www.codefactor.io/repository/github/marovira/rad/badge)](https://www.codefactor.io/repository/github/marovira/rad)
[![Generic badge](https://img.shields.io/badge/license-BSD--3--Clause-green)](https://opensource.org/license/bsd-3-clause)
[![Generic badge](https://img.shields.io/badge/c%2B%2B-20-blue)](https://en.cppreference.com/w/cpp/20)

## Table of Contents

1. [What is RAD](#what-is-rad)
2. [Dependecies](#dependencies)
3. [Building](BUILDING.md)
4. [Contributing](#contributing)

## What is RAD?

*RAD* (Research And Development) is a C++ library that serves as a platform for rapid
development of R&D applications in computer vision. It supports the following workflows:

* Image processing through OpenCV,
* Parallel development through oneAPI's TBB,
* Inference of neural networks through ONNXRuntime.

## Requirements

RAD requires a minimum version of 3.27 and supports the following platforms and compilers:

| Platform (Compiler) | Version |
|---------------------|---------|
| Windows (MSVC) | 19.38+ |
| Linux (GCC) | 13.1.0+ |
| Linux (LLVM Clang) | 15.0.7+ |

> **Warning:** RAD does not support macOS.

## Dependencies

| Library | Version |
|---------------------|---------|
| [Zeus](https://github.com/marovira/zeus) | 1.4.0 |
| [OpenCV](https://github.com/opencv/opencv) |4.9.0 |
| [TBB](https://github.com/oneapi-src/oneTBB) |2021.12.0 |
| [ONNXRuntime](https://github.com/microsoft/onnxruntime) | 1.17.3 |

> **Note:** ONNXRuntime is an optional dependency that may be removed when the library is
> built.

## Contributing

There are three ways in which you can contribute to RAD:

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

RAD is published under the BSD-3 license and can be viewed
[here](https://github.com/marovira/rad/blob/master/LICENSE).
