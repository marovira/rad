#!/bin/bash

if [ ! -d ~/deps ]; then
    if [[ $RUNNER_OS == "Windows" ]]; then
        python .github/workflows/scripts/generate_versions.py
    elif [[ $RUNNER_OS == "Linux" ]]; then
        python3 .github/workflows/scripts/generate_versions.py
    fi

    # Read the necessary versions.
    rad_deps=$(<~/rad_deps_version.txt)
    fmt_ver=$(<~/rad_fmt_version.txt)
    catch_ver="v$(<~/rad_catch_version.txt)"
    me_ver="v$(<~/rad_magic_enum_version.txt)"
    zeus_ver=$(<~/rad_zeus_version.txt)

    mkdir ~/deps
    git clone https://github.com/marovira/rad_deps.git --branch $rad_deps --depth 1
    cd ./rad_deps/linux
    tar --use-compress-program=unzstd -xvf onnxruntime.tar.zst -C ~/deps
    tar --use-compress-program=unzstd -xvf opencv.tar.zst -C ~/deps
    tar --use-compress-program=unzstd -xvf TBB.tar.zst -C ~/deps
    cd ../..

    git clone https://github.com/Neargye/magic_enum --branch $me_ver --depth 1
    cmake -S magic_enum -B magic_enum/build -DCMAKE_CXX_STANDARD=20 -DMAGIC_ENUM_OPT_BUILD_EXAMPLES=OFF -DMAGIC_ENUM_OPT_INSTALL=ON -DMAGIC_ENUM_OPT_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=~/deps/magic_enum
    cd ./magic_enum/build
    make
    make install
    cd ../..

    git clone https://github.com/marovira/zeus --branch $zeus_ver --depth 1
    cmake -S zeus -B zeus/build -DCMAKE_CXX_STANDARD=20 -DZEUS_INSTALL_TARGET=ON -DCMAKE_INSTALL_PREFIX=~/deps/zeus
    cd ./zeus/build
    make
    make install
    cd ../..

    git clone https://github.com/fmtlib/fmt.git --branch $fmt_ver --depth 1
    cmake -S fmt -B fmt/build -DCMAKE_CXX_STANDARD=20 -DFMT_INSTALL=ON -DFMT_TEST=OFF -DCMAKE_INSTALL_PREFIX=~/deps/fmt
    cd ./fmt/build
    make
    make install
    cd ../..
    git clone https://github.com/catchorg/Catch2.git --branch $catch_ver --depth 1
    cmake -S Catch2 -B Catch2/build -DCMAKE_CXX_STANDARD=20 -DCATCH_INSTALL_DOCS=OFF -DCMAKE_INSTALL_PREFIX=~/deps/Catch2
    cd ./Catch2/build
    make
    make install
fi
