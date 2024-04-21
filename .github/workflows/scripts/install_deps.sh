#!/bin/bash

if [ ! -d ~/deps ]; then
    mkdir ~/deps
    git clone https://github.com/marovira/rad_deps.git --branch 2.1.0 --depth 1
    cd ./rad_deps/linux
    tar --use-compress-program=unzstd -xvf onnxruntime.tar.zst -C ~/deps
    tar --use-compress-program=unzstd -xvf opencv.tar.zst -C ~/deps
    tar --use-compress-program=unzstd -xvf TBB.tar.zst -C ~/deps
    cd ../..

    git clone https://github.com/fmtlib/fmt.git --branch 10.2.0 --depth 1
    cmake -S fmt -B fmt/build -DCMAKE_CXX_STANDARD=20 -DFMT_INSTALL=ON -DFMT_TEST=OFF -DCMAKE_INSTALL_PREFIX=~/deps/fmt
    cd ./fmt/build
    make
    make install
    cd ../..
    git clone https://github.com/catchorg/Catch2.git --branch v3.5.1 --depth 1
    cmake -S Catch2 -B Catch2/build -DCMAKE_CXX_STANDARD=20 -DCATCH_INSTALL_DOCS=OFF -DCMAKE_INSTALL_PREFIX=~/deps/Catch2
    cd ./Catch2/build
    make
    make install
fi
