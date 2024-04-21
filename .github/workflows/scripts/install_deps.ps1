$c="C:/deps"
if(-not (Test-Path $c))
{
    git clone https://github.com/marovira/rad_deps.git --branch 2.1.0 --depth 1
    cd ./rad_deps/windows
    Expand-Archive ./onnxruntime.zip -DestinationPath C:/deps
    Expand-Archive ./opencv.zip -DestinationPath C:/deps
    Expand-Archive ./TBB.zip -DestinationPath C:/deps
    cd ../..

    git clone https://github.com/fmtlib/fmt.git --branch 10.2.0 --depth 1
    cmake -G "Visual Studio 17 2022" -S fmt -B fmt/build -DCMAKE_CXX_STANDARD=20 -DFMT_INSTALL=ON -DFMT_TEST=OFF -DCMAKE_INSTALL_PREFIX=C:/deps/fmt
    cd ./fmt/build
    cmake --build . --config $env:BUILD_TYPE
    cmake --build . --target INSTALL --config $env:BUILD_TYPE
    cd ../..

    git clone https://github.com/catchorg/Catch2.git --branch v3.5.1 --depth 1
    cmake -G "Visual Studio 17 2022" -S Catch2 -B Catch2/build -DCMAKE_CXX_STANDARD=20 -DCATCH_INSTALL_DOCS=OFF -DCMAKE_INSTALL_PREFIX=C:/deps/Catch2
    cd ./Catch2/build
    cmake --build . --config $env:BUILD_TYPE
    cmake --build . --target INSTALL --config $env:BUILD_TYPE
    cd ../..
}
