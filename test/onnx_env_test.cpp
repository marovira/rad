#include <rad/onnx/env.hpp>

#include <catch2/catch_test_macros.hpp>

namespace onnx = rad::onnx;

TEST_CASE("[env] - init_api")
{
    REQUIRE(onnx::init_ort_api());
}

TEST_CASE("[env] - create_environment", "[rad::onnx]")
{
    onnx::init_ort_api();

    auto env = onnx::create_environment("test_env");
    REQUIRE(static_cast<OrtEnv*>(env) != nullptr);
}
