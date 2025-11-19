#include <catch2/catch_test_macros.hpp>
#include <rad/onnx/env.hpp>
#include <rad/onnx/onnxruntime.hpp>

namespace onnx = rad::onnx;

TEST_CASE("[env] - create_environment", "[rad::onnx]")
{
    REQUIRE(onnx::init_ort_api());

    auto env = onnx::create_environment("test_env");
    REQUIRE(static_cast<OrtEnv*>(env) != nullptr);
}
