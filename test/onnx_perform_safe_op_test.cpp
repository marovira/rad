#include <catch2/catch_test_macros.hpp>
#include <opencv2/core.hpp>
#include <rad/onnx/onnxruntime.hpp>
#include <rad/onnx/perform_safe_op.hpp>

namespace onnx = rad::onnx;

TEST_CASE("[perform_safe_op] - perform_safe_op", "[rad::onnx]")
{
    SECTION("cv::Exception")
    {
        const bool ret = onnx::perform_safe_op([]() {
            throw cv::Exception{};
        });

        REQUIRE_FALSE(ret);
    }

    SECTION("Ort::Exception")
    {
        const bool ret = onnx::perform_safe_op([]() {
            throw Ort::Exception{"", ORT_FAIL};
        });

        REQUIRE_FALSE(ret);
    }

    SECTION("std::exception")
    {
        const bool ret = onnx::perform_safe_op([]() {
            throw std::runtime_error{""};
        });

        REQUIRE_FALSE(ret);
    }

    SECTION("No exception")
    {
        const bool ret = onnx::perform_safe_op([]() {});
        REQUIRE(ret);
    }
}
