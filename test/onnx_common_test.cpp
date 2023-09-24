#include <rad/onnx/common.hpp>

#include <catch2/catch_test_macros.hpp>

namespace onnx = rad::onnx;

TEST_CASE("[common] - perform_safe_op", "[rad::onnx]")
{
    SECTION("cv::Exception")
    {
        bool ret = onnx::perform_safe_op([]() {
            throw cv::Exception{};
        });

        REQUIRE_FALSE(ret);
    }

    SECTION("Ort::Exception")
    {
        bool ret = onnx::perform_safe_op([]() {
            throw Ort::Exception{"", ORT_FAIL};
        });

        REQUIRE_FALSE(ret);
    }

    SECTION("std::exception")
    {
        bool ret = onnx::perform_safe_op([]() {
            throw std::runtime_error{""};
        });

        REQUIRE_FALSE(ret);
    }

    SECTION("No exception")
    {
        bool ret = onnx::perform_safe_op([]() {});
        REQUIRE(ret);
    }
}
