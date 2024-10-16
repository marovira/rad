#include <rad/onnx/concepts.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

namespace onnx = rad::onnx;

TEMPLATE_TEST_CASE("[concepts] - TensorDataType",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t)
{
    STATIC_REQUIRE(onnx::TensorDataType<TestType>);
}

TEST_CASE("[concepts] - BaseTensorDataType", "[rad::onnx]")
{
    SECTION("Fundamental types")
    {
        STATIC_REQUIRE(std::is_same_v<float, onnx::BaseTensorDataType<float>::type>);
        STATIC_REQUIRE(
            std::is_same_v<std::uint8_t, onnx::BaseTensorDataType<std::uint8_t>::type>);
    }

    SECTION("Ort types")
    {
        STATIC_REQUIRE(std::is_same_v<std::uint16_t,
                                      onnx::BaseTensorDataType<Ort::Float16_t>::type>);
    }
}

TEMPLATE_TEST_CASE("[concepts] - OrtStringPath", "[rad::onnx]", char, wchar_t)
{
    STATIC_REQUIRE(std::is_same_v<std::basic_string<TestType>,
                                  typename onnx::OrtStringPath<TestType>::type>);
}
