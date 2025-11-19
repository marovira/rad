#include <catch2/catch_test_macros.hpp>
#include <rad/onnx/onnxruntime.hpp>
#include <rad/onnx/tensor_data_types.hpp>

#include <cstdint>

namespace onnx = rad::onnx;

TEST_CASE("[tensor_data_types] - get_element_data_type", "[rad::onnx]")
{
    STATIC_REQUIRE(onnx::get_element_data_type<float>()
                   == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    STATIC_REQUIRE(onnx::get_element_data_type<std::uint8_t>()
                   == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    STATIC_REQUIRE(onnx::get_element_data_type<Ort::Float16_t>()
                   == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    STATIC_REQUIRE(onnx::get_element_data_type<std::uint16_t>()
                   == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);
}
