#pragma once

#include "concepts.hpp"

namespace rad::onnx
{
    template<TensorDataType T>
    consteval ONNXTensorElementDataType get_element_data_type()
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        }
        else if constexpr (std::is_same_v<T, std::uint8_t>)
        {
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        }
        else if constexpr (std::is_same_v<T, Ort::Float16_t>)
        {
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        }
        else if constexpr (std::is_same_v<T, std::uint16_t>)
        {
            return ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
        }
    }
} // namespace rad::onnx
