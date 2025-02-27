#pragma once

#include "onnxruntime.hpp"
#include <zeus/concepts.hpp>

namespace rad::onnx
{
    template<typename T>
    concept TensorDataType =
        std::same_as<T, float> || std::same_as<T, std::uint8_t>
        || std::same_as<T, Ort::Float16_t> || std::same_as<T, std::uint16_t>
        || std::same_as<T, std::int64_t>;

    template<typename T>
    concept ImageTensorDataType = TensorDataType<T> && !std::same_as<T, std::int64_t>;

    template<TensorDataType T>
    struct BaseTensorDataType
    {
        using type = T;
    };

    template<>
    struct BaseTensorDataType<Ort::Float16_t>
    {
        using type = std::uint16_t;
    };

    template<typename T>
    struct OrtStringPath : std::false_type
    {};

    template<>
    struct OrtStringPath<char>
    {
        using type = std::string;
    };

    template<>
    struct OrtStringPath<wchar_t>
    {
        using type = std::wstring;
    };

} // namespace rad::onnx
