#pragma once

#include <cstdint>
#include <zeus/concepts.hpp>

namespace rad::onnx
{
    template<typename T>
    concept TensorDataType = std::same_as<T, float> || std::same_as<T, std::uint8_t>;

    template<typename T>
    struct OrtStringPath : std::false_type
    {};

    template<>
    struct OrtStringPath<char>
    {
        using value_type = std::string;
    };

    template<>
    struct OrtStringPath<wchar_t>
    {
        using value_type = std::wstring;
    };

} // namespace rad::onnx
