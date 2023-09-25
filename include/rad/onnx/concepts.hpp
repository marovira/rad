#pragma once

#include <cstdint>
#include <zeus/concepts.hpp>

namespace rad::onnx
{
    template<typename T>
    struct IsContiguousContainer : std::false_type
    {};

    template<zeus::ArithmeticType T>
    struct IsContiguousContainer<std::vector<T>> : std::true_type
    {};

    template<zeus::ArithmeticType T, std::size_t N>
    struct IsContiguousContainer<std::array<T, N>> : std::true_type
    {};

    template<typename T>
    concept ContiguousContainer = IsContiguousContainer<T>::value;

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
