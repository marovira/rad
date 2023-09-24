#pragma once

#include "../opencv.hpp"
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
    concept ImagePostProcessFunctor = requires(T fn, cv::Mat m) {
        // clang-format off
        {fn(m)} -> std::same_as<cv::Mat>;
        // clang-format on
    };

    template<typename T>
    concept TensorDataType = std::same_as<T, float> || std::same_as<T, std::uint8_t>;
} // namespace rad::onnx
