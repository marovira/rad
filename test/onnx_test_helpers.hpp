#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <rad/onnx/onnxruntime.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

template<typename T>
struct TestTensor
{
    Ort::Value tensor;
    std::vector<T> data;
};

inline cv::Size get_test_image_size()
{
    return cv::Size{64, 32};
}

template<typename T>
consteval int get_test_image_depth()
{
    if constexpr (std::is_same_v<T, float>)
    {
        return CV_32F;
    }
    else if constexpr (std::is_same_v<T, Ort::Float16_t>)
    {
        return CV_16S;
    }
    else if constexpr (std::is_same_v<T, std::uint16_t>)
    {
        return CV_16U;
    }
    else
    {
        return CV_8U;
    }
}

template<typename T>
int get_test_image_type(int channels)
{
    return CV_MAKETYPE(get_test_image_depth<T>(), channels);
}

template<typename T>
cv::Mat make_test_image(int channels)
{
    const auto size = get_test_image_size();
    const auto type = get_test_image_type<T>(channels);
    return cv::Mat::ones(size, type);
}

template<typename T>
constexpr T make_test_value()
{
    if constexpr (std::is_same_v<T, Ort::Float16_t>)
    {
        return Ort::Float16_t{1.0f};
    }
    else
    {
        return T{1};
    }
}

template<typename T>
constexpr T make_zero_test_value()
{
    if constexpr (std::is_same_v<T, Ort::Float16_t>)
    {
        return Ort::Float16_t{0.0f};
    }
    else
    {
        return T{0};
    }
}

template<typename T>
TestTensor<T> make_test_tensor_with_size(std::size_t size)
{
    std::vector<T> data(size, make_test_value<T>());
    const std::vector<std::int64_t> shape{1, static_cast<std::int64_t>(size)};

    Ort::Value tensor{nullptr};
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    tensor = Ort::Value::CreateTensor<T>(mem_info,
                                         data.data(),
                                         data.size(),
                                         shape.data(),
                                         shape.size());

    return {.tensor = std::move(tensor), .data = std::move(data)};
}

template<typename T>
TestTensor<T> make_test_tensor()
{
    return make_test_tensor_with_size<T>(10);
}

template<typename T>
consteval int get_onnx_element_type()
{
    if constexpr (std::is_same_v<T, float>)
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    else if constexpr (std::is_same_v<T, Ort::Float16_t>)
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    }
    else if constexpr (std::is_same_v<T, std::uint16_t>)
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    }
    else
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    }
}

inline bool image_equals(cv::Mat const& lhs, cv::Mat const& rhs)
{
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols)
    {
        return false;
    }

    cv::Scalar s = cv::sum(lhs - rhs);
    for (int i{0}; i < s.channels; ++i)
    {
        if (s[i] != 0)
        {
            return false;
        }
    }

    return true;
}

inline bool image_array_equals(std::vector<cv::Mat> const& lhs,
                               std::vector<cv::Mat> const& rhs)
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }

    for (std::size_t i{0}; i < lhs.size(); ++i)
    {
        if (!image_equals(lhs[i], rhs[i]))
        {
            return false;
        }
    }

    return true;
}
