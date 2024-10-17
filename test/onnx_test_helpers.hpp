#pragma once

#include <rad/onnx/onnxruntime.hpp>
#include <rad/opencv.hpp>

inline cv::Size get_test_image_size()
{
    return cv::Size{64, 32};
}

template<typename T>
int get_test_image_type(int channels)
{
    int depth = []() {
        if constexpr (std::is_same_v<T, float>)
        {
            return CV_32F;
        }
        else if constexpr (std::is_same_v<T, Ort::Float16_t>
                           || std::is_same_v<T, std::uint16_t>)
        {
            return CV_16S;
        }
        else
        {
            return CV_8U;
        }
    }();
    return CV_MAKETYPE(depth, channels);
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
