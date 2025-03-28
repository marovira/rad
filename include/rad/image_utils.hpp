#include "opencv.hpp"

#include <concepts>
#include <limits>
#include <zeus/concepts.hpp>

namespace rad
{

    // clang-format off
    template<int depth>
    concept IntegralDepth =
    (
        depth == CV_8U  ||
        depth == CV_8S  ||
        depth == CV_16U ||
        depth == CV_16S ||
        depth == CV_32S
    );

    template<int depth>
    concept FloatingPointDepth =
    (
        depth == CV_32F ||
        depth == CV_64F
    );
    // clang-format on

    constexpr bool is_integral_depth(int depth)
    {
        // clang-format off
        return
            depth == CV_8U  ||
            depth == CV_8S  ||
            depth == CV_16U ||
            depth == CV_16S ||
            depth == CV_32S;
        // clang-format on
    }

    constexpr bool is_floating_point_depth(int depth)
    {
        // clang-format off
        return
            depth == CV_32F ||
            depth == CV_64F;
        // clang-format on
    }

    template<int depth>
    consteval auto get_max_value_for_integral_depth()
    requires IntegralDepth<depth>
    {
        if constexpr (depth == CV_8U)
        {
            return std::numeric_limits<std::uint8_t>::max();
        }
        else if constexpr (depth == CV_8S)
        {
            return std::numeric_limits<std::int8_t>::max();
        }
        else if constexpr (depth == CV_16U)
        {
            return std::numeric_limits<std::uint16_t>::max();
        }
        else if constexpr (depth == CV_16S)
        {
            return std::numeric_limits<std::int16_t>::max();
        }
        else if constexpr (depth == CV_32S)
        {
            return std::numeric_limits<std::int32_t>::max();
        }
    }

    double get_max_value_for_integral_depth(int depth);

    cv::Mat change_colour_space(cv::Mat const& img, cv::ColorConversionCodes code);

    cv::Mat
    convert_to(cv::Mat const& img, int type, double alpha = 1.0, double beta = 0.0);

    cv::Mat to_normalised_float(cv::Mat const& img);
    cv::Mat to_normalised_float(cv::Mat const& img, int depth);
    cv::Mat to_normalised_float(cv::Mat const& img, cv::Scalar mean, cv::Scalar std);
    cv::Mat
    to_normalised_float(cv::Mat const& img, int depth, cv::Scalar mean, cv::Scalar std);

    cv::Mat from_normalised_float(cv::Mat const& img);
    cv::Mat from_normalised_float(cv::Mat const& img, int depth);

    cv::Mat to_fp16(cv::Mat const& img);
    cv::Mat to_fp32(cv::Mat const& img);

    cv::Mat downscale_to(cv::Mat const& img, cv::Size size);
    cv::Mat downscale_by_long_edge(cv::Mat const& img, int max_size);

    template<typename T>
    requires std::floating_point<T>
    constexpr float modified_wyvill_filter_falloff(T z)
    {
        T g{0};
        T base = T{1} - (z * z * z);
        g      = base * base * base * base;
        g      = g * g;
        g      = T{1} - g;
        g      = (g < T{1e-4}) ? T{0} : g;
        return g;
    }
} // namespace rad
