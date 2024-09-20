#include "opencv.hpp"
#include <concepts>

namespace rad
{
    cv::Mat change_colour_space(cv::Mat const& img, cv::ColorConversionCodes code);

    cv::Mat to_normalised_float(cv::Mat const& img);
    cv::Mat to_normalised_float(cv::Mat const& img, cv::Scalar mean, cv::Scalar std);

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
