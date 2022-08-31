#include "opencv.hpp"
#include <type_traits>

namespace rad
{
    cv::Mat to_normalised_float(cv::Mat const& img);
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
