#include "rad/image_utils.hpp"

namespace rad
{
    cv::Mat change_colour_space(cv::Mat const& img, cv::ColorConversionCodes code)
    {
        cv::Mat as_code;
        cvtColor(img, as_code, code);
        return as_code;
    }

    cv::Mat to_normalised_float(cv::Mat const& img)
    {
        return to_normalised_float(img, cv::Scalar::all(0), cv::Scalar::all(1));
    }

    cv::Mat to_normalised_float(cv::Mat const& img, cv::Scalar mean, cv::Scalar std)
    {
        cv::Mat as_float;
        img.convertTo(as_float, CV_32FC3, 1.0 / 255.0);

        as_float -= mean;
        as_float /= std;

        return as_float;
    }

    cv::Mat downscale_to(cv::Mat const& img, cv::Size size)
    {
        cv::Mat resized;
        cv::resize(img, resized, size, 0, 0, cv::INTER_CUBIC);
        return resized;
    }

    cv::Mat downscale_by_long_edge(cv::Mat const& img, int max_size)
    {
        cv::Size size = img.size();
        if (size.width <= max_size && size.height <= max_size)
        {
            return img;
        }

        float factor = static_cast<float>(max_size);
        cv::Size scale;
        if (img.cols > img.rows)
        {
            scale.width  = max_size;
            scale.height = static_cast<int>(std::round(img.rows / (img.cols / factor)));
        }
        else
        {
            scale.width  = static_cast<int>(std::round(img.cols / (img.rows / factor)));
            scale.height = max_size;
        }

        cv::Mat resized;
        cv::resize(img, resized, scale, 0, 0, cv::INTER_CUBIC);

        return resized;
    }
} // namespace rad
