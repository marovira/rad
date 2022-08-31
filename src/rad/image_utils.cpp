#include "image_utils.hpp"

namespace rad
{
    cv::Mat to_normalised_float(cv::Mat const& img)
    {
        cv::Mat as_float;
        img.convertTo(as_float, CV_32FC3, 1.0 / 255.0);

        as_float -= cv::Scalar{0.406f, 0.456f, 0.485f};
        as_float /= cv::Scalar{0.225f, 0.224f, 0.229f};

        return as_float;
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
