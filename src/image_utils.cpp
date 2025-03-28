#include "rad/image_utils.hpp"

namespace rad
{
    double get_max_value_for_integral_depth(int depth)
    {
        if (is_floating_point_depth(depth))
        {
            throw std::runtime_error{
                "error: cannot get maximum value for a floating point depth"};
        }

        switch (depth)
        {
        case CV_8U:
            return get_max_value_for_integral_depth<CV_8U>();

        case CV_8S:
            return get_max_value_for_integral_depth<CV_8S>();

        case CV_16U:
            return get_max_value_for_integral_depth<CV_16U>();

        case CV_16S:
            return get_max_value_for_integral_depth<CV_16S>();

        case CV_32S:
            return get_max_value_for_integral_depth<CV_32S>();

        default:
            throw std::runtime_error{"error: unknown integral depth"};
        }
    }

    cv::Mat change_colour_space(cv::Mat const& img, cv::ColorConversionCodes code)
    {
        cv::Mat as_code;
        cvtColor(img, as_code, code);
        return as_code;
    }

    cv::Mat convert_to(cv::Mat const& img, int type, double alpha, double beta)
    {
        cv::Mat target;
        img.convertTo(target, type, alpha, beta);
        return target;
    }

    cv::Mat to_normalised_float(cv::Mat const& img)
    {
        return to_normalised_float(img, CV_32F, cv::Scalar::all(0), cv::Scalar::all(1));
    }

    cv::Mat to_normalised_float(cv::Mat const& img, int depth)
    {
        return to_normalised_float(img, depth, cv::Scalar::all(0), cv::Scalar::all(1));
    }

    cv::Mat to_normalised_float(cv::Mat const& img, cv::Scalar mean, cv::Scalar std)
    {
        return to_normalised_float(img, CV_32F, mean, std);
    }

    cv::Mat
    to_normalised_float(cv::Mat const& img, int depth, cv::Scalar mean, cv::Scalar std)
    {
        if (!is_integral_depth(img.depth()))
        {
            throw std::runtime_error{
                "error: only conversions from integral depths are supported"};
        }

        if (!is_floating_point_depth(depth))
        {
            throw std::runtime_error{
                "error: only conversions to floating point depths are supported"};
        }

        const int channels = img.channels();
        if (channels != 1 && channels != 3 && channels != 4)
        {
            throw std::runtime_error{"error: only 1, 3, or 4 channels are supported"};
        }

        const int type = CV_MAKETYPE(depth, channels);
        cv::Mat as_float =
            convert_to(img, type, 1.0 / get_max_value_for_integral_depth(img.depth()));

        if (channels == 1)
        {
            as_float -= mean[0];
            as_float /= std[0];
        }
        else
        {
            as_float -= mean;
            as_float /= std;
        }

        return as_float;
    }

    cv::Mat from_normalised_float(cv::Mat const& img)
    {
        return from_normalised_float(img, CV_8U);
    }

    cv::Mat from_normalised_float(cv::Mat const& img, int depth)
    {
        if (!is_floating_point_depth(img.depth()))
        {
            throw std::runtime_error{
                "error: only conversions from floating point depths are supported"};
        }

        if (!is_integral_depth(depth))
        {
            throw std::runtime_error{
                "error: only conversions from integral depths are supported"};
        }

        const int type = CV_MAKETYPE(depth, img.channels());
        cv::Mat as_int = convert_to(img, type, get_max_value_for_integral_depth(depth));
        return as_int;
    }

    cv::Mat to_fp16(cv::Mat const& img)
    {
        if (img.depth() != CV_32F)
        {
            throw std::runtime_error(
                "error: only conversions from 32-bit to 16-bit floats are supported");
        }

        cv::Mat ret;
        cv::convertFp16(img, ret);
        return ret;
    }

    cv::Mat to_fp32(cv::Mat const& img)
    {
        if (img.depth() != CV_16S)
        {
            throw std::runtime_error(
                "error: only conversions from 32-bit to 16-bit floats are supported");
        }

        cv::Mat ret;
        cv::convertFp16(img, ret);
        return ret;
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

        auto factor = static_cast<float>(max_size);
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
