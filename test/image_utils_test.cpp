#include <coeus/image_utils.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("[image_utils] - change_colour_space", "[coeus]")
{
    cv::Mat orig = cv::Mat::zeros(cv::Size{1, 1}, CV_8UC3);
    orig         = cv::Scalar{1, 2, 3};

    cv::Mat as_rgb = coeus::change_colour_space(orig, cv::COLOR_BGR2RGB);
    auto p         = as_rgb.at<cv::Vec3b>(0, 0);
    REQUIRE(as_rgb.at<cv::Vec3b>(0, 0) == cv::Vec3b{3, 2, 1});
}

TEST_CASE("[image_utils] - to_normalised_float", "[coeus")
{
    cv::Mat orig = cv::Mat::zeros(cv::Size{1, 1}, CV_8UC3);
    orig         = cv::Scalar{127, 127, 127};

    SECTION("Default conversion to float")
    {
        const cv::Vec3f pix = cv::Vec3f::all(127 / 255.0f);
        cv::Mat as_float    = coeus::to_normalised_float(orig);
        REQUIRE(as_float.type() == CV_32FC3);
    }

    SECTION("Convert to float with mean/std")
    {
        const cv::Vec3f pix = cv::Vec3f::all(((127 / 255.0f) - 1) / 2.0f);
        cv::Mat as_float    = coeus::to_normalised_float(orig, 1, 2);
        REQUIRE(as_float.type() == CV_32FC3);
    }
}
