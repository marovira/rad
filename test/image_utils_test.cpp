#include <rad/image_utils.hpp>

#include <catch2/catch_test_macros.hpp>
#include <zeus/float.hpp>

static constexpr float epsilon{0.00001f};

TEST_CASE("[image_utils] - change_colour_space", "[rad]")
{
    cv::Mat orig = cv::Mat::zeros(cv::Size{1, 1}, CV_8UC3);
    orig         = cv::Scalar{1, 2, 3};

    cv::Mat as_rgb = rad::change_colour_space(orig, cv::COLOR_BGR2RGB);
    auto p         = as_rgb.at<cv::Vec3b>(0, 0);
    REQUIRE(as_rgb.at<cv::Vec3b>(0, 0) == cv::Vec3b{3, 2, 1});
}

TEST_CASE("[image_utils] - to_normalised_float", "[rad]")
{
    using zeus::almost_equal;

    cv::Mat orig = cv::Mat::zeros(cv::Size{1, 1}, CV_8UC3);
    orig         = cv::Scalar{255, 255, 255};

    auto compare_points = [](cv::Point3f const& lhs, cv::Point3f const& rhs) {
        return almost_equal<float>(lhs.x, rhs.x, epsilon)
               && almost_equal<float>(lhs.y, rhs.y, epsilon)
               && almost_equal<float>(lhs.z, rhs.z, epsilon);
    };

    SECTION("Default conversion to float")
    {
        const cv::Point3f pix = cv::Point3f{1.0f, 1.0f, 1.0f};
        cv::Mat as_float      = rad::to_normalised_float(orig);
        REQUIRE(as_float.type() == CV_32FC3);
        REQUIRE(compare_points(as_float.at<cv::Point3f>(0, 0), pix));
    }

    SECTION("Convert to float with mean/std")
    {
        const cv::Point3f pix = cv::Point3f{0.0f, 0.0f, 0.0f};
        cv::Mat as_float      = rad::to_normalised_float(orig, 1, 2);
        REQUIRE(as_float.type() == CV_32FC3);
        REQUIRE(compare_points(as_float.at<cv::Point3f>(0, 0), pix));
    }
}

TEST_CASE("[image_utils] - downscale_to", "[rad]")
{
    const cv::Size target{64, 64};
    const auto type = CV_8UC1;

    cv::Mat orig = cv::Mat::zeros(cv::Size{128, 128}, type);
    cv::Mat res  = rad::downscale_to(orig, target);

    REQUIRE(res.size() == target);
    REQUIRE(res.type() == type);
}

TEST_CASE("[image_utils] - downscale_by_long_edge", "[rad]")
{
    const int long_target{64};
    const auto type = CV_8UC1;

    SECTION("Width > Height")
    {
        cv::Mat orig = cv::Mat::zeros(cv::Size{512, 128}, type);
        cv::Mat res  = rad::downscale_by_long_edge(orig, long_target);

        REQUIRE(res.size() == cv::Size{long_target, 16});
        REQUIRE(res.type() == type);
    }

    SECTION("Height > Width")
    {
        cv::Mat orig = cv::Mat::zeros(cv::Size{128, 512}, type);
        cv::Mat res  = rad::downscale_by_long_edge(orig, long_target);

        REQUIRE(res.size() == cv::Size{16, long_target});
        REQUIRE(res.type() == type);
    }
}
