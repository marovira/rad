#include <rad/image_utils.hpp>

#include <catch2/catch_test_macros.hpp>
#include <zeus/float.hpp>

template<typename T>
consteval T epsilon()
{
    return static_cast<T>(0.00001f);
}

template<typename T>
consteval auto get_max()
{
    return std::numeric_limits<T>::max();
}

TEST_CASE("[image_utils] - IntegralDepth", "[rad]")
{
    STATIC_REQUIRE(rad::IntegralDepth<CV_8U>);
    STATIC_REQUIRE(rad::IntegralDepth<CV_8S>);
    STATIC_REQUIRE(rad::IntegralDepth<CV_16U>);
    STATIC_REQUIRE(rad::IntegralDepth<CV_16S>);
    STATIC_REQUIRE(rad::IntegralDepth<CV_32S>);
    STATIC_REQUIRE_FALSE(rad::IntegralDepth<CV_32F>);
    STATIC_REQUIRE_FALSE(rad::IntegralDepth<CV_64F>);
}

TEST_CASE("[image_utils] - FloatingPointDepth", "[rad]")
{
    STATIC_REQUIRE_FALSE(rad::FloatingPointDepth<CV_8U>);
    STATIC_REQUIRE_FALSE(rad::FloatingPointDepth<CV_8S>);
    STATIC_REQUIRE_FALSE(rad::FloatingPointDepth<CV_16U>);
    STATIC_REQUIRE_FALSE(rad::FloatingPointDepth<CV_16S>);
    STATIC_REQUIRE_FALSE(rad::FloatingPointDepth<CV_32S>);
    STATIC_REQUIRE(rad::FloatingPointDepth<CV_32F>);
    STATIC_REQUIRE(rad::FloatingPointDepth<CV_64F>);
}

TEST_CASE("[image_utils] - is_integral_depth", "[rad]")
{
    REQUIRE(rad::is_integral_depth(CV_8U));
    REQUIRE(rad::is_integral_depth(CV_8S));
    REQUIRE(rad::is_integral_depth(CV_16U));
    REQUIRE(rad::is_integral_depth(CV_16S));
    REQUIRE(rad::is_integral_depth(CV_32S));
    REQUIRE_FALSE(rad::is_integral_depth(CV_32F));
    REQUIRE_FALSE(rad::is_integral_depth(CV_64F));
}

TEST_CASE("[image_utils] - is_floating_point_depth", "[rad]")
{
    REQUIRE_FALSE(rad::is_floating_point_depth(CV_8U));
    REQUIRE_FALSE(rad::is_floating_point_depth(CV_8S));
    REQUIRE_FALSE(rad::is_floating_point_depth(CV_16U));
    REQUIRE_FALSE(rad::is_floating_point_depth(CV_16S));
    REQUIRE_FALSE(rad::is_floating_point_depth(CV_32S));
    REQUIRE(rad::is_floating_point_depth(CV_32F));
    REQUIRE(rad::is_floating_point_depth(CV_64F));
}

TEST_CASE("[image_utils] - get_max_value_for_integral_depth", "[rad]")
{
    SECTION("Compile-time")
    {
        STATIC_REQUIRE(rad::get_max_value_for_integral_depth<CV_8U>()
                       == get_max<std::uint8_t>());
        STATIC_REQUIRE(rad::get_max_value_for_integral_depth<CV_8S>()
                       == get_max<std::int8_t>());
        STATIC_REQUIRE(rad::get_max_value_for_integral_depth<CV_16U>()
                       == get_max<std::uint16_t>());
        STATIC_REQUIRE(rad::get_max_value_for_integral_depth<CV_16S>()
                       == get_max<std::int16_t>());
        STATIC_REQUIRE(rad::get_max_value_for_integral_depth<CV_32S>()
                       == get_max<std::int32_t>());
    }

    SECTION("Runtime")
    {
        REQUIRE(rad::get_max_value_for_integral_depth(CV_8U) == get_max<std::uint8_t>());
        REQUIRE(rad::get_max_value_for_integral_depth(CV_8S) == get_max<std::int8_t>());
        REQUIRE(rad::get_max_value_for_integral_depth(CV_16U)
                == get_max<std::uint16_t>());
        REQUIRE(rad::get_max_value_for_integral_depth(CV_16S) == get_max<std::int16_t>());
        REQUIRE(rad::get_max_value_for_integral_depth(CV_32S) == get_max<std::int32_t>());
        REQUIRE_THROWS(rad::get_max_value_for_integral_depth(CV_32F));
        REQUIRE_THROWS(rad::get_max_value_for_integral_depth(CV_64F));
    }
}

TEST_CASE("[image_utils] - change_colour_space", "[rad]")
{
    cv::Mat orig = cv::Mat::zeros(cv::Size{1, 1}, CV_8UC3);
    orig         = cv::Scalar{1, 2, 3};

    cv::Mat as_rgb = rad::change_colour_space(orig, cv::COLOR_BGR2RGB);
    REQUIRE(as_rgb.at<cv::Vec3b>(0, 0) == cv::Vec3b{3, 2, 1});
}

TEST_CASE("[image_utils] - convert_to", "[rad]")
{
    cv::Mat orig = cv::Mat::zeros(cv::Size{1, 1}, CV_8UC3);
    orig         = cv::Scalar{1, 2, 3};

    cv::Mat as_float = rad::convert_to(orig, CV_32FC3);
    REQUIRE(as_float.depth() == CV_32F);
    REQUIRE(as_float.at<cv::Vec3f>(0, 0) == cv::Vec3f{1, 2, 3});
}

TEST_CASE("[image_utils] - to_normalised_float", "[rad]")
{
    using zeus::almost_equal;

    auto compare_3d_points = [](auto const& lhs, auto const& rhs) {
        using T = std::remove_const_t<decltype(lhs.x)>;
        return almost_equal<T>(lhs.x, rhs.x, epsilon<T>())
               && almost_equal<T>(lhs.y, rhs.y, epsilon<T>())
               && almost_equal<T>(lhs.z, rhs.z, epsilon<T>());
    };

    auto compare_1d_points = [](auto lhs, auto rhs) {
        return almost_equal(lhs, rhs);
    };

    SECTION("Base function")
    {
        SECTION("RGB")
        {
            cv::Mat orig          = cv::Mat::zeros(cv::Size{1, 1}, CV_8UC3);
            orig                  = cv::Scalar{255, 255, 255};
            const cv::Scalar mean = cv::Scalar::all(1);
            const cv::Scalar std  = cv::Scalar::all(2);

            SECTION("To float")
            {
                const cv::Point3f pix = cv::Point3f{0.0f, 0.0f, 0.0f};

                cv::Mat as_float = rad::to_normalised_float(orig, CV_32F, mean, std);
                REQUIRE(as_float.type() == CV_32FC3);
                REQUIRE(compare_3d_points(as_float.at<cv::Point3f>(0, 0), pix));
            }

            SECTION("To double")
            {
                const cv::Point3d pix = cv::Point3d{0.0, 0.0, 0.0};

                cv::Mat as_float = rad::to_normalised_float(orig, CV_64F, mean, std);
                REQUIRE(as_float.type() == CV_64FC3);
                REQUIRE(compare_3d_points(as_float.at<cv::Point3d>(0, 0), pix));
            }
        }

        SECTION("Grayscale")
        {
            cv::Mat orig = cv::Mat::zeros(cv::Size{1, 1}, CV_8UC1);
            orig         = cv::Scalar{255};

            SECTION("To float")
            {
                const float pix{0.0f};

                cv::Mat as_float = rad::to_normalised_float(orig, CV_32F, 1, 2);
                REQUIRE(as_float.type() == CV_32F);
                REQUIRE(compare_1d_points(as_float.at<float>(0, 0), pix));
            }

            SECTION("To double")
            {
                const double pix{0.0};

                cv::Mat as_float = rad::to_normalised_float(orig, CV_64F, 1, 2);
                REQUIRE(as_float.type() == CV_64F);
                REQUIRE(compare_1d_points(as_float.at<double>(0, 0), pix));
            }
        }
    }

    SECTION("Overloads")
    {
        cv::Mat orig = cv::Mat::zeros(cv::Size{1, 1}, CV_8UC3);
        orig         = cv::Scalar{255, 255, 255};
        SECTION("Default depth, mean, and std")
        {
            const cv::Point3f pix = cv::Point3f{1, 1, 1};
            cv::Mat as_float      = rad::to_normalised_float(orig);
            REQUIRE(as_float.type() == CV_32FC3);
            REQUIRE(compare_3d_points(as_float.at<cv::Point3f>(0, 0), pix));
        }

        SECTION("Default mean and std")
        {
            const cv::Point3f pix = cv::Point3f{1, 1, 1};
            cv::Mat as_float      = rad::to_normalised_float(orig, CV_32F);
            REQUIRE(as_float.type() == CV_32FC3);
            REQUIRE(compare_3d_points(as_float.at<cv::Point3f>(0, 0), pix));
        }

        SECTION("Default depth")
        {
            const cv::Point3f pix = cv::Point3f{0, 0, 0};
            cv::Mat as_float =
                rad::to_normalised_float(orig, cv::Scalar::all(1), cv::Scalar::all(2));
            REQUIRE(as_float.type() == CV_32FC3);
            REQUIRE(compare_3d_points(as_float.at<cv::Point3f>(0, 0), pix));
        }
    }

    SECTION("Invalid inputs")
    {
        SECTION("Invalid input depth")
        {
            cv::Mat inv = cv::Mat::zeros(cv::Size{1, 1}, CV_32FC3);
            REQUIRE_THROWS(rad::to_normalised_float(inv, CV_32F, 0, 0));
        }

        SECTION("Invalid output depth")
        {
            cv::Mat val = cv::Mat::zeros(cv::Size{1, 1}, CV_8UC3);
            REQUIRE_THROWS(rad::to_normalised_float(val, CV_8U, 0, 0));
        }

        SECTION("Invalid depths")
        {
            cv::Mat inv = cv::Mat::zeros(cv::Size{1, 1}, CV_MAKETYPE(CV_8U, 2));
            REQUIRE_THROWS(rad::to_normalised_float(inv, CV_32F, 0, 0));

            inv = cv::Mat::zeros(cv::Size{1, 1}, CV_MAKETYPE(CV_8U, 5));
            REQUIRE_THROWS(rad::to_normalised_float(inv, CV_32F, 0, 0));
        }
    }
}

TEST_CASE("[image_utils] - from_normalised_float", "[rad]")
{
    using Point3 = cv::Point3_<std::uint8_t>;
    cv::Mat orig = cv::Mat::zeros(cv::Size{1, 1}, CV_32FC3);
    orig         = cv::Scalar::all(1);
    const Point3 pix{255, 255, 255};

    SECTION("Base function")
    {
        cv::Mat as_int = rad::from_normalised_float(orig, CV_8U);
        REQUIRE(as_int.type() == CV_8UC3);
        REQUIRE(as_int.at<Point3>(0, 0) == pix);
    }

    SECTION("Overloads")
    {
        cv::Mat as_int = rad::from_normalised_float(orig);

        REQUIRE(as_int.type() == CV_8UC3);
        REQUIRE(as_int.at<Point3>(0, 0) == pix);
    }

    SECTION("Invalid inputs")
    {
        cv::Mat inv = cv::Mat::zeros(cv::Size{1, 1}, CV_8UC3);
        REQUIRE_THROWS(rad::from_normalised_float(inv, CV_8U));

        cv::Mat val = cv::Mat::zeros(cv::Size{1, 1}, CV_64FC3);
        REQUIRE_THROWS(rad::from_normalised_float(val, CV_32F));
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
