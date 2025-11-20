#include "onnx_test_helpers.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <opencv2/core/mat.hpp>
#include <rad/onnx/onnxruntime.hpp>
#include <rad/onnx/tensor_conversion.hpp>
#include <rad/onnx/tensor_set.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace onnx = rad::onnx;

template<typename T>
struct TestDataType
{
    using Type = T;
};

template<>
struct TestDataType<Ort::Float16_t>
{
    using Type = std::uint16_t;
};

namespace
{
    template<typename T, typename U>
    bool array_equals(T const& lhs, U const& rhs)
    {
        if (lhs.size() != rhs.size())
        {
            return false;
        }

        for (std::size_t i{0}; i < lhs.size(); ++i)
        {
            if (lhs[i] != rhs[i])
            {
                return false;
            }
        }

        return true;
    }
} // namespace

TEMPLATE_TEST_CASE("[tensor_conversion] - image_batch_from_tensor",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    std::vector<cv::Mat> images(4);
    const auto size = get_test_image_size();
    onnx::TensorSet s;
    const auto post_process = [](cv::Mat img) {
        return img;
    };

    SECTION("Grayscale images")
    {
        constexpr int channels{1};
        const auto type = get_test_image_type<TestType>(channels);
        for (auto& img : images)
        {
            img = make_test_image<TestType>(channels);
        }

        s.insert_tensor_from_batched_images<TestType>(images);
        auto ret =
            onnx::image_batch_from_tensor<TestType>(s.front(), size, type, post_process);
        auto ret2 = onnx::image_batch_from_tensor<TestType>(s.front(), size, type);

        REQUIRE(image_array_equals(ret, ret2));
        REQUIRE(image_array_equals(ret, images));
    }

    SECTION("RGB images")
    {
        constexpr int channels{3};
        const auto type = get_test_image_type<TestType>(channels);
        for (auto& img : images)
        {
            img = make_test_image<TestType>(channels);
        }

        s.insert_tensor_from_batched_images<TestType>(images);
        auto ret =
            onnx::image_batch_from_tensor<TestType>(s.front(), size, type, post_process);
        auto ret2 = onnx::image_batch_from_tensor<TestType>(s.front(), size, type);

        REQUIRE(image_array_equals(ret, ret2));
        REQUIRE(image_array_equals(ret, images));
    }

    SECTION("RGBA images")
    {
        constexpr int channels{4};
        const auto type = get_test_image_type<TestType>(channels);
        for (auto& img : images)
        {
            img = make_test_image<TestType>(channels);
        }

        s.insert_tensor_from_batched_images<TestType>(images);
        auto ret =
            onnx::image_batch_from_tensor<TestType>(s.front(), size, type, post_process);
        auto ret2 = onnx::image_batch_from_tensor<TestType>(s.front(), size, type);

        REQUIRE(image_array_equals(ret, ret2));
        REQUIRE(image_array_equals(ret, images));
    }
}

TEMPLATE_TEST_CASE("[tensor_conversion] - image_from_tensor",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    onnx::TensorSet s;
    const auto post_process = [](cv::Mat img) {
        return img;
    };

    SECTION("Invalid batch")
    {
        std::vector<cv::Mat> images(2);
        for (auto& img : images)
        {
            img = make_test_image<TestType>(1);
        }

        s.insert_tensor_from_batched_images<TestType>(images);
        REQUIRE_THROWS(onnx::image_from_tensor<TestType>(s.tensors().front(),
                                                         images.front().size(),
                                                         images.front().type(),
                                                         post_process));
        REQUIRE_THROWS(onnx::image_from_tensor<TestType>(s.tensors().front(),
                                                         images.front().size(),
                                                         images.front().type()));
    }

    SECTION("Grayscale image")
    {
        constexpr int channels{1};
        const auto type   = get_test_image_type<TestType>(channels);
        const cv::Mat img = make_test_image<TestType>(channels);
        const auto size   = img.size();

        s.insert_tensor_from_image<TestType>(img);

        auto ret = onnx::image_from_tensor<TestType>(s.front(), size, type, post_process);
        auto ret2 = onnx::image_from_tensor<TestType>(s.front(), size, type);

        REQUIRE(image_equals(ret, ret2));
        REQUIRE(image_equals(ret, img));
    }

    SECTION("RGB image")
    {
        constexpr int channels{3};
        const auto type   = get_test_image_type<TestType>(channels);
        const cv::Mat img = make_test_image<TestType>(channels);
        const auto size   = img.size();

        s.insert_tensor_from_image<TestType>(img);

        auto ret = onnx::image_from_tensor<TestType>(s.front(), size, type, post_process);
        auto ret2 = onnx::image_from_tensor<TestType>(s.front(), size, type);

        REQUIRE(image_equals(ret, ret2));
        REQUIRE(image_equals(ret, img));
    }

    SECTION("RGBA image")
    {
        constexpr int channels{4};
        const auto type   = get_test_image_type<TestType>(channels);
        const cv::Mat img = make_test_image<TestType>(channels);
        const auto size   = img.size();

        s.insert_tensor_from_image<TestType>(img);

        auto ret = onnx::image_from_tensor<TestType>(s.front(), size, type, post_process);
        auto ret2 = onnx::image_from_tensor<TestType>(s.front(), size, type);

        REQUIRE(image_equals(ret, ret2));
        REQUIRE(image_equals(ret, img));
    }
}

TEMPLATE_TEST_CASE("[tensor_conversion] - array_batch_from_tensor",
                   "[rad::onnx]",
                   float,
                   std::uint8_t)
{
    onnx::TensorSet s;
    static constexpr std::size_t size{10};
    static constexpr std::size_t batch_size{4};

    const std::vector<std::vector<TestType>> batched_data(
        batch_size,
        std::vector<TestType>(size, TestType{1}));

    s.insert_tensor_from_batched_arrays<TestType>(batched_data);
    REQUIRE_FALSE(s.empty());

    auto ret = onnx::array_batch_from_tensor<TestType>(s.front(), size);
    REQUIRE(ret == batched_data);
}

TEMPLATE_TEST_CASE("[tensor_conversion] - array_from_tensor",
                   "[rad::onnx]",
                   float,
                   std::uint8_t)
{
    static constexpr std::size_t size{10};
    const std::vector<TestType> src_data(10, TestType{1});

    onnx::TensorSet s;
    s.insert_tensor_from_array<TestType>(src_data);
    REQUIRE_FALSE(s.empty());

    auto ret = onnx::array_from_tensor<TestType>(s.front(), size);
    REQUIRE(array_equals(src_data, ret));
}

TEMPLATE_TEST_CASE("[tensor_conversion] - scalar_from_tensor",
                   "[rad::onnx]",
                   float,
                   std::uint8_t)
{
    onnx::TensorSet s;
    static constexpr TestType value{1};
    s.insert_tensor_from_scalar<TestType>(value);
    REQUIRE(s.size() == 1);

    auto ret = onnx::scalar_from_tensor<TestType>(s.front());
    REQUIRE(ret == value);
}

TEMPLATE_TEST_CASE("[tensor_conversion] - batched_rects_from_tensor",
                   "[rad::onnx]",
                   float,
                   std::uint8_t)
{
    using Rect = std::array<TestType, 4>;
    const Rect box{TestType{0}, TestType{0}, TestType{1}, TestType{1}};
    const std::vector<Rect> boxes{box, box, box, box, box, box};
    const std::vector<std::int64_t> shape{2, 3, 4};

    onnx::TensorSet s;
    s.insert_tensor_from_data<TestType>(boxes[0].data(), shape);
    REQUIRE(s.size() == 1);

    auto ret = onnx::batched_rects_from_tensor<TestType>(s.front());
    REQUIRE(std::cmp_equal(ret.size(), shape[0]));
    for (std::int64_t i{0}; i < shape[0]; ++i)
    {
        REQUIRE(std::cmp_equal(ret[i].size(), shape[1]));
        for (auto b : ret[i])
        {
            REQUIRE(box == b);
        }
    }
}

TEMPLATE_TEST_CASE("[tensor_conversion] - rects_from_tensor",
                   "[rad::onnx]",
                   float,
                   std::uint8_t)
{
    using Rect = std::array<TestType, 4>;
    const Rect box{TestType{0}, TestType{0}, TestType{1}, TestType{1}};
    const std::vector<Rect> boxes{box, box, box, box, box, box};
    const std::vector<std::int64_t> shape{1, 6, 4};

    onnx::TensorSet s;
    s.insert_tensor_from_data<TestType>(boxes[0].data(), shape);
    REQUIRE(s.size() == 1);

    auto ret = onnx::batched_rects_from_tensor<TestType>(s.front());
    REQUIRE(std::cmp_equal(ret.size(), shape[0]));
    for (std::int64_t i{0}; i < shape[0]; ++i)
    {
        REQUIRE(std::cmp_equal(ret[i].size(), shape[1]));
        for (auto b : ret[i])
        {
            REQUIRE(box == b);
        }
    }
}

TEMPLATE_TEST_CASE("[tensor_conversion] - data_from_tensor",
                   "[rad::onnx]",
                   float,
                   std::uint8_t)
{
    const std::vector<TestType> data(10, TestType{1});
    const std::vector<std::int64_t> shape{1, 5, 2};

    onnx::TensorSet s;
    s.insert_tensor_from_data<TestType>(data.data(), shape);
    REQUIRE(s.size() == 1);

    auto ret = onnx::data_from_tensor<TestType>(s.front());
    REQUIRE(ret == data);
}
