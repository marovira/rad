#include <rad/onnx/tensor_conversion.hpp>

#include "onnx_test_helpers.hpp"
#include <rad/onnx/memory_backed_tensor_set.hpp>

#include <catch2/catch_template_test_macros.hpp>

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

TEMPLATE_TEST_CASE("[tensor_conversion] - image_batch_from_tensor",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t)
{
    std::vector<cv::Mat> images(4);
    const auto exp_size = get_test_image_size();
    onnx::MemoryBackedTensorSet<TestType> s;

    SECTION("Grayscale images")
    {
        constexpr int channels{1};
        for (auto& img : images)
        {
            img = make_test_image<TestType>(channels);
        }

        s.insert_tensor_from_batched_images(images);
        auto ret_imgs =
            onnx::image_batch_from_tensor<typename TestDataType<TestType>::Type>(
                s[0],
                exp_size,
                get_test_image_type<TestType>(channels));

        REQUIRE(ret_imgs.size() == images.size());
        for (auto ret_img : ret_imgs)
        {
            REQUIRE(ret_img.size() == exp_size);
            REQUIRE(ret_img.type() == get_test_image_type<TestType>(channels));
        }
    }

    SECTION("RGB images")
    {
        constexpr int channels{1};
        for (auto& img : images)
        {
            img = make_test_image<TestType>(channels);
        }

        s.insert_tensor_from_batched_images(images);
        auto ret_imgs =
            onnx::image_batch_from_tensor<typename TestDataType<TestType>::Type>(
                s[0],
                exp_size,
                get_test_image_type<TestType>(channels));

        REQUIRE(ret_imgs.size() == images.size());
        for (auto ret_img : ret_imgs)
        {
            REQUIRE(ret_img.size() == exp_size);
            REQUIRE(ret_img.type() == get_test_image_type<TestType>(channels));
        }
    }
}

TEMPLATE_TEST_CASE("[tensor_conversion] - image_from_tensor",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t)
{
    onnx::MemoryBackedTensorSet<TestType> s;

    SECTION("Grayscale image")
    {
        constexpr int channels{1};

        s.insert_tensor_from_image(make_test_image<TestType>(channels));

        cv::Mat ret = onnx::image_from_tensor<typename TestDataType<TestType>::Type>(
            s[0],
            get_test_image_size(),
            get_test_image_type<TestType>(channels));

        REQUIRE(ret.size() == get_test_image_size());
        REQUIRE(ret.type() == get_test_image_type<TestType>(channels));
    }

    SECTION("RGB image")
    {
        constexpr int channels{3};

        s.insert_tensor_from_image(make_test_image<TestType>(channels));

        cv::Mat ret = onnx::image_from_tensor<typename TestDataType<TestType>::Type>(
            s[0],
            get_test_image_size(),
            get_test_image_type<TestType>(channels));

        REQUIRE(ret.size() == get_test_image_size());
        REQUIRE(ret.type() == get_test_image_type<TestType>(channels));
    }
}

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

TEMPLATE_TEST_CASE("[tensor_conversion] - array_batch_from_tensor",
                   "[rad::onnx]",
                   float,
                   std::uint8_t)
{
    onnx::MemoryBackedTensorSet<TestType> s;
    static constexpr std::size_t size{10};
    static constexpr std::size_t batch_size{4};

    const std::vector<std::vector<TestType>> batched_data(
        batch_size,
        std::vector<TestType>(size, TestType{1}));

    s.insert_tensor_from_batched_arrays(batched_data);
    REQUIRE_FALSE(s.empty());

    auto ret = onnx::array_batch_from_tensor<TestType>(s[0], size);
    REQUIRE(ret == batched_data);
}

TEMPLATE_TEST_CASE("[tensor_conversion] - array_from_tensor",
                   "[rad::onnx]",
                   float,
                   std::uint8_t)
{
    static constexpr std::size_t size{10};
    const std::vector<TestType> src_data(10, TestType{1});

    onnx::MemoryBackedTensorSet<TestType> s;
    s.insert_tensor_from_array(src_data);
    REQUIRE_FALSE(s.empty());

    auto ret = onnx::array_from_tensor<TestType>(s[0], size);
    REQUIRE(array_equals(src_data, ret));
}
