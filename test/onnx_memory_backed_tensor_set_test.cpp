#include <rad/onnx/memory_backed_tensor_set.hpp>

#include "onnx_test_helpers.hpp"

#include <catch2/catch_template_test_macros.hpp>

namespace onnx = rad::onnx;

TEMPLATE_TEST_CASE("[MemoryBackedTensorSet] - compile-time constants",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t)
{
    STATIC_REQUIRE(
        std::is_same_v<typename onnx::MemoryBackedTensorSet<TestType>::value_type,
                       TestType>);
}

TEMPLATE_TEST_CASE("[MemoryBackedTensorSet] - Constructors",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t)
{
    SECTION("Default constructor")
    {
        onnx::MemoryBackedTensorSet<TestType> s;
        REQUIRE(s.tensors().empty());
    }

    SECTION("Move constructor")
    {
        onnx::MemoryBackedTensorSet<TestType> s;
        s.insert_tensor_from_image(make_test_image<TestType>(1));

        REQUIRE(s.tensors().size() == 1);

        auto s1{std::move(s)};

        REQUIRE(s.tensors().empty());
        REQUIRE(s1.tensors().size() == 1);
    }

    SECTION("Move-assign operator")
    {
        onnx::MemoryBackedTensorSet<TestType> s;
        s.insert_tensor_from_image(make_test_image<TestType>(1));

        REQUIRE(s.tensors().size() == 1);

        auto s1 = std::move(s);

        REQUIRE(s.tensors().empty());
        REQUIRE(s1.tensors().size() == 1);
    }
}

TEMPLATE_TEST_CASE("[MemoryBackedTensorSet] - array operator",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t)
{
    onnx::MemoryBackedTensorSet<TestType> s;
    s.insert_tensor_from_image(make_test_image<TestType>(1));
    REQUIRE(s.tensors().size() == 1);

    REQUIRE(s[0].IsTensor());
}

TEMPLATE_TEST_CASE("[MemoryBackedTensorSet] - ranged for-loop iterators",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t)
{
    onnx::MemoryBackedTensorSet<TestType> s;
    s.insert_tensor_from_image(make_test_image<TestType>(1));

    for (auto const& tensor : s)
    {
        REQUIRE(tensor.IsTensor());
    }
}

TEMPLATE_TEST_CASE("[MemoryBackedTensorSet] - size/empty",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t)
{
    SECTION("Empty set")
    {
        onnx::MemoryBackedTensorSet<TestType> s;
        REQUIRE(s.size() == 0);
        REQUIRE(s.empty());
    }

    SECTION("Non-empty set")
    {
        onnx::MemoryBackedTensorSet<TestType> s;
        s.insert_tensor_from_image(make_test_image<TestType>(1));

        REQUIRE_FALSE(s.empty());
        REQUIRE(s.size() == 1);
    }
}

TEMPLATE_TEST_CASE("[MemoryBackedTensorSet] - insert_tensor_from_batched_images",
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
        for (auto& img : images)
        {
            img = make_test_image<TestType>(1);
        }

        s.insert_tensor_from_batched_images(images);
        REQUIRE_FALSE(s.empty());

        auto& tensor = s[0];
        REQUIRE(tensor.IsTensor());

        auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
        REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

        auto shape = type_and_shape.GetShape();
        REQUIRE(shape.size() == 4);
        REQUIRE(shape[0] == 4);
        REQUIRE(shape[1] == 1);
        REQUIRE(shape[2] == exp_size.height);
        REQUIRE(shape[3] == exp_size.width);
    }

    SECTION("RGB images")
    {
        for (auto& img : images)
        {
            img = make_test_image<TestType>(3);
        }

        s.insert_tensor_from_batched_images(images);
        REQUIRE_FALSE(s.empty());

        auto& tensor = s[0];
        REQUIRE(tensor.IsTensor());

        auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
        REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

        auto shape = type_and_shape.GetShape();
        REQUIRE(shape.size() == 4);
        REQUIRE(shape[0] == 4);
        REQUIRE(shape[1] == 3);
        REQUIRE(shape[2] == exp_size.height);
        REQUIRE(shape[3] == exp_size.width);
    }
}

TEMPLATE_TEST_CASE("[MemoryBackedTensorSet] - insert_tensor_from_image",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t)
{
    onnx::MemoryBackedTensorSet<TestType> s;
    const auto exp_size = get_test_image_size();

    SECTION("Grayscale image")
    {
        cv::Mat img = make_test_image<TestType>(1);

        s.insert_tensor_from_image(img);
        REQUIRE_FALSE(s.empty());

        auto& tensor = s[0];
        REQUIRE(tensor.IsTensor());

        auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
        REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

        auto shape = type_and_shape.GetShape();
        REQUIRE(shape.size() == 4);
        REQUIRE(shape[0] == 1);
        REQUIRE(shape[1] == 1);
        REQUIRE(shape[2] == exp_size.height);
        REQUIRE(shape[3] == exp_size.width);
    }

    SECTION("RGB image")
    {
        cv::Mat img = make_test_image<TestType>(3);

        s.insert_tensor_from_image(img);
        REQUIRE_FALSE(s.empty());

        auto& tensor = s[0];
        REQUIRE(tensor.IsTensor());

        auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
        REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

        auto shape = type_and_shape.GetShape();
        REQUIRE(shape.size() == 4);
        REQUIRE(shape[0] == 1);
        REQUIRE(shape[1] == 3);
        REQUIRE(shape[2] == exp_size.height);
        REQUIRE(shape[3] == exp_size.width);
    }
}

TEMPLATE_TEST_CASE("[MemoryBackedTensorSet] - insert_tensor_from_batched_arrays",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t)
{
    onnx::MemoryBackedTensorSet<TestType> s;
    static constexpr std::size_t size{10};
    static constexpr std::size_t batch_size{4};

    const std::vector<std::vector<TestType>> batched_data(
        batch_size,
        std::vector<TestType>(size, make_test_value<TestType>()));

    s.insert_tensor_from_batched_arrays(batched_data);
    REQUIRE_FALSE(s.empty());

    for (auto const& tensor : s.tensors())
    {
        REQUIRE(tensor.IsTensor());

        auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
        REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

        auto shape = type_and_shape.GetShape();
        REQUIRE(shape.size() == 2);
        REQUIRE(shape[0] == batch_size);
        REQUIRE(shape[1] == size);
    }
}

TEMPLATE_TEST_CASE("[MemoryBackedTensorSet] - insert_tensor_from_array",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t)
{
    onnx::MemoryBackedTensorSet<TestType> s;
    static constexpr std::size_t size{10};

    const std::vector<TestType> data(size, make_test_value<TestType>());

    s.insert_tensor_from_array(data);
    REQUIRE_FALSE(s.empty());

    auto& tensor = s[0];
    REQUIRE(tensor.IsTensor());

    auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
    REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

    auto shape = type_and_shape.GetShape();
    REQUIRE(shape.size() == 2);
    REQUIRE(shape[0] == 1);
    REQUIRE(shape[1] == size);
}
