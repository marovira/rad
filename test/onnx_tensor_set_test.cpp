#include "onnx_test_helpers.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <opencv2/core/mat.hpp>
#include <rad/onnx/onnxruntime.hpp>
#include <rad/onnx/tensor_set.hpp>
#include <zeus/range.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace onnx = rad::onnx;

TEMPLATE_TEST_CASE("[TensorSet] - Constructors",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    SECTION("Default constructor")
    {
        const onnx::TensorSet s;
        REQUIRE(s.tensors().empty());
    }

    SECTION("Move constructor")
    {
        onnx::TensorSet s;
        s.insert_tensor_from_image<TestType>(make_test_image<TestType>(1));

        REQUIRE(s.tensors().size() == 1);

        auto s1{std::move(s)};

        // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
        REQUIRE(s.tensors().empty());
        REQUIRE(s1.tensors().size() == 1);
    }

    SECTION("Move-assign operator")
    {
        onnx::TensorSet s;
        s.insert_tensor_from_image<TestType>(make_test_image<TestType>(1));

        REQUIRE(s.tensors().size() == 1);

        auto s1 = std::move(s);

        // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
        REQUIRE(s.tensors().empty());
        REQUIRE(s1.tensors().size() == 1);
    }
}

TEMPLATE_TEST_CASE("[TensorSet] - array operator",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    onnx::TensorSet s;
    s.insert_tensor_from_image<TestType>(make_test_image<TestType>(1));
    REQUIRE(s.tensors().size() == 1);

    REQUIRE(s[0].IsTensor());
}

TEMPLATE_TEST_CASE("[TensorSet] - ranged for-loop iterators",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    onnx::TensorSet s;
    s.insert_tensor_from_image<TestType>(make_test_image<TestType>(1));

    for (auto const& tensor : s)
    {
        REQUIRE(tensor.IsTensor());
    }
}

TEMPLATE_TEST_CASE("[TensorSet] - size/empty",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    SECTION("Empty set")
    {
        const onnx::TensorSet s;
        REQUIRE(s.size() == 0);
        REQUIRE(s.empty());
    }

    SECTION("Non-empty set")
    {
        onnx::TensorSet s;
        s.insert_tensor_from_image<TestType>(make_test_image<TestType>(1));

        REQUIRE_FALSE(s.empty());
        REQUIRE(s.size() == 1);
    }
}

TEMPLATE_TEST_CASE("[TensorSet] - insert_tensor",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    onnx::TensorSet s;
    auto test_tensor = make_test_tensor<TestType>();

    s.insert_tensor(std::move(test_tensor.tensor));

    REQUIRE(s.size() == 1);
    REQUIRE(static_cast<OrtValue*>(test_tensor.tensor) == nullptr);

    auto& tensor = s[0];
    REQUIRE(tensor.IsTensor());

    auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
    REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

    auto shape = type_and_shape.GetShape();
    REQUIRE(shape.size() == 2);
    REQUIRE(std::cmp_equal(shape[0], 1));
    REQUIRE(std::cmp_equal(shape[1], test_tensor.data.size()));
}

TEMPLATE_TEST_CASE("[TensorSet] - insert_tensors",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    std::vector<Ort::Value> tensors;
    std::vector<std::vector<TestType>> data;
    for ([[maybe_unused]]
         auto _ : zeus::Range{10})
    {
        auto tensor = make_test_tensor<TestType>();
        tensors.emplace_back(std::move(tensor.tensor));
        data.emplace_back(std::move(tensor.data));
    }

    onnx::TensorSet s;
    s.insert_tensors(tensors);
    REQUIRE(tensors.empty());
    REQUIRE(s.size() == data.size());

    for (std::size_t i{0}; auto const& tensor : s)
    {
        REQUIRE(tensor.IsTensor());
        auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
        REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

        auto shape = type_and_shape.GetShape();
        REQUIRE(shape.size() == 2);
        REQUIRE(std::cmp_equal(shape[0], 1));
        REQUIRE(std::cmp_equal(shape[1], data[i].size()));
        ++i;
    }
}

TEMPLATE_TEST_CASE("[TensorSet] - insert_tensor_from_batched_images",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    std::vector<cv::Mat> images(4);
    const auto exp_size = get_test_image_size();

    onnx::TensorSet s;
    SECTION("Grayscale images")
    {
        for (auto& img : images)
        {
            img = make_test_image<TestType>(1);
        }

        s.insert_tensor_from_batched_images<TestType>(images);
        REQUIRE(s.size() == 1);

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

        s.insert_tensor_from_batched_images<TestType>(images);
        REQUIRE(s.size() == 1);

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

TEMPLATE_TEST_CASE("[TensorSet] - insert_tensor_from_image",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    onnx::TensorSet s;
    const auto exp_size = get_test_image_size();

    SECTION("Grayscale image")
    {
        const cv::Mat img = make_test_image<TestType>(1);

        s.insert_tensor_from_image<TestType>(img);
        REQUIRE(s.size() == 1);

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
        const cv::Mat img = make_test_image<TestType>(3);

        s.insert_tensor_from_image<TestType>(img);
        REQUIRE(s.size() == 1);

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

TEMPLATE_TEST_CASE("[TensorSet] - insert_tensor_from_batched_arrays",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    onnx::TensorSet s;
    static constexpr std::size_t size{10};
    static constexpr std::size_t batch_size{4};

    const std::vector<std::vector<TestType>> batched_data(
        batch_size,
        std::vector<TestType>(size, make_test_value<TestType>()));

    s.insert_tensor_from_batched_arrays<TestType>(batched_data);
    REQUIRE(s.size() == 1);

    auto& tensor = s[0];
    REQUIRE(tensor.IsTensor());

    auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
    REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

    auto shape = type_and_shape.GetShape();
    REQUIRE(shape.size() == 2);
    REQUIRE(std::cmp_equal(shape[0], batch_size));
    REQUIRE(std::cmp_equal(shape[1], size));
}

TEMPLATE_TEST_CASE("[TensorSet] - insert_tensor_from_array",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    onnx::TensorSet s;
    static constexpr std::size_t size{10};

    const std::vector<TestType> data(size, make_test_value<TestType>());

    s.insert_tensor_from_array<TestType>(data);
    REQUIRE(s.size() == 1);

    auto& tensor = s[0];
    REQUIRE(tensor.IsTensor());

    auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
    REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

    auto shape = type_and_shape.GetShape();
    REQUIRE(shape.size() == 2);
    REQUIRE(std::cmp_equal(shape[0], 1));
    REQUIRE(std::cmp_equal(shape[1], size));
}

TEMPLATE_TEST_CASE("[TensorSet] - insert_tensor_from_scalar",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    onnx::TensorSet s;
    const auto value = make_test_value<TestType>();

    s.insert_tensor_from_scalar<TestType>(value);
    REQUIRE(s.size() == 1);

    auto& tensor = s[0];
    REQUIRE(tensor.IsTensor());

    auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
    REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

    auto shape = type_and_shape.GetShape();
    REQUIRE(shape.size() == 1);
    REQUIRE(std::cmp_equal(shape[0], 1));
}

TEMPLATE_TEST_CASE("[TensorSet] - insert_tensor_from_data",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    onnx::TensorSet s;

    static constexpr std::size_t size{10};
    const std::vector<TestType> data(size, make_test_value<TestType>());
    const std::vector<std::int64_t> shape{1, size};

    s.insert_tensor_from_data<TestType>(data.data(), shape);
    REQUIRE(s.size() == 1);

    auto& tensor = s[0];
    REQUIRE(tensor.IsTensor());

    auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
    REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

    auto res_shape = type_and_shape.GetShape();
    REQUIRE(res_shape == shape);
}

TEMPLATE_TEST_CASE("[TensorSet] - replace_tensor_at",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    onnx::TensorSet s;

    REQUIRE_THROWS(s.replace_tensor_at(1, Ort::Value{nullptr}));

    auto init_tensor = make_test_tensor_with_size<TestType>(5);
    s.insert_tensor(std::move(init_tensor.tensor));
    REQUIRE(s.size() == 1);
    REQUIRE(static_cast<OrtValue*>(init_tensor.tensor) == nullptr);

    {
        auto& tensor = s[0];
        REQUIRE(tensor.IsTensor());

        auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
        REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

        auto shape = type_and_shape.GetShape();
        REQUIRE(shape.size() == 2);
        REQUIRE(std::cmp_equal(shape[0], 1));
        REQUIRE(std::cmp_equal(shape[1], init_tensor.data.size()));
    }

    auto new_tensor = make_test_tensor_with_size<TestType>(10);
    s.replace_tensor_at(0, std::move(new_tensor.tensor));
    REQUIRE(s.size() == 1);

    auto& tensor = s[0];
    REQUIRE(tensor.IsTensor());

    auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
    REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

    auto shape = type_and_shape.GetShape();
    REQUIRE(shape.size() == 2);
    REQUIRE(std::cmp_equal(shape[0], 1));
    REQUIRE(std::cmp_equal(shape[1], new_tensor.data.size()));
}

TEMPLATE_TEST_CASE("[TensorSet] - replace_tensor_with_batched_images_at",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    const auto exp_size = get_test_image_size();
    onnx::TensorSet s;
    std::vector<cv::Mat> init_images(2);
    std::vector<cv::Mat> new_images(4);
    SECTION("Grayscale images")
    {
        for (auto& img : init_images)
        {
            img = make_test_image<TestType>(1);
        }

        s.insert_tensor_from_batched_images<TestType>(init_images);
        REQUIRE(s.size() == 1);
        {
            auto& tensor = s[0];
            REQUIRE(tensor.IsTensor());

            auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
            REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

            auto shape = type_and_shape.GetShape();
            REQUIRE(shape.size() == 4);
            REQUIRE(std::cmp_equal(shape[0], init_images.size()));
            REQUIRE(shape[1] == 1);
            REQUIRE(shape[2] == exp_size.height);
            REQUIRE(shape[3] == exp_size.width);
        }

        for (auto& img : new_images)
        {
            img = make_test_image<TestType>(1);
        }

        s.replace_tensor_with_batched_images_at<TestType>(0, new_images);
        REQUIRE(s.size() == 1);
        auto& tensor = s[0];
        REQUIRE(tensor.IsTensor());

        auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
        REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

        auto shape = type_and_shape.GetShape();
        REQUIRE(shape.size() == 4);
        REQUIRE(std::cmp_equal(shape[0], new_images.size()));
        REQUIRE(shape[1] == 1);
        REQUIRE(shape[2] == exp_size.height);
        REQUIRE(shape[3] == exp_size.width);
    }

    SECTION("RGB images")
    {
        for (auto& img : init_images)
        {
            img = make_test_image<TestType>(3);
        }

        s.insert_tensor_from_batched_images<TestType>(init_images);
        REQUIRE(s.size() == 1);
        {
            auto& tensor = s[0];
            REQUIRE(tensor.IsTensor());

            auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
            REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

            auto shape = type_and_shape.GetShape();
            REQUIRE(shape.size() == 4);
            REQUIRE(std::cmp_equal(shape[0], init_images.size()));
            REQUIRE(shape[1] == 3);
            REQUIRE(shape[2] == exp_size.height);
            REQUIRE(shape[3] == exp_size.width);
        }

        for (auto& img : new_images)
        {
            img = make_test_image<TestType>(3);
        }

        s.replace_tensor_with_batched_images_at<TestType>(0, new_images);
        REQUIRE(s.size() == 1);
        auto& tensor = s[0];
        REQUIRE(tensor.IsTensor());

        auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
        REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

        auto shape = type_and_shape.GetShape();
        REQUIRE(shape.size() == 4);
        REQUIRE(std::cmp_equal(shape[0], new_images.size()));
        REQUIRE(shape[1] == 3);
        REQUIRE(shape[2] == exp_size.height);
        REQUIRE(shape[3] == exp_size.width);
    }
}

TEMPLATE_TEST_CASE("[TensorSet] - replace_tensor_with_image_at",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    onnx::TensorSet s;
    const auto exp_size = get_test_image_size();

    SECTION("Grayscale image")
    {
        const cv::Mat init_img = make_test_image<TestType>(1);
        const cv::Mat new_img  = make_test_image<TestType>(3);

        s.insert_tensor_from_image<TestType>(init_img);
        REQUIRE(s.size() == 1);

        {
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

        s.replace_tensor_with_image_at<TestType>(0, new_img);
        REQUIRE(s.size() == 1);

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

TEMPLATE_TEST_CASE("[TensorSet] - replace_tensor_with_batched_arrays_at",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    onnx::TensorSet s;
    static constexpr std::size_t size{5};
    static constexpr std::size_t batch_size{4};

    const std::vector<std::vector<TestType>> init_data(
        batch_size,
        std::vector<TestType>(size, make_test_value<TestType>()));
    const std::vector<std::vector<TestType>> new_data(
        batch_size,
        std::vector<TestType>(size * 2, make_test_value<TestType>()));

    s.insert_tensor_from_batched_arrays<TestType>(init_data);
    REQUIRE(s.size() == 1);
    {
        auto& tensor = s[0];
        REQUIRE(tensor.IsTensor());

        auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
        REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

        auto shape = type_and_shape.GetShape();
        REQUIRE(shape.size() == 2);
        REQUIRE(std::cmp_equal(shape[0], batch_size));
        REQUIRE(std::cmp_equal(shape[1], size));
    }

    s.replace_tensor_with_batched_arrays_at<TestType>(0, new_data);
    auto& tensor = s[0];
    REQUIRE(tensor.IsTensor());

    auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
    REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

    auto shape = type_and_shape.GetShape();
    REQUIRE(shape.size() == 2);
    REQUIRE(std::cmp_equal(shape[0], batch_size));
    REQUIRE(std::cmp_equal(shape[1], new_data[0].size()));
}

TEMPLATE_TEST_CASE("[TensorSet] - replace_tensor_with_array_at",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    onnx::TensorSet s;
    static constexpr std::size_t size{5};

    const std::vector<TestType> init_data(size, make_test_value<TestType>());
    const std::vector<TestType> new_data(size * 2, make_test_value<TestType>());

    s.insert_tensor_from_array<TestType>(init_data);
    REQUIRE(s.size() == 1);
    {
        auto& tensor = s[0];
        REQUIRE(tensor.IsTensor());

        auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
        REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

        auto shape = type_and_shape.GetShape();
        REQUIRE(shape.size() == 2);
        REQUIRE(std::cmp_equal(shape[0], 1));
        REQUIRE(std::cmp_equal(shape[1], size));
    }

    s.replace_tensor_with_array_at<TestType>(0, new_data);
    auto& tensor = s[0];
    REQUIRE(tensor.IsTensor());

    auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
    REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

    auto shape = type_and_shape.GetShape();
    REQUIRE(shape.size() == 2);
    REQUIRE(std::cmp_equal(shape[0], 1));
    REQUIRE(std::cmp_equal(shape[1], new_data.size()));
}

TEMPLATE_TEST_CASE("[TensorSet] - replace_tensor_with_scalar_at",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    onnx::TensorSet s;
    const auto init_value = make_test_value<TestType>();
    const auto new_value  = make_zero_test_value<TestType>();

    s.insert_tensor_from_scalar<TestType>(init_value);
    REQUIRE(s.size() == 1);
    {
        auto& tensor = s[0];
        REQUIRE(tensor.IsTensor());

        auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
        REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

        auto shape = type_and_shape.GetShape();
        REQUIRE(shape.size() == 1);
        REQUIRE(std::cmp_equal(shape[0], 1));
    }

    s.replace_tensor_with_scalar_at(0, new_value);
    REQUIRE(s.size() == 1);
    auto& tensor = s[0];
    REQUIRE(tensor.IsTensor());

    auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
    REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

    auto shape = type_and_shape.GetShape();
    REQUIRE(shape.size() == 1);
    REQUIRE(std::cmp_equal(shape[0], 1));
}

TEMPLATE_TEST_CASE("[TensorSet] - replace_tensor_and_data_at",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    onnx::TensorSet s;
    static constexpr std::size_t size{5};
    const std::vector<TestType> init_data(size, make_test_value<TestType>());
    const std::vector<std::int64_t> init_shape{1, size};
    const std::vector<TestType> new_data(size * 2, make_test_value<TestType>());
    const std::vector<std::int64_t> new_shape{1, size * 2};

    s.insert_tensor_from_data<TestType>(init_data.data(), init_shape);
    REQUIRE(s.size() == 1);
    {
        auto& tensor = s[0];
        REQUIRE(tensor.IsTensor());

        auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
        REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

        auto res_shape = type_and_shape.GetShape();
        REQUIRE(res_shape == init_shape);
    }

    s.replace_tensor_with_data_at(0, new_data.data(), new_shape);
    auto& tensor = s[0];
    REQUIRE(tensor.IsTensor());

    auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
    REQUIRE(type_and_shape.GetElementType() == get_onnx_element_type<TestType>());

    auto res_shape = type_and_shape.GetShape();
    REQUIRE(res_shape == new_shape);
}
