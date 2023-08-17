#include <rad/onnx/tensor.hpp>

#include <rad/onnx/env.hpp>

#include <catch2/catch_test_macros.hpp>

#include <catch2/reporters/catch_reporter_event_listener.hpp>

namespace onnx = rad::onnx;

TEST_CASE("[tensor] - images_to_tensor", "[rad]")
{
    onnx::init_ort_api();

    const cv::Size size{64, 64};
    const int type{CV_32F};

    std::vector<cv::Mat> images(4);

    SECTION("Grayscale images")
    {
        for (auto& img : images)
        {
            img = cv::Mat::ones(size, CV_MAKETYPE(type, 1));
        }

        auto input = onnx::images_to_tensor<float>(images);
        REQUIRE_FALSE(input.data.empty());

        auto& tensor = input.tensor;
        REQUIRE(tensor.IsTensor());

        auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
        REQUIRE(type_and_shape.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

        auto shape = type_and_shape.GetShape();
        REQUIRE(shape.size() == 4);
        REQUIRE(shape[0] == 4);
        REQUIRE(shape[1] == 1);
        REQUIRE(shape[2] == 64);
        REQUIRE(shape[3] == 64);
    }

    SECTION("RGB images")
    {
        for (auto& img : images)
        {
            img = cv::Mat::ones(size, CV_MAKETYPE(type, 3));
        }

        auto input = onnx::images_to_tensor<float>(images);
        REQUIRE_FALSE(input.data.empty());

        auto& tensor = input.tensor;
        REQUIRE(tensor.IsTensor());

        auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
        REQUIRE(type_and_shape.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

        auto shape = type_and_shape.GetShape();
        REQUIRE(shape.size() == 4);
        REQUIRE(shape[0] == 4);
        REQUIRE(shape[1] == 3);
        REQUIRE(shape[2] == 64);
        REQUIRE(shape[3] == 64);
    }
}

TEST_CASE("[tensor] - image_to_tensor", "[rad]")
{
    onnx::init_ort_api();

    const cv::Size size{64, 64};
    const int type = CV_32F;

    SECTION("Grayscale image")
    {
        cv::Mat img = cv::Mat::ones(size, CV_MAKETYPE(type, 1));

        auto input = onnx::image_to_tensor<float>(img);
        REQUIRE_FALSE(input.data.empty());

        auto& tensor = input.tensor;
        REQUIRE(tensor.IsTensor());

        auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
        REQUIRE(type_and_shape.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

        auto shape = type_and_shape.GetShape();
        REQUIRE(shape.size() == 4);
        REQUIRE(shape[0] == 1);
        REQUIRE(shape[1] == 1);
        REQUIRE(shape[2] == 64);
        REQUIRE(shape[3] == 64);
    }

    SECTION("RGB image")
    {
        cv::Mat img = cv::Mat::ones(size, CV_MAKETYPE(type, 3));

        auto input = onnx::image_to_tensor<float>(img);
        REQUIRE_FALSE(input.data.empty());

        auto& tensor = input.tensor;
        REQUIRE(tensor.IsTensor());

        auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
        REQUIRE(type_and_shape.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

        auto shape = type_and_shape.GetShape();
        REQUIRE(shape.size() == 4);
        REQUIRE(shape[0] == 1);
        REQUIRE(shape[1] == 3);
        REQUIRE(shape[2] == 64);
        REQUIRE(shape[3] == 64);
    }
}

TEST_CASE("[tensor] - array_to_tensor", "[rad]")
{
    onnx::init_ort_api();

    const std::vector<float> data{1.0f, 1.0f, 1.0f};

    auto input = onnx::array_to_tensor<float>(data);
    REQUIRE_FALSE(input.data.empty());

    auto& tensor = input.tensor;
    REQUIRE(tensor.IsTensor());

    auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
    REQUIRE(type_and_shape.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    auto shape = type_and_shape.GetShape();
    REQUIRE(shape.size() == 2);
    REQUIRE(shape[0] == 1);
    REQUIRE(shape[1] == 3);
}

TEST_CASE("[tensor] - image_from_tensor", "[rad]")
{
    onnx::init_ort_api();

    const int type = CV_32F;
    const cv::Size size{64, 64};

    SECTION("Grayscale image")
    {
        cv::Mat img = cv::Mat::ones(size, CV_MAKETYPE(type, 1));
        auto input  = onnx::image_to_tensor<float>(img);

        cv::Mat ret =
            onnx::image_from_tensor<float>(input.tensor, cv::Size{64, 64}, CV_32F);

        REQUIRE(ret.size() == size);
        REQUIRE(ret.type() == CV_32F);
    }

    SECTION("RGB image")
    {
        cv::Mat img = cv::Mat::ones(size, CV_MAKETYPE(type, 3));
        auto input  = onnx::image_to_tensor<float>(img);

        cv::Mat ret =
            onnx::image_from_tensor<float>(input.tensor, cv::Size{64, 64}, CV_32FC3);

        REQUIRE(ret.size() == size);
        REQUIRE(ret.type() == CV_32FC3);
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

TEST_CASE("[tensor] - array_from_tensor", "[rad]")
{
    onnx::init_ort_api();

    const std::vector<float> src_data{1.0f, 2.0f, 3.0f};

    auto input = onnx::array_to_tensor<float>(src_data);

    SECTION("Runtime array")
    {
        auto ret = onnx::array_from_tensor<float>(input.tensor, src_data.size());
        REQUIRE(ret == src_data);
    }

    SECTION("Compile-time array")
    {
        auto ret = onnx::array_from_tensor<float, 3>(input.tensor);
        REQUIRE(array_equals(src_data, ret));
    }
}
