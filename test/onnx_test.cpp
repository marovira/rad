#include <rad/onnx.hpp>

#include <catch2/catch_test_macros.hpp>
#include <type_traits>

namespace onnx = rad::onnx;

TEST_CASE("[onnx] - image_to_tensor", "[rad]")
{
    Ort::InitApi();

    const cv::Size size{64, 64};
    const int type = CV_32F;

    SECTION("Grayscale image")
    {
        cv::Mat img = cv::Mat::ones(size, CV_MAKETYPE(type, 1));

        onnx::TensorSet<float> inputs;
        onnx::image_to_tensor(img, inputs);

        REQUIRE(inputs.tensors.size() == 1);

        auto& tensor = inputs.tensors.front();
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

        onnx::TensorSet<float> inputs;
        onnx::image_to_tensor(img, inputs);

        REQUIRE(inputs.tensors.size() == 1);

        auto& tensor = inputs.tensors.front();
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

TEST_CASE("[onnx] - image_from_tensor", "[rad]")
{
    const int type = CV_32F;
    const cv::Size size{64, 64};

    SECTION("Grayscale image")
    {
        cv::Mat img = cv::Mat::ones(size, CV_MAKETYPE(type, 1));

        onnx::TensorSet<float> inputs;
        onnx::image_to_tensor(img, inputs);

        cv::Mat ret =
            onnx::image_from_tensor<float>(inputs.tensors[0], cv::Size{64, 64}, CV_32F);

        REQUIRE(ret.size() == size);
        REQUIRE(ret.type() == CV_32F);
    }

    SECTION("RGB image")
    {
        cv::Mat img = cv::Mat::ones(size, CV_MAKETYPE(type, 3));

        onnx::TensorSet<float> inputs;
        onnx::image_to_tensor(img, inputs);

        cv::Mat ret =
            onnx::image_from_tensor<float>(inputs.tensors[0], cv::Size{64, 64}, CV_32FC3);

        REQUIRE(ret.size() == size);
        REQUIRE(ret.type() == CV_32FC3);
    }
}
