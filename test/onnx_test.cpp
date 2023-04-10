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

TEST_CASE("[onnx] - array_to_tensor", "[rad]")
{
    const std::vector<float> data{1.0f, 1.0f, 1.0f};

    onnx::TensorSet<float> inputs;
    onnx::array_to_tensor(data, inputs);

    REQUIRE(inputs.tensors.size() == 1);

    auto& tensor = inputs.tensors.front();
    REQUIRE(tensor.IsTensor());

    auto type_and_shape = tensor.GetTensorTypeAndShapeInfo();
    REQUIRE(type_and_shape.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    auto shape = type_and_shape.GetShape();
    REQUIRE(shape.size() == 2);
    REQUIRE(shape[0] == 1);
    REQUIRE(shape[1] == 3);
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

TEST_CASE("[onnx] - array_from_tensor", "[rad]")
{
    const std::vector<float> src_data{1.0f, 2.0f, 3.0f};

    onnx::TensorSet<float> inputs;
    onnx::array_to_tensor(src_data, inputs);

    SECTION("Runtime array")
    {
        auto ret = onnx::array_from_tensor<float>(inputs.tensors[0], src_data.size());
        REQUIRE(ret == src_data);
    }

    SECTION("Compile-time array")
    {
        auto ret = onnx::array_from_tensor<float, 3>(inputs.tensors[0]);
        REQUIRE(array_equals(src_data, ret));
    }
}
