#include <rad/onnx/inference.hpp>

#include "model_file_manager.hpp"
#include <rad/onnx/env.hpp>

#include <catch2/catch_test_macros.hpp>

namespace onnx = rad::onnx;

TEST_CASE("[inference] - perform_inference", "[rad::onnx]")
{
    onnx::init_ort_api();
    ModelFileManager mgr;
    auto env = onnx::create_environment("inference_test");

    SECTION("CPU inference")
    {
        auto session =
            onnx::make_session_from_file(mgr.get_root(), env, [mgr](std::string) {
                return mgr.get_model(ModelFileManager::Type::static_axes);
            });
        REQUIRE(static_cast<OrtSession*>(session) != nullptr);

        onnx::InputTensorSet<float> set;
        {
            cv::Mat img = cv::Mat::ones(cv::Size{32, 32}, CV_32FC1);
            auto tensor = onnx::image_to_tensor<float>(img);

            set.insert(std::move(tensor));
        }

        auto out_tensors = onnx::perform_inference(session, set);
        REQUIRE(out_tensors.size() == 1);

        auto result = onnx::array_from_tensor<float>(out_tensors.front(), 10);
        REQUIRE(result.size() == 10);
    }

#if defined(RAD_ONNX_DML_ENABLED)
    SECTION("CPU inference")
    {
        auto opt = onnx::get_default_dml_session_options();
        auto session =
            onnx::make_session_from_file(mgr.get_root(), env, opt, [mgr](std::string) {
                return mgr.get_model(ModelFileManager::Type::static_axes);
            });
        REQUIRE(static_cast<OrtSession*>(session) != nullptr);

        onnx::InputTensorSet<float> set;
        {
            cv::Mat img = cv::Mat::ones(cv::Size{32, 32}, CV_32FC1);
            auto tensor = onnx::image_to_tensor<float>(img);

            set.insert(std::move(tensor));
        }

        auto out_tensors = onnx::perform_inference(session, set);
        REQUIRE(out_tensors.size() == 1);

        auto result = onnx::array_from_tensor<float>(out_tensors.front(), 10);
        REQUIRE(result.size() == 10);
    }
#endif
}
