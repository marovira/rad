#include "model_file_manager.hpp"

#include <catch2/catch_test_macros.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <rad/onnx/env.hpp>
#include <rad/onnx/memory_backed_tensor_set.hpp>
#include <rad/onnx/onnxruntime.hpp>
#include <rad/onnx/session.hpp>
#include <rad/onnx/tensor_conversion.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace onnx = rad::onnx;

namespace
{
    bool provider_present(std::vector<onnx::ExecutionProviders> const& providers,
                          onnx::ExecutionProviders provider)
    {
        return std::ranges::find(providers, provider) != providers.end();
    }

} // namespace

TEST_CASE("[session] - get_execution_providers", "[rad::onnx]")
{
    auto providers = onnx::get_execution_providers();
    REQUIRE_FALSE(providers.empty());

    // CPU must always be present.
    REQUIRE(provider_present(providers, onnx::ExecutionProviders::cpu));

#if defined(RAD_ONNX_DML_ENABLED)
    REQUIRE(provider_present(providers, onnx::ExecutionProviders::dml));
#endif

#if defined(RAD_ONNX_COREML_ENABLED)
    REQUIRE(provider_present(providers, onnx::ExecutionProviders::coreml));
#endif
}

TEST_CASE("[session] - is_provider_enabled", "[rad::onnx]")
{
    STATIC_REQUIRE(is_provider_enabled(rad::onnx::ExecutionProviders::cpu));

#if defined(RAD_ONNX_DML_ENABLED)
    STATIC_REQUIRE(is_provider_enabled(rad::onnx::ExecutionProviders::dml));
#endif
}

TEST_CASE("[session] - make_session_from_file", "[rad::onnx]")
{
    const ModelFileManager mgr;
    auto env = onnx::create_environment("session_test");

    SECTION("Invalid filename")
    {
        REQUIRE_THROWS(onnx::make_session_from_file(mgr.get_root(), env, [](std::string) {
            if constexpr (onnx::use_string_for_paths())
            {
                return std::string{};
            }
            else
            {
                return std::wstring{};
            };
        }));
    }

    SECTION("CPU session")
    {
        auto session =
            onnx::make_session_from_file(mgr.get_root(), env, [&mgr](std::string) {
                return mgr.get_model(ModelFileManager::Type::static_axes);
            });
        REQUIRE(static_cast<OrtSession*>(session) != nullptr);
    }

#if defined(RAD_ONNX_DML_ENABLED)
    SECTION("DML session")
    {
        auto opt = onnx::get_default_dml_session_options();
        auto session =
            onnx::make_session_from_file(mgr.get_root(), env, [&mgr](std::string) {
                return mgr.get_model(ModelFileManager::Type::static_axes);
            });
        REQUIRE(static_cast<OrtSession*>(session) != nullptr);
    }
#endif
}

TEST_CASE("[session] - get_input_shapes", "[rad::onnx]")
{
    const ModelFileManager mgr;
    auto env = onnx::create_environment("session_test");

    SECTION("Static axes")
    {
        auto session =
            onnx::make_session_from_file(mgr.get_root(), env, [&mgr](std::string) {
                return mgr.get_model(ModelFileManager::Type::static_axes);
            });
        REQUIRE(static_cast<OrtSession*>(session) != nullptr);

        auto shapes = onnx::get_input_shapes(session);
        REQUIRE(shapes.size() == 1);

        auto input_shape = shapes.front();
        REQUIRE(input_shape.size() == 4);
        REQUIRE(input_shape[0] == 1);
        REQUIRE(input_shape[1] == 1);
        REQUIRE(input_shape[2] == 32);
        REQUIRE(input_shape[3] == 32);
    }

    SECTION("Dynamic axes")
    {
        auto session =
            onnx::make_session_from_file(mgr.get_root(), env, [&mgr](std::string) {
                return mgr.get_model(ModelFileManager::Type::dynamic_axes);
            });
        REQUIRE(static_cast<OrtSession*>(session) != nullptr);

        auto shapes = onnx::get_input_shapes(session);
        REQUIRE(shapes.size() == 1);

        auto input_shape = shapes.front();
        REQUIRE(input_shape.size() == 4);
        REQUIRE(input_shape[0] == 1);
        REQUIRE(input_shape[1] == 1);
        REQUIRE(input_shape[2] == -1);
        REQUIRE(input_shape[3] == -1);
    }
}

TEST_CASE("[session] - get_output_shapes", "[rad::onnx]")
{
    const ModelFileManager mgr;
    auto env = onnx::create_environment("session_test");

    auto session = onnx::make_session_from_file(mgr.get_root(), env, [&mgr](std::string) {
        return mgr.get_model(ModelFileManager::Type::static_axes);
    });
    REQUIRE(static_cast<OrtSession*>(session) != nullptr);

    auto shapes = onnx::get_output_shapes(session);
    REQUIRE(shapes.size() == 1);

    auto output_shape = shapes.front();
    REQUIRE(output_shape.size() == 2);
    REQUIRE(output_shape[0] == 1);
    REQUIRE(output_shape[1] == 10);
}

TEST_CASE("[session] - get_input_types", "[rad::onnx]")
{
    const ModelFileManager mgr;
    auto env = onnx::create_environment("session_test");

    SECTION("32-bit float")
    {
        auto session =
            onnx::make_session_from_file(mgr.get_root(), env, [&mgr](std::string) {
                return mgr.get_model(ModelFileManager::Type::static_axes);
            });
        REQUIRE(static_cast<OrtSession*>(session) != nullptr);

        auto types = onnx::get_input_types(session);
        REQUIRE(types.size() == 1);
        REQUIRE(types[0] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    }

    SECTION("16-bit float")
    {
        auto session =
            onnx::make_session_from_file(mgr.get_root(), env, [&mgr](std::string) {
                return mgr.get_model(ModelFileManager::Type::half_float);
            });
        REQUIRE(static_cast<OrtSession*>(session) != nullptr);

        auto types = onnx::get_input_types(session);
        REQUIRE(types.size() == 1);

        REQUIRE(types[0] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    }
}

TEST_CASE("[session] - get_output_types", "[rad::onnx]")
{
    const ModelFileManager mgr;
    auto env = onnx::create_environment("session_test");

    SECTION("32-bit float")
    {
        auto session =
            onnx::make_session_from_file(mgr.get_root(), env, [&mgr](std::string) {
                return mgr.get_model(ModelFileManager::Type::static_axes);
            });
        REQUIRE(static_cast<OrtSession*>(session) != nullptr);

        auto types = onnx::get_output_types(session);
        REQUIRE(types.size() == 1);
        REQUIRE(types[0] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    }

    SECTION("16-bit float")
    {
        auto session =
            onnx::make_session_from_file(mgr.get_root(), env, [&mgr](std::string) {
                return mgr.get_model(ModelFileManager::Type::half_float);
            });
        REQUIRE(static_cast<OrtSession*>(session) != nullptr);

        auto types = onnx::get_output_types(session);
        REQUIRE(types.size() == 1);
        REQUIRE(types[0] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    }
}

TEST_CASE("[session] - get_input_names", "[rad::onnx]")
{
    const ModelFileManager mgr;
    auto env     = onnx::create_environment("session_test");
    auto session = onnx::make_session_from_file(mgr.get_root(), env, [&mgr](std::string) {
        return mgr.get_model(ModelFileManager::Type::static_axes);
    });

    auto names = onnx::get_input_names(session);
    REQUIRE(names.size() == 1);
    REQUIRE(names[0] == "input");
}

TEST_CASE("[session] - get_output_names", "[rad::onnx]")
{
    const ModelFileManager mgr;
    auto env     = onnx::create_environment("session_test");
    auto session = onnx::make_session_from_file(mgr.get_root(), env, [&mgr](std::string) {
        return mgr.get_model(ModelFileManager::Type::static_axes);
    });

    auto names = onnx::get_output_names(session);
    REQUIRE(names.size() == 1);
    REQUIRE(names[0] == "output");
}

TEST_CASE("[inference] - perform_inference", "[rad::onnx]")
{
    const ModelFileManager mgr;
    auto env = onnx::create_environment("inference_test");

    SECTION("CPU inference: 32-bit float")
    {
        auto session =
            onnx::make_session_from_file(mgr.get_root(), env, [&mgr](std::string) {
                return mgr.get_model(ModelFileManager::Type::static_axes);
            });
        REQUIRE(static_cast<OrtSession*>(session) != nullptr);

        onnx::MemoryBackedTensorSet<float> set;
        set.insert_tensor_from_image(cv::Mat::ones(cv::Size{32, 32}, CV_32FC1));

        auto out_tensors = onnx::perform_inference(session, set.tensors());
        REQUIRE(out_tensors.size() == 1);

        auto result = onnx::array_from_tensor<float>(out_tensors.front(), 10);
        REQUIRE(result.size() == 10);
    }

    SECTION("CPU inference: 16-bit float")
    {
        auto session =
            onnx::make_session_from_file(mgr.get_root(), env, [&mgr](std::string) {
                return mgr.get_model(ModelFileManager::Type::half_float);
            });
        REQUIRE(static_cast<OrtSession*>(session) != nullptr);

        onnx::MemoryBackedTensorSet<Ort::Float16_t> set;
        set.insert_tensor_from_image(cv::Mat::ones(cv::Size{32, 32}, CV_16SC1));

        auto out_tensors = onnx::perform_inference(session, set.tensors());
        REQUIRE(out_tensors.size() == 1);

        auto result = onnx::array_from_tensor<std::uint16_t>(out_tensors.front(), 10);
        REQUIRE(result.size() == 10);
    }

#if defined(RAD_ONNX_DML_ENABLED)
    SECTION("DML inference: 32-bit float")
    {
        auto opt = onnx::get_default_dml_session_options();
        auto session =
            onnx::make_session_from_file(mgr.get_root(), env, opt, [mgr](std::string) {
                return mgr.get_model(ModelFileManager::Type::static_axes);
            });
        REQUIRE(static_cast<OrtSession*>(session) != nullptr);

        onnx::MemoryBackedTensorSet<float> set;
        set.insert_tensor_from_image(cv::Mat::ones(cv::Size{32, 32}, CV_32FC1));

        auto out_tensors = onnx::perform_inference(session, set.tensors());
        REQUIRE(out_tensors.size() == 1);

        auto result = onnx::array_from_tensor<float>(out_tensors.front(), 10);
        REQUIRE(result.size() == 10);
    }

    SECTION("DML inference: 16-bit float")
    {
        auto opt = onnx::get_default_dml_session_options();
        auto session =
            onnx::make_session_from_file(mgr.get_root(), env, opt, [mgr](std::string) {
                return mgr.get_model(ModelFileManager::Type::half_float);
            });
        REQUIRE(static_cast<OrtSession*>(session) != nullptr);

        onnx::MemoryBackedTensorSet<Ort::Float16_t> set;
        set.insert_tensor_from_image(cv::Mat::ones(cv::Size{32, 32}, CV_16SC1));

        auto out_tensors = onnx::perform_inference(session, set.tensors());
        REQUIRE(out_tensors.size() == 1);

        auto result = onnx::array_from_tensor<std::uint16_t>(out_tensors.front(), 10);
        REQUIRE(result.size() == 10);
    }

#elif defined(RAD_ONNX_COREML_ENABLED) && !defined(RAD_CI_BUILD)
    SECTION("CoreML inference: 32-bit float CoreML all")
    {
        auto opt = onnx::get_default_coreml_session_options(
            {.compute_unit                = onnx::MLComputeUnit::all,
             .require_static_input_shapes = true});

        auto session =
            onnx::make_session_from_file(mgr.get_root(), env, opt, [mgr](std::string) {
                return mgr.get_model(ModelFileManager::Type::static_axes);
            });
        REQUIRE(static_cast<OrtSession*>(session) != nullptr);

        onnx::MemoryBackedTensorSet<float> set;
        set.insert_tensor_from_image(cv::Mat::ones(cv::Size{32, 32}, CV_32FC1));

        auto out_tensors = onnx::perform_inference(session, set.tensors());
        REQUIRE(out_tensors.size() == 1);

        auto result = onnx::array_from_tensor<float>(out_tensors.front(), 10);
        REQUIRE(result.size() == 10);
    }

    SECTION("CoreML inference: 16-bit float CoreML all")
    {
        auto opt = onnx::get_default_coreml_session_options(
            {.compute_unit = onnx::MLComputeUnit::all});

        auto session =
            onnx::make_session_from_file(mgr.get_root(), env, opt, [mgr](std::string) {
                return mgr.get_model(ModelFileManager::Type::half_float);
            });
        REQUIRE(static_cast<OrtSession*>(session) != nullptr);

        onnx::MemoryBackedTensorSet<Ort::Float16_t> set;
        set.insert_tensor_from_image(cv::Mat::ones(cv::Size{32, 32}, CV_16SC1));

        auto out_tensors = onnx::perform_inference(session, set.tensors());
        REQUIRE(out_tensors.size() == 1);

        auto result = onnx::array_from_tensor<std::uint16_t>(out_tensors.front(), 10);
        REQUIRE(result.size() == 10);
    }
#endif
}
