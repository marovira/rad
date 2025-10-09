#pragma once

#include "concepts.hpp"
#include "onnxruntime.hpp"
#include "perform_safe_op.hpp"

#include <fmt/printf.h>

namespace rad::onnx
{
    enum class ExecutionProviders
    {
        cpu = 0,
        dml,
        coreml,
    };

    template<typename T>
    concept SelectorFunctor = requires(T fn, std::string str) {
        { fn(str) } -> std::same_as<OrtStringPath<ORTCHAR_T>::type>;
    };

    consteval bool use_string_for_paths()
    {
        if constexpr (std::is_same_v<OrtStringPath<ORTCHAR_T>::type, std::string>)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    std::vector<ExecutionProviders> get_execution_providers();

    consteval bool is_provider_enabled(ExecutionProviders provider)
    {
        if (provider == ExecutionProviders::cpu)
        {
            return true;
        }

        if (provider == ExecutionProviders::dml)
        {
#if defined(RAD_ONNX_DML_ENABLED)
            return true;
#else
            return false;
#endif
        }

        if (provider == ExecutionProviders::coreml)
        {
#if defined(RAD_ONNX_COREML_ENABLED)
            return true;
#else
            return false;
#endif
        }

        return false;
    }

    Ort::SessionOptions get_default_session_options();

#if defined(RAD_ONNX_DML_ENABLED)
    Ort::SessionOptions get_default_dml_session_options(int device_id);
    Ort::SessionOptions get_default_dml_session_options();
#elif defined(RAD_ONNX_COREML_ENABLED)
    enum class MLComputeUnit
    {
        cpu_only,
        cpu_and_neural_engine,
        cpu_and_gpu,
        all,
    };

    enum class MLSpecialisationStrategy
    {
        base,
        fast_prediction,
    };

    struct CoreMLSettings
    {
        MLComputeUnit compute_unit{MLComputeUnit::all};
        bool require_static_input_shapes{false};
        bool enable_on_subgraphs{false};
        MLSpecialisationStrategy strategy{MLSpecialisationStrategy::base};
        bool profile_compute_plan{false};
        bool allow_low_precision_accumulation_on_gpu{false};
    };

    Ort::SessionOptions
    get_default_coreml_session_options(CoreMLSettings const& settings);
#endif

    template<SelectorFunctor T>
    Ort::Session make_session_from_file(std::string const& model_root,
                                        Ort::Env& env,
                                        Ort::SessionOptions const& opt,
                                        T selector)
    {
        auto model_to_load = selector(model_root);
        if (model_to_load.empty())
        {
            throw std::runtime_error{
                fmt::format("error: unable to find a model load in {}", model_root)};
        }

        Ort::Session session{env, model_to_load.c_str(), opt};
        return session;
    }

    template<SelectorFunctor T>
    Ort::Session
    make_session_from_file(std::string const& model_root, Ort::Env& env, T selector)
    {
        return make_session_from_file(model_root,
                                      env,
                                      get_default_session_options(),
                                      std::move(selector));
    }

    std::vector<std::vector<std::int64_t>> get_input_shapes(Ort::Session& session);
    std::vector<std::vector<std::int64_t>> get_output_shapes(Ort::Session& session);

    std::vector<std::string> get_input_names(Ort::Session& session);
    std::vector<std::string> get_output_names(Ort::Session& session);

    std::vector<ONNXTensorElementDataType> get_input_types(Ort::Session& session);
    std::vector<ONNXTensorElementDataType> get_output_types(Ort::Session& session);

    std::vector<Ort::Value> perform_inference(Ort::Session& session,
                                              std::vector<Ort::Value> const& inputs);
} // namespace rad::onnx
