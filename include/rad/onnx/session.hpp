#pragma once

#include "onnxruntime.hpp"

#include <fmt/printf.h>

namespace rad::onnx
{
    enum class ExecutionProviders
    {
        cpu = 0,
        dml,
    };

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

        return false;
    }

    Ort::SessionOptions get_default_session_options();

#if defined(RAD_ONNX_DML_ENABLED)
    Ort::SessionOptions get_default_dml_session_options(int device_id);
    Ort::SessionOptions get_default_dml_session_options();
#endif

    template<typename T>
    struct OrtStringPath : std::false_type
    {};

    template <>
    struct OrtStringPath<char>
    {
        using value_type = std::string;
    };

    template <>
    struct OrtStringPath<wchar_t>
    {
        using value_type = std::wstring;
    };

    template<typename T>
    consteval bool use_string()
    {
        if constexpr(std::is_same_v<T, char>)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    template<typename T>
    concept SelectorFunctor = requires(T fn, std::string str) {
        // clang-format off
        { fn(str) } -> std::same_as<OrtStringPath<ORTCHAR_T>::value_type>;
        // clang-format on
    };

    template<SelectorFunctor T>
    Ort::Session make_session_from_file(std::string const& model_root,
                                        Ort::Env& env,
                                        Ort::SessionOptions const& opt,
                                        T&& selector)
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
    make_session_from_file(std::string const& model_root, Ort::Env& env, T&& selector)
    {
        return make_session_from_file(model_root,
                                      env,
                                      get_default_session_options(),
                                      std::move(selector));
    }

    std::vector<std::vector<std::int64_t>> get_input_shapes(Ort::Session& session);
    std::vector<std::vector<std::int64_t>> get_output_shapes(Ort::Session& session);

    std::vector<Ort::Value> perform_inference(Ort::Session& session,
                                              std::vector<Ort::Value> const& inputs);
} // namespace rad::onnx
