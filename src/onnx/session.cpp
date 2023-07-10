#include "rad/onnx/session.hpp"

#if defined(RAD_ONNX_DML_ENABLED)
#    include <dml_provider_factory.h>
#endif

namespace rad::onnx
{
    ExecutionProviders convert_provider_name(std::string const& name)
    {
        if (name == "CPUExecutionProvider")
        {
            return ExecutionProviders::cpu;
        }

        if (name == "DmlExecutionProvider")
        {
            return ExecutionProviders::dml;
        }

        throw std::runtime_error{"error: unknown provider found"};
    }

    std::vector<ExecutionProviders> get_execution_providers()
    {
        auto provider_names = Ort::GetAvailableProviders();

        std::vector<ExecutionProviders> ret;
        for (auto const& name : provider_names)
        {
            ret.push_back(convert_provider_name(name));
        }

        return ret;
    }

    Ort::SessionOptions get_default_session_options()
    {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        return opts;
    }

    Ort::SessionOptions get_default_dml_session_options(int device_id)
    {
        if constexpr (!is_provider_enabled(ExecutionProviders::dml))
        {
            throw std::runtime_error{"error: RAD was not compiled with DML support"};
        }

        OrtApi const& ort_api = Ort::GetApi();

        OrtDmlApi const* dml_api{nullptr};
        ort_api.GetExecutionProviderApi("DML",
                                        ORT_API_VERSION,
                                        reinterpret_cast<void const**>(&dml_api));
        if (!dml_api)
        {
            throw std::runtime_error{"error: unable to acquire DML provider API"};
        }

        Ort::SessionOptions opt;
        opt.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        opt.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        opt.DisableMemPattern();
        dml_api->SessionOptionsAppendExecutionProvider_DML(opt, device_id);
        return opt;
    }

    Ort::SessionOptions get_default_dml_session_options()
    {
        return get_default_dml_session_options(0);
    }

    std::vector<std::vector<std::int64_t>> get_input_shapes(Ort::Session& session)
    {
        std::vector<std::vector<std::int64_t>> shapes;
        for (std::size_t i{0}; i < session.GetInputCount(); ++i)
        {
            shapes.push_back(
                session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        return shapes;
    }

    std::vector<std::vector<std::int64_t>> get_output_shapes(Ort::Session& session)
    {
        std::vector<std::vector<std::int64_t>> shapes;
        for (std::size_t i{0}; i < session.GetOutputCount(); ++i)
        {
            shapes.push_back(
                session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        return shapes;
    }

} // namespace rad::onnx
