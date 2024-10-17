#include "rad/onnx/session.hpp"

#include <fmt/printf.h>
#include <magic_enum.hpp>

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

        throw std::runtime_error{fmt::format("error: {} is an unknown provider", name)};
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

#if defined(RAD_ONNX_DML_ENABLED)
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
#endif

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

    std::vector<ONNXTensorElementDataType> get_input_types(Ort::Session& session)
    {
        std::vector<ONNXTensorElementDataType> types;
        for (std::size_t i{0}; i < session.GetInputCount(); ++i)
        {
            types.push_back(
                session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType());
        }

        return types;
    }

    std::vector<ONNXTensorElementDataType> get_output_types(Ort::Session& session)
    {
        std::vector<ONNXTensorElementDataType> types;
        for (std::size_t i{0}; i < session.GetOutputCount(); ++i)
        {
            types.push_back(session.GetOutputTypeInfo(i)
                                .GetTensorTypeAndShapeInfo()
                                .GetElementType());
        }

        return types;
    }

    std::vector<std::string> get_input_names(Ort::Session& session)
    {
        Ort::AllocatorWithDefaultOptions alloc;

        std::vector<std::string> names;
        std::size_t num_inputs = session.GetInputCount();
        for (std::size_t i{0}; i < num_inputs; ++i)
        {
            names.emplace_back(session.GetInputNameAllocated(i, alloc).get());
        }

        return names;
    }

    std::vector<std::string> get_output_names(Ort::Session& session)
    {
        Ort::AllocatorWithDefaultOptions alloc;

        std::vector<std::string> names;
        std::size_t num_inputs = session.GetOutputCount();
        for (std::size_t i{0}; i < num_inputs; ++i)
        {
            names.emplace_back(session.GetOutputNameAllocated(i, alloc).get());
        }

        return names;
    }

    std::vector<Ort::Value> perform_inference(Ort::Session& session,
                                              std::vector<Ort::Value> const& inputs)
    {
        Ort::AllocatorWithDefaultOptions alloc;

        // Query the number of inputs and the name of said inputs.
        auto input_names = get_input_names(session);

        // Verify that the number of inputs is the same as the number of tensors
        // we're receiving. For the sake of sanity, let's throw an exception
        // here.
        if (inputs.size() != input_names.size())
        {
            throw std::runtime_error{fmt::format(
                "error: invalid number of input tensors, expected {} but received {}",
                input_names.size(),
                inputs.size())};
        }

        // Next, query the dimensions and types of the inputs and use them to
        // ensure that the input tensors are correct.
        {
            std::vector<Ort::TypeInfo> input_types;
            for (std::size_t i{0}; i < input_names.size(); ++i)
            {
                input_types.push_back(session.GetInputTypeInfo(i));
            }

            // For each tensor, verify that the type and size matches.
            for (std::size_t i{0}; i < input_types.size(); ++i)
            {
                auto type_and_shape = inputs[i].GetTensorTypeAndShapeInfo();
                auto input_type     = input_types[i].GetTensorTypeAndShapeInfo();

                if (input_type.GetElementType() != type_and_shape.GetElementType())
                {
                    throw std::runtime_error{fmt::format(
                        "error: invalid input tensor type for tensor {}, expected {} but "
                        "received {}",
                        i,
                        magic_enum::enum_name(input_type.GetElementType()),
                        magic_enum::enum_name(type_and_shape.GetElementType()))};
                }

                // In order to verify the shapes, we have to be careful in case
                // we have dynamic sizes (which are represented as -1). If so,
                // those dimensions should be skipped.
                auto tensor_shape = type_and_shape.GetShape();
                auto input_shape  = input_type.GetShape();
                if (tensor_shape.size() != input_shape.size())
                {
                    throw std::runtime_error{
                        fmt::format("error: invalid input tensor shape for tensor {}, "
                                    "expected {} but received {}",
                                    i,
                                    input_shape.size(),
                                    tensor_shape.size())};
                }

                for (std::size_t k{0}; k < input_shape.size(); ++k)
                {
                    if (input_shape[k] == -1)
                    {
                        continue;
                    }

                    if (input_shape[k] != tensor_shape[k])
                    {
                        throw std::runtime_error{
                            fmt::format("error: on input tensor {}, dimension {}, "
                                        "expected {} but received {}",
                                        i,
                                        k,
                                        input_shape[k],
                                        tensor_shape[k])};
                    }
                }
            }
        }

        // Input tensors are correct, so now we need to query the names of the
        // output tensors. Note that we don't actually have to do any
        // allocations here, ORT will do it for us.
        auto output_names = get_output_names(session);

        // Everything is ready, so let's perform the inference. Make sure we convert the
        // name vectors into C-strings beforehand.
        std::vector<const char*> input_strings, output_strings;
        std::transform(input_names.begin(),
                       input_names.end(),
                       std::back_inserter(input_strings),
                       [](std::string const& str) {
                           return str.c_str();
                       });
        std::transform(output_names.begin(),
                       output_names.end(),
                       std::back_inserter(output_strings),
                       [](std::string const& str) {
                           return str.c_str();
                       });

        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                          input_strings.data(),
                                          inputs.data(),
                                          inputs.size(),
                                          output_strings.data(),
                                          output_strings.size());

        // Ensure that we got back the same number of tensors.
        if (output_names.size() != output_tensors.size())
        {
            throw std::runtime_error{fmt::format(
                "error: invalid number of output tensors, expected {} but received {}",
                output_names.size(),
                output_tensors.size())};
        }

        // Ensure that all of the tensors are actually tensors.
        for (std::size_t i{0}; auto const& tensor : output_tensors)
        {
            if (!tensor.IsTensor())
            {
                throw std::runtime_error{
                    fmt::format("error: output tensor {} is not a valid tensor", i)};
            }
            ++i;
        }

        return output_tensors;
    }
} // namespace rad::onnx
