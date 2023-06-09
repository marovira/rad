#pragma once

#include "onnxruntime.hpp"
#include "tensor.hpp"

namespace rad::onnx
{
    template<typename T>
    std::vector<Ort::Value> perform_inference(Ort::Session& session,
                                              InputTensorSet<T>& input_set)
    {
        Ort::AllocatorWithDefaultOptions alloc;

        // Query the number of inputs and the name of said inputs.
        std::vector<std::string> input_names;
        {
            std::size_t num_inputs = session.GetInputCount();
            for (std::size_t i{0}; i < num_inputs; ++i)
            {
                input_names.emplace_back(session.GetInputNameAllocated(i, alloc).get());
            }
        }

        // Verify that the number of inputs is the same as the number of tensors
        // we're receiving. For the sake of sanity, let's throw an exception
        // here.
        if (input_set.size() != input_names.size())
        {
            throw std::runtime_error{"error: invalid number of input tensors"};
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
                auto type_and_shape = input_set[i].GetTensorTypeAndShapeInfo();
                auto input_type     = input_types[i].GetTensorTypeAndShapeInfo();

                if (input_type.GetElementType() != type_and_shape.GetElementType())
                {
                    throw std::runtime_error{"error: invalid tensor types"};
                }

                // In order to verify the shapes, we have to be careful in case
                // we have dynamic sizes (which are represented as -1). If so,
                // those dimensions should be skipped.
                auto tensor_shape = type_and_shape.GetShape();
                auto input_shape  = input_type.GetShape();
                if (tensor_shape.size() != input_shape.size())
                {
                    throw std::runtime_error{"error: invalid shape types"};
                }

                for (std::size_t k{0}; k < input_shape.size(); ++k)
                {
                    if (input_shape[k] == -1)
                    {
                        continue;
                    }

                    if (input_shape[k] != tensor_shape[k])
                    {
                        throw std::runtime_error{"error: invalid shape types"};
                    }
                }
            }
        }

        // Input tensors are correct, so now we need to query the names of the
        // output tensors. Note that we don't actually have to do any
        // allocations here, ORT will do it for us.
        std::vector<std::string> output_names;
        {
            std::size_t num_outputs = session.GetOutputCount();
            for (std::size_t i{0}; i < num_outputs; ++i)
            {
                output_names.emplace_back(session.GetOutputNameAllocated(i, alloc).get());
            }
        }

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
                                          input_set.tensors(),
                                          input_set.size(),
                                          output_strings.data(),
                                          output_strings.size());

        // Ensure that we got back the same number of tensors.
        if (output_names.size() != output_tensors.size())
        {
            throw std::runtime_error{"error: invalid number of output tensors"};
        }

        // Ensure that all of the tensors are actually tensors.
        for (auto const& tensor : output_tensors)
        {
            if (!tensor.IsTensor())
            {
                throw std::runtime_error{"error: invalid output tensor"};
            }
        }

        return output_tensors;
    }
} // namespace rad::onnx
