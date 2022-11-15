#pragma once

#include "onnxruntime.hpp"
#include "opencv.hpp"

#include <fmt/printf.h>
#include <zeus/assert.hpp>

#if defined(RAD_ONNX_ENABLED)
namespace rad::onnx
{
    template<typename T>
    bool perform_safe_op(T&& fun)
    {
        try
        {
            fun();
        }
        catch (cv::Exception const& e)
        {
            fmt::print("{}\n", e.what());
            ASSERT(0);
            return false;
        }
        catch (Ort::Exception const& e)
        {
            fmt::print("{}\n", e.what());
            ASSERT(0);
            return false;
        }

        catch (std::exception const& e)
        {
            fmt::print("{}\n", e.what());
            ASSERT(0);
            return false;
        }

        return true;
    }

    void logger(void*,
                OrtLoggingLevel severity,
                const char* category,
                const char* log_id,
                const char* code_location,
                const char* message);

    Ort::Env create_environment(std::string const& name);

    template<typename ModelSelectorFun>
    Ort::Session make_session_from_file(std::string const& model_root,
                                        Ort::Env& env,
                                        ModelSelectorFun&& selector)
    {
        std::wstring model_to_load = selector(model_root);
        if (model_to_load.empty())
        {
            ASSERT(0);
            throw std::runtime_error{
                fmt::format("error: unable to find a model load in {}", model_root)
                    .c_str()};
        }

        Ort::SessionOptions opt;
        opt.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        Ort::Session session{env, model_to_load.c_str(), opt};
        return session;
    }

    template<typename T>
    struct TensorSet
    {
        TensorSet() = default;

        TensorSet(TensorSet const&)            = delete;
        TensorSet& operator=(TensorSet const&) = delete;

        std::vector<std::vector<T>> tensor_data;
        std::vector<Ort::Value> tensors;
    };

    template<typename T>
    void image_to_tensor(cv::Mat const& img, TensorSet<T>& set)
    {
        cv::Mat blob;
        cv::dnn::blobFromImage(img, blob);

        cv::MatSize sz = blob.size;
        std::vector<std::int64_t> tensor_dims(sz.dims());
        for (int i{0}; i < sz.dims(); ++i)
        {
            tensor_dims[i] = sz[i];
        }

        std::vector<T> tensor_data(blob.total());
        tensor_data.assign(blob.begin<T>(), blob.end<T>());

        Ort::Value tensor{nullptr};
        perform_safe_op([&tensor, &tensor_data, tensor_dims]() {
            Ort::MemoryInfo mem_info =
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            tensor = Ort::Value::CreateTensor<T>(mem_info,
                                                 tensor_data.data(),
                                                 tensor_data.size(),
                                                 tensor_dims.data(),
                                                 tensor_dims.size());
            return true;
        });

        ASSERT(tensor.IsTensor());

        set.tensor_data.emplace_back(std::move(tensor_data));
        set.tensors.emplace_back(std::move(tensor));
    }

    template<typename T>
    std::vector<Ort::Value> perform_inference(Ort::Session& session, TensorSet<T>& inputs)
    {
        auto& [_, input_tensors] = inputs;

        Ort::AllocatorWithDefaultOptions alloc;

        // Query the number of inputs and the name of said inputs.
        std::vector<std::string> input_names;
        {
            std::size_t num_inputs = session.GetInputCount();
            for (std::size_t i{0}; i < num_inputs; ++i)
            {
                input_names.push_back(session.GetInputNameAllocated(i, alloc).get());
            }
        }

        // Verify that the number of inputs is the same as the number of tensors
        // we're receiving. For the sake of sanity, let's throw an exception
        // here.
        if (input_tensors.size() != input_names.size())
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
                auto type_and_shape = input_tensors[i].GetTensorTypeAndShapeInfo();
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
                output_names.push_back(session.GetOutputNameAllocated(i, alloc).get());
            }
        }

        // Everything is ready, so let's perform the inference.
        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                          input_names.data(),
                                          input_tensors.data(),
                                          input_tensors.size(),
                                          output_names.data(),
                                          output_names.size());

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

    template<typename T, typename Fun>
    cv::Mat
    image_from_tensor(Ort::Value& tensor, cv::Size sz, int type, Fun&& post_process)
    {
        int num_channels = 1 + (type >> CV_CN_SHIFT);
        int depth        = type & CV_MAT_DEPTH_MASK;

        cv::Mat image;
        if (num_channels == 1)
        {
            // For grayscale images, just copy the data directly from the tensor into the
            // final image.
            cv::Mat from_tensor{sz, type, tensor.GetTensorMutableData<T>()};
            image = from_tensor.clone();
        }
        else if (num_channels == 3)
        {
            // 3-channeled images are a bit tricky. OpenCV expects the dimensions to be
            // HxWxC, but tensors come in CxHxW. To get around this problem, we're going
            // to "slice" up the tensor into its 3 channels and then merge them to the
            // final image.
            auto dims = tensor.GetTensorTypeAndShapeInfo().GetShape();
            if (dims.size() != 4)
            {
                auto msg = fmt::format(
                    "error: expected a 4-dimensional tensor but received {} dimensions",
                    dims.size());
                throw std::runtime_error{msg.c_str()};
            }

            if (dims[1] != 3)
            {
                auto msg = fmt::format(
                    "error: expected a 3-channel tensor but received {} channels",
                    dims[1]);
                throw std::runtime_error{msg.c_str()};
            }

            if (dims[2] != sz.height || dims[3] != sz.width)
            {
                auto msg = fmt::format("error: expected a tensor with dimensions {} x {} "
                                       "but received dimensions {} x {}",
                                       sz.width,
                                       sz.height,
                                       dims[3],
                                       dims[2]);
                throw std::runtime_error{msg.c_str()};
            }

            auto data = tensor.GetTensorMutableData<T>();
            std::vector<T> slice(dims[2] * dims[3]);
            std::vector<cv::Mat> channels(dims[1]);
            for (auto& channel : channels)
            {
                std::memcpy(slice.data(), data, sizeof(T) * slice.size());
                channel = cv::Mat{sz, CV_MAKE_TYPE(depth, 1), slice.data()}.clone();
                data += slice.size();
            }

            cv::merge(channels, image);
        }

        image = post_process(image);
        return image;
    }

    template<typename T>
    cv::Mat image_from_tensor(Ort::Value& tensor, cv::Size sz, int type)
    {
        return image_from_tensor<T>(tensor, sz, type, [](cv::Mat img) {
            return img;
        });
    }
} // namespace rad::onnx

#endif
