#pragma once

#include "concepts.hpp"
#include "onnxruntime.hpp"

#include <fmt/format.h>
#include <functional>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rad::onnx
{
    template<typename T>
    concept ImagePostProcessFunctor = requires(T fn, cv::Mat m) {
        { fn(m) } -> std::same_as<cv::Mat>;
    };

    template<ImageTensorDataType T, ImagePostProcessFunctor Fun>
    std::vector<cv::Mat> image_batch_from_tensor(Ort::Value const& tensor,
                                                 cv::Size sz,
                                                 int type,
                                                 Fun post_process)
    {
        const int num_channels = 1 + (type >> CV_CN_SHIFT);
        const int depth        = type & CV_MAT_DEPTH_MASK;

        const auto dims = tensor.GetTensorTypeAndShapeInfo().GetShape();
        if (dims.size() != 4)
        {
            throw std::runtime_error{
                fmt::format("error: expected a 4-dimensional image tensor but "
                            "received {} dimensions",
                            dims.size())};
        }

        if (dims[2] != sz.height || dims[3] != sz.width)
        {
            throw std::runtime_error{
                fmt::format("error: expected an image tensor with dimensions {} x {} "
                            "but received dimensions {} x {}",
                            sz.width,
                            sz.height,
                            dims[3],
                            dims[2])};
        }

        const auto batch_stride = dims[1] * dims[2] * dims[3];
        std::vector<cv::Mat> images(dims[0]);

        // OpenCV only takes pointers by void*, not void const*, so we need to cast away
        // the const of the returned pointer from the tensor.
        // Note: this is potentially dangerous and we have to make sure we NEVER modify
        // the data!
        auto batch_ptr = const_cast<T*>(tensor.GetTensorData<T>());
        for (auto& image : images)
        {
            if (num_channels == 1)
            {
                // For grayscale images, just copy the data directly from the tensor into
                // the final image.
                if (dims[1] != 1)
                {
                    throw std::runtime_error{
                        fmt::format("error: expected a 1-channel image tensor but "
                                    "received {} channels",
                                    dims[1])};
                }

                image = cv::Mat{sz, type, batch_ptr}.clone();
            }
            else
            {
                // Multi-channel images are a bit tricky. OpenCV expects the dimensions to
                // be HxWxC, but tensors come in CxHxW. To get around this problem, we're
                // going to "slice" up the tensor into its 3 channels and then merge them
                // to the final image.
                auto data                = batch_ptr;
                const std::size_t stride = dims[2] * dims[3];
                std::vector<cv::Mat> channels(dims[1]);
                for (auto& channel : channels)
                {
                    channel = cv::Mat{sz, CV_MAKE_TYPE(depth, 1), data};
                    data += stride;
                }

                cv::merge(channels, image);
            }

            image = post_process(image);
            batch_ptr += batch_stride;
        }

        return images;
    }

    template<ImageTensorDataType T>
    std::vector<cv::Mat>
    image_batch_from_tensor(Ort::Value const& tensor, cv::Size sz, int type)
    {
        return image_batch_from_tensor<T>(tensor, sz, type, [](cv::Mat img) {
            return img;
        });
    }

    template<ImageTensorDataType T, ImagePostProcessFunctor Fun>
    cv::Mat
    image_from_tensor(Ort::Value const& tensor, cv::Size sz, int type, Fun post_process)
    {
        auto images =
            image_batch_from_tensor<T, Fun>(tensor, sz, type, std::move(post_process));
        if (images.size() != 1)
        {
            throw std::runtime_error{fmt::format(
                "error: attempting to retrieve a single image from a batched "
                "tensor with {} images. Please use images_from_tensor instead",
                images.size())};
        }
        return images.front();
    }

    template<ImageTensorDataType T>
    cv::Mat image_from_tensor(Ort::Value const& tensor, cv::Size sz, int type)
    {
        return image_from_tensor<T>(tensor, sz, type, [](cv::Mat img) {
            return img;
        });
    }

    template<TensorDataType T>
    std::vector<std::vector<T>> array_batch_from_tensor(Ort::Value const& tensor,
                                                        std::size_t size)
    {
        const auto dims = tensor.GetTensorTypeAndShapeInfo().GetShape();
        if (dims.size() != 2)
        {
            throw std::runtime_error{fmt::format(
                "error: expected a 2-dimensional tensor but received {} dimensions",
                dims.size())};
        }

        if (!std::cmp_equal(dims[1], size))
        {
            throw std::runtime_error{
                fmt::format("error: expected an array tensor with length {} but "
                            "received length {}",
                            size,
                            dims[1])};
        }

        const auto batch_stride = dims[1];
        std::vector<std::vector<T>> arrays(dims[0]);
        auto batch_ptr = tensor.GetTensorData<T>();
        for (auto& a : arrays)
        {
            a = std::vector<T>(size);
            std::memcpy(a.data(), batch_ptr, size * sizeof(T));
            batch_ptr += batch_stride;
        }

        return arrays;
    }

    template<TensorDataType T>
    std::vector<T> array_from_tensor(Ort::Value const& tensor, std::size_t size)
    {
        const auto dims = tensor.GetTensorTypeAndShapeInfo().GetShape();
        if (dims[0] != 1)
        {
            throw std::runtime_error{fmt::format(
                "error: attempting to retrieve a single array from a batched tensor with "
                "{} arrays. Please use array_batch_from_tensor instead",
                dims[0])};
        }

        auto arrays = array_batch_from_tensor<T>(tensor, size);
        return arrays.front();
    }

    template<TensorDataType T>
    T scalar_from_tensor(Ort::Value const& tensor)
    {
        const auto dims = tensor.GetTensorTypeAndShapeInfo().GetShape();
        if (dims.size() != 1)
        {
            throw std::runtime_error{
                fmt::format("error: attempting to retrieve a scalar from a tensor with "
                            "more than one dimension")};
        }

        auto ptr = tensor.GetTensorData<T>();
        auto ret = *ptr;
        return ret;
    }

    template<TensorDataType T>
    std::vector<std::vector<std::array<T, 4>>>
    batched_rects_from_tensor(Ort::Value const& tensor)
    {
        const auto dims = tensor.GetTensorTypeAndShapeInfo().GetShape();
        if (dims.size() != 3)
        {
            throw std::runtime_error{
                fmt::format("error: expected either a 3 dimensional tensor, but "
                            "recieved {} dimensions",
                            dims.size())};
        }
        if (dims[2] != 4)
        {
            throw std::runtime_error{fmt::format(
                "error: expected the final dimension to be 4, but received {}",
                dims[2])};
        }

        const auto box_stride = dims[2];
        std::vector<std::vector<std::array<T, 4>>> batches(dims[0]);
        auto batch_ptr = tensor.GetTensorData<T>();
        for (auto& batch : batches)
        {
            batch.resize(dims[1]);
            for (auto& b : batch)
            {
                std::memcpy(b.data(), batch_ptr, dims[2] * sizeof(T));
                batch_ptr += box_stride;
            }
        }

        return batches;
    }

    template<TensorDataType T>
    std::vector<std::array<T, 4>> rects_from_tensor(Ort::Value const& tensor)
    {
        const auto dims = tensor.GetTensorTypeAndShapeInfo().GetShape();
        if (dims[0] != 1)
        {
            throw std::runtime_error{fmt::format(
                "error: attempting to retrieve a single array from a batched tensor with "
                "{} arrays. Please use array_batch_from_tensor instead",
                dims[0])};
        }

        auto boxes = batched_rects_from_tensor<T>(tensor);
        return boxes.front();
    }

    template<TensorDataType T>
    std::vector<T> data_from_tensor(Ort::Value const& tensor)
    {
        const auto dims = tensor.GetTensorTypeAndShapeInfo().GetShape();
        const auto size =
            std::accumulate(dims.begin(), dims.end(), 1ll, std::multiplies());

        std::vector<T> buffer(size);
        auto data_ptr = tensor.GetTensorData<T>();
        std::memcpy(buffer.data(), data_ptr, buffer.size() * sizeof(T));

        return buffer;
    }

} // namespace rad::onnx
