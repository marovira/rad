#pragma once

#include "../opencv.hpp"
#include "concepts.hpp"
#include "onnxruntime.hpp"
#include "perform_safe_op.hpp"

namespace rad::onnx::detail
{
    inline std::tuple<int, int, int>
    validate_batched_images(std::vector<cv::Mat> const& images)
    {
        auto& front_img        = images.front();
        const int num_channels = front_img.channels();
        const int rows         = front_img.rows;
        const int cols         = front_img.cols;

        for (std::size_t i{0}; auto const& img : images)
        {
            if (img.channels() != num_channels)
            {
                throw std::runtime_error(
                    fmt::format("error: for batched image {}, expected {} channels "
                                "but received {}",
                                i,
                                num_channels,
                                img.channels()));
            }

            if (img.rows != rows)
            {
                throw std::runtime_error(
                    fmt::format("error: for batched image {}, expected {} rows "
                                "but received {}",
                                i,
                                rows,
                                img.rows));
            }

            if (img.cols != cols)
            {
                throw std::runtime_error(
                    fmt::format("error: for batched image {}, expected {} columns "
                                "but received {}",
                                i,
                                cols,
                                img.cols));
            }
            ++i;
        }

        return {num_channels, rows, cols};
    }
} // namespace rad::onnx::detail

namespace rad::onnx
{
    template<typename T>
    concept ImagePostProcessFunctor = requires(T fn, cv::Mat m) {
        { fn(m) } -> std::same_as<cv::Mat>;
    };

    template<TensorDataType T>
    struct TensorBlob
    {
        std::vector<T> data;
        std::vector<std::int64_t> shape;
    };

    template<TensorDataType T>
    TensorBlob<T> image_batch_to_tensor_blob(std::vector<cv::Mat> const& images)
    {
        const auto [num_channels, rows, cols] = detail::validate_batched_images(images);
        const std::size_t stride              = rows * cols;

        std::vector<T> tensor_data(images.size() * num_channels * rows * cols);
        T* data_ptr = tensor_data.data();
        for (auto const& img : images)
        {
            std::vector<cv::Mat> ch(num_channels);
            for (auto& c : ch)
            {
                c = cv::Mat(rows, cols, img.depth(), data_ptr);
                data_ptr += stride;
            }

            cv::split(img, ch);
        }

        return {
            .data  = tensor_data,
            .shape = {static_cast<std::int64_t>(images.size()),
                      num_channels, rows,
                      cols}
        };
    }

    template<TensorDataType T>
    TensorBlob<T> image_to_tensor_blob(cv::Mat const& image)
    {
        return image_batch_to_tensor_blob<T>({image});
    }

    template<TensorDataType T, ImagePostProcessFunctor Fun>
    std::vector<cv::Mat> image_batch_from_tensor_blob(TensorBlob<T> const& blob,
                                                      cv::Size sz,
                                                      int type,
                                                      Fun post_process)
    {
        const auto batch_stride = blob.shape[1] * blob.shape[2] * blob.shape[3];
        std::vector<cv::Mat> images(blob.shape[0]);

        int num_channels = 1 + (type >> CV_CN_SHIFT);
        int depth        = type & CV_MAT_DEPTH_MASK;

        // OpenCV only takes pointers by void*, not void const*, so we need to cast away
        // the const of the returned pointer from the tensor.
        // Note: this is potentially dangerous and we have to make sure we NEVER modify
        // the data!
        auto batch_ptr = const_cast<T*>(blob.data.data());
        for (auto& image : images)
        {
            if (num_channels == 1)
            {
                // For single-channel images, just cpy the data directly from the blobl
                // into the final image.
                image = cv::Mat{sz, type, batch_ptr}.clone();
            }
            else
            {
                // For multi-channel images, slice the blob into individual channels and
                // then merge them together into the final image.
                auto data                = batch_ptr;
                const std::size_t stride = blob.shape[2] * blob.shape[3];
                std::vector<cv::Mat> channels(blob.shape[1]);
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

    template<TensorDataType T>
    std::vector<cv::Mat>
    image_batch_from_tensor_blob(TensorBlob<T> const& blob, cv::Size sz, int type)
    {
        return image_batch_from_tensor_blob<T>(blob, sz, type, [](cv::Mat img) {
            return img;
        });
    }

    template<TensorDataType T, ImagePostProcessFunctor Fun>
    cv::Mat image_from_tensor_blob(TensorBlob<T> const& blob,
                                   cv::Size sz,
                                   int type,
                                   Fun post_process)
    {
        auto images =
            image_batch_from_tensor_blob(blob, sz, type, std::move(post_process));
        if (images.size() != 1)
        {
            throw std::runtime_error{
                fmt::format("error: attempting to retrieve a single image from a batched "
                            "tensor blob with {} images. Please use "
                            "image_batch_from_tensor_blob instead",
                            images.size())};
        }

        return images.front();
    }

    template<TensorDataType T>
    cv::Mat image_from_tensor_blob(TensorBlob<T> const& blob, cv::Size sz, int type)
    {
        return image_from_tensor_blob<T>(blob, sz, type, [](cv::Mat img) {
            return img;
        });
    }

    template<TensorDataType T, ImagePostProcessFunctor Fun>
    std::vector<cv::Mat> image_batch_from_tensor(Ort::Value const& tensor,
                                                 cv::Size sz,
                                                 int type,
                                                 Fun post_process)
    {
        int num_channels = 1 + (type >> CV_CN_SHIFT);
        int depth        = type & CV_MAT_DEPTH_MASK;

        if (num_channels != 1 && num_channels != 3)
        {
            throw std::runtime_error{fmt::format(
                "error: invalid number of channels, expected 1 or 3 but received {}",
                num_channels)};
        }

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
            else if (num_channels == 3)
            {
                // 3-channeled images are a bit tricky. OpenCV expects the dimensions to
                // be HxWxC, but tensors come in CxHxW. To get around this problem, we're
                // going to "slice" up the tensor into its 3 channels and then merge them
                // to the final image.
                if (dims[1] != 3)
                {
                    throw std::runtime_error{fmt::format(
                        "error: expected a 3-channel image tensor but received "
                        "{} channels",
                        dims[1])};
                }

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

    template<TensorDataType T>
    std::vector<cv::Mat>
    image_batch_from_tensor(Ort::Value const& tensor, cv::Size sz, int type)
    {
        return image_batch_from_tensor<T>(tensor, sz, type, [](cv::Mat img) {
            return img;
        });
    }

    template<TensorDataType T, ImagePostProcessFunctor Fun>
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

    template<TensorDataType T>
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

        if (dims[1] != static_cast<std::int64_t>(size))
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

} // namespace rad::onnx
