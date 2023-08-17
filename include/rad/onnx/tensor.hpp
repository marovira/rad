#pragma once

#include "../opencv.hpp"
#include "onnxruntime.hpp"

namespace rad::onnx
{
    template<typename T>
    struct InputTensor
    {
        InputTensor(std::vector<T>&& d, Ort::Value&& t) :
            data{std::move(d)},
            tensor{std::move(t)}
        {}

        InputTensor(InputTensor const&)            = delete;
        InputTensor& operator=(InputTensor const&) = delete;

        std::vector<T> data;
        Ort::Value tensor;
    };

    template<typename T>
    class InputTensorSet
    {
    public:
        InputTensorSet() = default;

        InputTensorSet(InputTensorSet const&)            = delete;
        InputTensorSet& operator=(InputTensorSet const&) = delete;

        Ort::Value& operator[](std::size_t i)
        {
            return m_tensors[i];
        }

        Ort::Value const& operator[](std::size_t i) const
        {
            return m_tensors[i];
        }

        std::size_t size() const
        {
            return m_tensors.size();
        }

        bool empty() const
        {
            return m_tensors.empty();
        }

        void insert(InputTensor<T>&& input)
        {
            m_tensor_data.emplace_back(std::move(input.data));
            m_tensors.emplace_back(std::move(input.tensor));
        }

        Ort::Value* tensors()
        {
            return m_tensors.data();
        }

    private:
        std::vector<std::vector<T>> m_tensor_data;
        std::vector<Ort::Value> m_tensors;
    };

    template<typename T>
    InputTensor<T> images_to_tensor(std::vector<cv::Mat> const& images)
    {
        cv::Mat blob;
        cv::dnn::blobFromImages(images, blob);

        cv::MatSize sz = blob.size;
        std::vector<std::int64_t> tensor_dims(sz.dims());
        for (int i{0}; i < sz.dims(); ++i)
        {
            tensor_dims[i] = sz[i];
        }

        std::vector<T> tensor_data(blob.total());
        tensor_data.assign(blob.begin<T>(), blob.end<T>());

        Ort::Value tensor{nullptr};
        Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        tensor = Ort::Value::CreateTensor<T>(mem_info,
                                             tensor_data.data(),
                                             tensor_data.size(),
                                             tensor_dims.data(),
                                             tensor_dims.size());
        if (!tensor.IsTensor())
        {
            throw std::runtime_error{"error: could not create a valid tensor"};
        }

        return InputTensor<T>{std::move(tensor_data), std::move(tensor)};
    }

    template<typename T>
    InputTensor<T> image_to_tensor(cv::Mat const& img)
    {
        return images_to_tensor<T>({img});
    }

    template<typename T, typename Container>
    InputTensor<T> array_to_tensor(Container const& src_data)
    {
        static_assert(std::is_same_v<typename Container::value_type, T>);

        std::vector<std::int64_t> tensor_dims{1,
                                              static_cast<std::int64_t>(src_data.size())};

        std::vector<T> tensor_data(src_data.size());
        tensor_data.assign(src_data.begin(), src_data.end());

        Ort::Value tensor{nullptr};
        Ort::MemoryInfo mem_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        tensor = Ort::Value::CreateTensor<T>(mem_info,
                                             tensor_data.data(),
                                             tensor_data.size(),
                                             tensor_dims.data(),
                                             tensor_dims.size());

        if (!tensor.IsTensor())
        {
            throw std::runtime_error{"error: could not create a valid tensor"};
        }

        return InputTensor<T>{std::move(tensor_data), std::move(tensor)};
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
                throw std::runtime_error{
                    fmt::format("error: expected a 4-dimensional image tensor but "
                                "received {} dimensions",
                                dims.size())};
            }

            if (dims[1] != 3)
            {
                throw std::runtime_error{
                    fmt::format("error: expected a 3-channel image tensor but received "
                                "{} channels",
                                dims[1])};
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

    template<typename T>
    std::vector<T> array_from_tensor(Ort::Value& tensor, std::size_t size)
    {
        std::vector<T> ret(size);
        std::memcpy(ret.data(), tensor.GetTensorMutableData<T>(), size * sizeof(T));
        return ret;
    }

    template<typename T, std::size_t N>
    std::array<T, N> array_from_tensor(Ort::Value& tensor)
    {
        std::array<T, N> ret;
        std::memcpy(ret.data(), tensor.GetTensorMutableData<T>(), N * sizeof(T));
        return ret;
    }

} // namespace rad::onnx
