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

        InputTensor(InputTensor&& other) :
            data{std::move(other.data)},
            tensor{std::move(other.tensor)}
        {}

        InputTensor(InputTensor const&)            = delete;
        InputTensor& operator=(InputTensor const&) = delete;

        InputTensor& operator=(InputTensor&& other)
        {
            data   = std::move(other.data);
            tensor = std::move(other.tensor);
            return *this;
        }

        std::vector<T> data;
        Ort::Value tensor;
    };

    template<typename T>
    class InputTensorSet
    {
    public:
        using iterator       = std::vector<Ort::Value>::iterator;
        using const_iterator = std::vector<Ort::Value>::const_iterator;

        InputTensorSet()                      = default;
        InputTensorSet(InputTensorSet const&) = delete;

        InputTensorSet(InputTensorSet&& other) :
            m_tensor_data{std::move(other.m_tensor_data)},
            m_tensors{std::move(other.m_tensors)}
        {}

        InputTensorSet& operator=(InputTensorSet const&) = delete;

        InputTensorSet& operator=(InputTensorSet&& other)
        {
            m_tensor_data = std::move(other.m_tensor_data);
            m_tensors     = std::move(other.m_tensors);
            return *this;
        }

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

        iterator begin()
        {
            return m_tensors.begin();
        }

        const_iterator begin() const
        {
            return m_tensors.begin();
        }

        iterator end()
        {
            return m_tensors.end();
        }

        const_iterator end() const
        {
            return m_tensors.end();
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

    template<typename T>
    InputTensorSet<T> images_to_tensor_set(std::vector<cv::Mat> const& images)
    {
        InputTensorSet<T> inputs;
        auto tensors = images_to_tensor<T>(images);
        inputs.insert(std::move(tensors));
        return inputs;
    }

    template<typename T>
    InputTensorSet<T> image_to_tensor_set(cv::Mat const& img)
    {
        return images_to_tensor_set<T>({img});
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

    template<typename T, typename Container>
    InputTensorSet<T> array_to_tensor_set(Container const& src_data)
    {
        InputTensorSet<T> inputs;
        auto tensors = array_to_tensor(src_data);
        inputs.insert(std::move(tensors));
        return inputs;
    }

    template<typename T, typename Fun>
    std::vector<cv::Mat>
    images_from_tensor(Ort::Value& tensor, cv::Size sz, int type, Fun&& post_process)
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

        auto batch_ptr = tensor.GetTensorMutableData<T>();
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

                auto data = batch_ptr;
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
            batch_ptr += batch_stride;
        }

        return images;
    }

    template<typename T>
    std::vector<cv::Mat> images_from_tensor(Ort::Value& tensor, cv::Size sz, int type)
    {
        return images_from_tensor<T>(tensor, sz, type, [](cv::Mat img) {
            return img;
        });
    }

    template<typename T, typename Fun>
    cv::Mat
    image_from_tensor(Ort::Value& tensor, cv::Size sz, int type, Fun&& post_process)
    {
        auto images =
            images_from_tensor<T, Fun>(tensor, sz, type, std::move(post_process));
        if (images.size() != 1)
        {
            throw std::runtime_error{fmt::format(
                "error: attempting to retrieve a single image from a batched "
                "tensor with {} images. Please use images_from_tensor instead",
                images.size())};
        }
        return images.front();
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
