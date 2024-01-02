#pragma once

#include "../opencv.hpp"
#include "concepts.hpp"
#include "onnxruntime.hpp"
#include "perform_safe_op.hpp"

#include <zeus/container_traits.hpp>

namespace rad::onnx
{
    template<TensorDataType T>
    class MemoryBackedTensorSet
    {
    public:
        using const_iterator = std::vector<Ort::Value>::const_iterator;
        using value_type     = T;

        MemoryBackedTensorSet()                             = default;
        MemoryBackedTensorSet(MemoryBackedTensorSet const&) = delete;
        ~MemoryBackedTensorSet()                            = default;

        MemoryBackedTensorSet(MemoryBackedTensorSet&& other) :
            m_tensors{std::move(other.m_tensors)},
            m_tensor_data{std::move(other.m_tensor_data)}
        {}

        MemoryBackedTensorSet& operator=(MemoryBackedTensorSet const&) = delete;

        MemoryBackedTensorSet& operator=(MemoryBackedTensorSet&& other)
        {
            m_tensors     = std::move(other.m_tensors);
            m_tensor_data = std::move(other.m_tensor_data);
        }

        Ort::Value const& operator[](std::size_t i) const
        {
            return m_tensors[i];
        }

        const_iterator begin() const
        {
            return m_tensors.begin();
        }

        const_iterator end() const
        {
            return m_tensors.end();
        }

        std::size_t size() const
        {
            return m_tensors.size();
        }

        bool empty() const
        {
            return m_tensors.empty();
        }

        std::vector<Ort::Value> const& tensors() const
        {
            return m_tensors;
        }

        void insert_tensor_from_batched_images(std::vector<cv::Mat> const& images)
        {
            const auto depth = []() {
                if constexpr (std::is_same_v<value_type, float>)
                {
                    return CV_32F;
                }
                else
                {
                    return CV_8U;
                }
            }();

            auto blob = cv::dnn::blobFromImages(images,
                                                1.0,
                                                cv::Size{},
                                                cv::Scalar{},
                                                false,
                                                false,
                                                depth);

            cv::MatSize sz = blob.size;
            std::vector<std::int64_t> tensor_dims(sz.dims());
            for (int i{0}; i < sz.dims(); ++i)
            {
                tensor_dims[i] = sz[i];
            }

            std::vector<T> tensor_data(blob.total());
            tensor_data.assign(blob.template begin<T>(), blob.template end<T>());

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

            m_tensors.emplace_back(std::move(tensor));
            m_tensor_data.emplace_back(std::move(tensor_data));
        }

        void insert_tensor_from_image(cv::Mat const& img)
        {
            insert_tensor_from_batched_images({img});
        }

        template<zeus::ContiguousContainer U>
        void insert_tensor_from_array(U const& src_data)
        {
            static_assert(std::is_same_v<typename U::value_type, value_type>);

            std::vector<std::int64_t> tensor_dims{
                1,
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

            m_tensors.emplace_back(std::move(tensor));
            m_tensor_data.emplace_back(std::move(tensor_data));
        }

    private:
        std::vector<Ort::Value> m_tensors;
        std::vector<std::vector<T>> m_tensor_data;
    };
} // namespace rad::onnx
