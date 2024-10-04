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

        MemoryBackedTensorSet(MemoryBackedTensorSet&& other) noexcept :
            m_tensors{std::move(other.m_tensors)},
            m_tensor_data{std::move(other.m_tensor_data)}
        {}

        MemoryBackedTensorSet& operator=(MemoryBackedTensorSet const&) = delete;

        MemoryBackedTensorSet& operator=(MemoryBackedTensorSet&& other) noexcept
        {
            m_tensors     = std::move(other.m_tensors);
            m_tensor_data = std::move(other.m_tensor_data);
        }

        Ort::Value const& operator[](std::size_t i) const
        {
            return m_tensors[i];
        }

        [[nodiscard]]
        const_iterator begin() const
        {
            return m_tensors.begin();
        }

        [[nodiscard]]
        const_iterator end() const
        {
            return m_tensors.end();
        }

        [[nodiscard]]
        std::size_t size() const
        {
            return m_tensors.size();
        }

        [[nodiscard]]
        bool empty() const
        {
            return m_tensors.empty();
        }

        [[nodiscard]]
        std::vector<Ort::Value> const& tensors() const
        {
            return m_tensors;
        }

        void insert_tensor_from_batched_images(std::vector<cv::Mat> const& images)
        {
            const auto [num_channels, rows, cols] = validate_batched_images(images);
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

            std::vector<std::int64_t> tensor_dims{
                static_cast<std::int64_t>(images.size()),
                num_channels,
                rows,
                cols};

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
        void insert_tensor_from_batched_arrays(std::vector<U> const& src_data)
        {
            if (src_data.empty())
            {
                throw std::runtime_error{"error: array batch cannot be empty"};
            }

            std::vector<std::int64_t> tensor_dims{
                static_cast<std::int64_t>(src_data.size()),
                static_cast<std::int64_t>(src_data[0].size())};

            std::vector<T> tensor_data(tensor_dims[0] * tensor_dims[1]);
            auto data_ptr = tensor_data.data();
            for (auto const& elem : src_data)
            {
                std::memcpy(data_ptr, elem.data(), elem.size() * sizeof(T));
                data_ptr += elem.size();
            }

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

        template<zeus::ContiguousContainer U>
        void insert_tensor_from_array(U const& src_data)
        {
            insert_tensor_from_batched_arrays<U>({src_data});
        }

    private:
        std::tuple<int, int, int>
        validate_batched_images(std::vector<cv::Mat> const& images) const
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

        std::vector<Ort::Value> m_tensors;
        std::vector<std::vector<T>> m_tensor_data;
    };
} // namespace rad::onnx
