#pragma once

#include "concepts.hpp"
#include "onnxruntime.hpp"

#include <fmt/format.h>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <zeus/container_traits.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace rad::onnx
{
    class TensorSet
    {
    public:
        using const_iterator = std::vector<Ort::Value>::const_iterator;

        TensorSet()                 = default;
        TensorSet(TensorSet const&) = delete;
        ~TensorSet()                = default;

        // NOLINTNEXTLINE(bugprone-exception-escape)
        TensorSet(TensorSet&& other) noexcept :
            m_tensors{std::move(other.m_tensors)},
            m_tensor_data{std::move(other.m_tensor_data)}
        {}

        TensorSet& operator=(TensorSet const&) = delete;

        TensorSet& operator=(TensorSet&& other) noexcept
        {
            m_tensors     = std::move(other.m_tensors);
            m_tensor_data = std::move(other.m_tensor_data);
            return *this;
        }

        Ort::Value const& operator[](std::size_t i) const
        {
            return m_tensors.at(i);
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

        [[nodiscard]]
        Ort::Value const& front() const
        {
            if (m_tensors.empty())
            {
                throw std::runtime_error{
                    "error: cannot get the first element of an empty set"};
            }
            return m_tensors.front();
        }

        [[nodiscard]]
        Ort::Value const& back() const
        {
            if (m_tensors.empty())
            {
                throw std::runtime_error{
                    "error: cannot get the last element of an empty set"};
            }

            return m_tensors.back();
        }

        void clear()
        {
            m_tensors.clear();
            m_tensor_data.clear();
        }

        void insert_tensor(Ort::Value&& tensor)
        {
            m_tensors.emplace_back(std::move(tensor));
        }

        void insert_tensors(std::vector<Ort::Value>& tensors)
        {
            for (auto& tensor : tensors)
            {
                m_tensors.emplace_back(std::move(tensor));
            }
            tensors.clear();
        }

        template<ImageTensorDataType T>
        void insert_tensor_from_batched_images(std::vector<cv::Mat> const& images)
        {
            auto [tensor, data] = batched_images_to_tensor<T>(images);
            insert_tensor_and_data(std::move(tensor), std::move(data));
        }

        template<ImageTensorDataType T>
        void insert_tensor_from_image(cv::Mat const& img)
        {
            insert_tensor_from_batched_images<T>({img});
        }

        template<TensorDataType T, zeus::ContiguousContainer U>
        void insert_tensor_from_batched_arrays(std::vector<U> const& src_data)
        {
            auto [tensor, data] = batched_arrays_to_tensor<T, U>(src_data);
            insert_tensor_and_data(std::move(tensor), std::move(data));
        }

        template<TensorDataType T, zeus::ContiguousContainer U>
        void insert_tensor_from_array(U const& src_data)
        {
            insert_tensor_from_batched_arrays<T, U>({src_data});
        }

        template<TensorDataType T>
        void insert_tensor_from_scalar(T scalar)
        {
            auto [tensor, data] = tensor_from_scalar<T>(scalar);
            insert_tensor_and_data(std::move(tensor), std::move(data));
        }

        template<TensorDataType T>
        void insert_tensor_from_data(T const* src_data,
                                     std::vector<std::int64_t> const& shape)
        {
            auto [tensor, data] = tensor_from_data<T>(src_data, shape);
            insert_tensor_and_data(std::move(tensor), std::move(data));
        }

        void replace_tensor_at(std::size_t pos, Ort::Value&& tensor)
        {
            m_tensors.at(pos) = std::move(tensor);
            if (m_tensor_data.find(pos) != m_tensor_data.end())
            {
                m_tensor_data.erase(pos);
            }
        }

        template<ImageTensorDataType T>
        void replace_tensor_with_batched_images_at(std::size_t pos,
                                                   std::vector<cv::Mat> const& images)
        {
            auto [tensor, data] = batched_images_to_tensor<T>(images);
            replace_tensor_and_data_at(std::move(tensor), std::move(data), pos);
        }

        template<ImageTensorDataType T>
        void replace_tensor_with_image_at(std::size_t pos, cv::Mat const& img)
        {
            replace_tensor_with_batched_images_at<T>(pos, {img});
        }

        template<TensorDataType T, zeus::ContiguousContainer U>
        void replace_tensor_with_batched_arrays_at(std::size_t pos,
                                                   std::vector<U> const& src_data)
        {
            auto [tensor, data] = batched_arrays_to_tensor<T, U>(src_data);
            replace_tensor_and_data_at(std::move(tensor), std::move(data), pos);
        }

        template<TensorDataType T, zeus::ContiguousContainer U>
        void replace_tensor_with_array_at(std::size_t pos, U const& src_data)
        {
            replace_tensor_with_batched_arrays_at<T, U>(pos, {src_data});
        }

        template<TensorDataType T>
        void replace_tensor_with_scalar_at(std::size_t pos, T scalar)
        {
            auto [tensor, data] = tensor_from_scalar<T>(scalar);
            replace_tensor_and_data_at(std::move(tensor), std::move(data), pos);
        }

        template<TensorDataType T>
        void replace_tensor_with_data_at(std::size_t pos,
                                         T const* src_data,
                                         std::vector<std::int64_t> const& shape)
        {
            auto [tensor, data] = tensor_from_data<T>(src_data, shape);
            replace_tensor_and_data_at(std::move(tensor), std::move(data), pos);
        }

    private:
        [[nodiscard]]
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

        template<ImageTensorDataType T>
        [[nodiscard]]
        std::pair<Ort::Value, std::vector<std::byte>>
        batched_images_to_tensor(std::vector<cv::Mat> const& images) const
        {
            const auto [num_channels, rows, cols] = validate_batched_images(images);

            const auto stride = static_cast<std::size_t>(rows)
                                * static_cast<std::size_t>(cols) * sizeof(T);
            const auto size = images.size() * num_channels * rows * cols;

            std::vector<std::byte> tensor_data(size * sizeof(T));
            auto data_ptr = tensor_data.data();
            for (auto const& img : images)
            {
                std::vector<cv::Mat> ch(num_channels);
                for (auto& c : ch)
                {
                    c = cv::Mat(rows, cols, img.depth(), data_ptr);
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
                    data_ptr += stride;
                }

                cv::split(img, ch);
            }

            std::vector<std::int64_t> dims{
                static_cast<std::int64_t>(images.size()),
                num_channels,
                rows,
                cols,
            };

            Ort::Value tensor{nullptr};
            const Ort::MemoryInfo mem_info =
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            tensor = Ort::Value::CreateTensor<T>(
                mem_info,
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                reinterpret_cast<T*>(tensor_data.data()),
                size,
                dims.data(),
                dims.size());

            if (!tensor.IsTensor())
            {
                throw std::runtime_error{"error: could not create a valid tensor"};
            }

            return {std::move(tensor), std::move(tensor_data)};
        }

        template<TensorDataType T, zeus::ContiguousContainer U>
        [[nodiscard]]
        std::pair<Ort::Value, std::vector<std::byte>>
        batched_arrays_to_tensor(std::vector<U> const& src_data)
        {
            static_assert(std::is_same_v<T, typename U::value_type>);
            if (src_data.empty())
            {
                throw std::runtime_error{"error: array batch cannot be empty"};
            }

            const std::vector<std::int64_t> tensor_dims{
                static_cast<std::int64_t>(src_data.size()),
                static_cast<std::int64_t>(src_data[0].size())};

            const auto size = tensor_dims[0] * tensor_dims[1];
            std::vector<std::byte> tensor_data(size * sizeof(T));
            auto data_ptr = tensor_data.data();
            for (auto const& elem : src_data)
            {
                std::memcpy(data_ptr, elem.data(), elem.size() * sizeof(T));
                // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
                data_ptr += elem.size() * sizeof(T);
            }

            Ort::Value tensor{nullptr};
            Ort::MemoryInfo mem_info =
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            tensor = Ort::Value::CreateTensor<T>(
                mem_info,
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                reinterpret_cast<T*>(tensor_data.data()),
                size,
                tensor_dims.data(),
                tensor_dims.size());

            if (!tensor.IsTensor())
            {
                throw std::runtime_error{"error: could not create a valid tensor"};
            }

            return {std::move(tensor), std::move(tensor_data)};
        }

        template<TensorDataType T>
        [[nodiscard]]
        std::pair<Ort::Value, std::vector<std::byte>> tensor_from_scalar(T scalar)
        {
            const std::vector<std::int64_t> tensor_dims{1};
            std::vector<std::byte> tensor_data(sizeof(T));
            std::memcpy(tensor_data.data(), &scalar, tensor_data.size());

            Ort::Value tensor{nullptr};
            Ort::MemoryInfo mem_info =
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            tensor = Ort::Value::CreateTensor<T>(
                mem_info,
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                reinterpret_cast<T*>(tensor_data.data()),
                1,
                tensor_dims.data(),
                tensor_dims.size());

            if (!tensor.IsTensor())
            {
                throw std::runtime_error{"error: could not create a valid tensor"};
            }

            return {std::move(tensor), std::move(tensor_data)};
        }

        template<TensorDataType T>
        [[nodiscard]]
        std::pair<Ort::Value, std::vector<std::byte>>
        tensor_from_data(T const* data, std::vector<std::int64_t> const& shape)
        {
            const auto tensor_size =
                std::accumulate(shape.begin(), shape.end(), 1ll, std::multiplies());

            std::vector<std::byte> tensor_data(tensor_size * sizeof(T));
            std::memcpy(tensor_data.data(), data, tensor_size * sizeof(T));

            Ort::Value tensor{nullptr};
            const Ort::MemoryInfo mem_info =
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            tensor = Ort::Value::CreateTensor<T>(
                mem_info,
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                reinterpret_cast<T*>(tensor_data.data()),
                tensor_size,
                shape.data(),
                shape.size());

            if (!tensor.IsTensor())
            {
                throw std::runtime_error{"error: could not create a valid tensor"};
            }

            return {std::move(tensor), std::move(tensor_data)};
        }

        void replace_tensor_and_data_at(Ort::Value&& tensor,
                                        std::vector<std::byte>&& data,
                                        std::size_t pos)
        {
            m_tensors.at(pos) = Ort::Value{nullptr};
            m_tensors.at(pos) = std::move(tensor);
            if (m_tensor_data.find(pos) != m_tensor_data.end())
            {
                m_tensor_data.erase(pos);
            }
            m_tensor_data.insert(std::make_pair(pos, std::move(data)));
        }

        void insert_tensor_and_data(Ort::Value&& tensor, std::vector<std::byte>&& data)
        {
            m_tensors.emplace_back(std::move(tensor));
            m_tensor_data.insert(
                std::make_pair(m_tensor_data.size() - 1, std::move(data)));
        }

        std::vector<Ort::Value> m_tensors;
        std::map<std::size_t, std::vector<std::byte>> m_tensor_data;
    };
} // namespace rad::onnx
