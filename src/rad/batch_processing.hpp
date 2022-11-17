#pragma once

#include "opencv.hpp"
#include "processing_util.hpp"

#include <algorithm>
#include <functional>
#include <string>
#include <tbb/parallel_for_each.h>
#include <vector>

namespace rad
{
    struct Batch
    {
        std::size_t id{0};
        std::size_t count{0};
    };

    struct ImageBatch : public Batch
    {
        std::vector<std::pair<std::string, cv::Mat>> imgs;
    };

    struct FileBatch : public Batch
    {
        std::vector<std::string> paths;
    };

    template<typename T, typename TransformFun>
    decltype(auto) split_into_batches(std::vector<T> const& list,
                                      std::size_t batch_size,
                                      TransformFun fun)
    {
        if (list.empty())
        {
            throw std::runtime_error{"error: list to be split into batches is empty"};
        }

        using U = decltype(fun(list.front()));

        std::vector<std::vector<U>> batches;
        for (std::size_t i{0}, b{0}; i < list.size(); i += batch_size, ++b)
        {
            auto last = std::min(list.size(), i + batch_size);
            std::vector<T> tmp(list.begin() + i, list.begin() + last);

            std::vector<U> batch;
            std::transform(tmp.begin(), tmp.end(), std::back_inserter(batch), fun);
            batches.emplace_back(std::move(batch));
        }

        return batches;
    }

    template<typename ImageBatchProcessingFun>
    void process_image_batch(std::string const& root,
                             std::size_t batch_size,
                             ImageBatchProcessingFun&& fun)
    {
        auto file_list = get_file_paths_from_root(root);
        auto batches =
            split_into_batches(file_list, batch_size, [](fs::path const& path) {
                return load_image(path.string());
            });

        for (std::size_t i{0}; auto const& batch : batches)
        {
            ImageBatch b{.imgs = batch};
            b.id    = i;
            b.count = batches.size();

            fun(b);
            ++i;
        }
    }

    template<typename ImageBatchProcessingFun>
    void process_image_batch(std::string const& root,
                             std::vector<std::string> const& samples,
                             std::size_t batch_size,
                             ImageBatchProcessingFun&& fun)
    {
        auto batches =
            split_into_batches(samples, batch_size, [root](std::string const& sample) {
                std::string path = root + sample;
                return load_image(path);
            });

        for (std::size_t i{0}; auto const& batch : batches)
        {
            ImageBatch b{.imgs = batch};
            b.id    = i;
            b.count = batches.size();

            fun(b);
            ++i;
        }
    }

    template<typename FileBatchProcessingFun>
    void process_file_batch(std::string const& root,
                            std::size_t batch_size,
                            FileBatchProcessingFun&& fun)
    {
        auto file_list = get_file_paths_from_root(root);
        auto batches =
            split_into_batches(file_list, batch_size, [](fs::path const& path) {
                return path.string();
            });

        for (std::size_t i{0}; auto const& batch : batches)
        {
            FileBatch b{.paths = batch};
            b.id    = i;
            b.count = batches.size();

            fun(b);
            ++i;
        }
    }

    template<typename FileBatchProcessingFun>
    void process_file_batch(std::string const& root,
                            std::vector<std::string> const& samples,
                            std::size_t batch_size,
                            FileBatchProcessingFun&& fun)
    {
        auto batches =
            split_into_batches(samples, batch_size, [root](std::string const& sample) {
                return root + sample;
            });

        for (std::size_t i{0}; auto const& batch : batches)
        {
            FileBatch b{.paths = batch};
            b.id    = i;
            b.count = batches.size();

            fun(b);
            ++i;
        }
    }

} // namespace rad
