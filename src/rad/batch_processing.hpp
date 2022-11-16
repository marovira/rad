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
    using ImageBatch = std::vector<std::pair<std::string, cv::Mat>>;
    using FileBatch  = std::vector<std::string>;

    template<typename T, typename TransformFun>
    decltype(auto) split_into_batches(std::vector<T> const& list,
                                      std::size_t batch_size,
                                      TransformFun fun)
    {
        if (list.empty())
        {
            throw std::runtime_error{"error: list to be split into batches is empty"};
        }

        using U = decltype(fun(list.begin()));

        std::vector<std::vector<U>> batches;
        for (std::size_t i{0}; i < list.size(); i += batch_size)
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
        for (auto const& batch :
             split_into_batches(file_list, batch_size, [](fs::path const& path) {
                 return load_image(path.string());
             }))
        {
            fun(batch);
        }
    }

    template<typename ImageBatchProcessingFun>
    void process_image_batch(std::string const& root,
                             std::vector<std::string> const& samples,
                             std::size_t batch_size,
                             ImageBatchProcessingFun&& fun)
    {
        for (auto const& batch :
             split_into_batches(samples, batch_size, [root](std::string const& sample) {
                 std::string path = root + sample;
                 return load_image(path);
             }))
        {
            fun(batch);
        }
    }

    template<typename FileBatchProcessingFun>
    void process_file_batch(std::string const& root,
                            std::size_t batch_size,
                            FileBatchProcessingFun&& fun)
    {
        auto file_list = get_file_paths_from_root(root);
        for (auto const& batch :
             split_into_batches(file_list, batch_size, [](fs::path const& path) {
                 return path.string();
             }))
        {
            fun(batch);
        }
    }

    template<typename FileBatchProcessingFun>
    void process_file_batch(std::string const& root,
                            std::vector<std::string> const& samples,
                            std::size_t batch_size,
                            FileBatchProcessingFun&& fun)
    {
        for (auto const& batch :
             split_into_batches(samples, batch_size, [root](std::string const& sample) {
                 std::string path = root + sample;
                 return path;
             }))
        {
            fun(batch);
        }
    }

} // namespace rad
