#pragma once

#include "opencv.hpp"
#include "processing_util.hpp"

#include <functional>
#include <string>
#include <tbb/parallel_for_each.h>
#include <vector>

namespace coeus
{
    template<typename ImageProcessFun>
    void process_images(std::string const& root, ImageProcessFun&& fun)
    {
        for (auto const& entry : get_file_paths_from_root(root))
        {
            auto [filename, img] = load_image(entry.string());
            fun(filename, img);
        }
    }

    template<typename ImageProcessFun>
    void process_images(std::string const& root,
                        std::vector<std::string> const& samples,
                        ImageProcessFun&& fun)
    {
        for (auto sample : samples)
        {
            std::string path     = root + sample;
            auto [filename, img] = load_image(path);
            fun(filename, img);
        }
    }

    template<typename FileProcessFun>
    void process_files(std::string const& root, FileProcessFun&& fun)
    {
        for (auto const& entry : get_file_paths_from_root(root))
        {
            fun(entry.string());
        }
    }

    template<typename FileProcessFun>
    void process_files(std::string const& root,
                       std::vector<std::string> const& samples,
                       FileProcessFun&& fun)
    {
        for (auto sample : samples)
        {
            std::string path = root + sample;
            fun(path);
        }
    }

    template<typename ImageProcessFun>
    void process_images_parallel(std::string const& root, ImageProcessFun&& fun)
    {
        auto files = get_file_paths_from_root(root);
        std::filesystem::directory_iterator ite{root};

        tbb::parallel_for_each(files.begin(),
                               files.end(),
                               [fun](std::filesystem::path const& entry) {
                                   auto [filename, img] = load_image(entry.string());
                                   fun(filename, img);
                               });
    }

    template<typename ImageProcessFun>
    void process_images_parallel(std::string const& root,
                                 std::vector<std::string> const& samples,
                                 ImageProcessFun&& fun)
    {
        tbb::parallel_for_each(samples.begin(),
                               samples.end(),
                               [fun, root](std::string const& sample) {
                                   std::string path     = root + sample;
                                   auto [filename, img] = load_image(path);
                                   fun(filename, img);
                               });
    }

    template<typename FileProcessFun>
    void process_files_parallel(std::string const& root, FileProcessFun&& fun)
    {
        auto files = get_file_paths_from_root(root);
        std::filesystem::directory_iterator ite{root};
        tbb::parallel_for_each(files.begin(),
                               files.end(),
                               [fun](std::filesystem::path const& entry) {
                                   fun(entry.string());
                               });
    }

    template<typename FileProcessFun>
    void process_files_parallel(std::string const& root,
                                std::vector<std::string> const& samples,
                                FileProcessFun&& fun)
    {
        tbb::parallel_for_each(samples.begin(),
                               samples.end(),
                               [fun, root](std::string const& sample) {
                                   std::string path     = root + sample;
                                   auto [filename, img] = load_image(path);
                                   fun(filename, img);
                               });
    }

} // namespace coeus
