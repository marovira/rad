#pragma once

#include "opencv.hpp"
#include "processing_util.hpp"

#include <oneapi/tbb/parallel_for_each.h>
#include <string>
#include <vector>

namespace rad
{
    template<typename ImageProcessFun>
    void process_images(std::string const& root, ImageProcessFun fun, int flags)
    {
        for (auto const& entry : get_file_paths_from_root(root))
        {
            auto [filename, img] = load_image(entry.string(), flags);
            fun(filename, img);
        }
    }

    template<typename ImageProcessFun>
    void process_images(std::string const& root, ImageProcessFun fun)
    {
        process_images(root, fun, cv::IMREAD_COLOR);
    }

    template<typename ImageProcessFun>
    void process_images(std::string const& root,
                        std::vector<std::string> const& samples,
                        ImageProcessFun fun,
                        int flags)
    {
        for (auto sample : samples)
        {
            std::string path     = root + sample;
            auto [filename, img] = load_image(path, flags);
            fun(filename, img);
        }
    }

    template<typename ImageProcessFun>
    void process_images(std::string const& root,
                        std::vector<std::string> const& samples,
                        ImageProcessFun fun)
    {
        process_images(root, samples, fun, cv::IMREAD_COLOR);
    }

    template<typename FileProcessFun>
    void process_files(std::string const& root, FileProcessFun fun)
    {
        for (auto const& entry : get_file_paths_from_root(root))
        {
            fun(entry.string());
        }
    }

    template<typename FileProcessFun>
    void process_files(std::string const& root,
                       std::vector<std::string> const& samples,
                       FileProcessFun fun)
    {
        for (auto sample : samples)
        {
            std::string path = root + sample;
            fun(path);
        }
    }

    template<typename ImageProcessFun>
    void process_images_parallel(std::string const& root, ImageProcessFun fun, int flags)
    {
        auto files = get_file_paths_from_root(root);
        std::filesystem::directory_iterator ite{root};

        oneapi::tbb::parallel_for_each(files.begin(),
                                       files.end(),
                                       [fun, flags](std::filesystem::path const& entry) {
                                           auto [filename, img] =
                                               load_image(entry.string(), flags);
                                           fun(filename, img);
                                       });
    }

    template<typename ImageProcessFun>
    void process_images_parallel(std::string const& root, ImageProcessFun fun)
    {
        process_images_parallel(root, fun, cv::IMREAD_COLOR);
    }

    template<typename ImageProcessFun>
    void process_images_parallel(std::string const& root,
                                 std::vector<std::string> const& samples,
                                 ImageProcessFun fun,
                                 int flags)
    {
        oneapi::tbb::parallel_for_each(samples.begin(),
                                       samples.end(),
                                       [fun, root, flags](std::string const& sample) {
                                           std::string path     = root + sample;
                                           auto [filename, img] = load_image(path, flags);
                                           fun(filename, img);
                                       });
    }

    template<typename ImageProcessFun>
    void process_images_parallel(std::string const& root,
                                 std::vector<std::string> const& samples,
                                 ImageProcessFun fun)
    {
        process_images_parallel(root, samples, fun, cv::IMREAD_COLOR);
    }

    template<typename FileProcessFun>
    void process_files_parallel(std::string const& root, FileProcessFun fun)
    {
        auto files = get_file_paths_from_root(root);
        std::filesystem::directory_iterator ite{root};
        oneapi::tbb::parallel_for_each(files.begin(),
                                       files.end(),
                                       [fun](std::filesystem::path const& entry) {
                                           fun(entry.string());
                                       });
    }

    template<typename FileProcessFun>
    void process_files_parallel(std::string const& root,
                                std::vector<std::string> const& samples,
                                FileProcessFun fun)
    {
        oneapi::tbb::parallel_for_each(samples.begin(),
                                       samples.end(),
                                       [fun, root](std::string const& sample) {
                                           std::string path     = root + sample;
                                           auto [filename, img] = load_image(path);
                                           fun(filename, img);
                                       });
    }

} // namespace rad
