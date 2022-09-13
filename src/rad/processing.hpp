#pragma once

#include "opencv.hpp"

#include <filesystem>
#include <functional>
#include <string>
#include <tbb/parallel_for_each.h>
#include <vector>

namespace rad
{
    namespace fs = std::filesystem;

#if defined(RAD_ONNX_ENABLED)
    void prepare_training_directories(std::string const& root,
                                      std::string const& app_name);
#endif

    std::pair<std::string, cv::Mat> load_image(std::string const& path);

    template<typename ImageProcessFun>
    void process_images(std::string const& root, ImageProcessFun&& fun)
    {
        for (auto const& entry : fs::directory_iterator{root})
        {
            if (!entry.is_directory())
            {
                auto [filename, img] = load_image(entry.path().string());
                fun(filename, img);
            }
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

    template<typename ImageProcessFun>
    void process_images_parallel(std::string const& root, ImageProcessFun&& fun)
    {
        fs::directory_iterator ite{root};
        tbb::parallel_for_each(fs::begin(ite),
                               fs::end(ite),
                               [fun](fs::directory_entry const& entry) {
                                   auto [filename, img] =
                                       load_image(entry.path().string());
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

    void create_result_dir(std::string const& root, std::string const& app_name);
    void save_result(cv::Mat const& img,
                     std::string const& root,
                     std::string const& app_name,
                     std::string const& img_name);
} // namespace rad
