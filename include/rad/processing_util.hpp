#pragma once

#include "opencv.hpp"

#include <filesystem>

namespace rad
{
    std::vector<std::filesystem::path> get_file_paths_from_root(std::string const& root);

    std::pair<std::string, cv::Mat> load_image(std::string const& path, int flags);
    std::pair<std::string, cv::Mat> load_image(std::string const& path);

    void create_result_dir(std::string const& root, std::string const& app_name);
    void save_result(cv::Mat const& img,
                     std::string const& root,
                     std::string const& app_name,
                     std::string const& img_name);
} // namespace rad
