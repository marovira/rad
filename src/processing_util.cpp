#include "rad/processing_util.hpp"

namespace rad
{
    namespace fs = std::filesystem;

    std::vector<std::filesystem::path> get_file_paths_from_root(std::string const& root)
    {
        std::vector<fs::path> files;
        for (auto const& entry : fs::directory_iterator{root})
        {
            if (!entry.is_directory())
            {
                files.push_back(entry.path());
            }
        }

        return files;
    }

    std::pair<std::string, cv::Mat> load_image(std::string const& path, int flags)
    {
        fs::path entry{path};
        return {entry.stem().string(), cv::imread(entry.string(), flags)};
    }

    std::pair<std::string, cv::Mat> load_image(std::string const& path)
    {
        return load_image(path, cv::IMREAD_COLOR);
    }

    void create_result_dir(std::string const& root, std::string const& app_name)
    {
        fs::create_directories(root);

        std::string res_root = root + "/" + app_name + "/";
        fs::create_directories(res_root);
    }

    void save_result(cv::Mat const& img,
                     std::string const& root,
                     std::string const& app_name,
                     std::string const& img_name)
    {
        if (img.empty())
        {
            return;
        }

        cv::Mat result;
        if (img.channels() == 1)
        {
            double alpha = (img.type() == CV_32F) ? 255.0 : 1.0;
            img.convertTo(result, CV_8UC1, alpha);
        }
        else if (img.channels() == 3)
        {
            double alpha = (img.type() == CV_32FC3) ? 255.0 : 1.0;
            img.convertTo(result, CV_8UC3, alpha);
        }
        else
        {
            double alpha = (img.type() == CV_32FC4) ? 255.0 : 1.0;
            img.convertTo(result, CV_8UC4, alpha);
        }

        std::string res_root = root + "/" + app_name + "/";
        std::string path     = res_root + img_name;
        cv::imwrite(path, result);
    }

} // namespace rad
