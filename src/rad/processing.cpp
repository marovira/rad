#include "processing.hpp"

namespace rad
{
#if defined(RAD_ONNX_ENABLED)
    void prepare_training_directories(std::string const& root,
                                      std::string const& app_name)
    {
        // Create the top-level models directory first.
        std::string models_root{root};
        fs::create_directories(models_root);

        // Now inside that, create the following:
        // 1. A directory with the app name
        // 2. A sub-directory called "chkpt" where the checkpoints will be
        // placed,
        // 3. A sub-directory called "trained" where the final trained output
        // will be placed.
        models_root += app_name + "/";
        fs::create_directories(models_root);
        fs::create_directories(models_root + "chkpt/");
        fs::create_directories(models_root + "trained/");
    }
#endif

    std::pair<std::string, cv::Mat> load_image(std::string const& path)
    {
        std::string image_name;
        fs::path entry{path};
        cv::Mat img;
        if (entry.extension() == ".jpg" || entry.extension() == ".JPG"
            || entry.extension() == ".png")
        {
            image_name = entry.stem().string();
            img        = cv::imread(entry.string());
        }

        return {image_name, img};
    }

    void create_result_dir(std::string const& root, std::string const& app_name)
    {
        fs::create_directories(root);

        std::string res_root = root + app_name + "/";
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

        std::string res_root = root + app_name + "/";
        std::string path     = res_root + img_name;
        cv::imwrite(path, result);
    }

} // namespace rad
