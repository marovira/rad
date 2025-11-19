#pragma once

#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>

#include <fmt/format.h>

#include <filesystem>
#include <string>

class TestFileManager
{
public:
    struct Params
    {
        int num_files{10};
        std::string prefix{"test_img"};
        std::string ext{"jpg"};
        cv::Size size{64, 64};
        int type{CV_8UC3};
    };

    TestFileManager(Params const& params)
    {
        auto [num_files, prefix, ext, size, type] = params;

        m_root = std::filesystem::absolute("./test_files");
        std::filesystem::create_directories(m_root);

        for (int i{0}; i < num_files; ++i)
        {
            const cv::Mat img = cv::Mat::ones(size, type);
            auto img_name     = fmt::format("{}_{}.{}", prefix, i, ext);
            auto img_path     = m_root / img_name;
            cv::imwrite(img_path.string(), img);
        }
    }

    TestFileManager(TestFileManager const&) = default;
    TestFileManager(TestFileManager&&)      = default;

    ~TestFileManager()
    {
        try
        {
            std::filesystem::remove_all(m_root);
        }
        catch (std::filesystem::filesystem_error const&) // NOLINT(bugprone-empty-catch)
        {}
    }

    TestFileManager& operator=(TestFileManager const&) = default;
    TestFileManager& operator=(TestFileManager&&)      = default;

    [[nodiscard]]
    std::filesystem::path root() const
    {
        return m_root;
    }

private:
    std::filesystem::path m_root;
};
