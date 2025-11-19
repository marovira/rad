#include "test_file_manager.hpp"

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <rad/processing_util.hpp>
#include <zeus/platform.hpp> // NOLINT(misc-include-cleaner)

#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_set>

namespace fs = std::filesystem;

TEST_CASE("[processing_util] - get_file_paths_from_root", "[rad]")
{
    const TestFileManager mgr{TestFileManager::Params{}};

    auto files = rad::get_file_paths_from_root(mgr.root().string());

    REQUIRE(files.size() == 10);

    // Order is not guaranteed!
    std::unordered_set<fs::path> exp_paths;
    for (std::size_t i{0}; i < files.size(); ++i)
    {
        const fs::path exp = mgr.root() / fmt::format("test_img_{}.jpg", i);
        exp_paths.insert(exp);
    }

    for (auto file : files)
    {
        REQUIRE(exp_paths.find(file) != exp_paths.end());
    }
}

TEST_CASE("[processing_util] - load_image", "[rad]")
{
    TestFileManager::Params params{.num_files = 1};
    const std::string exp_name = "test_img_0";

    SECTION("jpg")
    {
        params.ext = "jpg";
        const TestFileManager mgr{params};

        const fs::path p = mgr.root() / fmt::format("{}.{}", exp_name, params.ext);
        auto [name, img] = rad::load_image(p.string());

        REQUIRE(name == exp_name);
        REQUIRE(img.size() == params.size);
        REQUIRE(img.type() == params.type);
    }

    SECTION("JPG")
    {
        params.ext = "JPG";
        const TestFileManager mgr{params};

        const fs::path p = mgr.root() / fmt::format("{}.{}", exp_name, params.ext);
        auto [name, img] = rad::load_image(p.string());

        REQUIRE(name == exp_name);
        REQUIRE(img.size() == params.size);
        REQUIRE(img.type() == params.type);
    }

#if !defined(ZEUS_PLATFORM_APPLE) || !defined(RAD_CI_BUILD)
    SECTION("png")
    {
        params.ext = "png";
        const TestFileManager mgr{params};

        const fs::path p = mgr.root() / fmt::format("{}.{}", exp_name, params.ext);
        auto [name, img] = rad::load_image(p.string());

        REQUIRE(name == exp_name);
        REQUIRE(img.size() == params.size);
        REQUIRE(img.type() == params.type);
    }
#endif
}

TEST_CASE("[processing_util] - create_result_dir", "[rad]")
{
    const fs::path root = fs::absolute("./test_root");
    rad::create_result_dir(root.string(), "app");

    REQUIRE(fs::exists(root / "app"));
    fs::remove_all(root);
}

TEST_CASE("[processing_util] - save_result", "[rad]")
{
    const fs::path root        = fs::absolute("./test_root");
    const std::string name     = "test_app";
    const std::string img_name = "test_img.jpg";
    const auto img_path        = root / (name + "/" + img_name);
    const cv::Size size{64, 64};

    fs::create_directories(root / name);

    SECTION("Empty image")
    {
        rad::save_result({}, root.string(), name, img_name);
        REQUIRE_FALSE(fs::exists(img_path));
    }

    SECTION("Single channel image")
    {
        SECTION("uint image")
        {
            const cv::Mat img = cv::Mat::ones(size, CV_8UC1);
            rad::save_result(img, root.string(), name, img_name);
            REQUIRE(fs::exists(img_path));
        }

        SECTION("float image")
        {
            const cv::Mat img = cv::Mat::ones(size, CV_32F);
            rad::save_result(img, root.string(), name, img_name);
            REQUIRE(fs::exists(img_path));
        }
    }

    SECTION("RGB image")
    {
        SECTION("uint image")
        {
            const cv::Mat img = cv::Mat::ones(size, CV_8UC3);
            rad::save_result(img, root.string(), name, img_name);
            REQUIRE(fs::exists(img_path));
        }

        SECTION("float image")
        {
            const cv::Mat img = cv::Mat::ones(size, CV_32FC3);
            rad::save_result(img, root.string(), name, img_name);
            REQUIRE(fs::exists(img_path));
        }
    }

    SECTION("RGBA image")
    {
        SECTION("uint image")
        {
            const cv::Mat img = cv::Mat::ones(size, CV_8UC4);
            rad::save_result(img, root.string(), name, img_name);
            REQUIRE(fs::exists(img_path));
        }

        SECTION("float image")
        {
            const cv::Mat img = cv::Mat::ones(size, CV_32FC4);
            rad::save_result(img, root.string(), name, img_name);
            REQUIRE(fs::exists(img_path));
        }
    }

    fs::remove_all(root);
}
