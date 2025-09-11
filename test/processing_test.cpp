#include <rad/processing.hpp>

#include "test_file_manager.hpp"

#include <catch2/catch_test_macros.hpp>
#include <zeus/platform.hpp>
#include <zeus/string.hpp>

#include <atomic>

namespace fs = std::filesystem;

TEST_CASE("processing - process_images", "[rad]")
{
    SECTION("Check files")
    {
        TestFileManager::Params params;
        TestFileManager mgr{params};

        std::vector<std::atomic<bool>> seen_files(params.num_files);

        auto fun = [&seen_files](std::string const& name, cv::Mat const&) {
            // Order may be arbitrary
            auto num           = zeus::split(name, '_')[2];
            auto as_int        = std::stoi(num);
            seen_files[as_int] = true;
        };

        SECTION("Full list")
        {
            rad::process_images(mgr.root().string(), fun);
            for (auto const& seen : seen_files)
            {
                REQUIRE(seen);
            }
        }

        SECTION("Full list - parallel")
        {
            rad::process_images_parallel(mgr.root().string(), fun);
            for (auto const& seen : seen_files)
            {
                REQUIRE(seen);
            }
        }

        SECTION("Partial list")
        {
            std::vector<std::string> samples{"test_img_0.jpg",
                                             "test_img_2.jpg",
                                             "test_img_4.jpg"};
            rad::process_images(mgr.root().string(), fun);
            REQUIRE(seen_files[0]);
            REQUIRE(seen_files[2]);
            REQUIRE(seen_files[4]);
        }

        SECTION("Partial list - parallel")
        {
            std::vector<std::string> samples{"test_img_0.jpg",
                                             "test_img_2.jpg",
                                             "test_img_4.jpg"};
            rad::process_images_parallel(mgr.root().string(), fun);
            REQUIRE(seen_files[0]);
            REQUIRE(seen_files[2]);
            REQUIRE(seen_files[4]);
        }
    }

    SECTION("Check loading")
    {
        static constexpr int num_files{10};
        SECTION("8-bit")
        {
            TestFileManager::Params params{.num_files = num_files};
            TestFileManager mgr{params};

            auto fun = [params](std::string const&, cv::Mat const& img) {
                REQUIRE(img.type() == params.type);
            };

            rad::process_images(mgr.root().string(), fun);
            rad::process_images_parallel(mgr.root().string(), fun);
        }

#if !defined(ZEUS_PLATFORM_APPLE) || !defined(RAD_CI_BUILD)
        SECTION("16-bit")
        {
            // This test needs to be enabled for all platforms with the exception of CI
            // builds on Apple.
            TestFileManager::Params params{.num_files = num_files,
                                           .ext       = "png",
                                           .type      = CV_16UC3};
            TestFileManager mgr{params};

            auto fun = [params](std::string const&, cv::Mat const& img) {
                REQUIRE(img.type() == params.type);
            };

            rad::process_images(mgr.root().string(), fun, cv::IMREAD_UNCHANGED);
            rad::process_images_parallel(mgr.root().string(), fun, cv::IMREAD_UNCHANGED);
        }
#endif
    }
}

TEST_CASE("processing - process_files", "[rad]")
{
    TestFileManager::Params params;
    TestFileManager mgr{params};

    std::vector<std::atomic<bool>> seen_files(params.num_files);

    auto fun = [&seen_files](std::string const& path) {
        fs::path p{path};
        std::string name = p.stem().string();

        auto num           = zeus::split(name, '_')[2];
        auto as_int        = std::stoi(num);
        seen_files[as_int] = true;
    };

    SECTION("Full list")
    {
        rad::process_files(mgr.root().string(), fun);
        for (auto const& seen : seen_files)
        {
            REQUIRE(seen);
        }
    }

    SECTION("Full list - parallel")
    {
        rad::process_files_parallel(mgr.root().string(), fun);
        for (auto const& seen : seen_files)
        {
            REQUIRE(seen);
        }
    }

    SECTION("Partial list")
    {
        std::vector<std::string> samples{"test_img_0.jpg",
                                         "test_img_2.jpg",
                                         "test_img_4.jpg"};
        rad::process_files(mgr.root().string(), fun);
        REQUIRE(seen_files[0]);
        REQUIRE(seen_files[2]);
        REQUIRE(seen_files[4]);
    }

    SECTION("Partial list - parallel")
    {
        std::vector<std::string> samples{"test_img_0.jpg",
                                         "test_img_2.jpg",
                                         "test_img_4.jpg"};
        rad::process_files_parallel(mgr.root().string(), fun);
        REQUIRE(seen_files[0]);
        REQUIRE(seen_files[2]);
        REQUIRE(seen_files[4]);
    }
}
