#include <coeus/processing.hpp>

#include "test_file_manager.hpp"

#include <catch2/catch_test_macros.hpp>
#include <zeus/string.hpp>

namespace fs = std::filesystem;

TEST_CASE("processing - process_images", "[coeus]")
{
    TestFileManager::Params params;
    TestFileManager mgr{params};

    std::vector<bool> seen_files(params.num_files, false);

    auto fun = [&seen_files](std::string const& name, cv::Mat const&) {
        // Order may be arbitrary
        auto num           = zeus::split(name, '_')[2];
        auto as_int        = std::stoi(num);
        seen_files[as_int] = true;
    };

    SECTION("Full list")
    {
        coeus::process_images(mgr.root().string(), fun);
        for (auto seen : seen_files)
        {
            REQUIRE(seen);
        }
    }

    SECTION("Full list - parallel")
    {
        coeus::process_images_parallel(mgr.root().string(), fun);
        for (auto seen : seen_files)
        {
            REQUIRE(seen);
        }
    }

    SECTION("Partial list")
    {
        std::vector<std::string> samples{"test_img_0.jpg",
                                         "test_img_2.jpg",
                                         "test_img_4.jpg"};
        coeus::process_images(mgr.root().string(), fun);
        REQUIRE(seen_files[0]);
        REQUIRE(seen_files[2]);
        REQUIRE(seen_files[4]);
    }

    SECTION("Partial list - parallel")
    {
        std::vector<std::string> samples{"test_img_0.jpg",
                                         "test_img_2.jpg",
                                         "test_img_4.jpg"};
        coeus::process_images_parallel(mgr.root().string(), fun);
        REQUIRE(seen_files[0]);
        REQUIRE(seen_files[2]);
        REQUIRE(seen_files[4]);
    }
}

TEST_CASE("processing - process_files", "[coeus]")
{
    TestFileManager::Params params;
    TestFileManager mgr{params};

    std::vector<bool> seen_files(params.num_files, false);

    auto fun = [&seen_files](std::string const& path) {
        fs::path p{path};
        std::string name = p.stem().string();

        auto num           = zeus::split(name, '_')[2];
        auto as_int        = std::stoi(num);
        seen_files[as_int] = true;
    };

    SECTION("Full list")
    {
        coeus::process_files(mgr.root().string(), fun);
        for (auto seen : seen_files)
        {
            REQUIRE(seen);
        }
    }

    SECTION("Full list - parallel")
    {
        coeus::process_files_parallel(mgr.root().string(), fun);
        for (auto seen : seen_files)
        {
            REQUIRE(seen);
        }
    }

    SECTION("Partial list")
    {
        std::vector<std::string> samples{"test_img_0.jpg",
                                         "test_img_2.jpg",
                                         "test_img_4.jpg"};
        coeus::process_files(mgr.root().string(), fun);
        REQUIRE(seen_files[0]);
        REQUIRE(seen_files[2]);
        REQUIRE(seen_files[4]);
    }

    SECTION("Partial list - parallel")
    {
        std::vector<std::string> samples{"test_img_0.jpg",
                                         "test_img_2.jpg",
                                         "test_img_4.jpg"};
        coeus::process_files_parallel(mgr.root().string(), fun);
        REQUIRE(seen_files[0]);
        REQUIRE(seen_files[2]);
        REQUIRE(seen_files[4]);
    }
}
