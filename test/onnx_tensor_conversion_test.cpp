#include <rad/onnx/tensor_conversion.hpp>

#include "onnx_test_helpers.hpp"
#include <rad/onnx/memory_backed_tensor_set.hpp>

#include <catch2/catch_template_test_macros.hpp>

namespace onnx = rad::onnx;

template<typename T>
struct TestDataType
{
    using Type = T;
};

template<>
struct TestDataType<Ort::Float16_t>
{
    using Type = std::uint16_t;
};

template<typename T, typename U>
bool array_equals(T const& lhs, U const& rhs)
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }

    for (std::size_t i{0}; i < lhs.size(); ++i)
    {
        if (lhs[i] != rhs[i])
        {
            return false;
        }
    }

    return true;
}

TEST_CASE("[tensor_conversion] - detail::validate_batched_images", "[rad::onnx]")
{
    const auto generate_batch = [](cv::Size sz, int type) {
        static constexpr auto batch_size{4};
        std::vector<cv::Mat> batch(batch_size);

        const cv::Mat img = cv::Mat::zeros(sz, type);

        for (auto& b : batch)
        {
            b = img.clone();
        }

        return batch;
    };

    SECTION("Valid batch")
    {
        auto batch                  = generate_batch(cv::Size{10, 20}, CV_8UC3);
        auto [channels, rows, cols] = onnx::detail::validate_batched_images(batch);

        REQUIRE(channels == 3);
        REQUIRE(rows == 20);
        REQUIRE(cols == 10);
    }

    SECTION("Invalid batch: channels")
    {
        const cv::Size sz{32, 32};
        auto batch   = generate_batch(sz, CV_8UC3);
        batch.back() = cv::Mat::zeros(cv::Size{32, 32}, CV_8UC1);

        REQUIRE_THROWS(onnx::detail::validate_batched_images(batch));
    }

    SECTION("Invalid batch: rows")
    {
        const cv::Size sz{32, 32};
        auto batch   = generate_batch(sz, CV_8UC3);
        batch.back() = cv::Mat::zeros(cv::Size{32, 10}, CV_8UC1);

        REQUIRE_THROWS(onnx::detail::validate_batched_images(batch));
    }

    SECTION("Invalid batch: cols")
    {
        const cv::Size sz{32, 32};
        auto batch   = generate_batch(sz, CV_8UC3);
        batch.back() = cv::Mat::zeros(cv::Size{10, 32}, CV_8UC1);

        REQUIRE_THROWS(onnx::detail::validate_batched_images(batch));
    }
}

TEMPLATE_TEST_CASE("[tensor_conversion] - image_batch_to_tensor_blob",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    static constexpr int num_images{4};
    std::vector<cv::Mat> images(num_images);
    const auto exp_size = get_test_image_size();

    SECTION("Grayscale images")
    {
        constexpr int channels{1};
        for (auto& img : images)
        {
            img = make_test_image<TestType>(channels);
        }

        auto blob = onnx::image_batch_to_tensor_blob<TestType>(images);
        REQUIRE_FALSE(blob.data.empty());
        REQUIRE_FALSE(blob.shape.empty());

        REQUIRE(blob.shape.size() == 4);
        REQUIRE(blob.shape[0] == num_images);
        REQUIRE(blob.shape[1] == channels);
        REQUIRE(blob.shape[2] == exp_size.height);
        REQUIRE(blob.shape[3] == exp_size.width);
    }

    SECTION("RGB images")
    {
        constexpr int channels{3};
        for (auto& img : images)
        {
            img = make_test_image<TestType>(channels);
        }

        auto blob = onnx::image_batch_to_tensor_blob<TestType>(images);
        REQUIRE_FALSE(blob.data.empty());
        REQUIRE_FALSE(blob.shape.empty());

        REQUIRE(blob.shape.size() == 4);
        REQUIRE(blob.shape[0] == num_images);
        REQUIRE(blob.shape[1] == channels);
        REQUIRE(blob.shape[2] == exp_size.height);
        REQUIRE(blob.shape[3] == exp_size.width);
    }

    SECTION("RGBA images")
    {
        constexpr int channels{4};
        for (auto& img : images)
        {
            img = make_test_image<TestType>(channels);
        }

        auto blob = onnx::image_batch_to_tensor_blob<TestType>(images);
        REQUIRE_FALSE(blob.data.empty());
        REQUIRE_FALSE(blob.shape.empty());

        REQUIRE(blob.shape.size() == 4);
        REQUIRE(blob.shape[0] == num_images);
        REQUIRE(blob.shape[1] == channels);
        REQUIRE(blob.shape[2] == exp_size.height);
        REQUIRE(blob.shape[3] == exp_size.width);
    }
}

TEMPLATE_TEST_CASE("[tensor_conversion] - image_to_tensor_blob",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    const auto exp_size = get_test_image_size();

    SECTION("Grayscale image")
    {
        constexpr int channels{1};

        auto blob =
            onnx::image_to_tensor_blob<TestType>(make_test_image<TestType>(channels));

        REQUIRE_FALSE(blob.data.empty());
        REQUIRE_FALSE(blob.shape.empty());

        REQUIRE(blob.shape.size() == 4);
        REQUIRE(blob.shape[0] == 1);
        REQUIRE(blob.shape[1] == channels);
        REQUIRE(blob.shape[2] == exp_size.height);
        REQUIRE(blob.shape[3] == exp_size.width);
    }

    SECTION("RGB image")
    {
        constexpr int channels{3};

        auto blob =
            onnx::image_to_tensor_blob<TestType>(make_test_image<TestType>(channels));

        REQUIRE_FALSE(blob.data.empty());
        REQUIRE_FALSE(blob.shape.empty());

        REQUIRE(blob.shape.size() == 4);
        REQUIRE(blob.shape[0] == 1);
        REQUIRE(blob.shape[1] == channels);
        REQUIRE(blob.shape[2] == exp_size.height);
        REQUIRE(blob.shape[3] == exp_size.width);
    }

    SECTION("RGBA image")
    {
        constexpr int channels{4};

        auto blob =
            onnx::image_to_tensor_blob<TestType>(make_test_image<TestType>(channels));

        REQUIRE_FALSE(blob.data.empty());
        REQUIRE_FALSE(blob.shape.empty());

        REQUIRE(blob.shape.size() == 4);
        REQUIRE(blob.shape[0] == 1);
        REQUIRE(blob.shape[1] == channels);
        REQUIRE(blob.shape[2] == exp_size.height);
        REQUIRE(blob.shape[3] == exp_size.width);
    }
}

TEMPLATE_TEST_CASE("[tensor_conversion] = image_batch_from_tensor_blob",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   std::uint16_t)
{
    static constexpr int num_images{4};
    const auto post_process = [](cv::Mat img) {
        return img;
    };
    std::vector<cv::Mat> images(num_images);
    const auto test_size = get_test_image_size();

    SECTION("Grayscale images")
    {
        constexpr int channels{1};
        const int type = get_test_image_type<TestType>(channels);
        for (auto& img : images)
        {
            img = make_test_image<TestType>(channels);
        }

        auto blob = onnx::image_batch_to_tensor_blob<TestType>(images);
        auto ret =
            onnx::image_batch_from_tensor_blob(blob, test_size, type, post_process);

        REQUIRE(image_array_equals(ret, images));

        auto ret2 = onnx::image_batch_from_tensor_blob(blob, test_size, type);
        REQUIRE(image_array_equals(ret, ret2));
    }

    SECTION("RGB images")
    {
        constexpr int channels{3};
        const int type = get_test_image_type<TestType>(channels);
        for (auto& img : images)
        {
            img = make_test_image<TestType>(channels);
        }

        auto blob = onnx::image_batch_to_tensor_blob<TestType>(images);
        auto ret =
            onnx::image_batch_from_tensor_blob(blob, test_size, type, post_process);

        REQUIRE(image_array_equals(ret, images));

        auto ret2 = onnx::image_batch_from_tensor_blob(blob, test_size, type);
        REQUIRE(image_array_equals(ret, ret2));
    }

    SECTION("RGBA images")
    {
        constexpr int channels{4};
        const int type = get_test_image_type<TestType>(channels);
        for (auto& img : images)
        {
            img = make_test_image<TestType>(channels);
        }

        auto blob = onnx::image_batch_to_tensor_blob<TestType>(images);
        auto ret =
            onnx::image_batch_from_tensor_blob(blob, test_size, type, post_process);

        REQUIRE(image_array_equals(ret, images));

        auto ret2 = onnx::image_batch_from_tensor_blob(blob, test_size, type);
        REQUIRE(image_array_equals(ret, ret2));
    }
}

TEMPLATE_TEST_CASE("[tensor_conversion] - image_from_tensor_blob",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    const auto exp_size     = get_test_image_size();
    const auto post_process = [](cv::Mat img) {
        return img;
    };

    SECTION("Invalid batch")
    {
        std::vector<cv::Mat> images(2);
        for (auto& img : images)
        {
            img = make_test_image<TestType>(1);
        }

        auto blob = onnx::image_batch_to_tensor_blob<TestType>(images);
        REQUIRE_THROWS(onnx::image_from_tensor_blob(blob,
                                                    exp_size,
                                                    images.front().type(),
                                                    post_process));

        REQUIRE_THROWS(
            onnx::image_from_tensor_blob(blob, exp_size, images.front().type()));
    }

    SECTION("Grayscale image")
    {
        constexpr int channels{1};
        const auto type   = get_test_image_type<TestType>(channels);
        const cv::Mat img = make_test_image<TestType>(channels);
        const auto size   = img.size();

        auto blob = onnx::image_to_tensor_blob<TestType>(img);
        auto ret  = onnx::image_from_tensor_blob(blob, size, type, post_process);
        auto ret2 = onnx::image_from_tensor_blob(blob, size, type);

        REQUIRE(image_equals(ret, ret2));
        REQUIRE(image_equals(ret, img));
    }

    SECTION("RGB image")
    {
        constexpr int channels{3};
        const auto type   = get_test_image_type<TestType>(channels);
        const cv::Mat img = make_test_image<TestType>(channels);
        const auto size   = img.size();

        auto blob = onnx::image_to_tensor_blob<TestType>(img);
        auto ret  = onnx::image_from_tensor_blob(blob, size, type, post_process);
        auto ret2 = onnx::image_from_tensor_blob(blob, size, type);

        REQUIRE(image_equals(ret, ret2));
        REQUIRE(image_equals(ret, img));
    }

    SECTION("RGBA image")
    {
        constexpr int channels{4};
        const auto type   = get_test_image_type<TestType>(channels);
        const cv::Mat img = make_test_image<TestType>(channels);
        const auto size   = img.size();

        auto blob = onnx::image_to_tensor_blob<TestType>(img);
        auto ret  = onnx::image_from_tensor_blob(blob, size, type, post_process);
        auto ret2 = onnx::image_from_tensor_blob(blob, size, type);

        REQUIRE(image_equals(ret, ret2));
        REQUIRE(image_equals(ret, img));
    }
}

TEMPLATE_TEST_CASE("[tensor_conversion] - image_batch_from_tensor",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    std::vector<cv::Mat> images(4);
    const auto size = get_test_image_size();
    onnx::MemoryBackedTensorSet<TestType> s;
    const auto post_process = [](cv::Mat img) {
        return img;
    };

    SECTION("Grayscale images")
    {
        constexpr int channels{1};
        const auto type = get_test_image_type<TestType>(channels);
        for (auto& img : images)
        {
            img = make_test_image<TestType>(channels);
        }

        s.insert_tensor_from_batched_images(images);
        auto ret =
            onnx::image_batch_from_tensor<TestType>(s[0], size, type, post_process);
        auto ret2 = onnx::image_batch_from_tensor<TestType>(s[0], size, type);

        REQUIRE(image_array_equals(ret, ret2));
        REQUIRE(image_array_equals(ret, images));
    }

    SECTION("RGB images")
    {
        constexpr int channels{3};
        const auto type = get_test_image_type<TestType>(channels);
        for (auto& img : images)
        {
            img = make_test_image<TestType>(channels);
        }

        s.insert_tensor_from_batched_images(images);
        auto ret =
            onnx::image_batch_from_tensor<TestType>(s[0], size, type, post_process);
        auto ret2 = onnx::image_batch_from_tensor<TestType>(s[0], size, type);

        REQUIRE(image_array_equals(ret, ret2));
        REQUIRE(image_array_equals(ret, images));
    }

    SECTION("RGBA images")
    {
        constexpr int channels{4};
        const auto type = get_test_image_type<TestType>(channels);
        for (auto& img : images)
        {
            img = make_test_image<TestType>(channels);
        }

        s.insert_tensor_from_batched_images(images);
        auto ret =
            onnx::image_batch_from_tensor<TestType>(s[0], size, type, post_process);
        auto ret2 = onnx::image_batch_from_tensor<TestType>(s[0], size, type);

        REQUIRE(image_array_equals(ret, ret2));
        REQUIRE(image_array_equals(ret, images));
    }
}

TEMPLATE_TEST_CASE("[tensor_conversion] - image_from_tensor",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t)
{
    onnx::MemoryBackedTensorSet<TestType> s;
    const auto post_process = [](cv::Mat img) {
        return img;
    };

    SECTION("Invalid batch")
    {
        std::vector<cv::Mat> images(2);
        for (auto& img : images)
        {
            img = make_test_image<TestType>(1);
        }

        s.insert_tensor_from_batched_images(images);
        REQUIRE_THROWS(onnx::image_from_tensor<TestType>(s.tensors().front(),
                                                         images.front().size(),
                                                         images.front().type(),
                                                         post_process));
        REQUIRE_THROWS(onnx::image_from_tensor<TestType>(s.tensors().front(),
                                                         images.front().size(),
                                                         images.front().type()));
    }

    SECTION("Grayscale image")
    {
        constexpr int channels{1};
        const auto type   = get_test_image_type<TestType>(channels);
        const cv::Mat img = make_test_image<TestType>(channels);
        const auto size   = img.size();

        s.insert_tensor_from_image(img);

        auto ret  = onnx::image_from_tensor<TestType>(s[0], size, type, post_process);
        auto ret2 = onnx::image_from_tensor<TestType>(s[0], size, type);

        REQUIRE(image_equals(ret, ret2));
        REQUIRE(image_equals(ret, img));
    }

    SECTION("RGB image")
    {
        constexpr int channels{3};
        const auto type   = get_test_image_type<TestType>(channels);
        const cv::Mat img = make_test_image<TestType>(channels);
        const auto size   = img.size();

        s.insert_tensor_from_image(img);

        auto ret  = onnx::image_from_tensor<TestType>(s[0], size, type, post_process);
        auto ret2 = onnx::image_from_tensor<TestType>(s[0], size, type);

        REQUIRE(image_equals(ret, ret2));
        REQUIRE(image_equals(ret, img));
    }

    SECTION("RGBA image")
    {
        constexpr int channels{4};
        const auto type   = get_test_image_type<TestType>(channels);
        const cv::Mat img = make_test_image<TestType>(channels);
        const auto size   = img.size();

        s.insert_tensor_from_image(img);

        auto ret  = onnx::image_from_tensor<TestType>(s[0], size, type, post_process);
        auto ret2 = onnx::image_from_tensor<TestType>(s[0], size, type);

        REQUIRE(image_equals(ret, ret2));
        REQUIRE(image_equals(ret, img));
    }
}

TEMPLATE_TEST_CASE("[tensor_conversion] - array_batch_from_tensor",
                   "[rad::onnx]",
                   float,
                   std::uint8_t)
{
    onnx::MemoryBackedTensorSet<TestType> s;
    static constexpr std::size_t size{10};
    static constexpr std::size_t batch_size{4};

    const std::vector<std::vector<TestType>> batched_data(
        batch_size,
        std::vector<TestType>(size, TestType{1}));

    s.insert_tensor_from_batched_arrays(batched_data);
    REQUIRE_FALSE(s.empty());

    auto ret = onnx::array_batch_from_tensor<TestType>(s[0], size);
    REQUIRE(ret == batched_data);
}

TEMPLATE_TEST_CASE("[tensor_conversion] - array_from_tensor",
                   "[rad::onnx]",
                   float,
                   std::uint8_t)
{
    static constexpr std::size_t size{10};
    const std::vector<TestType> src_data(10, TestType{1});

    onnx::MemoryBackedTensorSet<TestType> s;
    s.insert_tensor_from_array(src_data);
    REQUIRE_FALSE(s.empty());

    auto ret = onnx::array_from_tensor<TestType>(s[0], size);
    REQUIRE(array_equals(src_data, ret));
}
