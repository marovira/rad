#include "onnx_test_helpers.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <rad/onnx/onnxruntime.hpp>
#include <rad/onnx/tensor_conversion.hpp>
#include <rad/onnx/tensor_set.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <ranges>
#include <vector>

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

namespace
{
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

    template<typename T, typename U, typename P>
    bool array_equals(T const& lhs, U const& rhs, P const& pred)
    {
        if (lhs.size() != rhs.size())
        {
            return false;
        }

        for (std::size_t i{0}; i < lhs.size(); ++i)
        {
            if (!pred(lhs[i], rhs[i]))
            {
                return false;
            }
        }

        return true;
    }

} // namespace

TEMPLATE_TEST_CASE("[tensor_conversion] - TensorBlob::operator()",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t,
                   std::int64_t)
{
    static constexpr std::int64_t dim_b{7};
    static constexpr std::int64_t dim_n{5};
    static constexpr std::int64_t dim_m{3};
    const std::equal_to<const TestType> comparator;

    const Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    SECTION("One dimension")
    {
        std::vector<TestType> one_dim(dim_n);
        std::vector<TestType> one_dim_unsqueezed(dim_n);
        std::vector<std::int64_t> shape{dim_n};
        std::vector<std::int64_t> shape_unsqueezed{1, dim_n};
        for (std::int64_t n{0}; n < dim_n; n++)
        {
            const auto val        = static_cast<float>(n);
            one_dim[n]            = static_cast<TestType>(val);
            one_dim_unsqueezed[n] = static_cast<TestType>(val);
        }

        Ort::Value tensor{nullptr}, tensor_unsqueezed{nullptr};

        tensor = Ort::Value::CreateTensor<TestType>(mem_info,
                                                    one_dim.data(),
                                                    one_dim.size(),
                                                    shape.data(),
                                                    shape.size());

        tensor_unsqueezed = Ort::Value::CreateTensor<TestType>(mem_info,
                                                               one_dim_unsqueezed.data(),
                                                               one_dim_unsqueezed.size(),
                                                               shape_unsqueezed.data(),
                                                               shape_unsqueezed.size());

        auto blob            = onnx::blob_from_tensor<TestType>(tensor);
        auto blob_unsqueezed = onnx::blob_from_tensor<TestType>(tensor_unsqueezed);

        REQUIRE(array_equals(one_dim, blob_unsqueezed({0}), comparator));
        REQUIRE(array_equals(one_dim_unsqueezed, blob_unsqueezed({}), comparator));
        for (std::int64_t n{0}; n < dim_n; n++)
        {
            const TestType val            = blob({n}).front();
            const TestType val_unsqueezed = blob_unsqueezed({0, n}).front();
            REQUIRE(comparator(val, one_dim[n]));
            REQUIRE(comparator(val_unsqueezed, one_dim_unsqueezed[n]));
        }

        REQUIRE_THROWS(blob({dim_n}));
        REQUIRE_THROWS(blob({0, 0}));

        REQUIRE_THROWS(blob_unsqueezed({0, dim_n}));
        REQUIRE_THROWS(blob_unsqueezed({1, 0}));
        REQUIRE_THROWS(blob_unsqueezed({0, 0, 0}));
    }

    SECTION("Two dimensions")
    {
        std::vector<TestType> two_dims(dim_b * dim_n);
        std::vector<TestType> two_dims_unsqueezed(dim_b * dim_n);
        std::vector<std::int64_t> shape{dim_b, dim_n};
        std::vector<std::int64_t> shape_unsqueezed{1, dim_b, dim_n};
        for (std::int64_t b{0}; b < dim_b; b++)
        {
            for (std::int64_t n{0}; n < dim_n; n++)
            {
                const auto val             = static_cast<float>(100 * b + n);
                const std::int64_t index   = b * dim_n + n;
                two_dims[index]            = static_cast<TestType>(val);
                two_dims_unsqueezed[index] = static_cast<TestType>(val);
            }
        }

        Ort::Value tensor{nullptr}, tensor_unsqueezed{nullptr};

        tensor = Ort::Value::CreateTensor<TestType>(mem_info,
                                                    two_dims.data(),
                                                    two_dims.size(),
                                                    shape.data(),
                                                    shape.size());

        tensor_unsqueezed = Ort::Value::CreateTensor<TestType>(mem_info,
                                                               two_dims_unsqueezed.data(),
                                                               two_dims_unsqueezed.size(),
                                                               shape_unsqueezed.data(),
                                                               shape_unsqueezed.size());

        auto blob            = onnx::blob_from_tensor<TestType>(tensor);
        auto blob_unsqueezed = onnx::blob_from_tensor<TestType>(tensor_unsqueezed);
        REQUIRE(array_equals(two_dims, blob_unsqueezed({0}), comparator));
        REQUIRE(array_equals(two_dims_unsqueezed, blob_unsqueezed({}), comparator));
        for (std::int64_t b{0}; b < dim_b; b++)
        {
            auto batch = std::views::counted(two_dims.begin() + (b * dim_n), dim_n);
            auto batch_unsqueezed =
                std::views::counted(two_dims_unsqueezed.begin() + (b * dim_n), dim_n);
            REQUIRE(array_equals(batch, blob({b}), comparator));
            REQUIRE(array_equals(batch_unsqueezed, blob_unsqueezed({0, b}), comparator));
            for (std::int64_t n{0}; n < dim_n; n++)
            {
                const TestType val            = blob({b, n}).front();
                const TestType val_unsqueezed = blob_unsqueezed({0, b, n}).front();
                const std::int64_t index      = b * dim_n + n;
                REQUIRE(comparator(val, two_dims[index]));
                REQUIRE(comparator(val_unsqueezed, two_dims_unsqueezed[index]));
            }
        }

        REQUIRE_THROWS(blob({dim_b, 0}));
        REQUIRE_THROWS(blob({0, dim_n}));
        REQUIRE_THROWS(blob({0, 0, 0}));

        REQUIRE_THROWS(blob_unsqueezed({0, dim_b, 0}));
        REQUIRE_THROWS(blob_unsqueezed({0, 0, dim_n}));
        REQUIRE_THROWS(blob_unsqueezed({1, 0, 0}));
        REQUIRE_THROWS(blob_unsqueezed({0, 0, 0, 0}));
    }

    SECTION("Three dimensions")
    {
        std::vector<TestType> three_dims(dim_b * dim_m * dim_n);
        std::vector<TestType> three_dims_unsqueezed(dim_b * dim_m * dim_n);
        std::vector<std::int64_t> shape{dim_b, dim_m, dim_n};
        std::vector<std::int64_t> shape_unsqueezed{1, dim_b, dim_m, dim_n};
        for (std::int64_t b{0}; b < dim_b; b++)
        {
            for (std::int64_t m{0}; m < dim_m; m++)
            {
                for (std::int64_t n{0}; n < dim_n; n++)
                {
                    const auto val = static_cast<float>(100 * (100 * b + m) + n);
                    const std::int64_t index     = b * dim_m * dim_n + m * dim_n + n;
                    three_dims[index]            = static_cast<TestType>(val);
                    three_dims_unsqueezed[index] = static_cast<TestType>(val);
                }
            }
        }

        Ort::Value tensor{nullptr}, tensor_unsqueezed{nullptr};

        tensor = Ort::Value::CreateTensor<TestType>(mem_info,
                                                    three_dims.data(),
                                                    three_dims.size(),
                                                    shape.data(),
                                                    shape.size());

        tensor_unsqueezed =
            Ort::Value::CreateTensor<TestType>(mem_info,
                                               three_dims_unsqueezed.data(),
                                               three_dims_unsqueezed.size(),
                                               shape_unsqueezed.data(),
                                               shape_unsqueezed.size());

        auto blob            = onnx::blob_from_tensor<TestType>(tensor);
        auto blob_unsqueezed = onnx::blob_from_tensor<TestType>(tensor_unsqueezed);
        REQUIRE(array_equals(three_dims, blob_unsqueezed({0}), comparator));
        REQUIRE(array_equals(three_dims_unsqueezed, blob_unsqueezed({}), comparator));
        for (std::int64_t b{0}; b < dim_b; b++)
        {
            auto batch = std::views::counted(three_dims.begin() + (b * dim_m * dim_n),
                                             dim_m * dim_n);
            auto batch_unsqueezed =
                std::views::counted(three_dims_unsqueezed.begin() + (b * dim_m * dim_n),
                                    dim_m * dim_n);
            REQUIRE(array_equals(batch, blob({b}), comparator));
            REQUIRE(array_equals(batch_unsqueezed, blob_unsqueezed({0, b}), comparator));
            for (std::int64_t m{0}; m < dim_m; m++)
            {
                auto row = std::views::counted(three_dims.begin() + (b * dim_m * dim_n)
                                                   + (m * dim_n),
                                               dim_n);
                auto row_unsqueezed = std::views::counted(
                    three_dims_unsqueezed.begin() + (b * dim_m * dim_n) + (m * dim_n),
                    dim_n);
                REQUIRE(array_equals(row, blob({b, m}), comparator));
                REQUIRE(
                    array_equals(row_unsqueezed, blob_unsqueezed({0, b, m}), comparator));
                for (std::int64_t n{0}; n < dim_n; n++)
                {
                    const TestType val            = blob({b, m, n}).front();
                    const TestType val_unsqueezed = blob_unsqueezed({0, b, m, n}).front();
                    const std::int64_t index      = b * dim_m * dim_n + m * dim_n + n;
                    REQUIRE(comparator(val, three_dims[index]));
                    REQUIRE(comparator(val_unsqueezed, three_dims_unsqueezed[index]));
                }
            }
        }

        REQUIRE_THROWS(blob({dim_b, 0, 0}));
        REQUIRE_THROWS(blob({0, dim_m, 0}));
        REQUIRE_THROWS(blob({0, 0, dim_n}));
        REQUIRE_THROWS(blob({0, 0, 0, 0}));

        REQUIRE_THROWS(blob_unsqueezed({0, dim_b, 0, 0}));
        REQUIRE_THROWS(blob_unsqueezed({0, 0, dim_m, 0}));
        REQUIRE_THROWS(blob_unsqueezed({0, 0, 0, dim_n}));
        REQUIRE_THROWS(blob_unsqueezed({1, 0, 0, 0}));
        REQUIRE_THROWS(blob_unsqueezed({0, 0, 0, 0, 0}));
    }
}

TEMPLATE_TEST_CASE("[tensor_conversion] - blob_from_tensor",
                   "[rad::onnx]",
                   float,
                   std::uint8_t,
                   Ort::Float16_t,
                   std::uint16_t,
                   std::int64_t)
{
    static constexpr std::int64_t dim_b{7};
    static constexpr std::int64_t dim_n{5};
    static constexpr std::int64_t dim_m{3};
    const std::equal_to<const TestType> comparator;

    const Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    SECTION("One dimension")
    {
        std::vector<TestType> one_dim(dim_n);
        std::vector<TestType> one_dim_unsqueezed(dim_n);
        std::vector<std::int64_t> shape{dim_n};
        std::vector<std::int64_t> shape_unsqueezed{1, dim_n};
        for (std::int64_t n{0}; n < dim_n; n++)
        {
            const auto val        = static_cast<float>(n);
            one_dim[n]            = static_cast<TestType>(val);
            one_dim_unsqueezed[n] = static_cast<TestType>(val);
        }

        Ort::Value tensor{nullptr}, tensor_unsqueezed{nullptr};

        tensor = Ort::Value::CreateTensor<TestType>(mem_info,
                                                    one_dim.data(),
                                                    one_dim.size(),
                                                    shape.data(),
                                                    shape.size());

        tensor_unsqueezed = Ort::Value::CreateTensor<TestType>(mem_info,
                                                               one_dim_unsqueezed.data(),
                                                               one_dim_unsqueezed.size(),
                                                               shape_unsqueezed.data(),
                                                               shape_unsqueezed.size());

        const auto tensor_info  = tensor.GetTensorTypeAndShapeInfo();
        const auto tensor_dims  = tensor_info.GetShape();
        const auto tensor_count = tensor_info.GetElementCount();
        REQUIRE(tensor_count == dim_n);
        REQUIRE(array_equals(shape, tensor_dims));

        const auto tensor_unsqueezed_info = tensor_unsqueezed.GetTensorTypeAndShapeInfo();
        const auto tensor_unsqueezed_dims = tensor_unsqueezed_info.GetShape();
        const auto tensor_unsqueezed_count = tensor_unsqueezed_info.GetElementCount();
        REQUIRE(tensor_unsqueezed_count == dim_n);
        REQUIRE(array_equals(shape_unsqueezed, tensor_unsqueezed_dims));

        auto blob = onnx::blob_from_tensor<TestType>(tensor);
        REQUIRE(array_equals(shape, blob.shape));
        REQUIRE(array_equals(one_dim, blob.data, comparator));

        auto blob_unsqueezed = onnx::blob_from_tensor<TestType>(tensor_unsqueezed);
        REQUIRE(array_equals(shape_unsqueezed, blob_unsqueezed.shape));
        REQUIRE(array_equals(one_dim_unsqueezed, blob_unsqueezed.data, comparator));
    }

    SECTION("Two dimensions")
    {
        std::vector<TestType> two_dims(dim_b * dim_n);
        std::vector<TestType> two_dims_unsqueezed(dim_b * dim_n);
        std::vector<std::int64_t> shape{dim_b, dim_n};
        std::vector<std::int64_t> shape_unsqueezed{1, dim_b, dim_n};

        for (std::int64_t b{0}; b < dim_b; b++)
        {
            for (std::int64_t n{0}; n < dim_n; n++)
            {
                const auto val             = static_cast<float>(100 * b + n);
                const std::int64_t index   = b * dim_n + n;
                two_dims[index]            = static_cast<TestType>(val);
                two_dims_unsqueezed[index] = static_cast<TestType>(val);
            }
        }

        Ort::Value tensor{nullptr}, tensor_unsqueezed{nullptr};

        tensor = Ort::Value::CreateTensor<TestType>(mem_info,
                                                    two_dims.data(),
                                                    two_dims.size(),
                                                    shape.data(),
                                                    shape.size());

        tensor_unsqueezed = Ort::Value::CreateTensor<TestType>(mem_info,
                                                               two_dims_unsqueezed.data(),
                                                               two_dims_unsqueezed.size(),
                                                               shape_unsqueezed.data(),
                                                               shape_unsqueezed.size());

        const auto tensor_info  = tensor.GetTensorTypeAndShapeInfo();
        const auto tensor_dims  = tensor_info.GetShape();
        const auto tensor_count = tensor_info.GetElementCount();
        REQUIRE(tensor_count == dim_b * dim_n);
        REQUIRE(array_equals(shape, tensor_dims));

        const auto tensor_unsqueezed_info = tensor_unsqueezed.GetTensorTypeAndShapeInfo();
        const auto tensor_unsqueezed_dims = tensor_unsqueezed_info.GetShape();
        const auto tensor_unsqueezed_count = tensor_unsqueezed_info.GetElementCount();
        REQUIRE(tensor_unsqueezed_count == dim_b * dim_n);
        REQUIRE(array_equals(shape_unsqueezed, tensor_unsqueezed_dims));

        auto blob = onnx::blob_from_tensor<TestType>(tensor);
        REQUIRE(array_equals(shape, blob.shape));
        REQUIRE(array_equals(two_dims, blob.data, comparator));

        auto blob_unsqueezed = onnx::blob_from_tensor<TestType>(tensor_unsqueezed);
        REQUIRE(array_equals(shape_unsqueezed, blob_unsqueezed.shape));
        REQUIRE(array_equals(two_dims_unsqueezed, blob_unsqueezed.data, comparator));
    }

    SECTION("Three dimensions")
    {
        std::vector<TestType> three_dims(dim_b * dim_m * dim_n);
        std::vector<TestType> three_dims_unsqueezed(dim_b * dim_m * dim_n);
        std::vector<std::int64_t> shape{dim_b, dim_m, dim_n};
        std::vector<std::int64_t> shape_unsqueezed{1, dim_b, dim_m, dim_n};
        for (std::int64_t b{0}; b < dim_b; b++)
        {
            for (std::int64_t m{0}; m < dim_m; m++)
            {
                for (std::int64_t n{0}; n < dim_n; n++)
                {
                    const auto val = static_cast<float>(100 * (100 * b + m) + n);
                    const std::int64_t index     = b * dim_m * dim_n + m * dim_n + n;
                    three_dims[index]            = static_cast<TestType>(val);
                    three_dims_unsqueezed[index] = static_cast<TestType>(val);
                }
            }
        }

        Ort::Value tensor{nullptr}, tensor_unsqueezed{nullptr};

        tensor = Ort::Value::CreateTensor<TestType>(mem_info,
                                                    three_dims.data(),
                                                    three_dims.size(),
                                                    shape.data(),
                                                    shape.size());

        tensor_unsqueezed =
            Ort::Value::CreateTensor<TestType>(mem_info,
                                               three_dims_unsqueezed.data(),
                                               three_dims_unsqueezed.size(),
                                               shape_unsqueezed.data(),
                                               shape_unsqueezed.size());

        const auto tensor_info  = tensor.GetTensorTypeAndShapeInfo();
        const auto tensor_dims  = tensor_info.GetShape();
        const auto tensor_count = tensor_info.GetElementCount();
        REQUIRE(tensor_count == dim_b * dim_m * dim_n);
        REQUIRE(array_equals(shape, tensor_dims));

        const auto tensor_unsqueezed_info = tensor_unsqueezed.GetTensorTypeAndShapeInfo();
        const auto tensor_unsqueezed_dims = tensor_unsqueezed_info.GetShape();
        const auto tensor_unsqueezed_count = tensor_unsqueezed_info.GetElementCount();
        REQUIRE(tensor_unsqueezed_count == dim_b * dim_m * dim_n);
        REQUIRE(array_equals(shape_unsqueezed, tensor_unsqueezed_dims));

        auto blob = onnx::blob_from_tensor<TestType>(tensor);
        REQUIRE(array_equals(shape, blob.shape));
        REQUIRE(array_equals(three_dims, blob.data, comparator));

        auto blob_unsqueezed = onnx::blob_from_tensor<TestType>(tensor_unsqueezed);
        REQUIRE(array_equals(shape_unsqueezed, blob_unsqueezed.shape));
        REQUIRE(array_equals(three_dims_unsqueezed, blob_unsqueezed.data, comparator));
    }
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
    onnx::TensorSet s;
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

        s.insert_tensor_from_batched_images<TestType>(images);
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

        s.insert_tensor_from_batched_images<TestType>(images);
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

        s.insert_tensor_from_batched_images<TestType>(images);
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
    onnx::TensorSet s;
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

        s.insert_tensor_from_batched_images<TestType>(images);
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

        s.insert_tensor_from_image<TestType>(img);

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

        s.insert_tensor_from_image<TestType>(img);

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

        s.insert_tensor_from_image<TestType>(img);

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
    onnx::TensorSet s;
    static constexpr std::size_t size{10};
    static constexpr std::size_t batch_size{4};

    const std::vector<std::vector<TestType>> batched_data(
        batch_size,
        std::vector<TestType>(size, TestType{1}));

    s.insert_tensor_from_batched_arrays<TestType>(batched_data);
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

    onnx::TensorSet s;
    s.insert_tensor_from_array<TestType>(src_data);
    REQUIRE_FALSE(s.empty());

    auto ret = onnx::array_from_tensor<TestType>(s[0], size);
    REQUIRE(array_equals(src_data, ret));
}
