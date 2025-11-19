#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <rad/blending_functions.hpp>
#include <zeus/float.hpp>

template<typename T>
consteval T epsilon()
{
    return static_cast<T>(0.00001f);
}

namespace
{
    template<typename T>
    constexpr bool test_equal(T a, T b)
    {
        return zeus::almost_equal<T>(a, b, epsilon<T>());
    }
} // namespace

TEMPLATE_TEST_CASE("[blending_functions] - soft_blend", "[rad]", float, double)
{
    static constexpr TestType zero{0};
    static constexpr TestType one{1};
    static constexpr TestType half{0.5};

    SECTION("Base function")
    {
        static constexpr TestType r{2};
        REQUIRE(test_equal(rad::soft_blend<TestType>(zero, r), one));
        REQUIRE(test_equal(rad::soft_blend<TestType>(one, r), half));
        REQUIRE(test_equal(rad::soft_blend<TestType>(r, r), zero));
    }

    SECTION("Range 1")
    {
        REQUIRE(test_equal(rad::soft_blend<TestType>(zero), one));
        REQUIRE(test_equal(rad::soft_blend<TestType>(half), half));
        REQUIRE(test_equal(rad::soft_blend<TestType>(one), zero));
    }
}

TEMPLATE_TEST_CASE("[blending_functions] - wyvill_blend", "[rad]", float, double)
{
    static constexpr TestType zero{0};
    static constexpr TestType one{1};

    SECTION("Base function")
    {
        static constexpr TestType r{2};
        REQUIRE(test_equal(rad::wyvill_blend<TestType>(zero, r), one));
        REQUIRE(test_equal(rad::wyvill_blend<TestType>(r, r), zero));
    }

    SECTION("Range 1")
    {
        REQUIRE(test_equal(rad::wyvill_blend<TestType>(zero), one));
        REQUIRE(test_equal(rad::wyvill_blend<TestType>(one), zero));
    }
}

TEMPLATE_TEST_CASE("[blending_functions] sharp_mask_blend", "[rad]", float, double)
{
    static constexpr TestType zero{0};
    static constexpr TestType one{1};

    REQUIRE(test_equal(rad::sharp_mask_blend<TestType>(zero), zero));
    REQUIRE(test_equal(rad::sharp_mask_blend<TestType>(one), one));
}
