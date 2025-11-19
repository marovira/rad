#pragma once

#include <concepts>

namespace rad
{
    template<std::floating_point T>
    constexpr T soft_blend(T d, T r)
    {
        const auto d_6 = d * d * d * d * d * d;
        const auto r_6 = r * r * r * r * r * r;
        const auto d_4 = d * d * d * d;
        const auto r_4 = r * r * r * r;
        const auto d_2 = d * d;
        const auto r_2 = r * r;

        const auto a = (T{4} / T{9}) * (d_6 / r_6);
        const auto b = (T{17} / T{9}) * (d_4 / r_4);
        const auto c = (T{22} / T{9}) * (d_2 / r_2);

        return T{1} - a + b - c;
    }

    template<std::floating_point T>
    constexpr T soft_blend(T d)
    {
        return soft_blend(d, T{1});
    }

    template<std::floating_point T>
    constexpr T wyvill_blend(T d, T r)
    {
        const auto g = T{1} - ((d * d) / (r * r));
        return g * g * g;
    }

    template<std::floating_point T>
    constexpr T wyvill_blend(T d)
    {
        return wyvill_blend(d, T{1});
    }

    template<std::floating_point T>
    constexpr T sharp_mask_blend(T d)
    {
        const auto base = T{1} - (d * d * d);
        auto g          = base * base * base;
        g               = g * g;
        g               = T{1} - g;
        g               = g < T{1e-4} ? T{0} : g;
        return g;
    }
} // namespace rad
