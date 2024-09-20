#pragma once

#include <fmt/printf.h>
#include <zeus/assert.hpp>
#include <zeus/platform.hpp>

#if defined(ZEUS_PLATFORM_WINDOWS)
#    pragma warning(push)
#    pragma warning(disable : 5'054)
#endif

#include <opencv2/opencv.hpp>

#if defined(ZEUS_PLATFORM_WINDOWS)
#    pragma warning(pop)
#endif

namespace rad::opencv
{
    template<typename T>
    bool perform_safe_op(T fun)
    {
        try
        {
            fun();
        }
        catch (cv::Exception const& e)
        {
            fmt::print("{}\n", e.what());
            ASSERT(0);
            return false;
        }
        catch (std::exception const& e)
        {
            fmt::print("{}\n", e.what());
            ASSERT(0);
            return false;
        }

        return true;
    }
} // namespace rad::opencv
