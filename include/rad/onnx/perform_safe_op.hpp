#pragma once

#include "onnxruntime.hpp"

#include <fmt/base.h>
#include <opencv2/core.hpp>

#include <exception>

namespace rad::onnx
{
    template<typename T>
    [[nodiscard]]
    bool perform_safe_op(T fun)
    {
        try
        {
            fun();
        }
        catch (cv::Exception const& e)
        {
            fmt::print("{}\n", e.what());
            return false;
        }
        catch (Ort::Exception const& e)
        {
            fmt::print("{}\n", e.what());
            return false;
        }

        catch (std::exception const& e)
        {
            fmt::print("{}\n", e.what());
            return false;
        }

        return true;
    }
} // namespace rad::onnx
