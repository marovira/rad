#pragma once

#include "onnxruntime.hpp"

namespace rad::onnx
{
    bool init_ort_api();
    Ort::Env create_environment(std::string const& name);
} // namespace rad::onnx
