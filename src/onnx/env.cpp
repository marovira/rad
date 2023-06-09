#include "rad/onnx/env.hpp"

#include <fmt/printf.h>
#include <magic_enum.hpp>
#include <zeus/string.hpp>

namespace rad::onnx
{
    void init_ort_api()
    {
        if (!Ort::Global<void>::api_)
        {
            Ort::InitApi();
        }
    }

    std::string log_level_from_enum(OrtLoggingLevel level)
    {
        auto as_string = std::string{magic_enum::enum_name(level)};
        auto elems     = zeus::split(as_string, '_');
        auto name      = elems.back();
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
            return static_cast<unsigned char>(std::tolower(c));
        });
        return name;
    }

    void logger(void*,
                OrtLoggingLevel severity,
                const char* category,
                const char* log_id,
                const char* code_location,
                const char* message)
    {
        std::string level = log_level_from_enum(severity);
        fmt::print("{}: {}: ({}) at {}: {}\n",
                   level,
                   category,
                   log_id,
                   code_location,
                   message);
    }

    Ort::Env create_environment(std::string const& name)
    {
        return {OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                name.c_str(),
                logger,
                nullptr};
    }
} // namespace rad::onnx
