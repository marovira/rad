#include "rad/onnx.hpp"

#include <magic_enum.hpp>
#include <zeus/string.hpp>

namespace rad::onnx
{
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

    std::vector<std::vector<std::int64_t>> get_input_shapes(Ort::Session& session)
    {
        std::vector<std::vector<std::int64_t>> shapes;
        for (std::size_t i{0}; i < session.GetInputCount(); ++i)
        {
            shapes.push_back(
                session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        return shapes;
    }

    std::vector<std::vector<std::int64_t>> get_output_shapes(Ort::Session& session)
    {
        std::vector<std::vector<std::int64_t>> shapes;
        for (std::size_t i{0}; i < session.GetOutputCount(); ++i)
        {
            shapes.push_back(
                session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        return shapes;
    }
} // namespace rad::onnx
