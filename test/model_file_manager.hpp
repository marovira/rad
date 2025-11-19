#pragma once

#include "test_data_root.hpp"

#include <rad/onnx/session.hpp>

#include <filesystem>
#include <string>

class ModelFileManager
{
public:
    ModelFileManager()
    {
        m_root = std::filesystem::absolute(test::data_root);
    }

    [[nodiscard]]
    std::string get_root() const
    {
        return m_root.string();
    }

    enum class Type
    {
        static_axes,
        dynamic_axes,
        half_float,
    };

    [[nodiscard]]
    auto get_model(Type type) const
    {
        std::string name;
        switch (type)
        {
        case Type::static_axes:
            name = "test_static.onnx";
            break;
        case Type::dynamic_axes:
            name = "test_dynamic.onnx";
            break;

        case Type::half_float:
            name = "test_fp16.onnx";
            break;
        }

        auto ret = m_root / name;

        if constexpr (rad::onnx::use_string_for_paths())
        {
            return ret.string();
        }
        else
        {
            return ret.wstring();
        }
    }

private:
    std::filesystem::path m_root;
};
