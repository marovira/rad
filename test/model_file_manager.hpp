#pragma once

#include "test_data_root.hpp"

#include <rad/onnx/session.hpp>

#include <filesystem>
#include <fmt/printf.h>

class ModelFileManager
{
public:
    ModelFileManager()
    {
        m_root = std::filesystem::absolute(test::data_root);
    }

    std::string get_root() const
    {
        return m_root.string();
    }

    enum class Type
    {
        static_axes,
        dynamic_axes
    };

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
        }

        if constexpr(rad::onnx::use_string<ORTCHAR_T>())
        {
            return (m_root / name).string();
        }
        else
        {
            return (m_root / name).wstring();
        }
    }

private:
    std::filesystem::path m_root;
};
