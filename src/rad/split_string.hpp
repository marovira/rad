#pragma once

#include <string>
#include <vector>

namespace rad
{
    inline std::vector<std::string> split_string(std::string const& str, char delim)
    {
        std::vector<std::string> items;
        std::string cur;
        for (std::size_t i{0}; i < str.size(); ++i)
        {
            if (str[i] == delim)
            {
                items.push_back(cur);
                cur.clear();
                continue;
            }

            cur.push_back(str[i]);
        }
        items.push_back(cur);

        return items;
    }
} // namespace rad
