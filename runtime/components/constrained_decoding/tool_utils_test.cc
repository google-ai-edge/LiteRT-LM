// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/components/constrained_decoding/tool_utils.h"

#include <string>

#include <gtest/gtest.h>
#include "nlohmann/json.hpp"  // from @nlohmann_json

namespace litert::lm {
namespace {

TEST(ToolUtilsTest, GetToolsArrayDirectArray) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "test_tool"
  })json");
  nlohmann::ordered_json direct_array = nlohmann::ordered_json::array({tool});
  const auto& result = GetToolsArray(direct_array);
  EXPECT_TRUE(result.is_array());
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0]["name"], "test_tool");
}

TEST(ToolUtilsTest, GetToolsArrayWrappedObjects) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "test_tool"
  })json");
  nlohmann::ordered_json direct_array = nlohmann::ordered_json::array({tool});

  for (const std::string& key :
       {"oneOf", "one_of", "anyOf", "any_of", "tools"}) {
    nlohmann::ordered_json wrapped_obj =
        nlohmann::ordered_json::object({{key, direct_array}});
    const auto& unwrapped = GetToolsArray(wrapped_obj);
    EXPECT_TRUE(unwrapped.is_array());
    EXPECT_EQ(unwrapped.size(), 1);
    EXPECT_EQ(unwrapped[0]["name"], "test_tool");
  }
}

TEST(ToolUtilsTest, GetToolsArrayUnrecognizedObjectReturnsSelf) {
  nlohmann::ordered_json obj = nlohmann::ordered_json::parse(R"json({
    "unknown": 123
  })json");
  const auto& result = GetToolsArray(obj);
  EXPECT_TRUE(result.is_object());
  EXPECT_EQ(result["unknown"], 123);
}

}  // namespace
}  // namespace litert::lm
