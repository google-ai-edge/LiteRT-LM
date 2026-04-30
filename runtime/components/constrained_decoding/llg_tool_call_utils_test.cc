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

#include "runtime/components/constrained_decoding/llg_tool_call_utils.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/llguidance_schema_utils.h"

namespace litert::lm {
namespace {

using ::testing::ElementsAre;

TEST(LlgToolCallUtilsTest, ExtractToolProperties_NoParameters) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "get_time"
  })json");

  std::vector<std::string> tool_blocks;
  std::vector<std::string> required_props;
  std::vector<std::string> optional_props;

  ToolFormatConfig config = {
      .pair_separator = ":",
      .rule_suffix = "_args",
      .start_wrap = "",
      .end_wrap = "",
      .generate_value_rule = [](const nlohmann::ordered_json&, bool) {
        return "value";
      }};

  ExtractToolProperties(tool, "get_time", config, tool_blocks, required_props,
                        optional_props);

  EXPECT_TRUE(tool_blocks.empty());
  EXPECT_TRUE(required_props.empty());
  EXPECT_TRUE(optional_props.empty());
}

TEST(LlgToolCallUtilsTest, ExtractToolProperties_WithProperties) {
  nlohmann::ordered_json tool = nlohmann::ordered_json::parse(R"json({
    "name": "get_weather",
    "parameters": {
      "type": "object",
      "properties": {
        "location": { "type": "string" },
        "unit": { "type": "string" }
      },
      "required": ["location"]
    }
  })json");

  std::vector<std::string> tool_blocks;
  std::vector<std::string> required_props;
  std::vector<std::string> optional_props;

  ToolFormatConfig config = {
      .pair_separator = ":",
      .rule_suffix = "_args",
      .start_wrap = "{",
      .end_wrap = "}",
      .generate_value_rule = [](const nlohmann::ordered_json&, bool is_req) {
        return is_req ? "req_val" : "opt_val";
      }};

  ExtractToolProperties(tool, "get_weather", config, tool_blocks,
                        required_props, optional_props);

  // tool_blocks should contain rules for properties
  EXPECT_FALSE(tool_blocks.empty());

  // required_props should contain the location name
  EXPECT_THAT(required_props, ElementsAre("location"));

  // optional_props should contain the unit name
  EXPECT_THAT(optional_props, ElementsAre("unit"));
}

TEST(LlgToolCallUtilsTest, AppendRequiredProperties) {
  std::vector<std::string> required_props = {"a", "b"};
  std::vector<std::string> sequence;

  AppendRequiredProperties(required_props, "test", sequence);

  EXPECT_THAT(sequence, ElementsAre("test_req_a", "\",\"", "test_req_b"));
}

TEST(LlgToolCallUtilsTest, AppendOptionalProperties) {
  std::vector<std::string> optional_props = {"c", "d"};
  std::vector<std::string> tool_blocks;
  std::vector<std::string> sequence;

  AppendOptionalProperties(optional_props, "test", tool_blocks, sequence);

  EXPECT_THAT(sequence, ElementsAre("test_optional"));
  EXPECT_FALSE(tool_blocks.empty());
}

TEST(LlgToolCallUtilsTest, GetTextOnlyBlock) {
  LlgConstraintsOptions options;
  options.code_fence_start = "```";

  std::string block = GetTextOnlyBlock(options);
  EXPECT_THAT(block, testing::HasSubstr("SAFE_TEXT"));
}

}  // namespace
}  // namespace litert::lm
