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

#include "runtime/components/constrained_decoding/llg_fc_tool_calls.h"

#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/llg_tool_call_utils.h"
#include "runtime/components/constrained_decoding/llguidance_schema_utils.h"

namespace litert::lm {
namespace {

std::string GenerateFcValueRule(const nlohmann::ordered_json& prop_schema,
                                bool is_req) {
  if (prop_schema.contains("enum") && prop_schema["enum"].is_array()) {
    std::vector<std::string> enum_vals;
    for (const auto& val : prop_schema["enum"]) {
      if (val.is_string()) {
        enum_vals.push_back(absl::StrFormat(R"(fc_esc_open "%s" fc_esc_close)",
                                            val.get<std::string>()));
      } else if (val.is_number() || val.is_boolean()) {
        enum_vals.push_back(absl::StrFormat(R"("%s")", val.dump()));
      }
    }
    if (!enum_vals.empty()) {
      return absl::StrFormat("(%s)", absl::StrJoin(enum_vals, " | "));
    }
  }

  if (prop_schema.contains("type")) {
    const auto& type_node = prop_schema["type"];
    if (type_node.is_string()) {
      return GetRuleForType(type_node.get<std::string>(), "json_value");
    } else if (type_node.is_array()) {
      std::vector<std::string> types;
      for (const auto& t : type_node) {
        if (t.is_string()) {
          types.push_back(GetRuleForType(t.get<std::string>(), "json_value"));
        }
      }
      if (!types.empty()) {
        return absl::StrFormat("(%s)", absl::StrJoin(types, " | "));
      }
    }
  }

  return "json_value";
}

std::string GetFcGrammar(const LlgConstraintsOptions& options) {
  return absl::StrFormat(R"(
fc_esc_open: %s
fc_esc_close: %s

json_value: string | NUMBER | BOOLEAN | NULL | object | array

string: fc_esc_open /(.|\n)*/ fc_esc_close
array: "[" [json_value ("," json_value)*] "]"
object: "{" [pair ("," pair)*] "}"
pair: IDENTIFIER ":" json_value
IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/

// Primitives (Standard JSON)
NUMBER: /-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/
BOOLEAN: "true" | "false"
NULL: "null"
%%ignore /[ \t\r\n]+/)",
                         options.open_quote, options.close_quote);
}

std::string GetFcFunctionBlock(const std::vector<std::string>& tool_call_cases,
                               const LlgConstraintsOptions& options) {
  return absl::StrFormat(
      R"((fc_start (%s) fc_end)+ fc_resp
fc_start: %s
fc_end: %s
fc_resp: %s
)",
      absl::StrJoin(tool_call_cases, " | "), options.code_fence_start,
      options.code_fence_end, options.function_response_start);
}

}  // namespace

absl::StatusOr<std::string> CreateLarkGrammarForFcToolCalls(
    const nlohmann::ordered_json& tools, const LlgConstraintsOptions& options) {
  std::vector<std::string> tool_names;
  std::vector<std::string> tool_blocks;

  for (const auto& tool : tools) {
    if (!tool.contains("name") || !tool["name"].is_string()) {
      continue;
    }
    std::string tool_name = tool["name"].get<std::string>();
    tool_names.push_back(tool_name);
    const ToolFormatConfig fc_config = {
        .pair_separator = ":",
        .rule_suffix = "_object",
        .start_wrap = "{",
        .end_wrap = "}",
        .generate_value_rule = GenerateFcValueRule,
    };
    AppendToolRules(tool, tool_name, fc_config, tool_blocks);
  }

  std::string tool_union =
      absl::StrFormat(R"(TOOL_UNION: /%s/)", absl::StrJoin(tool_names, "|"));

  std::vector<std::string> tool_call_cases;
  tool_call_cases.reserve(tool_names.size());
  for (const auto& tool_name : tool_names) {
    tool_call_cases.push_back(
        absl::StrFormat(R"("call:" "%s" %s_object)", tool_name, tool_name));
  }

  std::string json_grammar = GetFcGrammar(options);
  std::string function_block = GetFcFunctionBlock(tool_call_cases, options);
  std::string text_only_block = GetTextOnlyBlock(options);

  std::string start_rule;
  switch (options.constraint_mode) {
    case LlgConstraintMode::kTextOnly: {
      return text_only_block;
    }
    case LlgConstraintMode::kFunctionCallsOnly: {
      if (tool_names.empty()) {
        return absl::InvalidArgumentError(
            "No tools provided for FunctionCallsOnly mode.");
      }
      start_rule = absl::StrCat("start: ", function_block, "\n");
      break;
    }
    case LlgConstraintMode::kTextAndOrFunctionCalls: {
      if (tool_names.empty()) {
        return text_only_block;
      }
      start_rule = absl::StrFormat(
          R"(
start: TEXT_CONTENT? function_block_opt
TEXT_CONTENT: /(.|\n)+/
function_block_opt: function_block |
function_block: %s
)",
          function_block);
      break;
    }
  }

  return absl::StrCat(tool_union, "\n", absl::StrJoin(tool_blocks, "\n"), "\n",
                      json_grammar, "\n", start_rule);
}

}  // namespace litert::lm
