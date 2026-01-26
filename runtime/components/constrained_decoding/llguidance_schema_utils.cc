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

#include "runtime/components/constrained_decoding/llguidance_schema_utils.h"

#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json

namespace litert::lm {

namespace {

// This works because special tokens cannot be in terminal.
static constexpr absl::string_view kTextOnlyLarkGrammar = R"(
start: TEXT
TEXT: /(.|\n)*/
)";

}  // namespace

absl::StatusOr<std::string> FormatToolsAsLarkGrammar(
    const nlohmann::ordered_json& tools, const LlgConstraintsOptions& options) {
  if (options.funcall_format == FuncallFormat::kEac) {
    return absl::UnimplementedError("EAC format not implemented for Lark.");
  }

  std::vector<std::string> tool_names;
  for (const auto& tool : tools) {
    if (tool.contains("name") && tool["name"].is_string()) {
      tool_names.push_back(tool["name"].get<std::string>());
    }
  }
  std::string tool_union =
      absl::StrFormat(R"(TOOL_UNION: /%s/)", absl::StrJoin(tool_names, "|"));

  // Ensures it's a valid JSON with string escapes, but doesn't constrain
  // against schema.
  std::string json_grammar =
      absl::StrFormat(R"(
fc2_esc_open: %s
fc2_esc_close: %s

key: IDENTIFIER
IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
json_value: custom_string | NUMBER | BOOLEAN | NULL | object | array

custom_string: fc2_esc_open /(.|\n)*/ fc2_esc_close
array: "[" [json_value ("," json_value)*] "]"
object: "{" [pair ("," pair)*] "}"
pair: key ":" json_value

// Primitives (Standard JSON)
NUMBER: /-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/
BOOLEAN: "true" | "false"
NULL: "null"
%%ignore /[ \t\r\n]+/)",
                      options.fc2_open_quote, options.fc2_close_quote);
  // Function calling syntax.
  std::string function_block = absl::StrFormat(
      R"((fc2_start "call:" TOOL_UNION object fc2_end)+ fc2_resp
fc2_start: %s
fc2_end: %s
fc2_resp: %s
)",
      options.fc2_code_fence_start, options.fc2_code_fence_end,
      options.fc2_function_response_start);

  std::string start_rule;
  switch (options.constraint_mode) {
    case LlgConstraintMode::kTextOnly: {
      return std::string(kTextOnlyLarkGrammar);
    }
    case LlgConstraintMode::kFunctionCallOnly: {
      if (tool_names.empty()) {
        return absl::InvalidArgumentError(
            "No tools provided for FunctionCallOnly mode.");
      }
      start_rule = absl::StrCat("start: ", function_block, "\n");
      break;
    }
    case LlgConstraintMode::kTextAndOr: {
      if (tool_names.empty()) {
        return std::string(kTextOnlyLarkGrammar);
      }
      std::string tool_names_regex = absl::StrJoin(tool_names, "|");
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

  return absl::StrCat(tool_union, "\n", json_grammar, "\n", start_rule);
}

}  // namespace litert::lm
