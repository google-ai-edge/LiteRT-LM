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

#include "runtime/components/constrained_decoding/llg_python_tool_calls.h"

#include <string>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/llg_tool_call_utils.h"
#include "runtime/components/constrained_decoding/llguidance_schema_utils.h"
#include "runtime/components/constrained_decoding/tool_utils.h"

namespace litert::lm {
namespace {

std::string GeneratePythonValueRule(const nlohmann::ordered_json& prop_schema,
                                    bool is_req) {
  std::string rule;
  if (prop_schema.contains("enum") && prop_schema["enum"].is_array()) {
    std::vector<std::string> enum_vals;
    for (const auto& val : prop_schema["enum"]) {
      if (val.is_string()) {
        enum_vals.push_back(absl::StrFormat(R"("\"%s\"" | "'%s'")",
                                            val.get<std::string>(),
                                            val.get<std::string>()));
      } else if (val.is_boolean()) {
        enum_vals.push_back(val.get<bool>() ? "\"True\"" : "\"False\"");
      } else if (val.is_number()) {
        enum_vals.push_back(absl::StrFormat(R"("%s")", val.dump()));
      }
    }
    if (!enum_vals.empty()) {
      rule = absl::StrFormat("(%s)", absl::StrJoin(enum_vals, " | "));
    }
  }

  if (rule.empty()) {
    for (const char* key : {"one_of", "oneOf", "any_of", "anyOf"}) {
      if (prop_schema.contains(key) && prop_schema[key].is_array()) {
        std::vector<std::string> sub_rules;
        sub_rules.reserve(prop_schema[key].size());
        for (const auto& sub_schema : prop_schema[key]) {
          sub_rules.push_back(
              GeneratePythonValueRule(sub_schema, /*is_req=*/true));
        }
        if (!sub_rules.empty()) {
          rule = absl::StrFormat("(%s)", absl::StrJoin(sub_rules, " | "));
          break;
        }
      }
    }
  }

  if (rule.empty() && prop_schema.contains("type")) {
    const auto& type_node = prop_schema["type"];
    if (type_node.is_string()) {
      rule = GetRuleForType(type_node.get<std::string>(), "python_value");
    } else if (type_node.is_array()) {
      std::vector<std::string> types;
      for (const auto& t : type_node) {
        if (t.is_string()) {
          types.push_back(GetRuleForType(t.get<std::string>(), "python_value"));
        }
      }
      if (!types.empty()) {
        rule = absl::StrFormat("(%s)", absl::StrJoin(types, " | "));
      }
    }
  }

  if (rule.empty()) {
    rule = "python_value";
  }

  if (!is_req) {
    return absl::StrFormat("(%s | \"None\")", rule);
  }

  return rule;
}

std::string GetPythonGrammar(const LlgConstraintsOptions& options) {
  return R"(
python_value: string | NUMBER | BOOLEAN | NULL | object | array

string: "\"" /([^"\\\\]|\\.)*/ "\"" | "'" /([^'\\\\]|\\.)*/ "'"
array: "[" [python_value ("," python_value)*] "]"
object: "{" [pair ("," pair)*] "}"
pair: string ":" python_value

// Primitives (Python-style)
// INTEGER is NUMBER without the fraction and exponent groups. Parameters
// declared as "type": "integer" bind to it so the model cannot emit 1000.0
// or 1e3 where an integer is required. This must stay in sync with the FC
// grammar because GetRuleForType is shared by both.
INTEGER: /-?(?:0|[1-9]\d*)/
NUMBER: /-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/
BOOLEAN: "True" | "False"
NULL: "None"
%ignore /[ \t]+/ )";
}

// `python_start` is the Lark expression that opens a tool call block.
std::string GetPythonFunctionBlock(
    const std::vector<std::string>& tool_call_cases,
    absl::string_view python_start) {
  return absl::StrFormat(
      R"(python_block
python_block: python_start (python_call)* python_end
python_start: %s
python_end: "\n```"
python_call: (%s) "\n"?
)",
      python_start, absl::StrJoin(tool_call_cases, " | "));
}

// Returns the terminals and the `function_block` rule for the modes that mix
// free text with tool call blocks. The code fence is plain text and LLGuidance
// lexes terminals greedily with one byte of lookahead, so a standalone
// free-text terminal would consume the fence and everything after it, and any
// call (even a malformed one) would be accepted as text. Instead, the text
// preceding each block is folded into the terminal that opens it:
// TEXT_AND_FENCE cannot contain the fence before its end, so it ends exactly at
// the first fence. TEXT_CONTENT only covers the optional trailing text after
// the last block.
std::string GetTextAndFunctionBlockRules(
    const std::vector<std::string>& tool_call_cases,
    absl::string_view code_fence_start) {
  const std::string escaped_fence =
      absl::StrReplaceAll(code_fence_start, {{"\n", "\\n"}});
  return absl::StrFormat(
      R"(
FORBIDDEN_CALL: /(.|\n)*%s(.|\n)*/
SAFE_TEXT: /(.|\n)*/ & ~FORBIDDEN_CALL
TEXT_AND_FENCE: SAFE_TEXT "%s"
TEXT_CONTENT: /(.|\n)+/ & ~FORBIDDEN_CALL
function_block: %s
)",
      escaped_fence, escaped_fence,
      GetPythonFunctionBlock(tool_call_cases, "TEXT_AND_FENCE"));
}

}  // namespace

absl::StatusOr<std::string> CreateLarkGrammarForPythonToolCalls(
    const nlohmann::ordered_json& tools, const LlgConstraintsOptions& options) {
  const nlohmann::ordered_json& tools_array = GetToolsArray(tools);
  if (!tools_array.is_array()) {
    return absl::InvalidArgumentError("tools must be an array.");
  }
  std::vector<std::string> tool_names;
  std::vector<std::string> tool_blocks;

  for (const auto& tool : tools_array) {
    if (!tool.contains("name") || !tool["name"].is_string()) {
      continue;
    }
    std::string tool_name = tool["name"].get<std::string>();
    tool_names.push_back(tool_name);
    // See `SanitizeLarkRuleName` for why `-args` uses a hyphen separator.
    const ToolFormatConfig python_config = {
        .pair_separator = "=",
        .rule_suffix = "-args",
        .start_wrap = "",
        .end_wrap = "",
        .generate_value_rule = GeneratePythonValueRule,
    };
    AppendToolRules(tool, tool_name, python_config, tool_blocks);
  }

  std::vector<std::string> tool_call_cases;
  if (options.forced_function_name.has_value()) {
    const std::string& forced_name = *options.forced_function_name;
    if (!absl::c_linear_search(tool_names, forced_name)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Forced function '%s' not found in provided tools.", forced_name));
    }
    tool_call_cases.push_back(
        absl::StrFormat(R"raw("%s" "(" %s-args ")")raw", forced_name,
                        SanitizeLarkRuleName(forced_name)));
  } else {
    tool_call_cases.reserve(tool_names.size());
    for (const auto& tool_name : tool_names) {
      tool_call_cases.push_back(
          absl::StrFormat(R"raw("%s" "(" %s-args ")")raw", tool_name,
                          SanitizeLarkRuleName(tool_name)));
    }
  }

  std::string python_grammar = GetPythonGrammar(options);
  std::string function_block =
      GetPythonFunctionBlock(tool_call_cases, R"("```tool_code\n")");
  std::string text_only_block = GetTextOnlyBlock(options);

  std::string start_rule;
  switch (options.constraint_mode) {
    case LlgConstraintMode::kTextOnly: {
      return text_only_block;
    }
    case LlgConstraintMode::kFunctionCallsOnly: {
      if (tool_call_cases.empty()) {
        return absl::InvalidArgumentError(
            "No tools provided for FunctionCallsOnly mode.");
      }
      if (options.forced_function_name.has_value()) {
        // The forced function must be called. FST doesn't support forcing a
        // function, so there is no FST behavior to match here.
        start_rule = absl::StrCat("start: ", function_block, "\n");
      } else {
        // Like FST's FUNCTION_CALL_ONLY mode, the model may also end its turn
        // without a call, e.g. once it has the tool responses it needs.
        start_rule = absl::StrCat("start: function_block_opt\n",
                                  "function_block_opt: function_block |\n",
                                  "function_block: ", function_block, "\n");
      }
      break;
    }
    case LlgConstraintMode::kTextAndOrFunctionCalls: {
      if (tool_call_cases.empty()) {
        return text_only_block;
      }
      start_rule = absl::StrCat("start: function_block* TEXT_CONTENT?",
                                GetTextAndFunctionBlockRules(
                                    tool_call_cases, options.code_fence_start));
      break;
    }
    case LlgConstraintMode::kFunctionCallsRequired: {
      if (tool_call_cases.empty()) {
        return absl::InvalidArgumentError(
            "No tools provided for FunctionCallsRequired mode.");
      }
      start_rule = absl::StrCat("start: function_block+ TEXT_CONTENT?",
                                GetTextAndFunctionBlockRules(
                                    tool_call_cases, options.code_fence_start));
      break;
    }
  }

  return absl::StrCat(absl::StrJoin(tool_blocks, "\n"), "\n", python_grammar,
                      "\n", start_rule);
}

}  // namespace litert::lm
