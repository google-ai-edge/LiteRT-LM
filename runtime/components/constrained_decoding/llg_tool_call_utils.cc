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
#include <unordered_set>
#include <vector>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/llguidance_schema_utils.h"

namespace litert::lm {

void ExtractToolProperties(const nlohmann::ordered_json& tool,
                           const std::string& tool_name,
                           const ToolFormatConfig& config,
                           std::vector<std::string>& tool_blocks,
                           std::vector<std::string>& required_props,
                           std::vector<std::string>& optional_props) {
  if (!tool.contains("parameters") || !tool["parameters"].is_object()) {
    return;
  }

  const auto& params = tool["parameters"];
  std::unordered_set<std::string> required_set;
  if (params.contains("required") && params["required"].is_array()) {
    for (const auto& req : params["required"]) {
      if (req.is_string()) {
        std::string req_str = req.get<std::string>();
        required_props.push_back(req_str);
        required_set.insert(req_str);
      }
    }
  }
  if (params.contains("properties") && params["properties"].is_object()) {
    for (const auto& [prop_name, prop_schema] : params["properties"].items()) {
      bool is_req = required_set.contains(prop_name);
      std::string pair_rule =
          absl::StrFormat(R"("%s" "%s" %s)", prop_name, config.pair_separator,
                          config.generate_value_rule(prop_schema, is_req));

      if (is_req) {
        tool_blocks.push_back(absl::StrFormat(R"(%s_req_%s: %s)", tool_name,
                                              prop_name, pair_rule));
      } else {
        optional_props.push_back(prop_name);
        tool_blocks.push_back(absl::StrFormat(R"(%s_opt_%s: %s)", tool_name,
                                              prop_name, pair_rule));
      }
    }
  }
}

void AppendRequiredProperties(const std::vector<std::string>& required_props,
                              const std::string& tool_name,
                              std::vector<std::string>& sequence) {
  for (const std::string& req : required_props) {
    if (!sequence.empty()) {
      sequence.push_back(R"(",")");
    }
    sequence.push_back(absl::StrFormat("%s_req_%s", tool_name, req));
  }
}

void AppendOptionalProperties(const std::vector<std::string>& optional_props,
                              const std::string& tool_name,
                              std::vector<std::string>& tool_blocks,
                              std::vector<std::string>& sequence) {
  if (optional_props.empty()) {
    return;
  }

  std::string opt_rule_name = absl::StrCat(tool_name, "_optional");

  std::vector<std::string> opt_pairs;
  std::vector<std::string> opt_pairs_with_comma;
  for (const std::string& opt : optional_props) {
    opt_pairs.push_back(absl::StrFormat("%s_opt_%s", tool_name, opt));
    opt_pairs_with_comma.push_back(
        absl::StrFormat(R"("," %s_opt_%s %s)", tool_name, opt, opt_rule_name));
  }
  std::string all_opts = absl::StrJoin(opt_pairs, " | ");
  std::string all_opts_with_comma = absl::StrJoin(opt_pairs_with_comma, " | ");

  if (!sequence.empty()) {
    sequence.push_back(opt_rule_name);
    tool_blocks.push_back(
        absl::StrFormat("%s: %s | \"\"", opt_rule_name, all_opts_with_comma));
  } else {
    sequence.push_back(opt_rule_name);
    tool_blocks.push_back(absl::StrFormat(R"(%s: "" | (%s) ("," (%s))*)",
                                          opt_rule_name, all_opts, all_opts));
  }
}

void AppendToolRules(const nlohmann::ordered_json& tool,
                     const std::string& tool_name,
                     const ToolFormatConfig& config,
                     std::vector<std::string>& tool_blocks) {
  std::string tool_rule = absl::StrCat(tool_name, config.rule_suffix);

  std::vector<std::string> required_props;
  std::vector<std::string> optional_props;

  ExtractToolProperties(tool, tool_name, config, tool_blocks, required_props,
                        optional_props);

  std::vector<std::string> sequence;
  AppendRequiredProperties(required_props, tool_name, sequence);
  AppendOptionalProperties(optional_props, tool_name, tool_blocks, sequence);

  std::string body = sequence.empty() ? R"("")" : absl::StrJoin(sequence, " ");
  if (config.start_wrap.empty() && config.end_wrap.empty()) {
    tool_blocks.push_back(absl::StrFormat(R"(%s: %s)", tool_rule, body));
  } else {
    tool_blocks.push_back(absl::StrFormat(R"(%s: "%s" %s "%s")", tool_rule,
                                          config.start_wrap, body,
                                          config.end_wrap));
  }
}

std::string GetTextOnlyBlock(const LlgConstraintsOptions& options) {
  std::string escaped_fence =
      absl::StrReplaceAll(options.code_fence_start, {{"\n", "\\n"}});
  return absl::StrFormat(
      R"(
FORBIDDEN_CALL : /(.|\n)*%s(.|\n)*/
SAFE_TEXT : /(.|\n)*/ & ~FORBIDDEN_CALL
start : SAFE_TEXT
)",
      escaped_fence);
}

std::string GetRuleForType(const std::string& type,
                           const std::string& fallback_rule) {
  if (type == "string") return "string";
  if (type == "number" || type == "integer") return "NUMBER";
  if (type == "boolean") return "BOOLEAN";
  if (type == "array") return "array";
  if (type == "object") return "object";
  if (type == "null") return "NULL";
  return fallback_rule;
}

}  // namespace litert::lm
