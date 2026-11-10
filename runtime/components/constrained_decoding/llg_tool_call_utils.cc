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

#include <cstddef>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/llguidance_schema_utils.h"

namespace litert::lm {

std::string SanitizeLarkRuleName(absl::string_view name) {
  std::string sanitized;
  sanitized.reserve(name.size());
  for (char c : name) {
    if (absl::ascii_islower(c) || absl::ascii_isdigit(c) || c == '_') {
      sanitized.push_back(c);
    } else if (absl::ascii_isupper(c)) {
      // Escape uppercase as `-u<lower>` to satisfy the lowercase-only rule
      // regex while preserving case distinction.
      sanitized.append("-u");
      sanitized.push_back(absl::ascii_tolower(c));
    } else {
      // Escape `-` and any non-identifier characters as `-x<hex>` so literal
      // hyphens cannot collide with `-u` escapes or `-req-`/`-opt-` separators.
      absl::StrAppendFormat(&sanitized, "-x%02x",
                            static_cast<unsigned char>(c));
    }
  }
  // Prefix with `r--` if the name does not start with `[a-z]`. This guarantees
  // the rule name matches `[a-z][_a-z0-9\-]*`, and cannot collide with a name
  // starting with `r` because unprefixed sanitized names never contain `--`
  // (hyphens only appear in `-u<lower>` and `-x<hex>`).
  if (name.empty() || !absl::ascii_islower(name[0])) {
    return absl::StrCat("r--", sanitized);
  }
  return sanitized;
}

void ExtractToolProperties(const nlohmann::ordered_json& tool,
                           const std::string& tool_name,
                           const ToolFormatConfig& config,
                           std::vector<std::string>& tool_blocks,
                           std::vector<std::string>& required_props,
                           std::vector<std::string>& optional_props) {
  if (!tool.contains("parameters") || !tool["parameters"].is_object()) {
    return;
  }

  const std::string sanitized_tool = SanitizeLarkRuleName(tool_name);
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
      std::string sanitized_prop = SanitizeLarkRuleName(prop_name);

      // Hyphenated `-req-`/`-opt-` separators prevent `(tool, prop)` boundary
      // collisions. See `SanitizeLarkRuleName` for details.
      if (is_req) {
        tool_blocks.push_back(absl::StrFormat(
            R"(%s-req-%s: %s)", sanitized_tool, sanitized_prop, pair_rule));
      } else {
        optional_props.push_back(prop_name);
        tool_blocks.push_back(absl::StrFormat(
            R"(%s-opt-%s: %s)", sanitized_tool, sanitized_prop, pair_rule));
      }
    }
  }
}

void AppendRequiredProperties(const std::vector<std::string>& required_props,
                              const std::string& tool_name,
                              std::vector<std::string>& sequence) {
  const std::string sanitized_tool = SanitizeLarkRuleName(tool_name);
  for (const std::string& req : required_props) {
    if (!sequence.empty()) {
      sequence.push_back(R"(",")");
    }
    sequence.push_back(absl::StrFormat("%s-req-%s", sanitized_tool,
                                       SanitizeLarkRuleName(req)));
  }
}

void AppendOptionalProperties(const std::vector<std::string>& optional_props,
                              const std::string& tool_name,
                              std::vector<std::string>& tool_blocks,
                              std::vector<std::string>& sequence) {
  if (optional_props.empty()) {
    return;
  }

  const std::string sanitized_tool = SanitizeLarkRuleName(tool_name);
  // See `SanitizeLarkRuleName` for why `-optional` uses a hyphen separator.
  std::string opt_rule_name = absl::StrCat(sanitized_tool, "-optional");

  std::vector<std::string> opt_pairs;
  std::vector<std::string> opt_pairs_with_comma;
  for (const std::string& opt : optional_props) {
    std::string sanitized_opt = SanitizeLarkRuleName(opt);
    opt_pairs.push_back(
        absl::StrFormat("%s-opt-%s", sanitized_tool, sanitized_opt));
    opt_pairs_with_comma.push_back(absl::StrFormat(
        R"("," %s-opt-%s %s)", sanitized_tool, sanitized_opt, opt_rule_name));
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

namespace {

// Returns the body of a tool's rule for
// `ToolFormatConfig::alphabetical_properties`: the properties in alphabetical
// order, ignoring case, separated by commas. Each required property must be
// there, and each optional one may be left out.
std::string GetAlphabeticalPropertiesBody(
    const std::vector<std::string>& required_props,
    const std::vector<std::string>& optional_props,
    absl::string_view tool_name) {
  struct Property {
    std::string sort_key;
    std::string name;
    std::string rule;
    bool is_required;
  };
  const std::string sanitized_tool = SanitizeLarkRuleName(tool_name);
  std::vector<Property> properties;
  std::unordered_set<std::string> names;
  const auto add = [&](const std::string& name, bool is_required) {
    if (!names.insert(name).second) {
      return;
    }
    properties.push_back({
        .sort_key = absl::AsciiStrToLower(name),
        .name = name,
        .rule = absl::StrFormat("%s-%s-%s", sanitized_tool,
                                is_required ? "req" : "opt",
                                SanitizeLarkRuleName(name)),
        .is_required = is_required,
    });
  };
  for (const std::string& name : required_props) {
    add(name, /*is_required=*/true);
  }
  for (const std::string& name : optional_props) {
    add(name, /*is_required=*/false);
  }
  // Jinja's dictsort compares the names in lowercase. Names that are the same
  // in lowercase are compared as they are, so the order is always the same.
  absl::c_sort(properties, [](const Property& a, const Property& b) {
    return std::tie(a.sort_key, a.name) < std::tie(b.sort_key, b.name);
  });

  if (properties.empty()) {
    return R"("")";
  }
  const auto first_required = absl::c_find_if(
      properties,
      [](const Property& property) { return property.is_required; });
  if (first_required == properties.end()) {
    // Any of the properties can come first, and it has no comma before it.
    std::vector<std::string> alternatives;
    alternatives.reserve(properties.size());
    for (size_t i = 0; i < properties.size(); ++i) {
      std::vector<std::string> sequence = {properties[i].rule};
      for (size_t j = i + 1; j < properties.size(); ++j) {
        sequence.push_back(absl::StrFormat(R"(("," %s)?)", properties[j].rule));
      }
      alternatives.push_back(absl::StrJoin(sequence, " "));
    }
    return absl::StrFormat("(%s)?", absl::StrJoin(alternatives, " | "));
  }
  // The first required property is always there, so the optional properties
  // before it are followed by a comma, and the properties after it are
  // preceded by one.
  std::vector<std::string> sequence;
  sequence.reserve(properties.size());
  for (auto it = properties.begin(); it != properties.end(); ++it) {
    if (it < first_required) {
      sequence.push_back(absl::StrFormat(R"((%s ",")?)", it->rule));
    } else if (it == first_required) {
      sequence.push_back(it->rule);
    } else if (it->is_required) {
      sequence.push_back(absl::StrFormat(R"("," %s)", it->rule));
    } else {
      sequence.push_back(absl::StrFormat(R"(("," %s)?)", it->rule));
    }
  }
  return absl::StrJoin(sequence, " ");
}

}  // namespace

void AppendToolRules(const nlohmann::ordered_json& tool,
                     const std::string& tool_name,
                     const ToolFormatConfig& config,
                     std::vector<std::string>& tool_blocks) {
  std::string tool_rule =
      absl::StrCat(SanitizeLarkRuleName(tool_name), config.rule_suffix);

  std::vector<std::string> required_props;
  std::vector<std::string> optional_props;

  ExtractToolProperties(tool, tool_name, config, tool_blocks, required_props,
                        optional_props);

  std::string body;
  if (config.alphabetical_properties) {
    body = GetAlphabeticalPropertiesBody(required_props, optional_props,
                                         tool_name);
  } else {
    std::vector<std::string> sequence;
    AppendRequiredProperties(required_props, tool_name, sequence);
    AppendOptionalProperties(optional_props, tool_name, tool_blocks, sequence);
    body = sequence.empty() ? R"("")" : absl::StrJoin(sequence, " ");
  }
  if (config.start_wrap.empty() && config.end_wrap.empty()) {
    tool_blocks.push_back(absl::StrFormat(R"(%s: %s)", tool_rule, body));
  } else {
    tool_blocks.push_back(absl::StrFormat(R"(%s: "%s" %s "%s")", tool_rule,
                                          config.start_wrap, body,
                                          config.end_wrap));
  }
}

std::string EscapeForLarkRegex(absl::string_view text) {
  std::string escaped;
  escaped.reserve(text.size());
  for (const char c : text) {
    switch (c) {
      case '\n':
        escaped.append("\\n");
        break;
      case '\r':
        escaped.append("\\r");
        break;
      case '\t':
        escaped.append("\\t");
        break;
      // The regex_syntax meta characters, as escaped by regex_syntax::escape,
      // and the `/` that ends a Lark regex.
      case '\\':
      case '/':
      case '.':
      case '^':
      case '$':
      case '|':
      case '?':
      case '*':
      case '+':
      case '(':
      case ')':
      case '[':
      case ']':
      case '{':
      case '}':
      case '#':
      case '&':
      case '-':
      case '~':
        escaped.push_back('\\');
        escaped.push_back(c);
        break;
      default:
        escaped.push_back(c);
    }
  }
  return escaped;
}

std::string GetStartRule(const LlgConstraintsOptions& options,
                         absl::string_view body) {
  if (options.channel_start.empty() || options.channel_end.empty()) {
    return absl::StrCat("start: ", body, "\n");
  }
  // The channel tokens are special tokens, so text can't match them. Only this
  // rule can, and only before `body`. Gemma 4 sometimes thinks in more than one
  // block, so any number of channels is allowed.
  return absl::StrFormat(
      R"(start: channel_block* %s
channel_block: %s CHANNEL_TEXT? %s
CHANNEL_TEXT: /(.|\n)+/
)",
      body, options.channel_start, options.channel_end);
}

std::string GetTextOnlyBlock(const LlgConstraintsOptions& options) {
  // The answer can't contain the code fence, so the model can't write a tool
  // call as text. It needs a character that isn't whitespace, so the model
  // can't end the turn without a visible answer, e.g. after only "\n". The
  // model can't end the turn before that character, so only two whitespace
  // characters can come before it. Otherwise the model could keep writing
  // whitespace instead of answering.
  return absl::StrCat(absl::StrFormat(
                          R"(
FORBIDDEN_CALL : /(.|\n)*%s(.|\n)*/
SAFE_TEXT : /\s{0,2}\S(.|\n)*/ & ~FORBIDDEN_CALL
)",
                          EscapeForLarkRegex(options.code_fence_start)),
                      GetStartRule(options, "SAFE_TEXT"));
}

std::string GetRuleForType(const std::string& type,
                           const std::string& fallback_rule) {
  if (type == "string") return "string";
  // "integer" and "number" are distinct JSON Schema types and must map to
  // distinct rules: NUMBER permits a fraction and an exponent, so sharing it
  // would let the model emit 1000.0 for an integer field. Both the FC and the
  // Python grammar define INTEGER; they have to stay in sync with this.
  if (type == "integer") return "INTEGER";
  if (type == "number") return "NUMBER";
  if (type == "boolean") return "BOOLEAN";
  if (type == "array") return "array";
  if (type == "object") return "object";
  if (type == "null") return "NULL";
  return fallback_rule;
}

}  // namespace litert::lm
