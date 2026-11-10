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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LLG_TOOL_CALL_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LLG_TOOL_CALL_UTILS_H_

#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/llguidance_schema_utils.h"

namespace litert::lm {

// Sanitizes a tool or property name into a valid, collision-free Lark rule
// identifier matching `[a-z][_a-z0-9\-]*`.
//
// LLGuidance Lark lexer only recognizes rule names matching
// `!?[_?]?[a-z][_a-z0-9\-]*`; anything else is lexed as a different token
// (e.g. an uppercase-leading name becomes a terminal) and grammar compilation
// fails.
//
// Encoding (injective):
// - `[a-z0-9_]`: unchanged.
// - `[A-Z]`: `-u` + lowercase letter.
// - Any other char (including `-`): `-x` + 2-digit lowercase hex.
// - Empty or non-`[a-z]` first character: prefixed with `r--`.
//
// Because `-` inside a sanitized name only ever appears as `-u<lower>`,
// `-x<hex>`, or the leading `r--`, joining sanitized names with hyphenated
// separators/suffixes (`-req-`, `-opt-`, `-optional`, `-object`, `-args`) is
// unambiguous and prevents boundary collisions such as
// `(tool="a_req", prop="b")` vs. `(tool="a", prop="req_b")`. Callers should
// keep raw tool/property names in quoted string literals (which constrain model
// output) and use `SanitizeLarkRuleName` only for non-terminal rule names.
std::string SanitizeLarkRuleName(absl::string_view name);

struct ToolFormatConfig {
  std::string pair_separator;
  std::string rule_suffix;
  std::string start_wrap;
  std::string end_wrap;
  absl::AnyInvocable<std::string(const nlohmann::ordered_json&, bool) const>
      generate_value_rule;
  // If true, the properties must be in alphabetical order, ignoring case, like
  // Jinja's dictsort sorts them, whether they're required or optional. Gemma 4
  // and FunctionGemma write the arguments of a tool call in this order. Each
  // optional property can be left out. If false, the required properties come
  // first, in the order of "required", followed by the optional ones in any
  // order.
  bool alphabetical_properties = false;
};

void ExtractToolProperties(const nlohmann::ordered_json& tool,
                           const std::string& tool_name,
                           const ToolFormatConfig& config,
                           std::vector<std::string>& tool_blocks,
                           std::vector<std::string>& required_props,
                           std::vector<std::string>& optional_props);

void AppendRequiredProperties(const std::vector<std::string>& required_props,
                              const std::string& tool_name,
                              std::vector<std::string>& sequence);

void AppendOptionalProperties(const std::vector<std::string>& optional_props,
                              const std::string& tool_name,
                              std::vector<std::string>& tool_blocks,
                              std::vector<std::string>& sequence);

void AppendToolRules(const nlohmann::ordered_json& tool,
                     const std::string& tool_name,
                     const ToolFormatConfig& config,
                     std::vector<std::string>& tool_blocks);

// Returns `text` escaped so that a Lark regex (`/.../`) matches it literally,
// e.g. "<|tool_call>" becomes "<\|tool_call>".
std::string EscapeForLarkRegex(absl::string_view text);

// Returns the `start` rule, which matches `body`. If `options` has channel
// tokens, the model may also open channels (e.g. the thought channel) before
// `body`.
std::string GetStartRule(const LlgConstraintsOptions& options,
                         absl::string_view body);

// Returns the grammar for text-only answers, i.e. the "none" tool choice. The
// answer can't contain the code fence, and it needs a character that isn't
// whitespace, even after a channel.
std::string GetTextOnlyBlock(const LlgConstraintsOptions& options);

std::string GetRuleForType(const std::string& type,
                           const std::string& fallback_rule);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LLG_TOOL_CALL_UTILS_H_
