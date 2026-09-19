// Copyright 2025 The Google AI Edge Authors.
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

#include "runtime/components/tool_use/fc_parser_utils.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/tool_use/parser_common.h"
#include "runtime/components/tool_use/rust/parsers.rs.h"
#include "re2/re2.h"  // from @com_googlesource_code_re2

namespace litert::lm {

namespace {

void SanitizeFcToolCalls(nlohmann::ordered_json& tool_calls) {
  if (!tool_calls.is_array()) {
    return;
  }
  for (auto& tool_call : tool_calls) {
    if (!tool_call.is_object() || !tool_call.contains("arguments")) {
      continue;
    }
    auto& args = tool_call["arguments"];
    if (!args.is_object()) {
      continue;
    }
    // Collect all argument names.
    std::vector<std::string> arg_names;
    for (auto it = args.begin(); it != args.end(); ++it) {
      arg_names.push_back(it.key());
    }

    // Check each string argument for trailing unquoted transitions of other arguments.
    for (auto it = args.begin(); it != args.end(); ++it) {
      if (!it.value().is_string()) {
        continue;
      }
      std::string val_str = it.value().get<std::string>();
      bool modified = false;
      for (const auto& other_key : arg_names) {
        if (other_key == it.key()) {
          continue;
        }
        std::string pattern = absl::StrCat(R"((?s)[\s`]*,\s*)",
                                           RE2::QuoteMeta(other_key),
                                           R"(\s*(?::.*)?$)");
        if (RE2::Replace(&val_str, pattern, "")) {
          modified = true;
          ABSL_LOG(INFO) << "Sanitized trailing parameter '" << other_key
                         << "' from argument '" << it.key() << "'";
        }
      }
      if (modified) {
        it.value() = val_str;
      }
    }
  }
}

}  // namespace

absl::StatusOr<nlohmann::ordered_json> ParseFcExpression(
    absl::string_view text) {
  auto tool_calls = parse_fc_expression(text.data());
  if (!tool_calls.is_ok) {
    absl::string_view error_message =
        absl::string_view(tool_calls.error.data(), tool_calls.error.size());
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse FC tool calls: ", error_message));
  }

  nlohmann::ordered_json tool_calls_json = nlohmann::ordered_json::array();
  for (const auto& tool_call : tool_calls.tool_calls) {
    tool_calls_json.push_back(ConvertJsonValue(tool_call));
  }
  SanitizeFcToolCalls(tool_calls_json);
  return tool_calls_json;
}

}  // namespace litert::lm
