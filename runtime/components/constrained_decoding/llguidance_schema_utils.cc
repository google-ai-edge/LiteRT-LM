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

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/llg_fc_tool_calls.h"
#include "runtime/components/constrained_decoding/llg_python_tool_calls.h"
#include "runtime/components/constrained_decoding/tool_utils.h"

namespace litert::lm {

absl::StatusOr<std::string> CreateLarkGrammarForTools(
    const nlohmann::ordered_json& tools, const LlgConstraintsOptions& options) {
  const nlohmann::ordered_json& tools_array = GetToolsArray(tools);
  if (!tools_array.is_array()) {
    return absl::InvalidArgumentError("tools must be an array.");
  }
  switch (options.funcall_format) {
    case FuncallFormat::kFc:
      return CreateLarkGrammarForFcToolCalls(tools_array, options);
    case FuncallFormat::kPython:
      return CreateLarkGrammarForPythonToolCalls(tools_array, options);
  }
  return absl::InvalidArgumentError("Unknown function call format.");
}

absl::Status ApplyToolChoiceToOptions(const nlohmann::ordered_json& tool_choice,
                                      LlgConstraintsOptions& options) {
  if (tool_choice.is_null()) {
    return absl::OkStatus();
  }
  if (tool_choice.is_string()) {
    const std::string mode_str = tool_choice.get<std::string>();
    if (mode_str == "auto") {
      options.constraint_mode = LlgConstraintMode::kTextAndOrFunctionCalls;
    } else if (mode_str == "none") {
      options.constraint_mode = LlgConstraintMode::kTextOnly;
    } else if (mode_str == "required") {
      options.constraint_mode = LlgConstraintMode::kFunctionCallsRequired;
    } else {
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported tool_choice string: %s", mode_str));
    }
    return absl::OkStatus();
  }
  if (tool_choice.is_object()) {
    if (tool_choice.contains("type") && tool_choice["type"] == "function" &&
        tool_choice.contains("function") &&
        tool_choice["function"].contains("name") &&
        tool_choice["function"]["name"].is_string()) {
      options.allowed_function_names = {
          tool_choice["function"]["name"].get<std::string>()};
      options.constraint_mode = LlgConstraintMode::kFunctionCallsRequired;
      return absl::OkStatus();
    }
    return absl::InvalidArgumentError(
        "Invalid tool_choice object format. Expected {\"type\": \"function\", "
        "\"function\": {\"name\": \"...\"}}");
  }
  return absl::InvalidArgumentError("tool_choice must be a string or object.");
}

}  // namespace litert::lm
