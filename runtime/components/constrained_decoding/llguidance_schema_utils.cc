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
#include <variant>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/llg_fc_tool_calls.h"
#include "runtime/components/constrained_decoding/llg_python_tool_calls.h"
#include "runtime/components/constrained_decoding/tool_choice.h"
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

namespace {

// Sets the constraint mode and the allowed functions for each kind of tool
// choice. std::visit doesn't compile if a ToolChoice alternative is missing.
struct ToolChoiceApplier {
  LlgConstraintsOptions* options;

  absl::Status operator()(const ToolChoice::Auto&) const {
    options->constraint_mode = LlgConstraintMode::kTextAndOrFunctionCalls;
    return absl::OkStatus();
  }
  absl::Status operator()(const ToolChoice::None&) const {
    options->constraint_mode = LlgConstraintMode::kTextOnly;
    return absl::OkStatus();
  }
  absl::Status operator()(const ToolChoice::Required&) const {
    options->constraint_mode = LlgConstraintMode::kFunctionCallsRequired;
    return absl::OkStatus();
  }
  absl::Status operator()(const ToolChoice::ForcedFunction& forced) const {
    if (forced.name.empty()) {
      return absl::InvalidArgumentError(
          "ToolChoice::ForcedFunction requires a name.");
    }
    options->constraint_mode = LlgConstraintMode::kFunctionCallsRequired;
    options->allowed_function_names = {forced.name};
    return absl::OkStatus();
  }
  absl::Status operator()(const ToolChoice::AllowedTools& allowed) const {
    if (allowed.names.empty()) {
      return absl::InvalidArgumentError(
          "ToolChoice::AllowedTools requires at least one name.");
    }
    options->constraint_mode = allowed.required
                                   ? LlgConstraintMode::kFunctionCallsRequired
                                   : LlgConstraintMode::kTextAndOrFunctionCalls;
    options->allowed_function_names = allowed.names;
    return absl::OkStatus();
  }
};

}  // namespace

absl::Status ApplyToolChoiceToOptions(const ToolChoice& tool_choice,
                                      LlgConstraintsOptions& options) {
  return std::visit(ToolChoiceApplier{&options}, tool_choice.choice);
}

}  // namespace litert::lm
