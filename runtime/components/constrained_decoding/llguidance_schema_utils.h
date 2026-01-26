// Copyright 2025 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LLGUIDANCE_SCHEMA_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LLGUIDANCE_SCHEMA_UTILS_H_

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json

namespace litert::lm {

// Supported function call formats.
enum class FuncallFormat {
  // Python-like funcall format used in go/extensions-as-code and go/eacv2.
  kEac,
  // Simplified JSON-based FC2.0 format (go/fc2).
  kFc2p0,
};

// Supported constraint modes.
enum class LlgConstraintMode {
  kTextAndOr,         // Optional text + optional function call.
  kFunctionCallOnly,  // Only function call is allowed.
  kTextOnly,          // Only text is allowed (no function calls).
};

// Options for formatting constraints.
struct LlgConstraintsOptions {
  FuncallFormat funcall_format = FuncallFormat::kFc2p0;
  LlgConstraintMode constraint_mode = LlgConstraintMode::kTextAndOr;

  // The FC2 control tokens.
  std::string fc2_code_fence_start;         // e.g. <start_function_call>
  std::string fc2_code_fence_end;           // e.g. <end_function_call>
  std::string fc2_open_quote;               // e.g. <escape>
  std::string fc2_close_quote;              // e.g. <escape>
  std::string fc2_function_response_start;  // e.g. <start_function_response>
};

// Converts tools to a Lark grammar string.
absl::StatusOr<std::string> FormatToolsAsLarkGrammar(
    const nlohmann::ordered_json& tools, const LlgConstraintsOptions& options);
}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LLGUIDANCE_SCHEMA_UTILS_H_
