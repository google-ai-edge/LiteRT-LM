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
  // Simplified JSON-based FC format.
  kFc,
  kPython,
};

// Supported constraint modes.
enum class LlgConstraintMode {
  kTextAndOrFunctionCalls,  // Optional text + optional function calls.
  kFunctionCallsOnly,       // Only function calls are allowed.
  kTextOnly,                // Only text is allowed (no function calls).
};

// Options for formatting constraints.
struct LlgConstraintsOptions {
  FuncallFormat funcall_format;
  LlgConstraintMode constraint_mode;

  std::string code_fence_start;
  std::string code_fence_end;
  std::string open_quote;
  std::string close_quote;
  std::string function_response_start;
};

// Converts tools to a Lark grammar string.
absl::StatusOr<std::string> CreateLarkGrammarForTools(
    const nlohmann::ordered_json& tools, const LlgConstraintsOptions& options);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LLGUIDANCE_SCHEMA_UTILS_H_
