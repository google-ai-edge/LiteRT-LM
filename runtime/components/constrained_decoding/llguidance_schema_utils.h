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

#include <optional>
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
// TODO(b/577134629): Consider replacing this enum with separate settings for
// text and for function calls, each allowed, disallowed or required.
enum class LlgConstraintMode {
  kTextAndOrFunctionCalls,  // Optional text + optional function calls ("auto").
  kFunctionCallsOnly,       // Only function calls are allowed (no text). Like
                            // FST's FUNCTION_CALL_ONLY mode, the turn may also
                            // end without a call, unless a function is forced.
  kTextOnly,                // Only text is allowed (no function calls, "none").
  kFunctionCallsRequired,   // Optional text + at least one required function
                            // call ("required").
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

  // If set, constrains tool calling strictly to this specific function name.
  // TODO(b/577134629): Consider a list of names, so that a call can be
  // required from any of several functions without allowing text.
  std::optional<std::string> forced_function_name = std::nullopt;
};

// Converts tools to a Lark grammar string.
absl::StatusOr<std::string> CreateLarkGrammarForTools(
    const nlohmann::ordered_json& tools, const LlgConstraintsOptions& options);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LLGUIDANCE_SCHEMA_UTILS_H_
