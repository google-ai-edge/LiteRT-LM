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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MODEL_TYPE_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MODEL_TYPE_UTILS_H_

#include <optional>
#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

using ::litert::support::Tokenizer;

// Try to infer the model type from the tokenizer. This is for backward
// compatibility, when the model type is not set in the model LlmMetadata.
absl::StatusOr<proto::LlmModelType> InferLlmModelType(
    const proto::LlmMetadata& metadata, Tokenizer* tokenizer);

// Get the default jinja prompt template for the given model type.
// This is for backwards compatibility when the deprecated prompt_templates of
// metadata field is used.
absl::StatusOr<std::string> GetDefaultJinjaPromptTemplate(
    const proto::PromptTemplates& prompt_templates,
    const proto::LlmModelType& llm_model_type);

// Get the model type name for the given model type.
absl::string_view GetModelTypeName(const proto::LlmModelType& model_type);

// Returns the ID of the stop token that the prompt template of `model_type`
// writes again at the start of the next input, or std::nullopt if there is
// none. Decoding rolls this token back when it stops on it, so the KV cache
// does not hold it twice (see StopTokenDetector::MarkForRollback()). Only
// Gemma 4 has one: the `function_response_start` of its model file (such as
// `<|tool_response>`), which starts each tool response. FunctionGemma also has
// a `function_response_start`, but its template ends a trailing tool call turn
// with it, so it is not written twice.
std::optional<int> GetRollbackStopTokenId(const proto::LlmModelType& model_type,
                                          Tokenizer& tokenizer);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MODEL_TYPE_UTILS_H_
