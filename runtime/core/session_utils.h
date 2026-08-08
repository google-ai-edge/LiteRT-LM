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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_UTILS_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

// The util function to get the BOS string if there is a valid BOS token id.
// Otherwise, return an empty string.
absl::StatusOr<std::string> MaybeGetBosString(
    const SessionConfig& session_config, support::Tokenizer& tokenizer);

// The util function to convert the string to processed input text.
absl::StatusOr<InputText> StringToProcessedInputText(
    absl::string_view text, const SessionConfig& session_config,
    support::Tokenizer& tokenizer,
    const std::optional<BenchmarkInfo>& benchmark_info);

// Util function for applying the prompt templates.
// contents: The input contents to apply the prompt templates.
// The output is the input after applying the proper prompt templates.
// This function is intended for basic text-only content. Will raise error if
// the input contains non-text contents and ApplyPromptTemplateInSession is
// true.
// The content_type is used to determine which prompt template to use.
// kFirst: User's turn first chunk.
// kMiddle: User's turn middle chunk.
// kLast: User's turn last chunk.
// kNA: Not applicable. Only used when ApplyPromptTemplateInSession is false.
enum class ContentType : int { kFirst, kMiddle, kLast, kNA };
absl::StatusOr<std::vector<InputData>> ApplyPromptTemplates(
    const std::vector<InputData>& contents, ContentType content_type,
    const SessionConfig& session_config, support::Tokenizer& tokenizer,
    bool is_first_turn);

// Preprocesses the input contents. This function is used for pre-processing
// the input contents before sending them to the LLM executor.
// Text input will be preprocessed by the tokenizer.
absl::StatusOr<std::vector<InputData>> PreprocessContents(
    const std::vector<InputData>& contents, const SessionConfig& session_config,
    support::Tokenizer& tokenizer,
    const std::optional<BenchmarkInfo>& benchmark_info);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_UTILS_H_
