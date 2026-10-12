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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_DATA_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_DATA_UTILS_H_

#include <memory>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/constraint.h"
#if !defined(LITERT_LM_FST_CONSTRAINTS_DISABLED)
#include "runtime/components/constrained_decoding/gemma_model_constraint_provider.h"
#endif
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/conversation/io_types.h"
#include "runtime/util/memory_mapped_file.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

// Loads the item data from the given JSON object to a MemoryMappedFile.
// The expected item content is:
// 1. Text item
//  {
//    "type": "text",
//    "text": "some text"
//  }
//
// 2. Image item
//  {
//    "type": "image",
//    "path": "/file/path/to/image",
//  }
//  {
//    "type": "image",
//    "blob": "base64 encoded image bytes as string",
//  }
//
// 3. Audio item
//  {
//    "type": "audio",
//    "path": "/file/path/to/audio",
//  }
//  {
//    "type": "audio",
//    "blob": "base64 encoded audio bytes as string",
//  }
//
// Note: though we support loading image and audio data from blob, this format
// is less efficient and less favorable.
absl::StatusOr<std::unique_ptr<MemoryMappedFile>> LoadItemData(
    const nlohmann::ordered_json& item);

// Normalizes a message's "content" value into a list of multimodal parts.
// - If "content" is a string, wraps it into [{"type": "text", "text": string}].
// - If "content" is an object, wraps it into [object].
// - If "content" is already an array, preserves it.
nlohmann::ordered_json NormalizeContent(const nlohmann::ordered_json& content);

// Normalizes the "content" field of a message into a list of multimodal parts.
// - If "content" is a string, wraps it into [{"type": "text", "text": string}].
// - If "content" is an object, wraps it into [object].
// - If "content" is already an array, preserves it.
// - If "content" is absent (e.g. assistant tool calls), returns message
// unchanged.
nlohmann::ordered_json NormalizeMessageContent(
    const nlohmann::ordered_json& message);

// Converts a raw model response text into an assistant Message JSON object.
// If `preface` contains non-empty tools, parses text and tool calls using the
// provided syntax and parser options; otherwise, wraps the response text into a
// standard text content array.
absl::StatusOr<nlohmann::ordered_json> ResponseTextToMessage(
    absl::string_view response_text, const std::optional<Preface>& preface,
    absl::string_view code_fence_start, absl::string_view code_fence_end,
    SyntaxType syntax_type, const ParserOptions& options);

#if !defined(LITERT_LM_FST_CONSTRAINTS_DISABLED)
using GemmaModelConstraintProviderPtr =
    std::unique_ptr<LiteRtLmGemmaModelConstraintProvider,
                    decltype(&LiteRtLmGemmaModelConstraintProvider_Destroy)>;

// Creates a GemmaModelConstraintProvider instance from the given tokenizer and
// stop token IDs. Returns a null unique_ptr if the tokenizer is not a
// SentencePiece tokenizer.
absl::StatusOr<GemmaModelConstraintProviderPtr>
CreateGemmaModelConstraintProvider(
    const ::litert::support::Tokenizer* tokenizer,
    const std::vector<std::vector<int>>& stop_token_ids);

// Creates a Constraint instance from tool declarations using the provided
// GemmaModelConstraintProvider and options. Returns nullptr if `provider` is
// null.
absl::StatusOr<std::unique_ptr<Constraint>> CreateGemmaConstraintFromTools(
    LiteRtLmGemmaModelConstraintProvider* provider,
    const nlohmann::ordered_json& tools,
    const LiteRtLmGemmaModelConstraintOptions& options);
#endif

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_DATA_UTILS_H_
