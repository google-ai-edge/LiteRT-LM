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

#include "runtime/conversation/model_data_processor/data_utils.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/escaping.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/constraint.h"
#if !defined(LITERT_LM_FST_CONSTRAINTS_DISABLED)
#include "runtime/components/constrained_decoding/gemma_model_constraint_provider.h"
#endif
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/conversation/io_types.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/status_macros.h"
#include "support/tokenizer/sentencepiece_tokenizer.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

using ::nlohmann::ordered_json;

absl::StatusOr<std::unique_ptr<MemoryMappedFile>> LoadItemData(
    const ordered_json& item) {
  if (!item.contains("type")) {
    // If `item` doesn't contain a type, it won't be loaded.
    return nullptr;
  }
  if (item["type"] == "text") {
    return InMemoryFile::Create(item["text"]);
  } else if (item["type"] == "image" || item["type"] == "audio") {
    if (item.contains("path")) {
      return MemoryMappedFile::Create(item["path"].get<std::string>());
    }
    if (item.contains("blob")) {
      std::string blob_b64 = item["blob"].get<std::string>();
      std::string blob;
      if (!absl::Base64Unescape(blob_b64, &blob)) {
        return absl::InvalidArgumentError("Failed to decode base64 blob.");
      }
      return InMemoryFile::Create(blob);
    }
    return absl::InvalidArgumentError(
        "Audio or image item must contain a path or blob.");
  } else if (item["type"] == "tool_response") {
    return nullptr;
  }
  return absl::UnimplementedError("Unsupported item type: " +
                                  item["type"].get<std::string>());
}

ordered_json NormalizeContent(const ordered_json& content) {
  if (content.is_string()) {
    return ordered_json::array(
        {{{"type", "text"}, {"text", content.get<std::string>()}}});
  }
  if (content.is_object()) {
    return ordered_json::array({content});
  }
  return content;
}

ordered_json NormalizeMessageContent(const ordered_json& message) {
  if (!message.contains("content")) {
    return message;
  }
  ordered_json result = message;
  result["content"] = NormalizeContent(message["content"]);
  return result;
}

absl::StatusOr<ordered_json> ResponseTextToMessage(
    absl::string_view response_text, const std::optional<Preface>& preface,
    absl::string_view code_fence_start, absl::string_view code_fence_end,
    SyntaxType syntax_type, const ParserOptions& options) {
  ordered_json message = {{"role", "assistant"}};
  if (preface.has_value() && std::holds_alternative<JsonPreface>(*preface) &&
      !std::get<JsonPreface>(*preface).tools.empty()) {
    ABSL_ASSIGN_OR_RETURN(
        ordered_json content_and_tool_calls,
        ParseTextAndToolCalls(response_text, code_fence_start, code_fence_end,
                              syntax_type, options));
    if (content_and_tool_calls.contains("content")) {
      message["content"] = content_and_tool_calls["content"];
    }
    if (content_and_tool_calls.contains("tool_calls")) {
      message["tool_calls"] = content_and_tool_calls["tool_calls"];
    }
  } else {
    message["content"] = ordered_json::array(
        {{{"type", "text"}, {"text", std::string(response_text)}}});
  }
  return message;
}

#if !defined(LITERT_LM_FST_CONSTRAINTS_DISABLED)
absl::StatusOr<GemmaModelConstraintProviderPtr>
CreateGemmaModelConstraintProvider(
    const ::litert::support::Tokenizer* tokenizer,
    const std::vector<std::vector<int>>& stop_token_ids) {
  GemmaModelConstraintProviderPtr constraint_provider(
      nullptr, &LiteRtLmGemmaModelConstraintProvider_Destroy);
  if (tokenizer->GetTokenizerType() !=
      ::litert::support::TokenizerType::kSentencePiece) {
    ABSL_LOG(WARNING) << "Constrained decoding is only supported for "
                         "SentencePiece tokenizer.";
    return constraint_provider;
  }
  std::vector<const int*> stop_token_ids_ptrs;
  std::vector<size_t> stop_token_lengths;
  stop_token_ids_ptrs.reserve(stop_token_ids.size());
  stop_token_lengths.reserve(stop_token_ids.size());
  for (const auto& stop_tokens : stop_token_ids) {
    stop_token_ids_ptrs.push_back(stop_tokens.data());
    stop_token_lengths.push_back(stop_tokens.size());
  }
  auto sp_tokenizer =
      reinterpret_cast<const ::litert::support::SentencePieceTokenizer*>(
          tokenizer);
  auto serialized_model_proto =
      sp_tokenizer->GetProcessor().model_proto().SerializeAsString();
  LiteRtLmGemmaModelConstraintProvider* provider =
      LiteRtLmGemmaModelConstraintProvider_Create(
          serialized_model_proto.data(), serialized_model_proto.size(),
          stop_token_ids_ptrs.data(), stop_token_lengths.data(),
          stop_token_ids.size());
  if (provider == nullptr) {
    return absl::InternalError(
        "Failed to create GemmaModelConstraintProvider.");
  }
  constraint_provider.reset(provider);
  return constraint_provider;
}

absl::StatusOr<std::unique_ptr<Constraint>> CreateGemmaConstraintFromTools(
    LiteRtLmGemmaModelConstraintProvider* provider,
    const nlohmann::ordered_json& tools,
    const LiteRtLmGemmaModelConstraintOptions& options) {
  if (provider == nullptr) {
    return nullptr;
  }
  if (!tools.is_array()) {
    return absl::InvalidArgumentError("Tools must be an array.");
  }
  nlohmann::ordered_json functions = nlohmann::ordered_json::array();
  for (const auto& tool : tools) {
    if (tool.contains("function")) {
      functions.push_back(tool["function"]);
    } else {
      functions.push_back(tool);
    }
  }
  std::string functions_str = functions.dump();
  LiteRtLmConstraint* constraint =
      LiteRtLmGemmaModelConstraintProvider_CreateConstraintFromTools(
          provider, functions_str.c_str(), &options);
  if (constraint == nullptr) {
    return absl::InternalError("Failed to create constraint with tools.");
  }
  return absl::WrapUnique(reinterpret_cast<Constraint*>(constraint));
}
#endif

}  // namespace litert::lm
