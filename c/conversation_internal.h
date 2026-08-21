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

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_CONVERSATION_INTERNAL_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_CONVERSATION_INTERNAL_H_

#include <memory>
#include <optional>
#include <string>

#include "c/conversation.h"
#include "runtime/components/constrained_decoding/no_repeat_ngram_config.h"
#include "runtime/components/constrained_decoding/repetition_penalty_config.h"
#include "runtime/components/constrained_decoding/suppress_tokens_config.h"
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/thinking_config.h"
#include "runtime/engine/engine_settings.h"

struct LiteRtLmThinkingConfig {
  litert::lm::ThinkingConfig thinking_config;
};

struct LiteRtLmConversationOptionalArgs {
  std::optional<litert::lm::RepetitionPenaltyConfig> repetition_penalty_config;
  std::optional<litert::lm::NoRepeatNgramConfig> no_repeat_ngram_config;
  std::optional<litert::lm::SuppressTokensConfig> suppress_tokens_config;
  std::optional<int> visual_token_budget;
  std::optional<int> max_output_tokens;
  std::optional<litert::lm::ThinkingConfig> thinking_config;
  LiteRtLmConstraintType constraint_type = kLiteRtLmConstraintTypeNone;
  std::string constraint_string;
};

struct LiteRtLmConversation {
  std::unique_ptr<litert::lm::Conversation> conversation;
  // This field stores the result of the last call to
  // `litert_lm_conversation_render_message_to_string`. This ties the lifetime
  // of the returned `const char*` to the `LiteRtLmConversation` object,
  // ensuring memory safety for the C API caller without requiring explicit
  // per-call deallocation.
  std::string last_rendered_message;
  // This field stores the result of the last call to
  // `litert_lm_conversation_render_preface_to_string`.
  std::string last_rendered_preface;
};

struct LiteRtLmJsonResponse {
  std::string json_string;
};

struct LiteRtLmConversationConfig {
  std::optional<litert::lm::SessionConfig> session_config;
  std::string system_message_json;
  std::string tools_json;
  std::string messages_json;
  std::string extra_context_json;
  std::string prompt_template;
  bool enable_constrained_decoding = false;
  std::optional<bool> filter_channel_content_from_kv_cache;
  bool stream_tool_calls = false;
  std::string stream_tool_calls_channel_name = "tool_call";
  std::optional<litert::lm::ThinkingConfig> thinking_config;
  std::optional<LiteRtLmConstraintProviderType> constraint_provider_type;
};

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_CONVERSATION_INTERNAL_H_
