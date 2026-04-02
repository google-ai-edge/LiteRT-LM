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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_IO_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_IO_TYPES_H_

#include <ostream>
#include <string>
#include <utility>
#include <variant>

#include "nlohmann/json.hpp"  // from @nlohmann_json

namespace litert::lm {

class Message : public nlohmann::ordered_json {
 public:
  using nlohmann::ordered_json::ordered_json;
  Message() = default;
  explicit Message(const nlohmann::ordered_json& json)
      : nlohmann::ordered_json(json) {}
  explicit Message(nlohmann::ordered_json&& json)
      : nlohmann::ordered_json(std::move(json)) {}
};

std::ostream& operator<<(std::ostream& os, const Message& message);

struct JsonPreface {
  // The messages in the preface. The messages provided the initial background
  // for the conversation. For example, the messages can be the conversation
  // history, prompt engineering instructions, few-shot examples, etc.
  nlohmann::ordered_json messages;
  // The tools able to be used by the model in the conversation.
  nlohmann::ordered_json tools;
  // The extra context that is not part of the messages or tools. This is can be
  // extended by the model to support other features. For example, configurable
  // template rendering or other model-specific features.
  nlohmann::ordered_json extra_context;
};

// Definition of a channel for responses, e.g. thinking channel.
struct Channel {
  // The channel name. Text from this channel will be written to
  // message["channels"][channel_name].
  std::string channel_name;

  // A string that marks the start of the channel, e.g. "<|channel>thought".
  std::string start;

  // A string that marks the end of the channel, e.g. "<channel|>".
  std::string end;
};

// Preface is the initial messages, tools and extra context for the
// conversation to begin with. It provides the initial background for the
// conversation.
using Preface = std::variant<JsonPreface>;

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_IO_TYPES_H_
