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

#include <initializer_list>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json

namespace litert::lm {

struct RawBlob {
  std::vector<uint8_t> bytes;
  std::string mime_type;
};

struct FilePath {
  std::string path;
};

struct PcmData {
  std::vector<float> frames;
};

struct TextPart {
  std::string text;
  std::string error;
  nlohmann::ordered_json extra_fields;
};

struct ImagePart {
  std::variant<RawBlob, FilePath> data;
};

struct AudioPart {
  std::variant<RawBlob, FilePath, PcmData> data;
};

struct FunctionCallPart {
  std::string name;
  nlohmann::ordered_json arguments;
  std::string id;
};

struct FunctionResponsePart {
  std::string name;
  std::string response;
  std::string id;
};

struct JsonPart {
  nlohmann::ordered_json json_data;
};

using Part = std::variant<TextPart, ImagePart, AudioPart, FunctionCallPart,
                          FunctionResponsePart, JsonPart>;

struct Message {
  std::string role;
  std::vector<Part> parts;
  absl::flat_hash_map<std::string, std::string> channels;
  bool content_is_string = false;
  bool content_is_object = false;

  // Enable implicit conversion to json
  explicit operator nlohmann::ordered_json() const;

  // Enable implicit construction from json
  Message() = default;
  Message(std::string role, std::vector<Part> parts)
      : role(std::move(role)), parts(std::move(parts)) {}
  explicit Message(const nlohmann::ordered_json& j);
  Message(std::initializer_list<std::pair<std::string, nlohmann::ordered_json>>
              init);

  // Check if message is empty/uninitialized
  bool empty() const;

  // Dump message as JSON string
  std::string dump() const;

  // Parse message from JSON string
  static Message parse(const std::string& json_str);
};

// JSON Serialization hooks
void to_json(nlohmann::ordered_json& j, const TextPart& p);
void to_json(nlohmann::ordered_json& j, const ImagePart& p);
void to_json(nlohmann::ordered_json& j, const AudioPart& p);
void to_json(nlohmann::ordered_json& j, const FunctionCallPart& p);
void to_json(nlohmann::ordered_json& j, const FunctionResponsePart& p);
void to_json(nlohmann::ordered_json& j, const Part& p);
void to_json(nlohmann::ordered_json& j, const Message& m);

nlohmann::ordered_json PartToJson(const Part& p, bool include_blobs = true);
nlohmann::ordered_json MessageToJson(const Message& m,
                                     bool include_blobs = true);

// JSON Deserialization hooks
void from_json(const nlohmann::ordered_json& j, TextPart& p);
void from_json(const nlohmann::ordered_json& j, ImagePart& p);
void from_json(const nlohmann::ordered_json& j, AudioPart& p);
void from_json(const nlohmann::ordered_json& j, FunctionCallPart& p);
void from_json(const nlohmann::ordered_json& j, FunctionResponsePart& p);
void from_json(const nlohmann::ordered_json& j, Part& p);
void from_json(const nlohmann::ordered_json& j, Message& m);

// Equality operators
bool operator==(const RawBlob& lhs, const RawBlob& rhs);
bool operator==(const FilePath& lhs, const FilePath& rhs);
bool operator==(const PcmData& lhs, const PcmData& rhs);
bool operator==(const TextPart& lhs, const TextPart& rhs);
bool operator==(const ImagePart& lhs, const ImagePart& rhs);
bool operator==(const AudioPart& lhs, const AudioPart& rhs);
bool operator==(const FunctionCallPart& lhs, const FunctionCallPart& rhs);
bool operator==(const FunctionResponsePart& lhs,
                const FunctionResponsePart& rhs);
bool operator==(const JsonPart& lhs, const JsonPart& rhs);
bool operator==(const Message& lhs, const Message& rhs);

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
