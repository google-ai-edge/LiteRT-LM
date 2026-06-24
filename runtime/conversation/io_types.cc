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

#include "runtime/conversation/io_types.h"

#include <initializer_list>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/escaping.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json

namespace litert::lm {

Message::operator nlohmann::ordered_json() const {
  nlohmann::ordered_json j;
  to_json(j, *this);
  return j;
}

Message::Message(const nlohmann::ordered_json& j) { from_json(j, *this); }

Message::Message(
    std::initializer_list<std::pair<std::string, nlohmann::ordered_json>>
        init) {
  nlohmann::ordered_json j = nlohmann::ordered_json::object();
  for (const auto& [key, val] : init) {
    j[key] = val;
  }
  from_json(j, *this);
}

bool Message::empty() const {
  return role.empty() && parts.empty() && channels.empty();
}

std::string Message::dump() const {
  return nlohmann::ordered_json(*this).dump();
}

Message Message::parse(const std::string& json_str) {
  return Message(nlohmann::ordered_json::parse(json_str));
}

// JSON Serialization hooks
void to_json(nlohmann::ordered_json& j, const TextPart& p) {
  j = p.extra_fields;
  if (!j.is_object()) {
    j = nlohmann::ordered_json::object();
  }
  j["type"] = "text";
  j["text"] = p.text;
  if (!p.error.empty()) {
    j["error"] = p.error;
  }
}

void to_json(nlohmann::ordered_json& j, const ImagePart& p) {
  j = nlohmann::ordered_json{{"type", "image"}};
  std::visit(
      [&j](const auto& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, FilePath>) {
          j["path"] = arg.path;
        } else if constexpr (std::is_same_v<T, RawBlob>) {
          std::string b64;
          absl::Base64Escape(
              std::string(arg.bytes.begin(), arg.bytes.end()), &b64);
          j["blob"] = b64;
        }
      },
      p.data);
}

void to_json(nlohmann::ordered_json& j, const AudioPart& p) {
  j = nlohmann::ordered_json{{"type", "audio"}};
  std::visit(
      [&j](const auto& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, FilePath>) {
          j["path"] = arg.path;
        } else if constexpr (std::is_same_v<T, RawBlob>) {
          std::string b64;
          absl::Base64Escape(
              std::string(arg.bytes.begin(), arg.bytes.end()), &b64);
          j["blob"] = b64;
        } else if constexpr (std::is_same_v<T, PcmData>) {
          // Do not serialize raw PCM to JSON.
        }
      },
      p.data);
}

void to_json(nlohmann::ordered_json& j, const FunctionCallPart& p) {
  j = nlohmann::ordered_json{
      {"type", "function"},
      {"function",
       nlohmann::ordered_json{{"name", p.name}, {"arguments", p.arguments}}}};
  if (!p.id.empty()) {
    j["id"] = p.id;
  }
}

void to_json(nlohmann::ordered_json& j, const FunctionResponsePart& p) {
  j = nlohmann::ordered_json{{"type", "function_response"},
                             {"name", p.name},
                             {"response", p.response}};
  if (!p.id.empty()) {
    j["id"] = p.id;
  }
}

void to_json(nlohmann::ordered_json& j, const Part& p) {
  j = PartToJson(p, /*include_blobs=*/true);
}

void to_json(nlohmann::ordered_json& j, const Message& m) {
  j = MessageToJson(m, /*include_blobs=*/true);
}

nlohmann::ordered_json PartToJson(const Part& p, bool include_blobs) {
  nlohmann::ordered_json j;
  if (!include_blobs) {
    if (std::holds_alternative<ImagePart>(p)) {
      const auto& ip = std::get<ImagePart>(p);
      if (std::holds_alternative<RawBlob>(ip.data)) {
        return nlohmann::ordered_json{{"type", "image"}};
      }
    } else if (std::holds_alternative<AudioPart>(p)) {
      const auto& ap = std::get<AudioPart>(p);
      if (std::holds_alternative<RawBlob>(ap.data) ||
          std::holds_alternative<PcmData>(ap.data)) {
        return nlohmann::ordered_json{{"type", "audio"}};
      }
    }
  }
  std::visit(
      [&j](const auto& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, JsonPart>) {
          j = arg.json_data;
        } else {
          to_json(j, arg);
        }
      },
      p);
  return j;
}

nlohmann::ordered_json MessageToJson(const Message& m, bool include_blobs) {
  nlohmann::ordered_json j = nlohmann::ordered_json{{"role", m.role}};
  auto content_array = nlohmann::ordered_json::array();
  auto tool_calls_array = nlohmann::ordered_json::array();

  for (const auto& part : m.parts) {
    if (std::holds_alternative<FunctionCallPart>(part)) {
      nlohmann::ordered_json tc_j;
      to_json(tc_j, std::get<FunctionCallPart>(part));
      tool_calls_array.push_back(tc_j);
    } else {
      nlohmann::ordered_json part_j = PartToJson(part, include_blobs);
      content_array.push_back(part_j);
    }
  }

  if (!content_array.empty()) {
    if (m.content_is_string && content_array.size() == 1 &&
        std::holds_alternative<TextPart>(m.parts[0])) {
      j["content"] = std::get<TextPart>(m.parts[0]).text;
    } else if (m.content_is_object && content_array.size() == 1) {
      j["content"] = content_array[0];
    } else {
      j["content"] = content_array;
    }
  }
  if (!tool_calls_array.empty()) {
    j["tool_calls"] = tool_calls_array;
  }
  if (!m.channels.empty()) {
    auto channels_j = nlohmann::ordered_json::object();
    for (const auto& [name, val] : m.channels) {
      channels_j[name] = val;
    }
    j["channels"] = channels_j;
  }
  return j;
}

// JSON Deserialization hooks
void from_json(const nlohmann::ordered_json& j, TextPart& p) {
  p.text = j.at("text").get<std::string>();
  if (j.contains("error")) {
    p.error = j.at("error").get<std::string>();
  }
  p.extra_fields = j;
  p.extra_fields.erase("type");
  p.extra_fields.erase("text");
  p.extra_fields.erase("error");
}

void from_json(const nlohmann::ordered_json& j, ImagePart& p) {
  if (j.contains("path")) {
    p.data = FilePath{j.at("path").get<std::string>()};
  } else if (j.contains("blob")) {
    std::string blob_b64 = j.at("blob").get<std::string>();
    std::string blob;
    if (absl::Base64Unescape(blob_b64, &blob)) {
      p.data = RawBlob{.bytes = std::vector<uint8_t>(blob.begin(), blob.end())};
    }
  }
}

void from_json(const nlohmann::ordered_json& j, AudioPart& p) {
  if (j.contains("path")) {
    p.data = FilePath{j.at("path").get<std::string>()};
  } else if (j.contains("blob")) {
    std::string blob_b64 = j.at("blob").get<std::string>();
    std::string blob;
    if (absl::Base64Unescape(blob_b64, &blob)) {
      p.data = RawBlob{.bytes = std::vector<uint8_t>(blob.begin(), blob.end())};
    }
  }
}

void from_json(const nlohmann::ordered_json& j, FunctionCallPart& p) {
  if (j.contains("function")) {
    const auto& fn = j.at("function");
    p.name = fn.at("name").get<std::string>();
    if (fn.contains("arguments")) {
      p.arguments = fn.at("arguments");
    }
  }
  if (j.contains("id")) {
    p.id = j.at("id").get<std::string>();
  }
}

void from_json(const nlohmann::ordered_json& j, FunctionResponsePart& p) {
  p.name = j.at("name").get<std::string>();
  p.response = j.at("response").get<std::string>();
  if (j.contains("id")) {
    p.id = j.at("id").get<std::string>();
  }
}

void from_json(const nlohmann::ordered_json& j, Part& p) {
  if (!j.contains("type")) {
    p = JsonPart{j};
    return;
  }
  std::string type = j.at("type").get<std::string>();
  if (type == "text") {
    TextPart tp;
    from_json(j, tp);
    p = tp;
  } else if (type == "image") {
    ImagePart ip;
    from_json(j, ip);
    p = ip;
  } else if (type == "audio") {
    AudioPart ap;
    from_json(j, ap);
    p = ap;
  } else if (type == "function") {
    FunctionCallPart fcp;
    from_json(j, fcp);
    p = fcp;
  } else if (type == "function_response") {
    FunctionResponsePart frp;
    from_json(j, frp);
    p = frp;
  } else {
    p = JsonPart{j};
  }
}

void from_json(const nlohmann::ordered_json& j, Message& m) {
  m.role = j.at("role").get<std::string>();
  m.content_is_string = false;
  m.content_is_object = false;
  if (j.contains("content")) {
    const auto& content = j.at("content");
    if (content.is_string()) {
      m.parts.push_back(TextPart{content.get<std::string>()});
      m.content_is_string = true;
    } else if (content.is_array()) {
      for (const auto& item : content) {
        Part p;
        from_json(item, p);
        m.parts.push_back(p);
      }
    } else if (content.is_object()) {
      Part p;
      from_json(content, p);
      m.parts.push_back(p);
      m.content_is_object = true;
    }
  }
  if (j.contains("tool_calls")) {
    const auto& tool_calls_j = j.at("tool_calls");
    if (tool_calls_j.is_array()) {
      for (const auto& tc_j : tool_calls_j) {
        FunctionCallPart fcp;
        from_json(tc_j, fcp);
        m.parts.push_back(fcp);
      }
    }
  } else if (j.contains("type") && j.at("type").is_string() &&
             j.at("type").get<std::string>() == "function" &&
             j.contains("function")) {
    FunctionCallPart fcp;
    from_json(j, fcp);
    m.parts.push_back(fcp);
  }
  if (j.contains("channels")) {
    const auto& channels_j = j.at("channels");
    if (channels_j.is_object()) {
      for (const auto& [name, val] : channels_j.items()) {
        m.channels[name] = val.get<std::string>();
      }
    }
  }
}

std::ostream& operator<<(std::ostream& os, const Message& message) {
  os << nlohmann::ordered_json(message).dump();
  return os;
}

bool operator==(const RawBlob& lhs, const RawBlob& rhs) {
  return lhs.bytes == rhs.bytes && lhs.mime_type == rhs.mime_type;
}

bool operator==(const FilePath& lhs, const FilePath& rhs) {
  return lhs.path == rhs.path;
}

bool operator==(const PcmData& lhs, const PcmData& rhs) {
  return lhs.frames == rhs.frames;
}

bool operator==(const TextPart& lhs, const TextPart& rhs) {
  return lhs.text == rhs.text && lhs.error == rhs.error &&
         lhs.extra_fields == rhs.extra_fields;
}

bool operator==(const ImagePart& lhs, const ImagePart& rhs) {
  return lhs.data == rhs.data;
}

bool operator==(const AudioPart& lhs, const AudioPart& rhs) {
  return lhs.data == rhs.data;
}

bool operator==(const FunctionCallPart& lhs, const FunctionCallPart& rhs) {
  return lhs.name == rhs.name && lhs.arguments == rhs.arguments &&
         lhs.id == rhs.id;
}

bool operator==(const FunctionResponsePart& lhs,
                const FunctionResponsePart& rhs) {
  return lhs.name == rhs.name && lhs.response == rhs.response &&
         lhs.id == rhs.id;
}

bool operator==(const JsonPart& lhs, const JsonPart& rhs) {
  return lhs.json_data == rhs.json_data;
}

bool operator==(const Message& lhs, const Message& rhs) {
  return lhs.role == rhs.role && lhs.parts == rhs.parts &&
         lhs.channels == rhs.channels &&
         lhs.content_is_string == rhs.content_is_string &&
         lhs.content_is_object == rhs.content_is_object;
}

}  // namespace litert::lm
