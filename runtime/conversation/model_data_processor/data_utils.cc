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
#include <utility>
#include <variant>

#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/escaping.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/conversation/io_types.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/status_macros.h"
#include "re2/re2.h"  // from @com_googlesource_code_re2

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
      if (item["blob"].is_binary()) {
        const auto& blob = item["blob"].get_binary();
        return InMemoryFile::Create(std::string(
            reinterpret_cast<const char*>(blob.data()), blob.size()));
      }
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
    if (content.get<std::string>().empty()) {
      return ordered_json::array();
    }
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
    ParserOptions options_with_tools = options;
    if (options_with_tools.regex_options.has_value() &&
        options_with_tools.regex_options->tools.empty()) {
      options_with_tools.regex_options->tools =
          std::get<JsonPreface>(*preface).tools;
    }
    ABSL_ASSIGN_OR_RETURN(
        ordered_json content_and_tool_calls,
        ParseTextAndToolCalls(response_text, code_fence_start, code_fence_end,
                              syntax_type, options_with_tools));
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

absl::StatusOr<ordered_json> FormatToolCalls(
    const ordered_json& tool_calls,
    absl::FunctionRef<absl::StatusOr<std::string>(const ordered_json&)>
        format_value) {
  if (!tool_calls.is_array()) {
    return absl::InvalidArgumentError("Tool calls must be an array.");
  }
  ordered_json formatted_tool_calls = ordered_json::array();
  for (const auto& tool_call : tool_calls) {
    if (!tool_call.is_object() || !tool_call.contains("function")) {
      continue;
    }
    const ordered_json& function = tool_call["function"];
    if (!function.is_object() || !function.contains("name")) {
      continue;
    }
    ordered_json tool_call_input = ordered_json::object();
    tool_call_input["type"] = "function";
    tool_call_input["function"]["name"] = function["name"];

    if (function.contains("arguments")) {
      if (function["arguments"].is_object()) {
        for (const auto& [key, value] : function["arguments"].items()) {
          ABSL_ASSIGN_OR_RETURN(std::string formatted_value,
                                format_value(value));
          tool_call_input["function"]["arguments"][key] = formatted_value;
        }
      } else {
        tool_call_input["function"]["arguments"] = function["arguments"];
      }
    }

    formatted_tool_calls.push_back(tool_call_input);
  }
  return formatted_tool_calls;
}

absl::StatusOr<ordered_json> FormatToolsArray(
    const ordered_json& tools,
    absl::FunctionRef<absl::StatusOr<std::string>(const ordered_json&)>
        format_tool) {
  if (!tools.is_array()) {
    return absl::InvalidArgumentError("Tools must be an array.");
  }
  ordered_json formatted_tools = ordered_json::array();
  for (const auto& tool : tools) {
    ABSL_ASSIGN_OR_RETURN(std::string formatted_tool, format_tool(tool));
    formatted_tools.push_back(formatted_tool);
  }
  return formatted_tools;
}

namespace {

constexpr absl::string_view kMultimodalEscapePrefix = "<<__litert_mm_esc__:";
constexpr absl::string_view kMultimodalEscapePrefixUpper =
    "<<__LITERT_MM_ESC__:";
constexpr absl::string_view kMultimodalEscapeSuffix = ">>";

std::string WrapEscapedBytes(absl::string_view raw) {
  return absl::StrCat(kMultimodalEscapePrefix, absl::BytesToHexString(raw),
                      kMultimodalEscapeSuffix);
}

bool IsMediaItemObject(const ordered_json& obj) {
  if (!obj.is_object() || !obj.contains("type") || !obj["type"].is_string()) {
    return false;
  }
  const std::string& type = obj["type"].get_ref<const std::string&>();
  return type == "image" || type == "audio";
}

bool IsProtectedStructuralField(absl::string_view key, absl::string_view val,
                                bool skip_media_items) {
  if (!skip_media_items) {
    return false;
  }
  if (key == "role") {
    return true;
  }
  if (key == "type" &&
      (val == "text" || val == "tool_response" || val == "function")) {
    return true;
  }
  return false;
}

bool StringNeedsEscaping(absl::string_view text, const RE2& delimiter_re) {
  if (absl::StrContains(text, kMultimodalEscapePrefix) ||
      absl::StrContains(text, kMultimodalEscapePrefixUpper)) {
    return true;
  }
  absl::string_view match;
  return delimiter_re.Match(text, 0, text.size(), RE2::UNANCHORED, &match, 1) &&
         !match.empty();
}

bool EscapeStringInPlace(std::string& text, const RE2& delimiter_re) {
  if (!StringNeedsEscaping(text, delimiter_re)) {
    return false;
  }
  std::string out;
  out.reserve(text.size() + 32);
  absl::string_view remaining = text;
  while (!remaining.empty()) {
    size_t best_pos = absl::string_view::npos;
    size_t best_len = 0;

    const size_t lower_pos = remaining.find(kMultimodalEscapePrefix);
    if (lower_pos != absl::string_view::npos) {
      best_pos = lower_pos;
      best_len = kMultimodalEscapePrefix.size();
    }

    const size_t upper_pos = remaining.find(kMultimodalEscapePrefixUpper);
    if (upper_pos != absl::string_view::npos && upper_pos < best_pos) {
      best_pos = upper_pos;
      best_len = kMultimodalEscapePrefixUpper.size();
    }

    absl::string_view delim_match;
    if (delimiter_re.Match(remaining, 0, remaining.size(), RE2::UNANCHORED,
                           &delim_match, 1) &&
        !delim_match.empty()) {
      const size_t delim_pos =
          static_cast<size_t>(delim_match.data() - remaining.data());
      if (delim_pos < best_pos ||
          (delim_pos == best_pos && delim_match.size() > best_len)) {
        best_pos = delim_pos;
        best_len = delim_match.size();
      }
    }

    if (best_pos == absl::string_view::npos) {
      out.append(remaining.data(), remaining.size());
      break;
    }

    out.append(remaining.data(), best_pos);
    out.append(WrapEscapedBytes(remaining.substr(best_pos, best_len)));
    remaining.remove_prefix(best_pos + best_len);
  }
  text = std::move(out);
  return true;
}

const std::string* GetTextItemString(const ordered_json& item) {
  if (item.is_string()) {
    return &item.get_ref<const std::string&>();
  }
  if (item.is_object() && item.contains("type") && item["type"].is_string() &&
      item["type"].get_ref<const std::string&>() == "text" &&
      item.contains("text") && item["text"].is_string()) {
    return &item["text"].get_ref<const std::string&>();
  }
  return nullptr;
}

std::string* GetMutableTextItemString(ordered_json& item) {
  if (item.is_string()) {
    return &item.get_ref<std::string&>();
  }
  if (item.is_object() && item.contains("type") && item["type"].is_string() &&
      item["type"].get_ref<const std::string&>() == "text" &&
      item.contains("text") && item["text"].is_string()) {
    return &item["text"].get_ref<std::string&>();
  }
  return nullptr;
}

size_t FindFirstUnclosedAngleBracket(absl::string_view text) {
  const absl::string_view trimmed = absl::StripTrailingAsciiWhitespace(text);
  const size_t last_gt = trimmed.rfind('>');
  return last_gt == absl::string_view::npos ? trimmed.find('<')
                                            : trimmed.find('<', last_gt + 1);
}

bool NeedsEscapingImpl(const ordered_json& value, const RE2& delimiter_re,
                       bool skip_media_items) {
  if (value.is_string()) {
    return StringNeedsEscaping(value.get_ref<const std::string&>(),
                               delimiter_re);
  }
  if (value.is_array()) {
    for (size_t i = 0; i < value.size(); ++i) {
      if (NeedsEscapingImpl(value[i], delimiter_re, skip_media_items)) {
        return true;
      }
      if (i + 1 < value.size()) {
        const std::string* curr_text = GetTextItemString(value[i]);
        const std::string* next_text = GetTextItemString(value[i + 1]);
        if (curr_text != nullptr && next_text != nullptr &&
            FindFirstUnclosedAngleBracket(*curr_text) !=
                absl::string_view::npos) {
          return true;
        }
      }
    }
    return false;
  }
  if (value.is_object()) {
    if (skip_media_items && IsMediaItemObject(value)) {
      return false;
    }
    for (const auto& [key, val] : value.items()) {
      if (StringNeedsEscaping(key, delimiter_re)) {
        return true;
      }
      if (val.is_string() &&
          IsProtectedStructuralField(key, val.get_ref<const std::string&>(),
                                     skip_media_items)) {
        continue;
      }
      if (NeedsEscapingImpl(val, delimiter_re, skip_media_items)) {
        return true;
      }
    }
  }
  return false;
}

void EscapeJsonImpl(ordered_json& value, const RE2& delimiter_re,
                    bool skip_media_items) {
  if (value.is_string()) {
    EscapeStringInPlace(value.get_ref<std::string&>(), delimiter_re);
    return;
  }
  if (value.is_array()) {
    for (auto& elem : value) {
      EscapeJsonImpl(elem, delimiter_re, skip_media_items);
    }
    for (size_t i = 0; i + 1 < value.size(); ++i) {
      std::string* curr_text = GetMutableTextItemString(value[i]);
      if (curr_text == nullptr ||
          GetMutableTextItemString(value[i + 1]) == nullptr) {
        continue;
      }
      const size_t first_lt = FindFirstUnclosedAngleBracket(*curr_text);
      if (first_lt == absl::string_view::npos) {
        continue;
      }
      const size_t trimmed_len =
          absl::StripTrailingAsciiWhitespace(*curr_text).size();
      std::string escaped;
      escaped.reserve(curr_text->size() + 32);
      escaped.append(curr_text->data(), first_lt);
      for (size_t pos = first_lt; pos < trimmed_len; ++pos) {
        if ((*curr_text)[pos] == '<') {
          escaped.append(WrapEscapedBytes("<"));
        } else {
          escaped.push_back((*curr_text)[pos]);
        }
      }
      escaped.append(curr_text->data() + trimmed_len,
                     curr_text->size() - trimmed_len);
      *curr_text = std::move(escaped);
    }
    return;
  }
  if (value.is_object()) {
    if (skip_media_items && IsMediaItemObject(value)) {
      return;
    }
    bool any_key_changed = false;
    for (auto& [key, val] : value.items()) {
      if (!any_key_changed && StringNeedsEscaping(key, delimiter_re)) {
        any_key_changed = true;
      }
      if (val.is_string() &&
          IsProtectedStructuralField(key, val.get_ref<const std::string&>(),
                                     skip_media_items)) {
        continue;
      }
      EscapeJsonImpl(val, delimiter_re, skip_media_items);
    }
    if (any_key_changed) {
      ordered_json rebuilt = ordered_json::object();
      for (auto& [key, val] : value.items()) {
        std::string clean_key = key;
        EscapeStringInPlace(clean_key, delimiter_re);
        rebuilt[std::move(clean_key)] = std::move(val);
      }
      value = std::move(rebuilt);
    }
  }
}

}  // namespace

bool NeedsMultimodalDelimiterEscaping(const ordered_json& value,
                                      absl::string_view delimiter_regex,
                                      bool skip_media_items) {
  if (delimiter_regex.empty()) {
    return false;
  }
  RE2 delimiter_re(delimiter_regex);
  if (!delimiter_re.ok()) {
    return false;
  }
  return NeedsEscapingImpl(value, delimiter_re, skip_media_items);
}

void EscapeMultimodalDelimitersInJson(ordered_json& value,
                                      absl::string_view delimiter_regex,
                                      bool skip_media_items) {
  if (delimiter_regex.empty()) {
    return;
  }
  RE2 delimiter_re(delimiter_regex);
  if (!delimiter_re.ok()) {
    return;
  }
  EscapeJsonImpl(value, delimiter_re, skip_media_items);
}

std::string UnescapeMultimodalDelimiters(absl::string_view text) {
  if (!absl::StrContains(text, kMultimodalEscapePrefix) &&
      !absl::StrContains(text, kMultimodalEscapePrefixUpper)) {
    return std::string(text);
  }
  static const LazyRE2 kUnescapeRe = {
      R"regex(<<__litert_mm_esc__:((?:[0-9a-f]{2})+)>>|<<__LITERT_MM_ESC__:((?:[0-9A-F]{2})+)>>)regex"};
  std::string out;
  out.reserve(text.size());
  absl::string_view remaining = text;
  absl::string_view submatches[3];
  while (kUnescapeRe->Match(remaining, 0, remaining.size(), RE2::UNANCHORED,
                            submatches, 3)) {
    const absl::string_view full_match = submatches[0];
    const absl::string_view lower_hex = submatches[1];
    const absl::string_view upper_hex = submatches[2];
    const size_t pos =
        static_cast<size_t>(full_match.data() - remaining.data());
    out.append(remaining.data(), pos);
    if (!lower_hex.empty()) {
      out.append(absl::HexStringToBytes(lower_hex));
    } else {
      std::string decoded = absl::HexStringToBytes(upper_hex);
      absl::AsciiStrToUpper(&decoded);
      out.append(decoded);
    }
    remaining.remove_prefix(pos + full_match.size());
  }
  out.append(remaining.data(), remaining.size());
  return out;
}

}  // namespace litert::lm
