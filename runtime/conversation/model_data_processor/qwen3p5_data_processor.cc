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

#include "runtime/conversation/model_data_processor/qwen3p5_data_processor.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/qwen3p5_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "re2/re2.h"  // from @com_googlesource_code_re2

namespace litert::lm {
namespace {

using ::nlohmann::ordered_json;

ordered_json TextPart(absl::string_view text) {
  return {{"type", "text"}, {"text", std::string(text)}};
}

// Returns the "properties" of the parameters of the tool named `name`, or
// nullptr if the tool or its properties are not declared. A tool is either
// {"type": "function", "function": {"name": ...}} or {"name": ...}.
const ordered_json* FindToolProperties(const ordered_json& tools,
                                       absl::string_view name) {
  for (const ordered_json& tool : tools) {
    const ordered_json& function =
        tool.contains("function") ? tool["function"] : tool;
    const auto name_it = function.find("name");
    if (name_it == function.end() || !name_it->is_string() ||
        name_it->get_ref<const std::string&>() != name) {
      continue;
    }
    const auto parameters_it = function.find("parameters");
    if (parameters_it == function.end()) {
      return nullptr;
    }
    const auto properties_it = parameters_it->find("properties");
    if (properties_it == parameters_it->end()) {
      return nullptr;
    }
    return &*properties_it;
  }
  return nullptr;
}

// The JSON schema types that a parameter allows.
struct AllowedTypes {
  bool string = false;
  bool boolean = false;
  bool null = false;
  // integer, number, object, array or null.
  bool json = false;
};

// Adds the type named `name` to `allowed`. Names are compared ignoring ASCII
// case; an unknown name adds nothing.
void AddTypeName(absl::string_view name, AllowedTypes& allowed) {
  if (absl::EqualsIgnoreCase(name, "string")) {
    allowed.string = true;
  } else if (absl::EqualsIgnoreCase(name, "boolean")) {
    allowed.boolean = true;
  } else if (absl::EqualsIgnoreCase(name, "null")) {
    allowed.null = true;
    allowed.json = true;
  } else if (absl::EqualsIgnoreCase(name, "integer") ||
             absl::EqualsIgnoreCase(name, "number") ||
             absl::EqualsIgnoreCase(name, "object") ||
             absl::EqualsIgnoreCase(name, "array")) {
    allowed.json = true;
  }
}

// Adds the types in the "type" of `schema`, a name or an array of names.
void AddSchemaTypes(const ordered_json& schema, AllowedTypes& allowed) {
  const auto type_it = schema.find("type");
  if (type_it == schema.end()) {
    return;
  }
  if (type_it->is_string()) {
    AddTypeName(type_it->get_ref<const std::string&>(), allowed);
    return;
  }
  if (type_it->is_array()) {
    for (const ordered_json& name : *type_it) {
      if (name.is_string()) {
        AddTypeName(name.get_ref<const std::string&>(), allowed);
      }
    }
  }
}

// Returns the types that parameter `key` allows in `properties` (nullptr when
// the tool declares none): the "type" of its schema or, without one, the
// "type" of each schema in its "anyOf" or "oneOf". Pydantic writes
// Optional[int] as {"anyOf": [{"type": "integer"}, {"type": "null"}]}. An
// undeclared parameter allows nothing.
AllowedTypes GetAllowedTypes(const ordered_json* properties,
                             const std::string& key) {
  AllowedTypes allowed;
  if (properties == nullptr) {
    return allowed;
  }
  const auto schema_it = properties->find(key);
  if (schema_it == properties->end()) {
    return allowed;
  }
  if (schema_it->contains("type")) {
    AddSchemaTypes(*schema_it, allowed);
    return allowed;
  }
  for (const char* keyword : {"anyOf", "oneOf"}) {
    const auto options_it = schema_it->find(keyword);
    if (options_it == schema_it->end() || !options_it->is_array()) {
      continue;
    }
    for (const ordered_json& option : *options_it) {
      AddSchemaTypes(option, allowed);
    }
  }
  return allowed;
}

// Converts a parameter value to its argument. The chat template writes objects
// and arrays as JSON and every other value with `| string`, which gives true,
// false and none in minijinja and True, False and None in Python's Jinja2 (the
// Hugging Face template). So when the type allows boolean or null, those words
// are read in any case. Otherwise a value whose type allows integer, number,
// object, array or null but not string is parsed as JSON; a boolean-only type
// is not, so 1 stays "1". A string or undeclared type, and a value that is not
// valid JSON, stay a string.
ordered_json ToArgument(absl::string_view value, const AllowedTypes& types) {
  if (types.boolean && absl::EqualsIgnoreCase(value, "true")) {
    return true;
  }
  if (types.boolean && absl::EqualsIgnoreCase(value, "false")) {
    return false;
  }
  if (types.null && (absl::EqualsIgnoreCase(value, "null") ||
                     absl::EqualsIgnoreCase(value, "none"))) {
    return nullptr;
  }
  if (types.json && !types.string) {
    ordered_json parsed = ordered_json::parse(value, /*cb=*/nullptr,
                                              /*allow_exceptions=*/false);
    if (!parsed.is_discarded()) {
      return parsed;
    }
  }
  return std::string(value);
}

// Parses one tool call block, the text between the code fences:
//   <function=NAME>
//   <parameter=KEY>
//   VALUE
//   </parameter>
//   ...
//   </function>
// Only whitespace, and "</parameter>" tags that close nothing, may surround the
// elements. NAME and KEY may contain spaces, as JSON schema property names can.
// One leading and one trailing line break ("\r\n" or "\n") of each value are
// dropped: the chat template writes them.
absl::StatusOr<ordered_json> ParseToolCallBlock(absl::string_view block,
                                                const ordered_json& tools) {
  // NAME, KEY: no ">" or line break, non-space first; outer spaces are dropped.
  static const LazyRE2 kFunctionRegex = {
      R"regex((?s)<function\s*=\s*([^>\s][^>\n]*?)\s*>(.*)</function>)regex"};
  // The value ends at the first "</parameter>" (the match is non-greedy), even
  // when the value itself contains that string.
  static const LazyRE2 kParameterRegex = {
      R"regex((?s)\s*<parameter\s*=\s*([^>\s][^>\n]*?)\s*>(.*?)</parameter>)regex"};
  static const LazyRE2 kParameterEndRegex = {R"regex(\s*</parameter>)regex"};

  std::string name;
  absl::string_view body;
  if (!RE2::FullMatch(absl::StripAsciiWhitespace(block), *kFunctionRegex, &name,
                      &body)) {
    return absl::InvalidArgumentError(
        "expected one <function=NAME>...</function> element");
  }
  const ordered_json* properties = FindToolProperties(tools, name);
  ordered_json arguments = ordered_json::object();
  std::string key;
  absl::string_view value;
  while (true) {
    if (RE2::Consume(&body, *kParameterRegex, &key, &value)) {
      // "<parameter" in the value: this parameter is not closed before the
      // next one, whose key and value would be lost inside this value.
      if (absl::StrContains(value, "<parameter")) {
        return absl::InvalidArgumentError(
            "expected </parameter> before the next <parameter=KEY>");
      }
      if (!absl::ConsumePrefix(&value, "\r\n")) {
        absl::ConsumePrefix(&value, "\n");
      }
      if (!absl::ConsumeSuffix(&value, "\r\n")) {
        absl::ConsumeSuffix(&value, "\n");
      }
      // A repeated parameter keeps its last value.
      arguments[key] = ToArgument(value, GetAllowedTypes(properties, key));
      continue;
    }
    // Qwen3.5-0.8B may write a "</parameter>" that closes nothing; skip it.
    if (!RE2::Consume(&body, *kParameterEndRegex)) {
      break;
    }
  }
  if (!absl::StripAsciiWhitespace(body).empty()) {
    return absl::InvalidArgumentError(
        "expected only closed <parameter=KEY>...</parameter> elements inside "
        "<function=NAME>...</function>");
  }
  return ordered_json{
      {"type", "function"},
      {"function", {{"name", name}, {"arguments", std::move(arguments)}}}};
}

}  // namespace

Qwen3p5DataProcessor::Qwen3p5DataProcessor(Qwen3p5DataProcessorConfig config,
                                           std::optional<Preface> preface,
                                           std::unique_ptr<RE2> fence_regex)
    : config_(std::move(config)),
      preface_(std::move(preface)),
      fence_regex_(std::move(fence_regex)) {}

Qwen3p5DataProcessor::~Qwen3p5DataProcessor() = default;

absl::StatusOr<std::unique_ptr<ModelDataProcessor>>
Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig config,
                             std::optional<Preface> preface) {
  // The fences delimit the blocks, so neither may be empty.
  if (config.code_fence_start.empty() || config.code_fence_end.empty()) {
    return absl::InvalidArgumentError(
        "code_fence_start and code_fence_end must not be empty.");
  }
  // (text) <start> (block) <end>, as in ParseTextAndToolCalls; (?s) lets the
  // text and the block span lines.
  auto fence_regex = std::make_unique<RE2>(
      config.escape_fence_strings
          ? absl::StrCat("(?s)(.*?)", RE2::QuoteMeta(config.code_fence_start),
                         "(.*?)", RE2::QuoteMeta(config.code_fence_end))
          : absl::StrCat("(?s)(.*?)", config.code_fence_start, "(.*?)",
                         config.code_fence_end));
  if (!fence_regex->ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid regex: ", fence_regex->pattern(),
                     " error: ", fence_regex->error()));
  }
  return absl::WrapUnique(new Qwen3p5DataProcessor(
      std::move(config), std::move(preface), std::move(fence_regex)));
}

absl::StatusOr<Message> Qwen3p5DataProcessor::ToMessageImpl(
    const Responses& responses,
    const Qwen3p5DataProcessorArguments& args) const {
  absl::string_view response_text = responses.GetTexts()[0];
  // Without tools the response is one text part; so is an empty response, as
  // in ParseTextAndToolCalls.
  if (response_text.empty() || !preface_.has_value() ||
      !std::holds_alternative<JsonPreface>(*preface_) ||
      std::get<JsonPreface>(*preface_).tools.empty()) {
    return ordered_json{
        {"role", "assistant"},
        {"content", ordered_json::array({TextPart(response_text)})}};
  }
  const ordered_json& tools = std::get<JsonPreface>(*preface_).tools;

  ordered_json content = ordered_json::array();
  ordered_json tool_calls = ordered_json::array();
  absl::string_view remaining = response_text;
  absl::string_view text;
  absl::string_view block;
  size_t remaining_size = remaining.size();
  while (RE2::Consume(&remaining, *fence_regex_, &text, &block)) {
    // A fence regex that can match empty text would not move on.
    if (remaining.size() == remaining_size) {
      return absl::InvalidArgumentError("fence regex matched an empty string");
    }
    remaining_size = remaining.size();
    if (!text.empty()) {
      content.push_back(TextPart(text));
    }
    // An empty block has no call and is skipped, as in ParseTextAndToolCalls.
    if (block.empty()) {
      continue;
    }
    absl::StatusOr<ordered_json> tool_call = ParseToolCallBlock(block, tools);
    if (tool_call.ok()) {
      tool_calls.push_back(*std::move(tool_call));
      continue;
    }
    if (ReturnErrorOnParseFailure()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to parse tool calls from code block: ", block,
                       "\nfull response: ", response_text,
                       "\nerror: ", tool_call.status().message()));
    }
    ordered_json part = TextPart(block);
    part["error"] = absl::StrCat("Failed to parse tool calls: ",
                                 tool_call.status().message());
    content.push_back(std::move(part));
  }
  // The text after the last block.
  if (!remaining.empty()) {
    content.push_back(TextPart(remaining));
  }

  ordered_json message = {{"role", "assistant"}};
  if (!content.empty()) {
    message["content"] = std::move(content);
  }
  if (!tool_calls.empty()) {
    message["tool_calls"] = std::move(tool_calls);
  }
  return message;
}

absl::string_view Qwen3p5DataProcessor::CodeFenceStart() const {
  return config_.code_fence_start;
}

absl::string_view Qwen3p5DataProcessor::CodeFenceEnd() const {
  return config_.code_fence_end;
}

}  // namespace litert::lm
