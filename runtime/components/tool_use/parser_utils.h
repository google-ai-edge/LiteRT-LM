// Copyright 2025 The Google AI Edge Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_PARSER_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_PARSER_UTILS_H_

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json

namespace litert::lm {

// The syntax type of the tool calls.
enum class SyntaxType {
  kUnknown = 0,
  kPython = 1,
  kJson = 2,
  kFc = 3,
};

// Maps from string to SyntaxType.
SyntaxType GetSyntaxType(absl::string_view syntax_type);

// Options for parsing text and tool calls.
struct ParserOptions {
  // If true, regex special characters within the fence strings will be escaped.
  bool escape_fence_strings = true;

  // A regex with a capture group used to filter each line of the tool call
  // string. If provided, only the captured substring will be parsed as a tool
  // call.
  std::string tool_code_regex = "";

  // If true, returns an error status when a tool call fails to parse. If false,
  // the string containing the invalid tool call is returned as text content
  // along with an `error` field containing the error message from the parser.
  bool return_error_on_parse_failure = true;
};

// Parses a string into text and tool calls.
//
// Tool calls are parsed from tool code blocks. A tool code block is delimited
// by `code_fence_start` and `code_fence_end`.
//
// Args:
//   `response_str`: The raw string response from the model.
//   `code_fence_start`: The string marking the beginning of the code block.
//   `code_fence_end`: The string marking the end of the code block.
//   `syntax_type`: The syntax type of the tool calls.
//   `options`: Additional parsing options.
//
// Returns:
//   A JSON object with two fields:
//     - `content`: A list of JSON objects representing the message content.
//     - `tool_calls`: A list of JSON objects representing the tool calls.
absl::StatusOr<nlohmann::ordered_json> ParseTextAndToolCalls(
    absl::string_view response_str, absl::string_view code_fence_start,
    absl::string_view code_fence_end, SyntaxType syntax_type,
    const ParserOptions& options = {});

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_TOOL_USE_PARSER_UTILS_H_
