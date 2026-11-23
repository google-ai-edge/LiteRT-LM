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

#include "models/json_schema_validator.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <iterator>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK_AND_ASSIGN, EXPECT_OK, StatusIs

namespace litert::lm {
namespace {

using json = nlohmann::ordered_json;
using ::testing::HasSubstr;
using ::testing::status::StatusIs;

std::filesystem::path ResolveSchemaPath() {
  constexpr absl::string_view kRelPath =
      "models/chat_template_input_schema.json";
  const auto src_dir = std::filesystem::path(::testing::SrcDir());
  if (std::filesystem::exists(src_dir / kRelPath)) {
    return src_dir / kRelPath;
  }
  if (std::filesystem::exists(src_dir / "litert_lm" / kRelPath)) {
    return src_dir / "litert_lm" / kRelPath;
  }
  if (std::filesystem::exists(src_dir / "_main" / kRelPath)) {
    return src_dir / "_main" / kRelPath;
  }
  return src_dir / kRelPath;
}

absl::StatusOr<json> LoadChatTemplateInputSchema() {
  const std::filesystem::path schema_path = ResolveSchemaPath();
  std::ifstream input_stream(schema_path);
  if (!input_stream.is_open()) {
    return absl::InternalError(
        absl::StrCat("Could not open schema file: ", schema_path.string()));
  }
  const std::string content((std::istreambuf_iterator<char>(input_stream)),
                            std::istreambuf_iterator<char>());
  json parsed = json::parse(content, nullptr, /*allow_exceptions=*/false);
  if (parsed.is_discarded()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse schema file: ", schema_path.string()));
  }
  return parsed;
}

TEST(JsonSchemaValidatorTest, ValidMinimalChatTemplateInput) {
  ASSERT_OK_AND_ASSIGN(const json schema, LoadChatTemplateInputSchema());
  const json instance = json::parse(R"({
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Hello"}
        ]
      }
    ]
  })");
  EXPECT_OK(ValidateJsonSchema(instance, schema));
}

TEST(JsonSchemaValidatorTest, ValidFullChatTemplateInput) {
  ASSERT_OK_AND_ASSIGN(const json schema, LoadChatTemplateInputSchema());
  const json instance = json::parse(R"({
    "bos_token": "<bos>",
    "enable_thinking": true,
    "add_generation_prompt": true,
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather.",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
          },
          "response": {
            "type": "object",
            "description": "Weather payload"
          }
        }
      }
    ],
    "messages": [
      {
        "role": "system",
        "content": [{"type": "text", "text": "You are helpful."}]
      },
      {
        "role": "user",
        "content": [
          {"type": "image"},
          {"type": "audio"},
          {"type": "video"},
          {"type": "text", "text": "What is the weather?"}
        ]
      },
      {
        "role": "assistant",
        "tool_calls": [
          {
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": {"location": "London", "units": "metric"}
            }
          },
          {
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\": \"Paris\"}"
            }
          }
        ]
      },
      {
        "role": "tool",
        "content": [
          {
            "type": "tool_response",
            "name": "get_weather",
            "response": {"temperature": "18C", "nested": [1, 2, 3]}
          }
        ]
      }
    ]
  })");
  EXPECT_OK(ValidateJsonSchema(instance, schema));
}

TEST(JsonSchemaValidatorTest, RejectsExtraTopLevelProperty) {
  ASSERT_OK_AND_ASSIGN(const json schema, LoadChatTemplateInputSchema());
  const json instance = json::parse(R"({
    "messages": [],
    "extra_field": true
  })");
  EXPECT_THAT(
      ValidateJsonSchema(instance, schema),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("unexpected additional property 'extra_field'")));
}

TEST(JsonSchemaValidatorTest, RejectsExtraMessageProperty) {
  ASSERT_OK_AND_ASSIGN(const json schema, LoadChatTemplateInputSchema());
  const json instance = json::parse(R"({
    "messages": [
      {
        "role": "assistant",
        "reasoning_content": "thinking...",
        "content": [{"type": "text", "text": "42"}]
      }
    ]
  })");
  EXPECT_THAT(ValidateJsonSchema(instance, schema),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("$.messages[0]: unexpected additional "
                                 "property 'reasoning_content'")));
}

TEST(JsonSchemaValidatorTest, RejectsExtraContentPartProperty) {
  ASSERT_OK_AND_ASSIGN(const json schema, LoadChatTemplateInputSchema());
  const json instance = json::parse(R"({
    "messages": [
      {
        "role": "user",
        "content": [{"type": "text", "text": "hi", "unexpected": 1}]
      }
    ]
  })");
  EXPECT_THAT(ValidateJsonSchema(instance, schema),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("$.messages[0].content[0]")));
}

TEST(JsonSchemaValidatorTest, RejectsExtraToolProperty) {
  ASSERT_OK_AND_ASSIGN(const json schema, LoadChatTemplateInputSchema());
  const json instance = json::parse(R"({
    "messages": [],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "fn",
          "description": "desc",
          "extra_fn_prop": 123
        }
      }
    ]
  })");
  EXPECT_THAT(ValidateJsonSchema(instance, schema),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("$.tools[0].function: unexpected additional "
                                 "property 'extra_fn_prop'")));
}

TEST(JsonSchemaValidatorTest, RejectsMissingRequiredMessages) {
  ASSERT_OK_AND_ASSIGN(const json schema, LoadChatTemplateInputSchema());
  const json instance = json::parse(R"({"add_generation_prompt": true})");
  EXPECT_THAT(ValidateJsonSchema(instance, schema),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("missing required property 'messages'")));
}

TEST(JsonSchemaValidatorTest, RejectsInvalidRoleEnum) {
  ASSERT_OK_AND_ASSIGN(const json schema, LoadChatTemplateInputSchema());
  const json instance = json::parse(R"({
    "messages": [{"role": "developer"}]
  })");
  EXPECT_THAT(ValidateJsonSchema(instance, schema),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("$.messages[0].role")));
}

TEST(JsonSchemaValidatorTest, RejectsInvalidType) {
  ASSERT_OK_AND_ASSIGN(const json schema, LoadChatTemplateInputSchema());
  const json instance = json::parse(R"({
    "messages": [{"role": "user", "content": "plain string"}]
  })");
  EXPECT_THAT(
      ValidateJsonSchema(instance, schema),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("$.messages[0].content: expected type 'array'")));
}

TEST(JsonSchemaValidatorTest, RejectsInvalidOneOfArguments) {
  ASSERT_OK_AND_ASSIGN(const json schema, LoadChatTemplateInputSchema());
  const json instance = json::parse(R"({
    "messages": [
      {
        "role": "assistant",
        "tool_calls": [
          {
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": 42
            }
          }
        ]
      }
    ]
  })");
  EXPECT_THAT(ValidateJsonSchema(instance, schema),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("$.messages[0].tool_calls[0].function."
                                 "arguments: does not match any schema in "
                                 "oneOf")));
}

TEST(JsonSchemaValidatorTest, SupportsPrimitiveTypesAndTypeArrays) {
  const json schema = json::parse(R"({
    "type": "object",
    "properties": {
      "i": {"type": "integer"},
      "n": {"type": "number"},
      "b": {"type": "boolean"},
      "z": {"type": "null"},
      "multi": {"type": ["string", "integer"]}
    },
    "additionalProperties": {"type": "string"}
  })");

  EXPECT_OK(ValidateJsonSchema(
      json::parse(R"({"i": 3, "n": 1.5, "b": false, "z": null, "multi": "ok",
                      "extra": "str"})"),
      schema));
  EXPECT_THAT(ValidateJsonSchema(json::parse(R"({"i": 1.5})"), schema),
              StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("$.i")));
  EXPECT_THAT(
      ValidateJsonSchema(json::parse(R"({"multi": true})"), schema),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("$.multi")));
  EXPECT_THAT(
      ValidateJsonSchema(json::parse(R"({"extra": 123})"), schema),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("$.extra")));
}

TEST(JsonSchemaValidatorTest, RejectsOneOfMatchingMultipleSchemas) {
  const json schema = json::parse(R"({
    "oneOf": [
      {"type": "integer"},
      {"type": "number"}
    ]
  })");
  EXPECT_THAT(ValidateJsonSchema(json(5), schema),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("matched 2 schemas in oneOf")));
}

TEST(JsonSchemaValidatorTest, RejectsInvalidSchemaDefinition) {
  EXPECT_THAT(ValidateJsonSchema(json::object(), json(123)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("schema must be a JSON object")));
  EXPECT_THAT(
      ValidateJsonSchema(json(nullptr), json::parse(R"({"type": "string"})")),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("got 'null'")));
  EXPECT_THAT(
      ValidateJsonSchema(json("x"), json::parse(R"({"type": "bogus"})")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("expected type 'bogus'")));
  EXPECT_THAT(ValidateJsonSchema(json("x"), json::parse(R"({"type": 123})")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("schema 'type' must be a string or array")));
  EXPECT_THAT(ValidateJsonSchema(json("x"), json::parse(R"({"type": [123]})")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("schema 'type' array must contain strings")));
  EXPECT_THAT(ValidateJsonSchema(json("x"), json::parse(R"({"enum": "bad"})")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("schema 'enum' must be an array")));
  EXPECT_THAT(ValidateJsonSchema(json("x"), json::parse(R"({"anyOf": {}})")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("schema 'anyOf' must be an array")));
  EXPECT_THAT(ValidateJsonSchema(json("x"), json::parse(R"({"oneOf": {}})")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("schema 'oneOf' must be an array")));
  EXPECT_THAT(
      ValidateJsonSchema(json::object(), json::parse(R"({"required": "a"})")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("schema 'required' must be an array")));
  EXPECT_THAT(
      ValidateJsonSchema(json::object(), json::parse(R"({"required": [1]})")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("schema 'required' entries must be strings")));
  EXPECT_THAT(
      ValidateJsonSchema(json::object(), json::parse(R"({"properties": []})")),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("schema 'properties' must be an object")));
  EXPECT_THAT(
      ValidateJsonSchema(json::object(),
                         json::parse(R"({"additionalProperties": "false"})")),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "schema 'additionalProperties' must be a boolean or object")));
  EXPECT_THAT(ValidateJsonSchema(json::array(), json::parse(R"({"items": 1})")),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("schema 'items' must be an object")));
}

}  // namespace
}  // namespace litert::lm
