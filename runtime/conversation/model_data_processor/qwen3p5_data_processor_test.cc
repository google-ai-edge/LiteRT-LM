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
#include <string>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/conversation/model_data_processor/qwen3p5_data_processor_config.h"
#include "runtime/conversation/model_data_processor/test_utils.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using json = nlohmann::ordered_json;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::StartsWith;

// Qwen3.5-2B's reply to "What is the weather in Paris right now?" when given a
// get_weather tool and models/qwen3_5/chat_template.jinja, with
// enable_thinking=False.
constexpr char kGetWeatherCall[] =
    "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n"
    "</parameter>\n</function>\n</tool_call>";

constexpr char kMalformedCall[] =
    "<tool_call>\nnot a function block\n</tool_call>";

// Returns a tool declaration in the {"type": "function", "function": ...}
// shape.
json Tool(const std::string& name, const json& properties) {
  return {{"type", "function"},
          {"function",
           {{"name", name},
            {"parameters", {{"type", "object"}, {"properties", properties}}}}}};
}

json GetWeatherTool() {
  return Tool("get_weather", {{"location", {{"type", "string"}}}});
}

absl::StatusOr<std::unique_ptr<ModelDataProcessor>> CreateWithTools(
    const json& tools) {
  JsonPreface preface;
  preface.tools = tools;
  return Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig{}, preface);
}

absl::StatusOr<Message> ReplyToMessage(const ModelDataProcessor& processor,
                                       const std::string& reply) {
  return processor.ToMessage(Responses(TaskState::kProcessing, {reply}),
                             std::monostate{});
}

TEST(Qwen3p5DataProcessorTest, ToInputDataVector) {
  ASSERT_OK_AND_ASSIGN(auto processor, Qwen3p5DataProcessor::Create(
                                           Qwen3p5DataProcessorConfig{}));
  const std::string rendered_template_prompt =
      "<|im_start|>user\ntest prompt<|im_end|>\n<|im_start|>assistant\n";
  const json messages = {
      {"role", "user"},
      {"content", "test prompt"},
  };
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_template_prompt, messages, {}));

  InputText expected_text(rendered_template_prompt);
  EXPECT_THAT(input_data, ElementsAre(HasInputText(&expected_text)));
}

TEST(Qwen3p5DataProcessorTest, ToMessageWithoutToolsReturnsText) {
  ASSERT_OK_AND_ASSIGN(
      auto processor, Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig{},
                                                   JsonPreface{}));

  ASSERT_OK_AND_ASSIGN(const Message message,
                       ReplyToMessage(*processor, kGetWeatherCall));

  EXPECT_EQ(
      message,
      json({{"role", "assistant"},
            {"content", {{{"type", "text"}, {"text", kGetWeatherCall}}}}}));
}

TEST(Qwen3p5DataProcessorTest, ToMessageWithToolsButNoCallReturnsText) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({GetWeatherTool()})));
  constexpr char kReply[] = "It is sunny in Paris.\n\nAnything else?";

  ASSERT_OK_AND_ASSIGN(const Message message,
                       ReplyToMessage(*processor, kReply));

  EXPECT_EQ(message,
            json({{"role", "assistant"},
                  {"content", {{{"type", "text"}, {"text", kReply}}}}}));
}

TEST(Qwen3p5DataProcessorTest, ToMessageSingleCall) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({GetWeatherTool()})));

  ASSERT_OK_AND_ASSIGN(const Message message,
                       ReplyToMessage(*processor, kGetWeatherCall));

  EXPECT_EQ(message, json::parse(R"json({
    "role": "assistant",
    "tool_calls": [
      {
        "type": "function",
        "function": {"name": "get_weather", "arguments": {"location": "Paris"}}
      }
    ]
  })json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageTwoCalls) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({GetWeatherTool()})));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      ReplyToMessage(*processor,
                     absl::StrCat(kGetWeatherCall, "\n",
                                  "<tool_call>\n<function=get_weather>\n"
                                  "<parameter=location>\nLondon\n</parameter>\n"
                                  "</function>\n</tool_call>")));

  // The newline between the blocks is kept as text.
  EXPECT_EQ(message, json::parse(R"json({
    "role": "assistant",
    "content": [{"type": "text", "text": "\n"}],
    "tool_calls": [
      {
        "type": "function",
        "function": {"name": "get_weather", "arguments": {"location": "Paris"}}
      },
      {
        "type": "function",
        "function": {"name": "get_weather", "arguments": {"location": "London"}}
      }
    ]
  })json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageMultiLineValue) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({Tool(
                           "write_note", {{"text", {{"type", "string"}}},
                                          {"padded", {{"type", "string"}}},
                                          {"crlf", {{"type", "string"}}}})})));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      ReplyToMessage(
          *processor,
          "<tool_call>\n<function=write_note>\n"
          "<parameter=text>\nline 1\nline 2\n\nline 4\n</parameter>\n"
          "<parameter=padded>\n\n  indented\n\n</parameter>\n"
          "<parameter=crlf>\r\nline 1\r\nline 2\r\n</parameter>\n"
          "</function>\n</tool_call>"));

  // Only one line break after the opening tag and one before the closing tag
  // are dropped, "\r\n" or else "\n".
  EXPECT_EQ(message, json::parse(R"json({
    "role": "assistant",
    "tool_calls": [
      {
        "type": "function",
        "function": {
          "name": "write_note",
          "arguments": {
            "text": "line 1\nline 2\n\nline 4",
            "padded": "\n  indented\n",
            "crlf": "line 1\r\nline 2"
          }
        }
      }
    ]
  })json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageTextAroundCall) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({GetWeatherTool()})));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      ReplyToMessage(*processor,
                     absl::StrCat("I will look it up.\n\n", kGetWeatherCall,
                                  "\nOne moment, please.")));

  EXPECT_EQ(message, json::parse(R"json({
    "role": "assistant",
    "content": [
      {"type": "text", "text": "I will look it up.\n\n"},
      {"type": "text", "text": "\nOne moment, please."}
    ],
    "tool_calls": [
      {
        "type": "function",
        "function": {"name": "get_weather", "arguments": {"location": "Paris"}}
      }
    ]
  })json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageUnclosedFenceIsText) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({GetWeatherTool()})));
  constexpr char kTruncatedCall[] =
      "<tool_call>\n<function=get_weather>\n<parameter=location>\nPa";

  ASSERT_OK_AND_ASSIGN(const Message message,
                       ReplyToMessage(*processor, kTruncatedCall));

  EXPECT_EQ(
      message,
      json({{"role", "assistant"},
            {"content", {{{"type", "text"}, {"text", kTruncatedCall}}}}}));
}

TEST(Qwen3p5DataProcessorTest, ToMessageMalformedBlockReturnsError) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({GetWeatherTool()})));

  EXPECT_THAT(ReplyToMessage(*processor, kMalformedCall),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr(absl::StrCat(
                           "Failed to parse tool calls from code block: "
                           "\nnot a function block\n",
                           "\nfull response: ", kMalformedCall, "\nerror: "))));
}

TEST(Qwen3p5DataProcessorTest,
     ToMessageMalformedBlockAsTextWhenNotReturningError) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({GetWeatherTool()})));
  processor->SetReturnErrorOnParseFailure(false);

  ASSERT_OK_AND_ASSIGN(const Message message,
                       ReplyToMessage(*processor, kMalformedCall));

  EXPECT_FALSE(message.contains("tool_calls"));
  ASSERT_EQ(message.at("content").size(), 1);
  const json& part = message.at("content").at(0);
  EXPECT_EQ(part.at("type"), "text");
  EXPECT_EQ(part.at("text"), "\nnot a function block\n");
  EXPECT_THAT(part.at("error").get<std::string>(),
              StartsWith("Failed to parse tool calls: "));
}

TEST(Qwen3p5DataProcessorTest, ToMessageParameterNotClosedIsMalformed) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array(
                           {Tool("f", {{"a", {{"type", "integer"}}}})})));

  EXPECT_THAT(
      ReplyToMessage(*processor,
                     "<tool_call>\n<function=f>\n<parameter=a>\n1\n</function>"
                     "\n</tool_call>"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Failed to parse tool calls from code block")));
  // A "</parameter>" that closes nothing is skipped, but the parameter opened
  // after it still has to be closed.
  EXPECT_THAT(
      ReplyToMessage(*processor,
                     "<tool_call>\n<function=f>\n</parameter>\n"
                     "<parameter=a>\n1\n</function>\n</tool_call>"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Failed to parse tool calls from code block")));
}

TEST(Qwen3p5DataProcessorTest, ToMessageParameterEndThatClosesNothing) {
  ASSERT_OK_AND_ASSIGN(auto processor, CreateWithTools(json::parse(R"json([
    {
      "type": "function",
      "function": {
        "name": "get_current_time",
        "description": "Get the current local time.",
        "parameters": {"type": "object", "properties": {}, "required": []}
      }
    }
  ])json")));

  // Qwen3.5-0.8B's reply to "What time is it now?" when given this tool and
  // models/qwen3_5/chat_template.jinja, with enable_thinking=False: the
  // "</parameter>" closes nothing and is skipped.
  ASSERT_OK_AND_ASSIGN(
      const Message message,
      ReplyToMessage(*processor,
                     "<tool_call>\n<function=get_current_time>\n</parameter>\n"
                     "</function>\n</tool_call>"));

  EXPECT_EQ(message, json::parse(R"json({
    "role": "assistant",
    "tool_calls": [
      {
        "type": "function",
        "function": {"name": "get_current_time", "arguments": {}}
      }
    ]
  })json"));
}

TEST(Qwen3p5DataProcessorTest,
     ToMessageParameterNotClosedBeforeNextIsMalformed) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array(
                           {Tool("f", {{"a", {{"type", "integer"}}},
                                       {"b", {{"type", "integer"}}}})})));
  constexpr char kBlock[] =
      "\n<function=f>\n<parameter=a>\n1\n<parameter=b>\n2\n</parameter>\n"
      "</function>\n";
  const std::string reply = absl::StrCat("<tool_call>", kBlock, "</tool_call>");

  // Read as one parameter, "a" would get the value "1\n<parameter=b>\n2" and
  // "b" would be lost.
  EXPECT_THAT(ReplyToMessage(*processor, reply),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("expected </parameter> before the next "
                                 "<parameter=KEY>")));

  processor->SetReturnErrorOnParseFailure(false);
  ASSERT_OK_AND_ASSIGN(const Message message,
                       ReplyToMessage(*processor, reply));
  EXPECT_FALSE(message.contains("tool_calls"));
  ASSERT_EQ(message.at("content").size(), 1);
  const json& part = message.at("content").at(0);
  EXPECT_EQ(part.at("type"), "text");
  EXPECT_EQ(part.at("text"), kBlock);
  EXPECT_EQ(part.at("error"),
            "Failed to parse tool calls: expected </parameter> before the next "
            "<parameter=KEY>");
}

TEST(Qwen3p5DataProcessorTest, ToMessageFunctionWithoutParameters) {
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      CreateWithTools(json::array({Tool("get_time", json::object())})));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      ReplyToMessage(*processor,
                     "<tool_call>\n<function=get_time>\n</function>\n"
                     "</tool_call>"));

  EXPECT_EQ(message, json::parse(R"json({
    "role": "assistant",
    "tool_calls": [
      {
        "type": "function",
        "function": {"name": "get_time", "arguments": {}}
      }
    ]
  })json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageTypedArguments) {
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      CreateWithTools(json::array(
          {Tool("configure",
                {{"count", {{"type", "integer"}}},
                 {"ratio", {{"type", "number"}}},
                 {"flag", {{"type", "boolean"}}},
                 {"enabled", {{"type", "boolean"}}},
                 {"toggle", {{"type", json::array({"boolean"})}}},
                 {"mixed", {{"type", json::array({"boolean", "integer"})}}},
                 {"opts", {{"type", "object"}}},
                 {"ids", {{"type", "array"}}},
                 {"name", {{"type", "string"}}},
                 {"limit", {{"type", json::array({"integer", "null"})}}},
                 {"label", {{"type", json::array({"string", "null"})}}}})})));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      ReplyToMessage(*processor,
                     "<tool_call>\n<function=configure>\n"
                     "<parameter=count>\n3\n</parameter>\n"
                     "<parameter=ratio>\n0.5\n</parameter>\n"
                     "<parameter=flag>\ntrue\n</parameter>\n"
                     "<parameter=opts>\n{\"a\": 1}\n</parameter>\n"
                     "<parameter=ids>\n[1, 2]\n</parameter>\n"
                     "<parameter=name>\n42\n</parameter>\n"
                     "<parameter=extra>\n7\n</parameter>\n"
                     "<parameter=limit>\n5\n</parameter>\n"
                     "<parameter=label>\n5\n</parameter>\n"
                     "</function>\n</tool_call>\n"
                     "<tool_call>\n<function=configure>\n"
                     "<parameter=count>\nabc\n</parameter>\n"
                     "</function>\n</tool_call>\n"
                     "<tool_call>\n<function=configure>\n"
                     "<parameter=flag>\nTrue\n</parameter>\n"
                     "<parameter=enabled>\nFalse\n</parameter>\n"
                     "<parameter=limit>\nNone\n</parameter>\n"
                     "<parameter=label>\nNone\n</parameter>\n"
                     "</function>\n</tool_call>\n"
                     "<tool_call>\n<function=configure>\n"
                     "<parameter=flag>\n1\n</parameter>\n"
                     "<parameter=enabled>\nTRUE\n</parameter>\n"
                     "<parameter=toggle>\n0\n</parameter>\n"
                     "<parameter=mixed>\n1\n</parameter>\n"
                     "</function>\n</tool_call>"));

  // Declared non-string types are parsed as JSON, except a boolean-only type,
  // which reads only true and false in any case (1 stays a string); a string
  // type, an undeclared parameter and a value that is not valid JSON stay
  // strings. Types that allow boolean or null also read True, False and None,
  // as Python's Jinja2 writes them, even when they allow string too.
  EXPECT_EQ(message.at("tool_calls"), json::parse(R"json([
    {
      "type": "function",
      "function": {
        "name": "configure",
        "arguments": {
          "count": 3,
          "ratio": 0.5,
          "flag": true,
          "opts": {"a": 1},
          "ids": [1, 2],
          "name": "42",
          "extra": "7",
          "limit": 5,
          "label": "5"
        }
      }
    },
    {
      "type": "function",
      "function": {"name": "configure", "arguments": {"count": "abc"}}
    },
    {
      "type": "function",
      "function": {
        "name": "configure",
        "arguments": {
          "flag": true,
          "enabled": false,
          "limit": null,
          "label": null
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "configure",
        "arguments": {"flag": "1", "enabled": true, "toggle": "0", "mixed": 1}
      }
    }
  ])json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageTypeNameCaseUnknownTypeAndAnyOf) {
  ASSERT_OK_AND_ASSIGN(auto processor, CreateWithTools(json::parse(R"json([
    {
      "type": "function",
      "function": {
        "name": "search",
        "parameters": {
          "type": "object",
          "properties": {
            "code": {"type": "STRING"},
            "count": {"type": "Integer"},
            "when": {"type": "date"},
            "page": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "after": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "cursor": {"oneOf": [{"type": "integer"}, {"type": "null"}]}
          }
        }
      }
    }
  ])json")));

  ASSERT_OK_AND_ASSIGN(const Message message,
                       ReplyToMessage(*processor,
                                      "<tool_call>\n<function=search>\n"
                                      "<parameter=code>\n12345\n</parameter>\n"
                                      "<parameter=count>\n7\n</parameter>\n"
                                      "<parameter=when>\n2026\n</parameter>\n"
                                      "<parameter=page>\n5\n</parameter>\n"
                                      "<parameter=after>\nnull\n</parameter>\n"
                                      "<parameter=cursor>\n3\n</parameter>\n"
                                      "</function>\n</tool_call>"));

  // Type names are compared ignoring case, an unknown type stays a string, and
  // a schema without "type" takes the types in its "anyOf" or "oneOf".
  EXPECT_EQ(message.at("tool_calls"), json::parse(R"json([
    {
      "type": "function",
      "function": {
        "name": "search",
        "arguments": {
          "code": "12345",
          "count": 7,
          "when": "2026",
          "page": 5,
          "after": null,
          "cursor": 3
        }
      }
    }
  ])json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageBareToolShapeInPreface) {
  ASSERT_OK_AND_ASSIGN(auto processor, CreateWithTools(json::parse(R"json([
    {
      "name": "get_weather",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {"type": "string"},
          "days": {"type": "integer"}
        }
      }
    }
  ])json")));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      ReplyToMessage(*processor,
                     "<tool_call>\n<function=get_weather>\n"
                     "<parameter=location>\nParis\n</parameter>\n"
                     "<parameter=days>\n3\n</parameter>\n"
                     "</function>\n</tool_call>"));

  EXPECT_EQ(message.at("tool_calls"), json::parse(R"json([
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": {"location": "Paris", "days": 3}
      }
    }
  ])json"));
}

TEST(Qwen3p5DataProcessorTest, CodeFence) {
  ASSERT_OK_AND_ASSIGN(auto processor, Qwen3p5DataProcessor::Create(
                                           Qwen3p5DataProcessorConfig{}));
  EXPECT_EQ(processor->CodeFenceStart(), "<tool_call>");
  EXPECT_EQ(processor->CodeFenceEnd(), "</tool_call>");
}

TEST(Qwen3p5DataProcessorTest, TemplateRoundTrip) {
  const json tools = json::array(
      {Tool("get_weather",
            {{"location", {{"type", "string"}}},
             {"days", {{"type", "integer"}}},
             {"metric", {{"type", "boolean"}}},
             {"hour", {{"type", json::array({"integer", "null"})}}}})});
  // The template writes the boolean and null values with `| string`, which
  // gives true, false and none in minijinja.
  const json tool_calls = json::parse(R"json([
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": {"location": "Paris", "days": 3, "metric": true,
                      "hour": null}
      }
    },
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": {"location": "London", "metric": false}
      }
    }
  ])json");
  ASSERT_OK_AND_ASSIGN(
      const std::string template_content,
      GetContents(GetModelPath("qwen3_5/chat_template.jinja")));
  PromptTemplate prompt_template(template_content);
  PromptTemplateInput input{
      .messages = json::array(
          {{{"role", "user"},
            {"content", json::array({{{"type", "text"},
                                      {"text",
                                       "What is the weather in Paris and "
                                       "London?"}}})}},
           {{"role", "assistant"}, {"tool_calls", tool_calls}}}),
      .tools = tools,
      .add_generation_prompt = false,
  };
  ASSERT_OK_AND_ASSIGN(const std::string rendered,
                       prompt_template.Apply(input));

  // Cut out the assistant turn, which is what the model generates.
  constexpr absl::string_view kTurnStart = "<|im_start|>assistant\n";
  const size_t turn_start = rendered.rfind(kTurnStart);
  ASSERT_NE(turn_start, std::string::npos);
  const size_t reply_start = turn_start + kTurnStart.size();
  const size_t reply_end = rendered.find("<|im_end|>", reply_start);
  ASSERT_NE(reply_end, std::string::npos);
  const std::string reply =
      rendered.substr(reply_start, reply_end - reply_start);

  ASSERT_OK_AND_ASSIGN(auto processor, CreateWithTools(tools));
  ASSERT_OK_AND_ASSIGN(const Message message,
                       ReplyToMessage(*processor, reply));

  EXPECT_EQ(message.at("tool_calls"), tool_calls);
}

TEST(Qwen3p5DataProcessorTest, ToMessageFunctionNotClosedIsMalformed) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({GetWeatherTool()})));

  EXPECT_THAT(
      ReplyToMessage(*processor,
                     "<tool_call>\n<function=get_weather>\n"
                     "<parameter=location>\nParis\n</parameter>\n"
                     "</tool_call>"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Failed to parse tool calls from code block")));
}

TEST(Qwen3p5DataProcessorTest, ToMessageMissingNameOrKeyIsMalformed) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({GetWeatherTool()})));

  EXPECT_THAT(
      ReplyToMessage(*processor,
                     "<tool_call>\n<function>\n"
                     "<parameter=location>\nParis\n</parameter>\n"
                     "</function>\n</tool_call>"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Failed to parse tool calls from code block")));
  EXPECT_THAT(
      ReplyToMessage(*processor,
                     "<tool_call>\n<function=get_weather>\n"
                     "<parameter>\nParis\n</parameter>\n"
                     "</function>\n</tool_call>"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Failed to parse tool calls from code block")));
  // A name of only spaces.
  EXPECT_THAT(
      ReplyToMessage(*processor,
                     "<tool_call>\n<function= >\n"
                     "<parameter=location>\nParis\n</parameter>\n"
                     "</function>\n</tool_call>"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Failed to parse tool calls from code block")));
  // A key without its ">": a key does not run past the end of its line.
  EXPECT_THAT(
      ReplyToMessage(*processor,
                     "<tool_call>\n<function=get_weather>\n"
                     "<parameter=location\nParis\n</parameter>\n"
                     "<parameter=days>\n3\n</parameter>\n"
                     "</function>\n</tool_call>"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Failed to parse tool calls from code block")));
}

TEST(Qwen3p5DataProcessorTest, ToMessageTwoFunctionsInOneBlockIsMalformed) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({GetWeatherTool()})));

  EXPECT_THAT(
      ReplyToMessage(*processor,
                     "<tool_call>\n<function=get_weather>\n"
                     "<parameter=location>\nParis\n</parameter>\n"
                     "</function>\n<function=get_weather>\n"
                     "<parameter=location>\nLondon\n</parameter>\n"
                     "</function>\n</tool_call>"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Failed to parse tool calls from code block")));
}

TEST(Qwen3p5DataProcessorTest, ToMessageStringTypeKeepsJsonLikeValues) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({Tool(
                           "save", {{"json_object", {{"type", "string"}}},
                                    {"json_number", {{"type", "string"}}},
                                    {"json_boolean", {{"type", "string"}}},
                                    {"json_null", {{"type", "string"}}}})})));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      ReplyToMessage(*processor,
                     "<tool_call>\n<function=save>\n"
                     "<parameter=json_object>\n{\"a\":1}\n</parameter>\n"
                     "<parameter=json_number>\n123\n</parameter>\n"
                     "<parameter=json_boolean>\ntrue\n</parameter>\n"
                     "<parameter=json_null>\nnull\n</parameter>\n"
                     "</function>\n</tool_call>"));

  EXPECT_EQ(message.at("tool_calls"), json::parse(R"json([
    {
      "type": "function",
      "function": {
        "name": "save",
        "arguments": {
          "json_object": "{\"a\":1}",
          "json_number": "123",
          "json_boolean": "true",
          "json_null": "null"
        }
      }
    }
  ])json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageRepeatedParameterKeepsLastValue) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({GetWeatherTool()})));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      ReplyToMessage(*processor,
                     "<tool_call>\n<function=get_weather>\n"
                     "<parameter=location>\nParis\n</parameter>\n"
                     "<parameter=location>\nLondon\n</parameter>\n"
                     "</function>\n</tool_call>"));

  EXPECT_EQ(message.at("tool_calls"), json::parse(R"json([
    {
      "type": "function",
      "function": {"name": "get_weather", "arguments": {"location": "London"}}
    }
  ])json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageTextBetweenCalls) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({GetWeatherTool()})));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      ReplyToMessage(*processor,
                     absl::StrCat(kGetWeatherCall, "\nAnd London:\n",
                                  "<tool_call>\n<function=get_weather>\n"
                                  "<parameter=location>\nLondon\n</parameter>\n"
                                  "</function>\n</tool_call>")));

  EXPECT_EQ(message, json::parse(R"json({
    "role": "assistant",
    "content": [{"type": "text", "text": "\nAnd London:\n"}],
    "tool_calls": [
      {
        "type": "function",
        "function": {"name": "get_weather", "arguments": {"location": "Paris"}}
      },
      {
        "type": "function",
        "function": {"name": "get_weather", "arguments": {"location": "London"}}
      }
    ]
  })json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageUndeclaredFunctionKeepsStrings) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({GetWeatherTool()})));

  ASSERT_OK_AND_ASSIGN(const Message message,
                       ReplyToMessage(*processor,
                                      "<tool_call>\n<function=get_time>\n"
                                      "<parameter=zone>\nUTC\n</parameter>\n"
                                      "<parameter=offset>\n9\n</parameter>\n"
                                      "<parameter=dst>\ntrue\n</parameter>\n"
                                      "</function>\n</tool_call>"));

  EXPECT_EQ(message.at("tool_calls"), json::parse(R"json([
    {
      "type": "function",
      "function": {
        "name": "get_time",
        "arguments": {"zone": "UTC", "offset": "9", "dst": "true"}
      }
    }
  ])json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageFunctionNameIsCaseSensitive) {
  const json tool = Tool("get_weather", {{"location", {{"type", "string"}}},
                                         {"days", {{"type", "integer"}}}});
  ASSERT_OK_AND_ASSIGN(auto processor, CreateWithTools(json::array({tool})));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      ReplyToMessage(*processor,
                     "<tool_call>\n<function=Get_Weather>\n"
                     "<parameter=location>\nParis\n</parameter>\n"
                     "<parameter=days>\n3\n</parameter>\n"
                     "</function>\n</tool_call>\n"
                     "<tool_call>\n<function=get_weather>\n"
                     "<parameter=location>\nParis\n</parameter>\n"
                     "<parameter=days>\n3\n</parameter>\n"
                     "</function>\n</tool_call>"));

  // Get_Weather is not a declared tool, so its values stay strings; the call is
  // still returned, with the name the model wrote.
  EXPECT_EQ(message.at("tool_calls"), json::parse(R"json([
    {
      "type": "function",
      "function": {
        "name": "Get_Weather",
        "arguments": {"location": "Paris", "days": "3"}
      }
    },
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": {"location": "Paris", "days": 3}
      }
    }
  ])json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageMultiByteValue) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array(
                           {Tool("post", {{"text", {{"type", "string"}}}})})));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      ReplyToMessage(*processor,
                     "<tool_call>\n<function=post>\n"
                     "<parameter=text>\n東京は雨です 🌧️\n</parameter>\n"
                     "</function>\n</tool_call>"));

  EXPECT_EQ(message.at("tool_calls"), json::parse(R"json([
    {
      "type": "function",
      "function": {"name": "post", "arguments": {"text": "東京は雨です 🌧️"}}
    }
  ])json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageValueWithoutNewlinesAndEmptyValue) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array(
                           {Tool("f", {{"location", {{"type", "string"}}},
                                       {"x", {{"type", "string"}}}})})));

  ASSERT_OK_AND_ASSIGN(const Message message,
                       ReplyToMessage(*processor,
                                      "<tool_call>\n<function=f>\n"
                                      "<parameter=location>Paris</parameter>\n"
                                      "<parameter=x>\n\n</parameter>\n"
                                      "</function>\n</tool_call>"));

  EXPECT_EQ(message.at("tool_calls"), json::parse(R"json([
    {
      "type": "function",
      "function": {"name": "f", "arguments": {"location": "Paris", "x": ""}}
    }
  ])json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageTextBeforeCallWithThinkTags) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({GetWeatherTool()})));

  // Any text before a block is content. Channel text, such as the thought
  // channel of models/qwen3_5/LlmMetadataProto.pbtext, is taken out of the
  // response by the conversation before ToMessage, outside this processor.
  ASSERT_OK_AND_ASSIGN(
      const Message message,
      ReplyToMessage(*processor, absl::StrCat("<think>reasoning</think>\n",
                                              kGetWeatherCall)));

  EXPECT_EQ(message, json::parse(R"json({
    "role": "assistant",
    "content": [{"type": "text", "text": "<think>reasoning</think>\n"}],
    "tool_calls": [
      {
        "type": "function",
        "function": {"name": "get_weather", "arguments": {"location": "Paris"}}
      }
    ]
  })json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageEmptyBlockIsSkipped) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({GetWeatherTool()})));

  // As in Qwen3DataProcessor, an empty block gives neither a call nor an error.
  ASSERT_OK_AND_ASSIGN(const Message empty_block,
                       ReplyToMessage(*processor, "<tool_call></tool_call>"));
  EXPECT_EQ(empty_block, json::parse(R"json({"role": "assistant"})json"));

  ASSERT_OK_AND_ASSIGN(
      const Message text_and_empty_block,
      ReplyToMessage(*processor, "Let me check.<tool_call></tool_call>"));
  EXPECT_EQ(text_and_empty_block, json::parse(R"json({
    "role": "assistant",
    "content": [{"type": "text", "text": "Let me check."}]
  })json"));

  // A block of only whitespace is malformed.
  EXPECT_THAT(
      ReplyToMessage(*processor, "<tool_call>\n</tool_call>"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Failed to parse tool calls from code block")));
}

TEST(Qwen3p5DataProcessorTest, ToMessageRegexFences) {
  JsonPreface preface;
  preface.tools = json::array({GetWeatherTool()});
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      Qwen3p5DataProcessor::Create(
          Qwen3p5DataProcessorConfig{.code_fence_start = R"(<tool_call>\s*)",
                                     .code_fence_end = R"(\s*</tool_call>)",
                                     .escape_fence_strings = false},
          preface));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      ReplyToMessage(*processor, absl::StrCat("Sure.\n", kGetWeatherCall)));

  EXPECT_EQ(message, json::parse(R"json({
    "role": "assistant",
    "content": [{"type": "text", "text": "Sure.\n"}],
    "tool_calls": [
      {
        "type": "function",
        "function": {"name": "get_weather", "arguments": {"location": "Paris"}}
      }
    ]
  })json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageEmptyResponse) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({GetWeatherTool()})));

  ASSERT_OK_AND_ASSIGN(const Message message, ReplyToMessage(*processor, ""));

  EXPECT_EQ(message, json::parse(R"json({
    "role": "assistant",
    "content": [{"type": "text", "text": ""}]
  })json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageCallAndMalformedBlockReturnsError) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({GetWeatherTool()})));

  // The valid call before the malformed block is not returned either.
  EXPECT_THAT(
      ReplyToMessage(*processor, absl::StrCat(kGetWeatherCall, kMalformedCall)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Failed to parse tool calls from code block: "
                         "\nnot a function block\n")));
}

TEST(Qwen3p5DataProcessorTest,
     ToMessageCallAndMalformedBlockWhenNotReturningError) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({GetWeatherTool()})));
  processor->SetReturnErrorOnParseFailure(false);

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      ReplyToMessage(*processor,
                     absl::StrCat(kGetWeatherCall, kMalformedCall)));

  EXPECT_EQ(message.at("tool_calls"), json::parse(R"json([
    {
      "type": "function",
      "function": {"name": "get_weather", "arguments": {"location": "Paris"}}
    }
  ])json"));
  ASSERT_EQ(message.at("content").size(), 1);
  const json& part = message.at("content").at(0);
  EXPECT_EQ(part.at("type"), "text");
  EXPECT_EQ(part.at("text"), "\nnot a function block\n");
  EXPECT_THAT(part.at("error").get<std::string>(),
              StartsWith("Failed to parse tool calls: "));
}

TEST(Qwen3p5DataProcessorTest, ToMessageToolWithoutPropertiesKeepsStrings) {
  ASSERT_OK_AND_ASSIGN(auto processor, CreateWithTools(json::parse(R"json([
    {
      "type": "function",
      "function": {"name": "no_properties", "parameters": {"type": "object"}}
    },
    {
      "type": "function",
      "function": {"name": "no_parameters"}
    }
  ])json")));

  ASSERT_OK_AND_ASSIGN(const Message message,
                       ReplyToMessage(*processor,
                                      "<tool_call>\n<function=no_properties>\n"
                                      "<parameter=n>\n5\n</parameter>\n"
                                      "</function>\n</tool_call>\n"
                                      "<tool_call>\n<function=no_parameters>\n"
                                      "<parameter=n>\ntrue\n</parameter>\n"
                                      "</function>\n</tool_call>"));

  EXPECT_EQ(message.at("tool_calls"), json::parse(R"json([
    {
      "type": "function",
      "function": {"name": "no_properties", "arguments": {"n": "5"}}
    },
    {
      "type": "function",
      "function": {"name": "no_parameters", "arguments": {"n": "true"}}
    }
  ])json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageValueEndsAtFirstParameterClose) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array(
                           {Tool("post", {{"text", {{"type", "string"}}}})})));

  // The value ends at the first "</parameter>", so the rest of the text meant
  // as the value is outside any element and the block is malformed.
  EXPECT_THAT(ReplyToMessage(*processor,
                             "<tool_call>\n<function=post>\n"
                             "<parameter=text>\nClose it with </parameter>.\n"
                             "</parameter>\n</function>\n</tool_call>"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("expected only closed <parameter=KEY>")));

  // With only whitespace before the next "</parameter>", that one closes
  // nothing and is skipped, so the value is cut there without an error.
  ASSERT_OK_AND_ASSIGN(
      const Message message,
      ReplyToMessage(*processor,
                     "<tool_call>\n<function=post>\n"
                     "<parameter=text>\nClose it with </parameter>\n"
                     "</parameter>\n</function>\n</tool_call>"));
  EXPECT_EQ(message.at("tool_calls"), json::parse(R"json([
    {
      "type": "function",
      "function": {"name": "post", "arguments": {"text": "Close it with "}}
    }
  ])json"));
}

TEST(Qwen3p5DataProcessorTest, ToMessageNameAndKeyWithSpaces) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateWithTools(json::array({Tool(
                           "book", {{"start date", {{"type", "string"}}},
                                    {"nights", {{"type", "integer"}}}})})));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      ReplyToMessage(*processor,
                     "<tool_call>\n<function= book >\n"
                     "<parameter=start date>\n2026-10-12\n</parameter>\n"
                     "<parameter= nights >\n2\n</parameter>\n"
                     "</function>\n</tool_call>"));

  // A JSON schema property name may contain spaces; the spaces around a name
  // or a key are dropped.
  EXPECT_EQ(message.at("tool_calls"), json::parse(R"json([
    {
      "type": "function",
      "function": {
        "name": "book",
        "arguments": {"start date": "2026-10-12", "nights": 2}
      }
    }
  ])json"));
}

TEST(Qwen3p5DataProcessorTest, CreateWithEmptyFenceFails) {
  EXPECT_THAT(Qwen3p5DataProcessor::Create(
                  Qwen3p5DataProcessorConfig{.code_fence_start = ""}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(Qwen3p5DataProcessor::Create(
                  Qwen3p5DataProcessorConfig{.code_fence_end = ""}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Qwen3p5DataProcessorTest, CreateWithInvalidFenceRegexFails) {
  EXPECT_THAT(
      Qwen3p5DataProcessor::Create(Qwen3p5DataProcessorConfig{
          .code_fence_start = "(", .escape_fence_strings = false}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Invalid regex")));
  // Escaped, the same fence is plain text.
  EXPECT_OK(Qwen3p5DataProcessor::Create(
      Qwen3p5DataProcessorConfig{.code_fence_start = "("}));
}

TEST(Qwen3p5DataProcessorTest, ToMessageFenceRegexMatchingEmptyTextFails) {
  JsonPreface preface;
  preface.tools = json::array({GetWeatherTool()});
  const Qwen3p5DataProcessorConfig config{.code_fence_start = "(?:x)?",
                                          .code_fence_end = "(?:y)?",
                                          .escape_fence_strings = false};
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Qwen3p5DataProcessor::Create(config, preface));

  // Both fences match empty text, so the fence regex matches without moving
  // on; ToMessage returns an error instead of looping.
  EXPECT_THAT(ReplyToMessage(*processor, kGetWeatherCall),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("fence regex matched an empty string")));
}

}  // namespace
}  // namespace litert::lm
