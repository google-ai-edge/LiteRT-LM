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

#include "runtime/conversation/internal_callback_util.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::status::StatusIs;

nlohmann::ordered_json TextMessage(absl::string_view text) {
  nlohmann::ordered_json message;
  message["role"] = "assistant";
  message["content"] = {{{"type", "text"}, {"text", text}}};
  return message;
}

nlohmann::ordered_json ChannelMessage(absl::string_view text,
                                      absl::string_view channel_name) {
  nlohmann::ordered_json message;
  message["role"] = "assistant";
  message["channels"] = {{channel_name, text}};
  return message;
}

absl::AnyInvocable<void(absl::StatusOr<Message>)> CreateUserMessageCallback(
    std::vector<nlohmann::ordered_json>& output, bool& done,
    absl::Status& status) {
  return [&](absl::StatusOr<Message> message) {
    if (!message.ok()) {
      done = true;
      status = message.status();
      return;
    }
    if (message->is_null()) {
      done = true;
    } else {
      output.push_back(*message);
    }
  };
}

class InternalCallbackTest : public testing::Test {
 protected:
  void SetUp() override {
    Gemma3DataProcessorConfig config;

    // Need a tool in the preface to trigger tool call parsing. The actual tool
    // definition is unimportant.
    JsonPreface preface{.tools = nlohmann::ordered_json::parse(R"json([{
                  "name": "tool_name",
                  "parameters": { "properties": { "x": { "type": "integer" } } }
                }])json")};
    ASSERT_OK_AND_ASSIGN(model_data_processor_,
                         Gemma3DataProcessor::Create(config, preface));

    processor_args_ = DataProcessorArguments();
  }

  std::unique_ptr<Gemma3DataProcessor> model_data_processor_;
  std::vector<nlohmann::ordered_json> output_;
  bool done_ = false;
  absl::Status status_;
  DataProcessorArguments processor_args_;
  std::vector<Channel> channels_;
};

TEST_F(InternalCallbackTest, OnDone) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kDone));

  EXPECT_THAT(output_, IsEmpty());
  EXPECT_TRUE(done_);
  EXPECT_OK(status_);
}

TEST_F(InternalCallbackTest, OnError) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(absl::InternalError("error"));

  EXPECT_THAT(output_, IsEmpty());
  EXPECT_TRUE(done_);
  EXPECT_THAT(status_, StatusIs(absl::StatusCode::kInternal, "error"));
}

TEST_F(InternalCallbackTest, Text) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"this "}));
  callback(Responses(TaskState::kProcessing, {"is "}));
  callback(Responses(TaskState::kProcessing, {"some "}));
  callback(Responses(TaskState::kProcessing, {"text"}));

  EXPECT_THAT(output_, ElementsAre(TextMessage("this "), TextMessage("is "),
                                   TextMessage("some "), TextMessage("text")));
}

TEST_F(InternalCallbackTest, ToolCall) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name"}));
  callback(Responses(TaskState::kProcessing, {"(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n```"}));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbackTest, TextAndToolCall) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"this "}));
  callback(Responses(TaskState::kProcessing, {"is "}));
  callback(Responses(TaskState::kProcessing, {"some "}));
  callback(Responses(TaskState::kProcessing, {"text\n"}));
  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name"}));
  callback(Responses(TaskState::kProcessing, {"(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n```"}));

  EXPECT_THAT(output_, ElementsAre(TextMessage("this "), TextMessage("is "),
                                   TextMessage("some "), TextMessage("text\n"),
                                   nlohmann::ordered_json::parse(R"json({
                            "role": "assistant",
                            "tool_calls": [
                              {
                                "type": "function",
                                "function": {
                                  "name": "tool_name",
                                  "arguments": {
                                    "x": 1
                                  }
                                }
                              }
                            ]
                          })json")));
}

TEST_F(InternalCallbackTest, SplitCodeFenceStart) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_"}));
  callback(Responses(TaskState::kProcessing, {"code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name"}));
  callback(Responses(TaskState::kProcessing, {"(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n```"}));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbackTest, TextBeforeSplitCodeFenceStart) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"text```tool_"}));
  callback(Responses(TaskState::kProcessing, {"code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name"}));
  callback(Responses(TaskState::kProcessing, {"(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n```"}));

  EXPECT_THAT(output_, ElementsAre(TextMessage("text"),
                                   nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbackTest, ToolCallAfterSplitCodeFenceStart) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```"}));
  callback(Responses(TaskState::kProcessing, {"tool_code\ntool_name"}));
  callback(Responses(TaskState::kProcessing, {"(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n```"}));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbackTest, TextOnBothSidesOfCodeFenceStart) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"text```tool_code\ntool_name"}));
  callback(Responses(TaskState::kProcessing, {"(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n```"}));

  EXPECT_THAT(output_, ElementsAre(TextMessage("text"),
                                   nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbackTest, SplitCodeFenceEnd) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n`"}));
  callback(Responses(TaskState::kProcessing, {"``"}));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbackTest, TextBeforeSplitCodeFenceEnd) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name(x="}));
  callback(Responses(TaskState::kProcessing, {"1)\n``"}));
  callback(Responses(TaskState::kProcessing, {"`"}));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                "role": "assistant",
                "tool_calls": [
                  {
                    "type": "function",
                    "function": {
                      "name": "tool_name",
                      "arguments": {
                        "x": 1
                      }
                    }
                  }
                ]
              })json")));
}

TEST_F(InternalCallbackTest, TextAfterSplitCodeFenceEnd) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n`"}));
  callback(Responses(TaskState::kProcessing, {"``text"}));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                            "role": "assistant",
                            "tool_calls": [
                              {
                                "type": "function",
                                "function": {
                                  "name": "tool_name",
                                  "arguments": {
                                    "x": 1
                                  }
                                }
                              }
                            ]
                          })json"),
                                   TextMessage("text")));
}

TEST_F(InternalCallbackTest, OnNextTextOnBothSidesOfSplitCodeFenceEnd) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name(x="}));
  callback(Responses(TaskState::kProcessing, {"1)\n`"}));
  callback(Responses(TaskState::kProcessing, {"``text"}));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                            "role": "assistant",
                            "tool_calls": [
                              {
                                "type": "function",
                                "function": {
                                  "name": "tool_name",
                                  "arguments": {
                                    "x": 1
                                  }
                                }
                              }
                            ]
                          })json"),
                                   TextMessage("text")));
}

TEST_F(InternalCallbackTest, ParallelToolCalls) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_a(x=1)\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_b(y='z')"}));
  callback(Responses(TaskState::kProcessing, {"\n```"}));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json(
                {
                  "role": "assistant",
                  "tool_calls": [
                    {
                      "type": "function",
                      "function": {
                        "name": "tool_a",
                        "arguments": {
                          "x": 1
                        }
                      }
                    },
                    {
                      "type": "function",
                      "function": {
                        "name": "tool_b",
                        "arguments": {
                          "y": "z"
                        }
                      }
                    }
                  ]
                }
                )json")));
}

TEST_F(InternalCallbackTest, TwoConsecutiveToolCodeBlocks) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_a(x=1)\n"}));
  callback(Responses(TaskState::kProcessing, {"``````tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_b(y='z')\n"}));
  callback(Responses(TaskState::kProcessing, {"```"}));

  EXPECT_THAT(output_, ElementsAre(nlohmann::ordered_json::parse(R"json({
                            "role": "assistant",
                            "tool_calls": [
                              {
                                "type": "function",
                                "function": {
                                  "name": "tool_a",
                                  "arguments": {
                                    "x": 1
                                  }
                                }
                              }
                            ]
                          })json"),
                                   nlohmann::ordered_json::parse(R"json({
                            "role": "assistant",
                            "tool_calls": [
                              {
                                "type": "function",
                                "function": {
                                  "name": "tool_b",
                                  "arguments": {
                                    "y": "z"
                                  }
                                }
                              }
                            ]
                          })json")));
}

TEST_F(InternalCallbackTest, IncompleteToolCodeBlock) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name(x=1)"}));
  callback(Responses(TaskState::kDone));

  // The incomplete tool code block is sent to the callback as a text message.
  EXPECT_THAT(output_,
              ElementsAre(TextMessage("```tool_code\ntool_name(x=1)")));
}

TEST_F(InternalCallbackTest, WrongCodeFenceStart) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n```"}));
  callback(Responses(TaskState::kDone));

  EXPECT_THAT(output_, ElementsAre(TextMessage("```tool\n"),
                                   TextMessage("tool_name(x=1)"),
                                   TextMessage("\n"), TextMessage("```")));
}

TEST_F(InternalCallbackTest, WrongCodeFenceEnd) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"tool_name(x=1)"}));
  callback(Responses(TaskState::kProcessing, {"\n``x"}));
  callback(Responses(TaskState::kDone));

  EXPECT_THAT(output_,
              ElementsAre(TextMessage("```tool_code\ntool_name(x=1)\n``x")));
}

TEST_F(InternalCallbackTest, InvalidFunctionCall) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"```tool_code\n"}));
  callback(Responses(TaskState::kProcessing, {"not a function call"}));
  callback(Responses(TaskState::kProcessing, {"\n```"}));

  EXPECT_TRUE(done_);
  EXPECT_THAT(status_, StatusIs(absl::StatusCode::kInvalidArgument));
}

class InternalCallbackChannelTest : public testing::Test {
 protected:
  void SetUp() override {
    Gemma3DataProcessorConfig config;

    // Need a tool in the preface to trigger tool call parsing. The actual tool
    // definition is unimportant.
    JsonPreface preface{.tools = nlohmann::ordered_json::parse(R"json([{
                  "name": "tool_name",
                  "parameters": { "properties": { "x": { "type": "integer" } } }
                }])json")};
    ASSERT_OK_AND_ASSIGN(auto gemma3_processor,
                         Gemma3DataProcessor::Create(config, preface));

    channels_ = {{"thought", "<|channel>thought\n", "<channel|>"}};
    model_data_processor_ = std::move(gemma3_processor);

    processor_args_ = DataProcessorArguments();
  }

  std::unique_ptr<Gemma3DataProcessor> model_data_processor_;
  std::vector<nlohmann::ordered_json> output_;
  bool done_ = false;
  absl::Status status_;
  DataProcessorArguments processor_args_;
  std::vector<Channel> channels_;
};

TEST_F(InternalCallbackChannelTest, ChannelStream) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"<|channel>thought\n"}));
  callback(Responses(TaskState::kProcessing, {"I "}));
  callback(Responses(TaskState::kProcessing, {"am "}));
  callback(Responses(TaskState::kProcessing, {"thinking"}));
  callback(Responses(TaskState::kProcessing, {"<channel|>"}));

  EXPECT_THAT(output_, ElementsAre(ChannelMessage("I ", "thought"),
                                   ChannelMessage("am ", "thought"),
                                   ChannelMessage("thinking", "thought")));
}

TEST_F(InternalCallbackChannelTest, SplitChannelEnd) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"<|channel>thought\n"}));
  callback(Responses(TaskState::kProcessing, {"partial "}));
  callback(Responses(TaskState::kProcessing, {"<chan"}));
  callback(Responses(TaskState::kProcessing, {"nel|>"}));

  EXPECT_THAT(output_, ElementsAre(ChannelMessage("partial ", "thought")));
}

TEST_F(InternalCallbackChannelTest, ChannelAndText) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"some "}));
  callback(Responses(TaskState::kProcessing, {"text\n"}));
  callback(Responses(TaskState::kProcessing, {"<|chan"}));
  callback(Responses(TaskState::kProcessing, {"nel>thought\n"}));
  callback(Responses(TaskState::kProcessing, {"I "}));
  callback(Responses(TaskState::kProcessing, {"am "}));
  callback(Responses(TaskState::kProcessing, {"thinking"}));
  callback(Responses(TaskState::kProcessing, {"<channel|>"}));
  callback(Responses(TaskState::kProcessing, {" more text"}));

  EXPECT_THAT(output_, ElementsAre(TextMessage("some "), TextMessage("text\n"),
                                   ChannelMessage("I ", "thought"),
                                   ChannelMessage("am ", "thought"),
                                   ChannelMessage("thinking", "thought"),
                                   TextMessage(" more text")));
}

TEST_F(InternalCallbackChannelTest, IncompleteChannel) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  auto callback =
      CreateInternalCallback(*model_data_processor_, processor_args_, channels_,
                             std::move(user_callback));

  callback(Responses(TaskState::kProcessing, {"<|channel>thought\n"}));
  callback(Responses(TaskState::kProcessing, {"this is "}));
  callback(Responses(TaskState::kDone));

  EXPECT_THAT(output_, ElementsAre(ChannelMessage("this is ", "thought")));
}

TEST_F(InternalCallbackChannelTest, ChannelStreamWithCompleteMessageCallback) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  Message final_message;
  bool final_done = false;
  auto complete_message_callback = [&](const Message& message) {
    final_message = message;
    final_done = true;
  };

  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, channels_,
      std::move(user_callback), /*cancel_callback=*/nullptr,
      std::move(complete_message_callback));

  callback(Responses(TaskState::kProcessing, {"Hello"}));
  callback(Responses(TaskState::kProcessing, {"<|channel>thought\n"}));
  callback(Responses(TaskState::kProcessing, {"I am thinking"}));
  callback(Responses(TaskState::kProcessing, {"<channel|>"}));
  callback(Responses(TaskState::kProcessing, {" World!"}));
  callback(Responses(TaskState::kDone));

  EXPECT_TRUE(final_done);
  EXPECT_THAT(final_message, testing::Eq(Message::parse(R"json({
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello World!"}],
                "channels": {
                  "thought": "I am thinking"
                }
              })json")));
}

TEST_F(InternalCallbackChannelTest,
       ChannelStreamUnclosedWithCompleteMessageCallback) {
  auto user_callback = CreateUserMessageCallback(output_, done_, status_);
  Message final_message;
  bool final_done = false;
  auto complete_message_callback = [&](const Message& message) {
    final_message = message;
    final_done = true;
  };

  auto callback = CreateInternalCallback(
      *model_data_processor_, processor_args_, channels_,
      std::move(user_callback), /*cancel_callback=*/nullptr,
      std::move(complete_message_callback));

  callback(Responses(TaskState::kProcessing, {"<|channel>thought\n"}));
  callback(Responses(TaskState::kProcessing, {"I am thinking"}));
  callback(Responses(TaskState::kDone));

  EXPECT_TRUE(final_done);
  EXPECT_TRUE(final_message.contains("channels"));
  EXPECT_EQ(final_message["channels"]["thought"], "I am thinking");
}

}  // namespace
}  // namespace litert::lm
