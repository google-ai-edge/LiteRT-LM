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

#include "runtime/conversation/model_data_processor/lfm2_data_processor.h"

#include <string>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/lfm2_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using json = nlohmann::ordered_json;
using ::testing::ElementsAre;

MATCHER_P(HasInputText, text_input, "") {
  if (!std::holds_alternative<InputText>(arg)) {
    return false;
  }
  auto text_bytes = std::get<InputText>(arg).GetRawTextString();
  auto expected_bytes = text_input->GetRawTextString();
  if (!text_bytes.ok() || !expected_bytes.ok()) {
    return false;
  }
  return text_bytes.value() == expected_bytes.value();
}

TEST(Lfm2DataProcessorTest, ToInputDataVector) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Lfm2DataProcessor::Create(Lfm2DataProcessorConfig{}));
  const std::string rendered_template_prompt =
      "<|im_start|>user\ntest "
      "prompt<|im_end|>\n<|im_start|>assistant\ntest "
      "response<|im_end|>\n";
  const nlohmann::ordered_json messages = {
      {"role", "user"},
      {"content", "test prompt"},
      {"role", "assistant"},
      {"content", "test response"},
  };
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_template_prompt, messages, {}));

  InputText expected_text(
      "<|im_start|>user\ntest "
      "prompt<|im_end|>\n<|im_start|>assistant\ntest "
      "response<|im_end|>\n");
  EXPECT_THAT(input_data, ElementsAre(HasInputText(&expected_text)));
}

TEST(Lfm2DataProcessorTest, ToMessageDefault) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Lfm2DataProcessor::Create(Lfm2DataProcessorConfig{}));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(Responses(TaskState::kProcessing, {"test response"}),
                           std::monostate{}));

  EXPECT_EQ(
      message,
      json({{"role", "assistant"},
            {"content", {{{"type", "text"}, {"text", "test response"}}}}}));
}

TEST(Lfm2DataProcessorTest, ToMessageWithToolCallAndText) {
  JsonPreface preface;
  preface.tools = nlohmann::ordered_json::array();
  preface.tools.push_back(json{
      {"type", "function"}, {"function", {{"name", "get_candidate_status"}}}});
  ASSERT_OK_AND_ASSIGN(auto processor, Lfm2DataProcessor::Create(
                                           Lfm2DataProcessorConfig{}, preface));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(
          Responses(
              TaskState::kProcessing,
              {"<|tool_call_start|>[get_candidate_status(candidate_id=\"12345\""
               ")]<|tool_call_end|>Checking candidate status."}),
          std::monostate{}));

  EXPECT_EQ(message, nlohmann::ordered_json::parse(R"json({
              "role": "assistant",
              "content": [
                {
                  "type": "text",
                  "text": "Checking candidate status."
                }
              ],
              "tool_calls": [
                {
                  "type": "function",
                  "function": {
                    "name": "get_candidate_status",
                    "arguments": {
                      "candidate_id": "12345"
                    }
                  }
                }
              ]
            })json"));
}

TEST(Lfm2DataProcessorTest, ToMessageWithToolCallOnly) {
  JsonPreface preface;
  preface.tools = nlohmann::ordered_json::array();
  preface.tools.push_back(json{
      {"type", "function"}, {"function", {{"name", "get_candidate_status"}}}});
  ASSERT_OK_AND_ASSIGN(auto processor, Lfm2DataProcessor::Create(
                                           Lfm2DataProcessorConfig{}, preface));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(
          Responses(
              TaskState::kProcessing,
              {"<|tool_call_start|>[get_candidate_status(candidate_id=\"12345\""
               ")]<|tool_call_end|>"}),
          std::monostate{}));

  EXPECT_EQ(message, nlohmann::ordered_json::parse(R"json({
              "role": "assistant",
              "tool_calls": [
                {
                  "type": "function",
                  "function": {
                    "name": "get_candidate_status",
                    "arguments": {
                      "candidate_id": "12345"
                    }
                  }
                }
              ]
            })json"));
}

TEST(Lfm2DataProcessorTest, ToMessageWithParallelToolCalls) {
  JsonPreface preface;
  preface.tools = nlohmann::ordered_json::array();
  preface.tools.push_back(
      json{{"type", "function"}, {"function", {{"name", "get_weather"}}}});
  ASSERT_OK_AND_ASSIGN(auto processor, Lfm2DataProcessor::Create(
                                           Lfm2DataProcessorConfig{}, preface));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(
          Responses(TaskState::kProcessing,
                    {"<|tool_call_start|>[get_weather(location=\"London\"), "
                     "get_weather(location=\"Paris\")]<|tool_call_end|>"}),
          std::monostate{}));

  EXPECT_EQ(message, nlohmann::ordered_json::parse(R"json({
              "role": "assistant",
              "tool_calls": [
                {
                  "type": "function",
                  "function": {
                    "name": "get_weather",
                    "arguments": {
                      "location": "London"
                    }
                  }
                },
                {
                  "type": "function",
                  "function": {
                    "name": "get_weather",
                    "arguments": {
                      "location": "Paris"
                    }
                  }
                }
              ]
            })json"));
}

TEST(Lfm2DataProcessorTest, CodeFence) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       Lfm2DataProcessor::Create(Lfm2DataProcessorConfig{}));
  EXPECT_EQ(processor->CodeFenceStart(), "<|tool_call_start|>");
  EXPECT_EQ(processor->CodeFenceEnd(), "<|tool_call_end|>");
}

}  // namespace
}  // namespace litert::lm
