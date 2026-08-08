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

#include "runtime/conversation/model_data_processor/minicpm5_data_processor.h"

#include <filesystem>  // NOLINT
#include <fstream>
#include <iterator>
#include <string>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/minicpm5_data_processor_config.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using json = nlohmann::ordered_json;
using ::testing::ElementsAre;

constexpr char kTestdataDir[] =
    "litert_lm/runtime/components/testdata/";

std::string GetTestdataPath(const std::string& file_name) {
  return (std::filesystem::path(::testing::SrcDir()) / kTestdataDir / file_name)
      .string();
}

absl::StatusOr<std::string> GetContents(const std::string& path) {
  std::ifstream input_stream(path);
  if (!input_stream.is_open()) {
    return absl::InternalError(absl::StrCat("Could not open file: ", path));
  }
  return std::string((std::istreambuf_iterator<char>(input_stream)),
                     (std::istreambuf_iterator<char>()));
}

MATCHER_P(HasInputText, text_input, "") {
  if (!std::holds_alternative<InputText>(arg)) {
    return false;
  }
  auto text_bytes = std::get<InputText>(arg).GetRawTextString();
  auto expected_bytes = text_input->GetRawTextString();
  if (!text_bytes.ok() || !expected_bytes.ok()) {
    return false;
  }
  return *text_bytes == *expected_bytes;
}

TEST(MiniCpm5DataProcessorTest, ToInputDataVector) {
  ASSERT_OK_AND_ASSIGN(auto processor, MiniCpm5DataProcessor::Create(
                                           MiniCpm5DataProcessorConfig{}));
  const std::string rendered_template_prompt =
      "<|im_start|>user\ntest prompt<|im_end|>\n<|im_start|>assistant\n";
  const nlohmann::ordered_json messages = {
      {"role", "user"},
      {"content", "test prompt"},
  };
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_template_prompt, messages, {}));

  InputText expected_text(
      "<|im_start|>user\ntest prompt<|im_end|>\n<|im_start|>assistant\n");
  EXPECT_THAT(input_data, ElementsAre(HasInputText(&expected_text)));
}

TEST(MiniCpm5DataProcessorTest, ToMessageDefault) {
  ASSERT_OK_AND_ASSIGN(auto processor, MiniCpm5DataProcessor::Create(
                                           MiniCpm5DataProcessorConfig{}));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(Responses(TaskState::kProcessing, {"test response"}),
                           std::monostate{}));

  EXPECT_EQ(
      message,
      json({{"role", "assistant"},
            {"content", {{{"type", "text"}, {"text", "test response"}}}}}));
}

TEST(MiniCpm5DataProcessorTest, ToMessageToolCallWithCdataAndQuotes) {
  JsonPreface preface;
  preface.tools = json::array({json({{"name", "get_current_weather"}})});
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      MiniCpm5DataProcessor::Create(MiniCpm5DataProcessorConfig{}, preface));

  const std::string raw_response =
      "I will call weather tool.\n"
      "<function name='get_current_weather'>"
      "<param name=\"location\"><![CDATA[Tokyo, Japan]]></param>"
      "</function>";

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(Responses(TaskState::kProcessing, {raw_response}),
                           std::monostate{}));

  EXPECT_EQ(
      message,
      json({{"role", "assistant"},
            {"content",
             {{{"type", "text"}, {"text", "I will call weather tool.\n"}}}},
            {"tool_calls",
             {{{"type", "function"},
               {"function",
                {{"name", "get_current_weather"},
                 {"arguments", {{"location", "Tokyo, Japan"}}}}}}}}}));
}

TEST(MiniCpm5DataProcessorTest, ToMessageToolCallWithSpaces) {
  JsonPreface preface;
  preface.tools = json::array({json({{"name", "get_current_weather"}})});
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      MiniCpm5DataProcessor::Create(MiniCpm5DataProcessorConfig{}, preface));

  const std::string raw_response =
      "<function  name = \"get_current_weather\" >"
      "<param   name  =  'location' >Tokyo</param>"
      "</function>";

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(Responses(TaskState::kProcessing, {raw_response}),
                           std::monostate{}));

  EXPECT_EQ(message, json({{"role", "assistant"},
                           {"tool_calls",
                            {{{"type", "function"},
                              {"function",
                               {{"name", "get_current_weather"},
                                {"arguments", {{"location", "Tokyo"}}}}}}}}}));
}

TEST(MiniCpm5DataProcessorTest, CodeFence) {
  ASSERT_OK_AND_ASSIGN(auto processor, MiniCpm5DataProcessor::Create(
                                           MiniCpm5DataProcessorConfig{}));
  EXPECT_EQ(processor->CodeFenceStart(), "<function name=");
  EXPECT_EQ(processor->CodeFenceEnd(), "</function>");
}

TEST(MiniCpm5DataProcessorTest, TemplateEquivalenceThinkingDisabled) {
  const std::string template_path = GetTestdataPath("MiniCPM5-1B.jinja");
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(template_path));

  PromptTemplate prompt_template(template_content);
  PromptTemplateInput input{
      .messages = json::array(
          {{{"role", "user"}, {"content", "What is quantum computing?"}}}),
      .add_generation_prompt = true,
      .extra_context = json::object({{"enable_thinking", false}}),
  };

  ASSERT_OK_AND_ASSIGN(std::string rendered, prompt_template.Apply(input));

  const std::string expected =
      "<|im_start|>user\nWhat is quantum computing?<|im_end|>\n"
      "<|im_start|>assistant\n<think>\n\n</think>\n\n";

  EXPECT_EQ(rendered, expected);
}

TEST(MiniCpm5DataProcessorTest, TemplateEquivalenceThinkingEnabled) {
  const std::string template_path = GetTestdataPath("MiniCPM5-1B.jinja");
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(template_path));

  PromptTemplate prompt_template(template_content);
  PromptTemplateInput input{
      .messages = json::array(
          {{{"role", "user"}, {"content", "What is quantum computing?"}}}),
      .add_generation_prompt = true,
      .extra_context = json::object({{"enable_thinking", true}}),
  };

  ASSERT_OK_AND_ASSIGN(std::string rendered, prompt_template.Apply(input));

  const std::string expected =
      "<|im_start|>user\nWhat is quantum computing?<|im_end|>\n"
      "<|im_start|>assistant\n<think>\n";

  EXPECT_EQ(rendered, expected);
}

TEST(MiniCpm5DataProcessorTest, TemplateEquivalenceFunctionCalling) {
  const std::string template_path = GetTestdataPath("MiniCPM5-1B.jinja");
  ASSERT_OK_AND_ASSIGN(const std::string template_content,
                       GetContents(template_path));

  PromptTemplate prompt_template(template_content);
  PromptTemplateInput input{
      .messages = json::array(
          {{{"role", "user"}, {"content", "whats the weather in tokyo?"}}}),
      .tools = json::array(
          {{{"name", "get_current_weather"}, {"description", "Get weather"}}}),
      .add_generation_prompt = true,
      .extra_context = json::object({{"enable_thinking", false}}),
  };

  ASSERT_OK_AND_ASSIGN(std::string rendered, prompt_template.Apply(input));

  EXPECT_THAT(
      rendered,
      ::testing::HasSubstr(
          "<tools>\n{\"name\": \"get_current_weather\", \"description\": \"Get "
          "weather\"}\n</tools>"));

  EXPECT_THAT(
      rendered,
      ::testing::HasSubstr(
          "Tool usage guidelines:\n- You may call zero or more functions."));
  EXPECT_THAT(rendered,
              ::testing::HasSubstr("<|im_start|>user\nwhats the weather in "
                                   "tokyo?<|im_end|>"));
  EXPECT_THAT(rendered, ::testing::HasSubstr("<|im_start|>assistant\n"));
}

}  // namespace

}  // namespace litert::lm
