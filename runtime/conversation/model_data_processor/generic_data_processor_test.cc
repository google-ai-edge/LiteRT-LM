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

#include "runtime/conversation/model_data_processor/generic_data_processor.h"

#include <string>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/generic_data_processor_config.h"
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
  if (!text_bytes.ok()) {
    return false;
  }
  return text_bytes.value() == text_input->GetRawTextString().value();
}

TEST(GenericDataProcessorTest, ToInputDataVector) {
  ASSERT_OK_AND_ASSIGN(auto processor, GenericDataProcessor::Create());
  const std::string rendered_template_prompt =
      "<start_of_turn>user\ntest "
      "prompt\n<end_of_turn>\n<start_of_turn>assistant\ntest "
      "response\n<end_of_turn>";
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
      "<start_of_turn>user\ntest "
      "prompt\n<end_of_turn>\n<start_of_turn>assistant\ntest "
      "response\n<end_of_turn>");
  EXPECT_THAT(input_data, ElementsAre(HasInputText(&expected_text)));
}

TEST(GenericDataProcessorTest, ToMessageDefault) {
  ASSERT_OK_AND_ASSIGN(auto processor, GenericDataProcessor::Create());

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(Responses(TaskState::kProcessing, {"test response"}),
                           std::monostate{}));

  ASSERT_TRUE(std::holds_alternative<nlohmann::ordered_json>(message));
  const nlohmann::ordered_json& json_message =
      std::get<nlohmann::ordered_json>(message);
  EXPECT_EQ(
      json_message,
      json({{"role", "assistant"},
            {"content", {{{"type", "text"}, {"text", "test response"}}}}}));
}

TEST(GenericDataProcessorTest, ToMessageModelRole) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       GenericDataProcessor::Create(
                           GenericDataProcessorConfig{.model_role = "model"}));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(Responses(TaskState::kProcessing, {"test response"}),
                           std::monostate{}));

  ASSERT_TRUE(std::holds_alternative<nlohmann::ordered_json>(message));
  const nlohmann::ordered_json& json_message =
      std::get<nlohmann::ordered_json>(message);
  EXPECT_EQ(
      json_message,
      json({{"role", "model"},
            {"content", {{{"type", "text"}, {"text", "test response"}}}}}));
}

TEST(GenericDataProcessorTest, ToTemplateInputNoTypedContent) {
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      GenericDataProcessor::Create(
          GenericDataProcessorConfig{.model_role = "model"},
          PromptTemplateCapabilities{.requires_typed_content = false}));
  ASSERT_OK_AND_ASSIGN(const json template_input_1,
                       processor->MessageToTemplateInput(json(
                           {{"role", "user"}, {"content", "test prompt"}})));
  EXPECT_EQ(template_input_1,
            json({{"role", "user"}, {"content", "test prompt"}}));
  ASSERT_OK_AND_ASSIGN(
      const json template_input_2,
      processor->MessageToTemplateInput(
          json({{"role", "user"},
                {"content", {{{"type", "text"}, {"text", "test prompt"}}}}})));
  EXPECT_EQ(template_input_2,
            json({{"role", "user"}, {"content", "test prompt"}}));
}

TEST(GenericDataProcessorTest, ToTemplateInputTypedContent) {
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      GenericDataProcessor::Create(
          GenericDataProcessorConfig{.model_role = "model"},
          PromptTemplateCapabilities{.requires_typed_content = true}));
  ASSERT_OK_AND_ASSIGN(const json template_input_1,
                       processor->MessageToTemplateInput(json(
                           {{"role", "user"}, {"content", "test prompt"}})));
  EXPECT_EQ(template_input_1,
            json({{"role", "user"},
                  {"content", {{{"type", "text"}, {"text", "test prompt"}}}}}));
  ASSERT_OK_AND_ASSIGN(
      const json template_input_2,
      processor->MessageToTemplateInput(
          json({{"role", "user"},
                {"content", {{{"type", "text"}, {"text", "test prompt"}}}}})));
  EXPECT_EQ(template_input_2,
            json({{"role", "user"},
                  {"content", {{{"type", "text"}, {"text", "test prompt"}}}}}));
}

}  // namespace
}  // namespace litert::lm
