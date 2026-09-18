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

#include "runtime/conversation/prompt_utils.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using json = nlohmann::ordered_json;

TEST(PromptUtilsTest, StripBlobsFromTemplateInput_Messages) {
  PromptTemplateInput input{
      .messages = json::array({
          json::object({
              {"role", "user"},
              {"content",
               json::array({
                   json::object({{"type", "text"}, {"text", "hello"}}),
                   json::object({{"type", "image"},
                                 {"blob", "LARGE_BLOB_DATA"},
                                 {"path", "path/to/img"}}),
               })},
          }),
      }),
  };

  StripBlobsFromTemplateInput(input);

  // Verify blob is removed.
  const auto& parts = input.messages[0]["content"];
  EXPECT_EQ(parts[0]["type"], "text");
  EXPECT_EQ(parts[0]["text"], "hello");
  EXPECT_FALSE(parts[0].contains("blob"));

  EXPECT_EQ(parts[1]["type"], "image");
  EXPECT_EQ(parts[1]["path"], "path/to/img");
  EXPECT_FALSE(parts[1].contains("blob"));
}

TEST(PromptUtilsTest, StripBlobsFromTemplateInput_ExtraContextMessage) {
  PromptTemplateInput input{
      .extra_context = json::object({
          {"message",
           json::object({
               {"role", "user"},
               {"content", json::array({
                               json::object({{"type", "image"},
                                             {"blob", "LARGE_BLOB_DATA"}}),
                           })},
           })},
      }),
  };

  StripBlobsFromTemplateInput(input);

  // Verify blob is removed.
  const auto& content = input.extra_context["message"]["content"];
  EXPECT_EQ(content[0]["type"], "image");
  EXPECT_FALSE(content[0].contains("blob"));
}

TEST(PromptUtilsTest, RenderSingleTurnTemplate_DummyUserMessageNormalized) {
  ASSERT_OK_AND_ASSIGN(auto processor, Gemma3DataProcessor::Create());
  PromptTemplate prompt_template(
      "{% for msg in messages %}"
      "{{ msg.role }}:{% for part in msg.content %}[{{ part.type }}:{{ "
      "part.text }}]{% endfor %}|"
      "{% endfor %}");
  std::vector<Message> history;
  Preface preface = JsonPreface{
      .messages = {{{"role", "system"}, {"content", "System prompt"}}}};
  ASSERT_OK_AND_ASSIGN(
      auto result, RenderSingleTurnTemplate(
                       *processor, history, preface, Message(), prompt_template,
                       /*current_is_appending_message=*/false,
                       /*append_message=*/false));
  EXPECT_EQ(result.text, "system:[text:System prompt]|user:[text:]|");
}

}  // namespace
}  // namespace litert::lm
