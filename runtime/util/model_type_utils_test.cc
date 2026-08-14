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

#include "runtime/util/model_type_utils.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/components/prompt_template.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/util/test_utils.h"  // NOLINT
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

using ::testing::NiceMock;
using ::testing::Return;
using TokenizerType = ::litert::support::TokenizerType;

class MockTokenizer : public Tokenizer {
 public:
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText,
              (absl::Span<const int> token_id, bool skip_special_tokens),
              (override));
  MOCK_METHOD(absl::StatusOr<std::vector<int>>, TextToTokenIds,
              (absl::string_view text), (override));
  MOCK_METHOD(absl::StatusOr<int>, TokenToId, (absl::string_view token),
              (override));
  MOCK_METHOD(TokenizerType, GetTokenizerType, (), (const, override));
  MOCK_METHOD(std::vector<std::string>, GetTokens, (), (const, override));
  MOCK_METHOD(int, GetVocabSize, (), (const, override));
};

TEST(ModelTypeUtilsTest, InferLlmModelTypeGemma3N) {
  NiceMock<MockTokenizer> tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText)
      .Times(testing::AnyNumber())
      .WillRepeatedly(Return("<start_of_turn>"));
  EXPECT_CALL(tokenizer, TextToTokenIds("<start_of_audio>"))
      .WillRepeatedly(Return(std::vector<int>{256000}));
  ASSERT_OK_AND_ASSIGN(auto model_type,
                       InferLlmModelType(proto::LlmMetadata(), &tokenizer));
  EXPECT_THAT(model_type.has_gemma3n(), true);
}

TEST(ModelTypeUtilsTest, InferLlmModelTypeGemma3NWrongAudioToken) {
  NiceMock<MockTokenizer> tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText)
      .Times(testing::AnyNumber())
      .WillRepeatedly(Return("<start_of_turn>"));
  EXPECT_CALL(tokenizer, TextToTokenIds("<start_of_audio>"))
      .WillRepeatedly(Return(
          // The encoded ids for "<start_of_audio>" in the Gemma3 1B tokenizer.
          std::vector<int>{256001}));
  ASSERT_OK_AND_ASSIGN(auto model_type,
                       InferLlmModelType(proto::LlmMetadata(), &tokenizer));
  EXPECT_THAT(model_type.has_gemma3n(), false);
}

TEST(ModelTypeUtilsTest, InferLlmModelTypeGemma3) {
  NiceMock<MockTokenizer> tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText)
      .Times(testing::AnyNumber())
      .WillRepeatedly(Return("<start_of_turn>"));
  EXPECT_CALL(tokenizer, TextToTokenIds("<start_of_audio>"))
      .WillRepeatedly(Return(
          // The encoded ids for "<start_of_audio>" in the Gemma3 1B tokenizer.
          std::vector<int>{236820, 3041, 236779, 1340, 236779, 20156, 236813}));
  ASSERT_OK_AND_ASSIGN(auto model_type,
                       InferLlmModelType(proto::LlmMetadata(), &tokenizer));
  EXPECT_THAT(model_type.has_gemma3(), true);
}

TEST(ModelTypeUtilsTest, InferLlmModelTypeGenericModel) {
  NiceMock<MockTokenizer> tokenizer;
  EXPECT_CALL(tokenizer, TokenIdsToText).WillRepeatedly(Return("Hello"));
  EXPECT_CALL(tokenizer, TextToTokenIds("<start_of_audio>"))
      .WillRepeatedly(Return(std::vector<int>{256000}));
  ASSERT_OK_AND_ASSIGN(auto model_type,
                       InferLlmModelType(proto::LlmMetadata(), &tokenizer));
  EXPECT_THAT(model_type.has_generic_model(), true);
}

TEST(ModelTypeUtilsTest, GetDefaultJinjaPromptTemplate) {
  proto::PromptTemplates prompt_templates;
  prompt_templates.mutable_user()->set_prefix("<start_of_turn>user\n");
  prompt_templates.mutable_model()->set_prefix("<start_of_turn>model\n");
  prompt_templates.mutable_system()->set_prefix("<start_of_turn>system\n");
  prompt_templates.mutable_user()->set_suffix("<end_of_turn>\n");
  prompt_templates.mutable_model()->set_suffix("<end_of_turn>\n");
  prompt_templates.mutable_system()->set_suffix("<end_of_turn>\n");
  proto::LlmModelType llm_model_type;
  llm_model_type.mutable_generic_model();
  ASSERT_OK_AND_ASSIGN(
      auto jinja_prompt_template,
      GetDefaultJinjaPromptTemplate(prompt_templates, llm_model_type));
  PromptTemplate prompt_template(jinja_prompt_template);
  PromptTemplateInput prompt_template_input;
  prompt_template_input.messages = {
      {{"role", "system"}, {"content", "This is a system message"}},
      {{"role", "user"}, {"content", "This is a user message"}},
      {{"role", "model"}, {"content", "This is a model message"}}};
  ASSERT_OK_AND_ASSIGN(auto rendered_prompt,
                       prompt_template.Apply(prompt_template_input));
  EXPECT_EQ(rendered_prompt,
            "<start_of_turn>system\nThis is a system message<end_of_turn>\n"
            "<start_of_turn>user\nThis is a user message<end_of_turn>\n"
            "<start_of_turn>model\nThis is a model message<end_of_turn>\n"
            "<start_of_turn>model\n");
}

TEST(ModelTypeUtilsTest, GetDefaultJinjaPromptTemplateEmpty) {
  proto::PromptTemplates prompt_templates;
  proto::LlmModelType llm_model_type;
  llm_model_type.mutable_generic_model();
  ASSERT_OK_AND_ASSIGN(
      auto jinja_prompt_template,
      GetDefaultJinjaPromptTemplate(prompt_templates, llm_model_type));
  PromptTemplate prompt_template(jinja_prompt_template);
  PromptTemplateInput prompt_template_input;
  prompt_template_input.messages = {
      {{"role", "system"}, {"content", "This is a system message"}},
      {{"role", "user"}, {"content", "This is a user message"}},
      {{"role", "model"}, {"content", "This is a model message"}}};
  ASSERT_OK_AND_ASSIGN(auto rendered_prompt,
                       prompt_template.Apply(prompt_template_input));
  EXPECT_EQ(rendered_prompt,
            "This is a system message"
            "This is a user message"
            "This is a model message");
}

TEST(ModelTypeUtilsTest, GetDefaultJinjaPromptTemplateWithImageAndAudio) {
  proto::PromptTemplates prompt_templates;
  prompt_templates.mutable_user()->set_prefix("<start_of_turn>user\n");
  prompt_templates.mutable_model()->set_prefix("<start_of_turn>model\n");
  prompt_templates.mutable_system()->set_prefix("<start_of_turn>system\n");
  prompt_templates.mutable_user()->set_suffix("<end_of_turn>\n");
  prompt_templates.mutable_model()->set_suffix("<end_of_turn>\n");
  prompt_templates.mutable_system()->set_suffix("<end_of_turn>\n");
  proto::LlmModelType llm_model_type;
  llm_model_type.mutable_generic_model();
  ASSERT_OK_AND_ASSIGN(
      auto jinja_prompt_template,
      GetDefaultJinjaPromptTemplate(prompt_templates, llm_model_type));
  PromptTemplate prompt_template(jinja_prompt_template);
  PromptTemplateInput prompt_template_input;
  prompt_template_input.messages = {
      {{"role", "system"}, {"content", "This is a system message"}},
      {
          {"role", "user"},
          {"content",
           {
               {{"type", "text"}, {"text", "Here is a user image "}},
               {{"type", "image"}, {"image", "image_bytes"}},
               {{"type", "text"}, {"text", " Here is a user audio "}},
               {{"type", "audio"}, {"audio", "audio_bytes"}},
           }},
      },
      {{"role", "model"}, {"content", "This is a model message"}}};
  ASSERT_OK_AND_ASSIGN(auto rendered_prompt,
                       prompt_template.Apply(prompt_template_input));
  EXPECT_EQ(rendered_prompt,
            "<start_of_turn>system\nThis is a system message<end_of_turn>\n"
            "<start_of_turn>user\nHere is a user image <start_of_image> Here "
            "is a user audio <start_of_audio><end_of_turn>\n"
            "<start_of_turn>model\nThis is a model message<end_of_turn>\n"
            "<start_of_turn>model\n");
}

TEST(ModelTypeUtilsTest, GetModelTypeName) {
  proto::LlmModelType model_type;
  EXPECT_EQ(GetModelTypeName(model_type), "Not set");

  model_type.mutable_generic_model();
  EXPECT_EQ(GetModelTypeName(model_type), "generic_model");

  model_type.mutable_gemma3n();
  EXPECT_EQ(GetModelTypeName(model_type), "gemma3n");

  model_type.mutable_function_gemma();
  EXPECT_EQ(GetModelTypeName(model_type), "function_gemma");

  model_type.mutable_gemma3();
  EXPECT_EQ(GetModelTypeName(model_type), "gemma3");

  model_type.mutable_qwen3();
  EXPECT_EQ(GetModelTypeName(model_type), "qwen3");

  model_type.mutable_qwen2p5();
  EXPECT_EQ(GetModelTypeName(model_type), "qwen2p5");

  model_type.mutable_gemma4();
  EXPECT_EQ(GetModelTypeName(model_type), "gemma4");

  model_type.mutable_fast_vlm();
  EXPECT_EQ(GetModelTypeName(model_type), "fast_vlm");

  model_type.mutable_lfm2();
  EXPECT_EQ(GetModelTypeName(model_type), "lfm2");

  model_type.mutable_minicpm5();
  EXPECT_EQ(GetModelTypeName(model_type), "minicpm5");
}

}  // namespace
}  // namespace litert::lm
