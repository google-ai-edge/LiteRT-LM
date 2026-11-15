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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/components/prompt_template.h"
#include "runtime/components/tool_use/parser_utils.h"
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/fastvlm_data_processor_config.h"
#include "runtime/conversation/model_data_processor/gemma3_data_processor_config.h"
#include "runtime/conversation/model_data_processor/gemma4_data_processor_config.h"
#include "runtime/conversation/model_data_processor/model_data_processor_factory.h"
#include "runtime/conversation/model_data_processor/qwen3_data_processor_config.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/llm_model_type.pb.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "support/tokenizer/huggingface_tokenizer.h"

namespace litert::lm {
namespace {

using ::testing::HasSubstr;
using ::testing::status::StatusIs;

constexpr absl::string_view kTestLlmPath =
    "litert_lm/runtime/testdata/test_lm.litertlm";

constexpr absl::string_view kTestHfTokenizerLlmPath =
    "litert_lm/runtime/testdata/test_hf_tokenizer.litertlm";

std::string GetTestdataPath(absl::string_view file_path) {
  return absl::StrCat(::testing::SrcDir(), "/", file_path);
}

TEST(ConversationLiteTest, PromptTemplateIsNoopInLiteTarget) {
  PromptTemplate tmpl("{{ message }}");
  PromptTemplateInput input;
  EXPECT_THAT(tmpl.Apply(input), StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(ConversationLiteTest, HuggingFaceTokenizerIsNoopInLiteTarget) {
  EXPECT_THAT(litert::support::HuggingFaceTokenizer::CreateFromJson("{}"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("HuggingFace tokenizer is disabled")));
}

TEST(ConversationLiteTest,
     ModelWithHuggingFaceTokenizerReportsDisabledErrorInLiteTarget) {
  auto model_file = ScopedFile::Open(GetTestdataPath(kTestHfTokenizerLlmPath));
  ASSERT_OK(model_file);
  ASSERT_OK_AND_ASSIGN(auto loader,
                       LitertLmLoader::Create(std::move(*model_file)));
  ASSERT_OK_AND_ASSIGN(auto resources,
                       ModelResourcesLitertLm::Create(std::move(loader)));
  EXPECT_THAT(resources->GetTokenizer(),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("HuggingFace tokenizer is disabled")));
}

TEST(ConversationLiteTest, ToolUseParserLiteReturnsErrorWithoutParsers) {
  EXPECT_THAT(
      ParseTextAndToolCalls("```tool_code\nfoo(x=1)\n```", "```tool_code\n",
                            "\n```", SyntaxType::kPython),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ConversationLiteTest, OnlyGemmaAndGenericProcessorsAreLinkedInLiteTarget) {
  EXPECT_OK(CreateModelDataProcessor(Gemma3DataProcessorConfig()));
  EXPECT_OK(CreateModelDataProcessor(Gemma4DataProcessorConfig()));
  EXPECT_THAT(CreateModelDataProcessor(Qwen3DataProcessorConfig()),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(CreateModelDataProcessor(FastVlmDataProcessorConfig()),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(ConversationLiteTest,
     CreateDataProcessorConfigFromLlmModelTypeInLiteTarget) {
  proto::LlmModelType gemma3_type;
  gemma3_type.mutable_gemma3();
  EXPECT_OK(CreateDataProcessorConfigFromLlmModelType(gemma3_type));

  proto::LlmModelType gemma3n_type;
  gemma3n_type.mutable_gemma3n();
  EXPECT_OK(CreateDataProcessorConfigFromLlmModelType(gemma3n_type));

  proto::LlmModelType gemma4_type;
  gemma4_type.mutable_gemma4();
  EXPECT_OK(CreateDataProcessorConfigFromLlmModelType(gemma4_type));

  proto::LlmModelType generic_type;
  generic_type.mutable_generic_model();
  EXPECT_OK(CreateDataProcessorConfigFromLlmModelType(generic_type));

  proto::LlmModelType qwen3_type;
  qwen3_type.mutable_qwen3();
  EXPECT_THAT(CreateDataProcessorConfigFromLlmModelType(qwen3_type),
              StatusIs(absl::StatusCode::kUnimplemented));

  proto::LlmModelType fastvlm_type;
  fastvlm_type.mutable_fast_vlm();
  EXPECT_THAT(CreateDataProcessorConfigFromLlmModelType(fastvlm_type),
              StatusIs(absl::StatusCode::kUnimplemented));

  proto::LlmModelType lfm2_type;
  lfm2_type.mutable_lfm2();
  EXPECT_THAT(CreateDataProcessorConfigFromLlmModelType(lfm2_type),
              StatusIs(absl::StatusCode::kUnimplemented));

  proto::LlmModelType function_gemma_type;
  function_gemma_type.mutable_function_gemma();
  EXPECT_THAT(CreateDataProcessorConfigFromLlmModelType(function_gemma_type),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(ConversationLiteTest,
     PositiveGemmaDataProcessorAndEngineSessionRoundTripSucceeds) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       CreateModelDataProcessor(Gemma3DataProcessorConfig()));
  const Message user_message = {{"role", "user"}, {"content", "Hello world!"}};
  ASSERT_OK_AND_ASSIGN(
      std::vector<InputData> input_data,
      processor->ToInputDataVector(
          "<start_of_turn>user\nHello world!<end_of_turn>\n",
          nlohmann::ordered_json::array({user_message}), std::monostate()));
  ASSERT_FALSE(input_data.empty());

  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  ASSERT_OK_AND_ASSIGN(auto engine,
                       EngineFactory::CreateDefault(engine_settings));
  ASSERT_OK_AND_ASSIGN(auto config, ConversationConfig::CreateDefault(*engine));
  EXPECT_OK(Conversation::Create(*engine, config));

  ASSERT_OK_AND_ASSIGN(auto session,
                       engine->CreateSession(config.GetSessionConfig()));
  ASSERT_OK(session->RunPrefill(input_data));
  ASSERT_OK_AND_ASSIGN(Responses responses, session->RunDecode());
  ASSERT_FALSE(responses.GetTexts().empty());

  ASSERT_OK_AND_ASSIGN(Message assistant_message,
                       processor->ToMessage(responses, std::monostate()));
  EXPECT_EQ(assistant_message["role"], "assistant");
  ASSERT_TRUE(assistant_message["content"].is_array());
  ASSERT_FALSE(assistant_message["content"].empty());
}

TEST(ConversationLiteTest, ConversationSendMessageEndToEndSucceedsInLiteTarget) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings,
                       EngineSettings::CreateDefault(model_assets,
                                                     Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(32);
  ASSERT_OK_AND_ASSIGN(auto engine,
                       EngineFactory::CreateDefault(engine_settings));
  ASSERT_OK_AND_ASSIGN(auto config, ConversationConfig::CreateDefault(*engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*engine, config));

  ASSERT_OK_AND_ASSIGN(
      Message response1,
      conversation->SendMessage({{"role", "user"}, {"content", "Hello!"}},
                                OptionalArgs{.max_output_tokens = 3}));
  EXPECT_EQ(response1["role"], "assistant");
  ASSERT_TRUE(response1["content"].is_array());
  ASSERT_FALSE(response1["content"].empty());

  ASSERT_OK_AND_ASSIGN(
      Message response2,
      conversation->SendMessage({{"role", "user"}, {"content", "Follow up"}},
                                OptionalArgs{.max_output_tokens = 3}));
  EXPECT_EQ(response2["role"], "assistant");
  EXPECT_EQ(conversation->GetHistory().size(), 4);
}

TEST(ConversationLiteTest,
     FallbackPromptTemplateRendersGemma3AndMultimodalMessages) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings,
                       EngineSettings::CreateDefault(model_assets,
                                                     Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(15);
  ASSERT_OK_AND_ASSIGN(auto engine,
                       EngineFactory::CreateDefault(engine_settings));

  JsonPreface preface{
      .messages = nlohmann::ordered_json::array(
          {{{"role", "system"}, {"content", "You are a translator."}}})};
  ASSERT_OK_AND_ASSIGN(
      auto config,
      ConversationConfig::Builder()
          .SetPreface(preface)
          .SetOverwriteProcessorConfig(Gemma3DataProcessorConfig())
          .Build(*engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*engine, config));

  ASSERT_OK_AND_ASSIGN(std::string rendered_preface,
                       conversation->RenderPrefaceIntoString(OptionalArgs()));
  EXPECT_EQ(rendered_preface,
            "<start_of_turn>system\nYou are a translator.<end_of_turn>\n");

  Message multimodal_message = {
      {"role", "user"},
      {"content", nlohmann::ordered_json::array(
                      {{{"type", "text"}, {"text", "Translate: "}},
                       {{"type", "audio"}, {"blob", "dummy_audio"}},
                       {{"type", "image"}, {"blob", "dummy_image"}}})}};
  ASSERT_OK_AND_ASSIGN(
      std::string rendered_turn,
      conversation->RenderMessageIntoString(multimodal_message, OptionalArgs()));
  EXPECT_EQ(rendered_turn,
            "<start_of_turn>system\nYou are a translator.<end_of_turn>\n"
            "<start_of_turn>user\nTranslate: "
            "<start_of_audio><start_of_image><end_of_turn>\n"
            "<start_of_turn>model\n");
}

TEST(ConversationLiteTest,
     FallbackPromptTemplateRendersGemma4AndCustomPromptTemplates) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings,
                       EngineSettings::CreateDefault(model_assets,
                                                     Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(15);
  ASSERT_OK_AND_ASSIGN(auto engine,
                       EngineFactory::CreateDefault(engine_settings));

  ASSERT_OK_AND_ASSIGN(
      auto gemma4_config,
      ConversationConfig::Builder()
          .SetOverwriteProcessorConfig(Gemma4DataProcessorConfig())
          .Build(*engine));
  ASSERT_OK_AND_ASSIGN(auto gemma4_conversation,
                       Conversation::Create(*engine, gemma4_config));
  ASSERT_OK_AND_ASSIGN(
      std::string gemma4_rendered,
      gemma4_conversation->RenderMessageIntoString(
          {{"role", "user"},
           {"content",
            nlohmann::ordered_json::array(
                {{{"type", "text"}, {"text", "Transcribe: "}},
                 {{"type", "audio"}, {"blob", "dummy_audio"}}})}},
          OptionalArgs()));
  EXPECT_EQ(gemma4_rendered,
            "<|turn>user\nTranscribe: <start_of_audio><turn|>\n<|turn>model\n");

  SessionConfig custom_session_config = SessionConfig::CreateDefault();
  custom_session_config.GetMutablePromptTemplates().mutable_user()->set_prefix(
      "<u_start>");
  custom_session_config.GetMutablePromptTemplates().mutable_user()->set_suffix(
      "<u_end>");
  custom_session_config.GetMutablePromptTemplates().mutable_model()->set_prefix(
      "<m_start>");
  custom_session_config.GetMutablePromptTemplates().mutable_model()->set_suffix(
      "<m_end>");
  ASSERT_OK_AND_ASSIGN(auto custom_config,
                       ConversationConfig::Builder()
                           .SetSessionConfig(custom_session_config)
                           .Build(*engine));
  ASSERT_OK_AND_ASSIGN(auto custom_conversation,
                       Conversation::Create(*engine, custom_config));
  ASSERT_OK_AND_ASSIGN(
      std::string custom_rendered,
      custom_conversation->RenderMessageIntoString(
          {{"role", "user"}, {"content", "Ping"}}, OptionalArgs()));
  EXPECT_EQ(custom_rendered, "<u_start>Ping<u_end><m_start>");
}

}  // namespace
}  // namespace litert::lm
