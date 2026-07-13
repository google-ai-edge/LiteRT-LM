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

#include "c/engine.h"

#include <fcntl.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/logits_processor/no_repeat_ngram_config.h"
#include "runtime/components/logits_processor/repetition_penalty_config.h"
#include "runtime/components/logits_processor/suppress_tokens_config.h"
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/thinking_config.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"

using ::litert::lm::Conversation;
using ::litert::lm::EngineSettings;
using ::litert::lm::NoRepeatNgramConfig;
using ::litert::lm::RepetitionPenaltyConfig;
using ::litert::lm::SessionConfig;
using ::litert::lm::SuppressTokensConfig;

struct LiteRtLmEngineSettings {
  std::unique_ptr<EngineSettings> settings;
};

struct LiteRtLmSessionConfig {
  std::unique_ptr<SessionConfig> config;
};

struct LiteRtLmConversationOptionalArgs {
  std::optional<RepetitionPenaltyConfig> repetition_penalty_config;
  std::optional<NoRepeatNgramConfig> no_repeat_ngram_config;
  std::optional<SuppressTokensConfig> suppress_tokens_config;
  std::optional<int> visual_token_budget;
  std::optional<int> max_output_tokens;
  std::optional<litert::lm::ThinkingConfig> thinking_config;
};

struct LiteRtLmConversationConfig {
  std::optional<SessionConfig> session_config;
  std::string system_message_json;
  std::string tools_json;
  std::string messages_json;
  std::string extra_context_json;
  bool enable_constrained_decoding = false;
  bool filter_channel_content_from_kv_cache = false;
  bool stream_tool_calls = false;
  std::string stream_tool_calls_channel_name = "tool_call";
  std::optional<litert::lm::ThinkingConfig> thinking_config;
};

struct LiteRtLmConversation {
  std::unique_ptr<Conversation> conversation;
};

struct LiteRtLmJsonResponse {
  std::string json_string;
};

namespace {

std::string GetTestdataPath(const std::string& filename) {
  std::string srcdir = ::testing::SrcDir();
  // On Windows, SrcDir() may return paths with backslashes. The LiteRT LM C API
  // expects forward slashes.
  std::replace(srcdir.begin(), srcdir.end(), '\\', '/');
  return srcdir + "/" + filename;
}

// Use unique_ptr for automatic resource management of C API objects.
using EngineSettingsPtr =
    std::unique_ptr<LiteRtLmEngineSettings,
                    decltype(&litert_lm_engine_settings_delete)>;
using EnginePtr =
    std::unique_ptr<LiteRtLmEngine, decltype(&litert_lm_engine_delete)>;
using SessionPtr =
    std::unique_ptr<LiteRtLmSession, decltype(&litert_lm_session_delete)>;
using ResponsesPtr =
    std::unique_ptr<LiteRtLmResponses, decltype(&litert_lm_responses_delete)>;
using InputDataPtr =
    std::unique_ptr<LiteRtLmInputData, decltype(&litert_lm_input_data_delete)>;
using ConversationPtr =
    std::unique_ptr<LiteRtLmConversation,
                    decltype(&litert_lm_conversation_delete)>;
using JsonResponsePtr =
    std::unique_ptr<LiteRtLmJsonResponse,
                    decltype(&litert_lm_json_response_delete)>;
using SessionConfigPtr =
    std::unique_ptr<LiteRtLmSessionConfig,
                    decltype(&litert_lm_session_config_delete)>;
using SamplerParamsPtr =
    std::unique_ptr<LiteRtLmSamplerParams,
                    decltype(&litert_lm_sampler_params_delete)>;
using ConversationConfigPtr =
    std::unique_ptr<LiteRtLmConversationConfig,
                    decltype(&litert_lm_conversation_config_delete)>;
using RepetitionPenaltyConfigPtr =
    std::unique_ptr<LiteRtLmRepetitionPenaltyConfig,
                    decltype(&litert_lm_repetition_penalty_config_delete)>;
using NoRepeatNgramConfigPtr =
    std::unique_ptr<LiteRtLmNoRepeatNgramConfig,
                    decltype(&litert_lm_no_repeat_ngram_config_delete)>;
using SuppressTokensConfigPtr =
    std::unique_ptr<LiteRtLmSuppressTokensConfig,
                    decltype(&litert_lm_suppress_tokens_config_delete)>;
using OptionalArgsPtr =
    std::unique_ptr<LiteRtLmConversationOptionalArgs,
                    decltype(&litert_lm_conversation_optional_args_delete)>;
using TokenizeResultPtr =
    std::unique_ptr<LiteRtLmTokenizeResult,
                    decltype(&litert_lm_tokenize_result_delete)>;
using DetokenizeResultPtr =
    std::unique_ptr<LiteRtLmDetokenizeResult,
                    decltype(&litert_lm_detokenize_result_delete)>;
using TokenUnionPtr = std::unique_ptr<LiteRtLmTokenUnion,
                                      decltype(&litert_lm_token_union_delete)>;
using TokenUnionsPtr =
    std::unique_ptr<LiteRtLmTokenUnions,
                    decltype(&litert_lm_token_unions_delete)>;

TEST(EngineCTest, CreateSettingsWithNoVisionAndAudioBackend) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  EXPECT_FALSE(settings->settings->GetVisionExecutorSettings().has_value());
  EXPECT_FALSE(settings->settings->GetAudioExecutorSettings().has_value());
}

TEST(EngineCTest, CreateSettingsWithVisionAndAudioBackend) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ "gpu",
                                       /* audio_backend_str */ "cpu"),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  EXPECT_TRUE(settings->settings->GetVisionExecutorSettings().has_value());
  EXPECT_TRUE(settings->settings->GetAudioExecutorSettings().has_value());
  EXPECT_EQ(settings->settings->GetVisionExecutorSettings()->GetBackend(),
            litert::lm::Backend::GPU);
  EXPECT_EQ(settings->settings->GetAudioExecutorSettings()->GetBackend(),
            litert::lm::Backend::CPU);
}

TEST(EngineCTest, CreateSettingsWithInvalidVisionBackend) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ "dummy_backend",
                                       /* audio_backend_str */ "cpu"),
      &litert_lm_engine_settings_delete);
  ASSERT_EQ(settings, nullptr);
}

TEST(EngineCTest, SetCacheDir) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  const std::string cache_dir = "test_cache_dir";
  litert_lm_engine_settings_set_cache_dir(settings.get(), cache_dir.c_str());
  EXPECT_EQ(settings->settings->GetMainExecutorSettings().GetCacheDir(),
            cache_dir);
}

TEST(EngineCTest, SetCacheDirWithVisionAndAudio) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ "gpu",
                                       /* audio_backend_str */ "cpu"),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  EXPECT_TRUE(settings->settings->GetVisionExecutorSettings().has_value());
  EXPECT_TRUE(settings->settings->GetAudioExecutorSettings().has_value());

  const std::string cache_dir = "test_cache_dir";
  litert_lm_engine_settings_set_cache_dir(settings.get(), cache_dir.c_str());

  EXPECT_EQ(settings->settings->GetMainExecutorSettings().GetCacheDir(),
            cache_dir);
  EXPECT_EQ(settings->settings->GetVisionExecutorSettings()->GetCacheDir(),
            cache_dir);
  EXPECT_EQ(settings->settings->GetAudioExecutorSettings()->GetCacheDir(),
            cache_dir);
}

TEST(EngineCTest, SetMaxNumImages) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_images(settings.get(), 10);
  EXPECT_EQ(settings->settings->GetMainExecutorSettings().GetMaxNumImages(),
            10);
}

TEST(EngineCTest, SetPrefillChunkSize) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  int prefill_chunk_size = 128;
  litert_lm_engine_settings_set_prefill_chunk_size(settings.get(),
                                                   prefill_chunk_size);
  auto config = settings->settings->GetMainExecutorSettings()
                    .GetBackendConfig<litert::lm::CpuConfig>();
  ASSERT_TRUE(config.ok());
  EXPECT_EQ(config->prefill_chunk_size, prefill_chunk_size);
}
TEST(EngineCTest, SetParallelFileSectionLoading) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);

  // Default should be true.
  EXPECT_TRUE(settings->settings->GetParallelFileSectionLoading());

  litert_lm_engine_settings_set_parallel_file_section_loading(settings.get(),
                                                              false);
  EXPECT_FALSE(settings->settings->GetParallelFileSectionLoading());

  litert_lm_engine_settings_set_parallel_file_section_loading(settings.get(),
                                                              true);
  EXPECT_TRUE(settings->settings->GetParallelFileSectionLoading());
}

TEST(EngineCTest, BenchmarkSettings) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);

  litert_lm_engine_settings_enable_benchmark(settings.get());
  litert_lm_engine_settings_set_num_prefill_tokens(settings.get(), 100);
  litert_lm_engine_settings_set_num_decode_tokens(settings.get(), 200);

  const auto& params = settings->settings->GetBenchmarkParams();
  EXPECT_EQ(params->num_prefill_tokens(), 100);
  EXPECT_EQ(params->num_decode_tokens(), 200);
}

TEST(EngineCTest, SetEnableSpeculativeDecoding) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);

  // Default should be false.
  EXPECT_FALSE(settings->settings->GetMainExecutorSettings()
                   .GetAdvancedSettings()
                   .value_or(litert::lm::AdvancedSettings())
                   .enable_speculative_decoding);

  litert_lm_engine_settings_set_enable_speculative_decoding(settings.get(),
                                                            true);
  EXPECT_TRUE(settings->settings->GetMainExecutorSettings()
                  .GetAdvancedSettings()
                  .value_or(litert::lm::AdvancedSettings())
                  .enable_speculative_decoding);

  litert_lm_engine_settings_set_enable_speculative_decoding(settings.get(),
                                                            false);
  EXPECT_FALSE(settings->settings->GetMainExecutorSettings()
                   .GetAdvancedSettings()
                   .value_or(litert::lm::AdvancedSettings())
                   .enable_speculative_decoding);
}

TEST(EngineCTest, CreateSettingsFromRawFileDescriptor) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");
  int fd = open(task_path.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);
  EngineSettingsPtr settings(
      litert_lm_engine_settings_create_from_raw_file_descriptor(
          fd, "cpu", /* vision_backend_str */ nullptr,
          /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  EXPECT_TRUE(settings->settings->GetMainExecutorSettings()
                  .GetModelAssets()
                  .HasScopedFile());
  EXPECT_FALSE(settings->settings->GetMainExecutorSettings()
                   .GetModelAssets()
                   .GetPath()
                   .ok());
}

TEST(EngineCTest, CreateSessionConfigWithSamplerParams) {
  SamplerParamsPtr sampler_params(
      litert_lm_sampler_params_create(kLiteRtLmSamplerTypeTopP),
      &litert_lm_sampler_params_delete);
  ASSERT_NE(sampler_params, nullptr);
  litert_lm_sampler_params_set_top_k(sampler_params.get(), 10);
  litert_lm_sampler_params_set_top_p(sampler_params.get(), 0.5f);
  litert_lm_sampler_params_set_temperature(sampler_params.get(), 0.1f);
  litert_lm_sampler_params_set_seed(sampler_params.get(), 1234);

  SessionConfigPtr config(litert_lm_session_config_create(),
                          &litert_lm_session_config_delete);
  ASSERT_NE(config, nullptr);
  litert_lm_session_config_set_sampler_params(config.get(),
                                              sampler_params.get());

  const auto& params = config->config->GetSamplerParams();
  EXPECT_EQ(params.k(), 10);
  EXPECT_FLOAT_EQ(params.p(), 0.5f);
  EXPECT_FLOAT_EQ(params.temperature(), 0.1f);
  EXPECT_EQ(params.seed(), 1234);
}

TEST(EngineCTest, CreateSessionConfigWithNoSamplerParams) {
  SessionConfigPtr config(litert_lm_session_config_create(),
                          &litert_lm_session_config_delete);
  ASSERT_NE(config, nullptr);

  // Verify that the default sampler parameters are used.
  const auto& params = config->config->GetSamplerParams();
  EXPECT_EQ(params.type(),
            litert::lm::proto::SamplerParameters::TYPE_UNSPECIFIED);
}

TEST(EngineCTest, CreateSessionConfigWithApplyPromptTemplate) {
  SessionConfigPtr config(litert_lm_session_config_create(),
                          &litert_lm_session_config_delete);
  ASSERT_NE(config, nullptr);

  // By default, it is true.
  EXPECT_TRUE(config->config->GetApplyPromptTemplateInSession());

  litert_lm_session_config_set_apply_prompt_template(config.get(), false);
  EXPECT_FALSE(config->config->GetApplyPromptTemplateInSession());

  litert_lm_session_config_set_apply_prompt_template(config.get(), true);
  EXPECT_TRUE(config->config->GetApplyPromptTemplateInSession());
}

TEST(EngineCTest, CreateConversationConfig) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create Sampler Params.
  SamplerParamsPtr sampler_params(
      litert_lm_sampler_params_create(kLiteRtLmSamplerTypeTopP),
      &litert_lm_sampler_params_delete);
  ASSERT_NE(sampler_params, nullptr);
  litert_lm_sampler_params_set_top_k(sampler_params.get(), 10);
  litert_lm_sampler_params_set_top_p(sampler_params.get(), 0.5f);
  litert_lm_sampler_params_set_temperature(sampler_params.get(), 0.1f);
  litert_lm_sampler_params_set_seed(sampler_params.get(), 1234);
  SessionConfigPtr session_config(litert_lm_session_config_create(),
                                  &litert_lm_session_config_delete);
  ASSERT_NE(session_config, nullptr);
  litert_lm_session_config_set_sampler_params(session_config.get(),
                                              sampler_params.get());

  // 3. Create a Conversation Config with the Engine Handle, Session Config
  // and System Message.
  const std::string system_message =
      R"({"type":"text","text":"You are a helpful assistant."})";
  ConversationConfigPtr conversation_config(
      litert_lm_conversation_config_create(),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_session_config(conversation_config.get(),
                                                   session_config.get());
  litert_lm_conversation_config_set_system_message(conversation_config.get(),
                                                   system_message.c_str());

  // 4. Test to see if the Conversation has the Sampler Params.
  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const auto& params = conversation->conversation->GetConfig()
                           .GetSessionConfig()
                           .GetSamplerParams();
  EXPECT_EQ(params.k(), 10);
  EXPECT_FLOAT_EQ(params.p(), 0.5f);
  EXPECT_FLOAT_EQ(params.temperature(), 0.1f);
  EXPECT_EQ(params.seed(), 1234);

  // 5. Test to see if the Conversation has the correct System Message.
  const auto& preface = std::get<litert::lm::JsonPreface>(
      conversation->conversation->GetConfig().GetPreface());
  nlohmann::ordered_json message;
  message["role"] = "system";
  message["content"] = nlohmann::ordered_json::parse(system_message);
  nlohmann::ordered_json expected_messages =
      nlohmann::ordered_json::array({message});
  EXPECT_EQ(preface.messages, expected_messages);
}

TEST(EngineCTest, CreateConversationConfigWithNoSamplerParams) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create a Conversation Config with the System Message.
  const std::string system_message =
      R"({"type":"text","text":"You are a helpful assistant."})";
  ConversationConfigPtr conversation_config(
      litert_lm_conversation_config_create(),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_system_message(conversation_config.get(),
                                                   system_message.c_str());

  // 3. Test to see if the Conversation has the correct System Message.
  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const auto& preface = std::get<litert::lm::JsonPreface>(
      conversation->conversation->GetConfig().GetPreface());
  nlohmann::ordered_json message;
  message["role"] = "system";
  message["content"] = nlohmann::ordered_json::parse(system_message);
  nlohmann::ordered_json expected_messages =
      nlohmann::ordered_json::array({message});
  EXPECT_EQ(preface.messages, expected_messages);
}

TEST(EngineCTest, CreateConversationConfigWithNoSamplerParamsNoSystemMessage) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create a Conversation Config with the Session Config.
  SessionConfigPtr session_config(litert_lm_session_config_create(),
                                  &litert_lm_session_config_delete);
  ASSERT_NE(session_config, nullptr);
  ConversationConfigPtr conversation_config(
      litert_lm_conversation_config_create(),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_session_config(conversation_config.get(),
                                                   session_config.get());

  // 4. Test to see if the Conversation has the correct System Message.
  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const auto& preface = std::get<litert::lm::JsonPreface>(
      conversation->conversation->GetConfig().GetPreface());
  EXPECT_EQ(preface.messages, nullptr);
}

TEST(EngineCTest, CreateConversationConfigWithSamplerBackend) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionConfigPtr session_config(litert_lm_session_config_create(),
                                  &litert_lm_session_config_delete);
  ASSERT_NE(session_config, nullptr);
  session_config->config->SetSamplerBackend(litert::lm::Backend::GPU);

  ConversationConfigPtr conversation_config(
      litert_lm_conversation_config_create(),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_session_config(conversation_config.get(),
                                                   session_config.get());

  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const auto& final_session_config =
      conversation->conversation->GetConfig().GetSessionConfig();
  EXPECT_EQ(final_session_config.GetSamplerBackend(), litert::lm::Backend::GPU);
}

TEST(EngineCTest, CreateConversationConfigWithTools) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create a Conversation Config with tools.
  const std::string tools_json = R"([
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["location"]
        }
      }
    }
  ])";

  ConversationConfigPtr conversation_config(
      litert_lm_conversation_config_create(),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_tools(conversation_config.get(),
                                          tools_json.c_str());

  // 3. Test to see if the Conversation has the correct tools.
  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const auto& preface = std::get<litert::lm::JsonPreface>(
      conversation->conversation->GetConfig().GetPreface());
  EXPECT_EQ(preface.tools, nlohmann::ordered_json::parse(tools_json));
}

TEST(EngineCTest, CreateConversationConfigWithInvalidTools) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create a Conversation Config with an invalid tools json.
  const std::string tools_json = R"({"type": "function"})";  // Not an array

  ConversationConfigPtr conversation_config(
      litert_lm_conversation_config_create(),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_tools(conversation_config.get(),
                                          tools_json.c_str());

  // 3. Test to see if the Conversation has no tools.
  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const auto& preface = std::get<litert::lm::JsonPreface>(
      conversation->conversation->GetConfig().GetPreface());
  EXPECT_TRUE(preface.tools.is_null());
}

TEST(EngineCTest, CreateConversationConfigWithEmptyToolsArray) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create a Conversation Config with an empty tools array.
  const std::string tools_json = R"([])";

  ConversationConfigPtr conversation_config(
      litert_lm_conversation_config_create(),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_tools(conversation_config.get(),
                                          tools_json.c_str());

  // 3. Test to see if the Conversation has empty tools.
  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const auto& preface = std::get<litert::lm::JsonPreface>(
      conversation->conversation->GetConfig().GetPreface());
  EXPECT_TRUE(preface.tools.is_array());
  EXPECT_TRUE(preface.tools.empty());
}

TEST(EngineCTest, CreateConversationConfigWithMalformedToolsJson) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create a Conversation Config with malformed tools json.
  const std::string tools_json = R"([{"type": "function", ...}])";

  ConversationConfigPtr conversation_config(
      litert_lm_conversation_config_create(),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_tools(conversation_config.get(),
                                          tools_json.c_str());

  // 3. Test to see if the Conversation has no tools.
  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const auto& preface = std::get<litert::lm::JsonPreface>(
      conversation->conversation->GetConfig().GetPreface());
  EXPECT_TRUE(preface.tools.is_null());
}

TEST(EngineCTest, CreateConversationConfigWithNoSystemMessage) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create Sampler Params.
  SamplerParamsPtr sampler_params(
      litert_lm_sampler_params_create(kLiteRtLmSamplerTypeTopP),
      &litert_lm_sampler_params_delete);
  ASSERT_NE(sampler_params, nullptr);
  litert_lm_sampler_params_set_top_k(sampler_params.get(), 10);
  litert_lm_sampler_params_set_top_p(sampler_params.get(), 0.5f);
  litert_lm_sampler_params_set_temperature(sampler_params.get(), 0.1f);
  litert_lm_sampler_params_set_seed(sampler_params.get(), 1234);
  SessionConfigPtr session_config(litert_lm_session_config_create(),
                                  &litert_lm_session_config_delete);
  ASSERT_NE(session_config, nullptr);
  litert_lm_session_config_set_sampler_params(session_config.get(),
                                              sampler_params.get());

  // 3. Create a Conversation Config with the Session Config.
  ConversationConfigPtr conversation_config(
      litert_lm_conversation_config_create(),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_session_config(conversation_config.get(),
                                                   session_config.get());

  // 4. Test to see if the Conversation has the default Sampler Params.
  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const auto& params = conversation->conversation->GetConfig()
                           .GetSessionConfig()
                           .GetSamplerParams();
  EXPECT_EQ(params.k(), 10);
  EXPECT_FLOAT_EQ(params.p(), 0.5f);
  EXPECT_FLOAT_EQ(params.temperature(), 0.1f);
  EXPECT_EQ(params.seed(), 1234);

  // 5. Test to see if the Conversation has the correct System Message.
  const auto& preface = std::get<litert::lm::JsonPreface>(
      conversation->conversation->GetConfig().GetPreface());
  EXPECT_EQ(preface.messages, nullptr);
}

TEST(EngineCTest, ThinkingConfig) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm.litertlm");
  EngineSettingsPtr settings(litert_lm_engine_settings_create(
                                 task_path.c_str(), "cpu", nullptr, nullptr),
                             &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionConfigPtr session_config(litert_lm_session_config_create(),
                                  &litert_lm_session_config_delete);
  ASSERT_NE(session_config, nullptr);

  ConversationConfigPtr conversation_config(
      litert_lm_conversation_config_create(),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_session_config(conversation_config.get(),
                                                   session_config.get());

  // Set thinking_config on conversation config.
  LiteRtLmThinkingConfig* thinking_config = litert_lm_thinking_config_create();
  ASSERT_NE(thinking_config, nullptr);
  litert_lm_thinking_config_set_enable_thinking(thinking_config, true);
  litert_lm_thinking_config_set_thinking_token_budget(thinking_config, 42);
  litert_lm_conversation_config_set_thinking_config(conversation_config.get(),
                                                    thinking_config);
  litert_lm_thinking_config_delete(thinking_config);

  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  ASSERT_TRUE(
      conversation->conversation->GetConfig().thinking_config().has_value());
  EXPECT_TRUE(conversation->conversation->GetConfig()
                  .thinking_config()
                  ->enable_thinking());
  EXPECT_EQ(conversation->conversation->GetConfig()
                .thinking_config()
                ->thinking_token_budget(),
            42);

  // Test resetting thinking_config to nullptr on conversation_config.
  litert_lm_conversation_config_set_thinking_config(conversation_config.get(),
                                                    nullptr);
  EXPECT_FALSE(conversation_config->thinking_config.has_value());
}

TEST(EngineCTest, OptionalArgsThinkingConfig) {
  LiteRtLmConversationOptionalArgs* optional_args =
      litert_lm_conversation_optional_args_create();
  ASSERT_NE(optional_args, nullptr);

  LiteRtLmThinkingConfig* thinking_config = litert_lm_thinking_config_create();
  ASSERT_NE(thinking_config, nullptr);
  litert_lm_thinking_config_set_enable_thinking(thinking_config, false);
  litert_lm_thinking_config_set_thinking_token_budget(thinking_config, 0);
  litert_lm_conversation_optional_args_set_thinking_config(optional_args,
                                                           thinking_config);
  litert_lm_thinking_config_delete(thinking_config);

  ASSERT_TRUE(optional_args->thinking_config.has_value());
  EXPECT_FALSE(optional_args->thinking_config->enable_thinking());
  EXPECT_EQ(optional_args->thinking_config->thinking_token_budget(), 0);

  // Test resetting thinking_config to nullptr on optional_args.
  litert_lm_conversation_optional_args_set_thinking_config(optional_args,
                                                           nullptr);
  EXPECT_FALSE(optional_args->thinking_config.has_value());

  litert_lm_conversation_optional_args_delete(optional_args);
}

TEST(EngineCTest, TokenizerTest) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm.litertlm");
  EngineSettingsPtr settings(litert_lm_engine_settings_create(
                                 task_path.c_str(), "cpu", nullptr, nullptr),
                             &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  const char* text = "hello";
  TokenizeResultPtr tokenize_result(
      litert_lm_engine_tokenize(engine.get(), text),
      &litert_lm_tokenize_result_delete);
  ASSERT_NE(tokenize_result, nullptr);
  size_t num_tokens =
      litert_lm_tokenize_result_get_num_tokens(tokenize_result.get());
  EXPECT_GT(num_tokens, 0);

  const int* tokens =
      litert_lm_tokenize_result_get_tokens(tokenize_result.get());
  DetokenizeResultPtr detokenize_result(
      litert_lm_engine_detokenize(engine.get(), tokens, num_tokens),
      &litert_lm_detokenize_result_delete);
  ASSERT_NE(detokenize_result, nullptr);
  EXPECT_STREQ(litert_lm_detokenize_result_get_string(detokenize_result.get()),
               text);

  TokenUnionPtr start_token(litert_lm_engine_get_start_token(engine.get()),
                            &litert_lm_token_union_delete);
  if (start_token != nullptr) {
    if (litert_lm_token_union_get_type(start_token.get()) ==
        kLiteRtLmTokenUnionTypeIds) {
      const int* ids;
      size_t num_ids;
      EXPECT_EQ(
          litert_lm_token_union_get_ids(start_token.get(), &ids, &num_ids), 0);
      EXPECT_GT(num_ids, 0);
    } else {
      EXPECT_NE(litert_lm_token_union_get_string(start_token.get()), nullptr);
    }
  }

  TokenUnionsPtr stop_tokens(litert_lm_engine_get_stop_tokens(engine.get()),
                             &litert_lm_token_unions_delete);
  if (stop_tokens != nullptr) {
    size_t num_tokens =
        litert_lm_token_unions_get_num_tokens(stop_tokens.get());
    for (size_t i = 0; i < num_tokens; ++i) {
      TokenUnionPtr stop_token(
          litert_lm_token_unions_get_token_at(stop_tokens.get(), i),
          &litert_lm_token_union_delete);
      ASSERT_NE(stop_token, nullptr);
      if (litert_lm_token_union_get_type(stop_token.get()) ==
          kLiteRtLmTokenUnionTypeIds) {
        const int* ids;
        size_t num_ids;
        EXPECT_EQ(
            litert_lm_token_union_get_ids(stop_token.get(), &ids, &num_ids), 0);
        EXPECT_GT(num_ids, 0);
      } else {
        EXPECT_NE(litert_lm_token_union_get_string(stop_token.get()), nullptr);
      }
    }
  }
}

TEST(EngineCTest, GenerateContent) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(litert_lm_engine_create_session(
                         engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt = "Hello world!";
  InputDataPtr input_data(
      litert_lm_input_data_create(kLiteRtLmInputDataTypeText, prompt,
                                  strlen(prompt)),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};
  ResponsesPtr responses(
      litert_lm_session_generate_content(session.get(), inputs, 1),
      &litert_lm_responses_delete);
  ASSERT_NE(responses, nullptr);

  EXPECT_EQ(litert_lm_responses_get_num_candidates(responses.get()), 1);
  const char* response_text =
      litert_lm_responses_get_response_text_at(responses.get(), 0);
  ASSERT_NE(response_text, nullptr);
  EXPECT_GT(strlen(response_text), 0);
}

TEST(EngineCTest, CreateSessionWithMaxOutputTokens) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // Test with max_output_tokens=1. The response length should be short (<10).
  {
    SessionConfigPtr session_config(litert_lm_session_config_create(),
                                    &litert_lm_session_config_delete);
    ASSERT_NE(session_config, nullptr);
    litert_lm_session_config_set_max_output_tokens(session_config.get(), 1);

    SessionPtr session(
        litert_lm_engine_create_session(engine.get(), session_config.get()),
        &litert_lm_session_delete);
    ASSERT_NE(session, nullptr);

    const char* prompt = "Hello world!";
    InputDataPtr input_data(
        litert_lm_input_data_create(kLiteRtLmInputDataTypeText, prompt,
                                    strlen(prompt)),
        &litert_lm_input_data_delete);
    ASSERT_NE(input_data, nullptr);
    const LiteRtLmInputData* inputs[] = {input_data.get()};
    ResponsesPtr responses(
        litert_lm_session_generate_content(session.get(), inputs, 1),
        &litert_lm_responses_delete);
    ASSERT_NE(responses, nullptr);

    EXPECT_EQ(litert_lm_responses_get_num_candidates(responses.get()), 1);
    const char* response_text =
        litert_lm_responses_get_response_text_at(responses.get(), 0);
    ASSERT_NE(response_text, nullptr);
    EXPECT_GT(strlen(response_text), 0);
    EXPECT_LT(strlen(response_text), 10);
  }

  // Test without max_output_tokens. The response length should be long (>=10).
  {
    SessionConfigPtr session_config(litert_lm_session_config_create(),
                                    &litert_lm_session_config_delete);
    ASSERT_NE(session_config, nullptr);

    SessionPtr session(
        litert_lm_engine_create_session(engine.get(), session_config.get()),
        &litert_lm_session_delete);
    ASSERT_NE(session, nullptr);

    const char* prompt = "Hello world!";
    InputDataPtr input_data(
        litert_lm_input_data_create(kLiteRtLmInputDataTypeText, prompt,
                                    strlen(prompt)),
        &litert_lm_input_data_delete);
    ASSERT_NE(input_data, nullptr);
    const LiteRtLmInputData* inputs[] = {input_data.get()};
    ResponsesPtr responses(
        litert_lm_session_generate_content(session.get(), inputs, 1),
        &litert_lm_responses_delete);
    ASSERT_NE(responses, nullptr);

    EXPECT_EQ(litert_lm_responses_get_num_candidates(responses.get()), 1);
    const char* response_text =
        litert_lm_responses_get_response_text_at(responses.get(), 0);
    ASSERT_NE(response_text, nullptr);
    EXPECT_GT(strlen(response_text), 10);
  }
}

TEST(EngineCTest, ConversationSendMessage) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(),
                                    /*conversation_config=*/nullptr),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  JsonResponsePtr response(
      litert_lm_conversation_send_message(conversation.get(), message_json,
                                          /*extra_context=*/nullptr,
                                          /*optional_args=*/nullptr),
      &litert_lm_json_response_delete);
  ASSERT_NE(response, nullptr);

  const char* response_str = litert_lm_json_response_get_string(response.get());
  ASSERT_NE(response_str, nullptr);
  EXPECT_GT(strlen(response_str), 0);
}

TEST(EngineCTest, ConversationRenderPreface) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm.litertlm");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  ConversationConfigPtr conversation_config(
      litert_lm_conversation_config_create(),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);

  const char* messages_json =
      R"([{"role": "system", "content": "You are a helpful assistant."}])";
  litert_lm_conversation_config_set_messages(conversation_config.get(),
                                             messages_json);

  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const char* rendered =
      litert_lm_conversation_render_preface_to_string(conversation.get());
  ASSERT_NE(rendered, nullptr);
  EXPECT_THAT(rendered, testing::HasSubstr("You are a helpful assistant."));
}

TEST(EngineCTest, ConversationSendMessageWithConfig) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm.litertlm");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create Sampler Params.
  SamplerParamsPtr sampler_params(
      litert_lm_sampler_params_create(kLiteRtLmSamplerTypeTopP),
      &litert_lm_sampler_params_delete);
  ASSERT_NE(sampler_params, nullptr);
  litert_lm_sampler_params_set_top_k(sampler_params.get(), 10);
  litert_lm_sampler_params_set_top_p(sampler_params.get(), 0.5f);
  litert_lm_sampler_params_set_temperature(sampler_params.get(), 0.1f);
  litert_lm_sampler_params_set_seed(sampler_params.get(), 1234);
  SessionConfigPtr session_config(litert_lm_session_config_create(),
                                  &litert_lm_session_config_delete);
  ASSERT_NE(session_config, nullptr);
  litert_lm_session_config_set_sampler_params(session_config.get(),
                                              sampler_params.get());

  // 3. Create a Conversation Config with the Session Config
  // and System Message.
  const std::string system_message =
      R"({"type":"text","text":"You are a helpful assistant."})";
  ConversationConfigPtr conversation_config(
      litert_lm_conversation_config_create(),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_session_config(conversation_config.get(),
                                                   session_config.get());
  litert_lm_conversation_config_set_system_message(conversation_config.get(),
                                                   system_message.c_str());

  // 4. Create a Conversation with the Conversation Config.
  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  // 5. Send a message to the conversation.
  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  JsonResponsePtr response(
      litert_lm_conversation_send_message(conversation.get(), message_json,
                                          /*extra_context=*/nullptr,
                                          /*optional_args=*/nullptr),
      &litert_lm_json_response_delete);
  ASSERT_NE(response, nullptr);

  const char* response_str = litert_lm_json_response_get_string(response.get());
  ASSERT_NE(response_str, nullptr);
  EXPECT_GT(strlen(response_str), 0);
}

TEST(EngineCTest, ConversationSendMessageWithExtraContext) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm.litertlm");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create a Conversation Config.
  ConversationConfigPtr conversation_config(
      litert_lm_conversation_config_create(),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);

  // 3. Create a Conversation with the Conversation Config.
  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  // 4. Send a message to the conversation with extra context.
  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  const char* extra_context = R"({"key": "value"})";
  JsonResponsePtr response(
      litert_lm_conversation_send_message(conversation.get(), message_json,
                                          /*extra_context=*/extra_context,
                                          /*optional_args=*/nullptr),
      &litert_lm_json_response_delete);
  ASSERT_NE(response, nullptr);

  const char* response_str = litert_lm_json_response_get_string(response.get());
  ASSERT_NE(response_str, nullptr);
  EXPECT_GT(strlen(response_str), 0);
}

TEST(EngineCTest, ConversationSendMessageWithOptionalArgs) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm.litertlm");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create a Conversation Config.
  ConversationConfigPtr conversation_config(
      litert_lm_conversation_config_create(),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);

  // 3. Create a Conversation with the Conversation Config.
  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  // 4. Create Optional Args.
  RepetitionPenaltyConfigPtr repetition_penalty_config(
      litert_lm_repetition_penalty_config_create(),
      &litert_lm_repetition_penalty_config_delete);
  ASSERT_NE(repetition_penalty_config, nullptr);
  litert_lm_repetition_penalty_config_set_repetition_penalty(
      repetition_penalty_config.get(), 1.2f);
  litert_lm_repetition_penalty_config_set_presence_penalty(
      repetition_penalty_config.get(), 0.1f);
  litert_lm_repetition_penalty_config_set_frequency_penalty(
      repetition_penalty_config.get(), 0.2f);
  litert_lm_repetition_penalty_config_set_window_size(
      repetition_penalty_config.get(), 10);

  OptionalArgsPtr optional_args(litert_lm_conversation_optional_args_create(),
                                &litert_lm_conversation_optional_args_delete);
  ASSERT_NE(optional_args, nullptr);

  NoRepeatNgramConfigPtr no_repeat_ngram_config(
      litert_lm_no_repeat_ngram_config_create(),
      &litert_lm_no_repeat_ngram_config_delete);
  ASSERT_NE(no_repeat_ngram_config, nullptr);
  litert_lm_no_repeat_ngram_config_set_no_repeat_ngram_size(
      no_repeat_ngram_config.get(), 3);
  litert_lm_no_repeat_ngram_config_set_window_size(no_repeat_ngram_config.get(),
                                                   10);

  SuppressTokensConfigPtr suppress_tokens_config(
      litert_lm_suppress_tokens_config_create(),
      &litert_lm_suppress_tokens_config_delete);
  ASSERT_NE(suppress_tokens_config, nullptr);
  int suppress_tokens[] = {10, 20, 30};
  litert_lm_suppress_tokens_config_set_suppress_tokens(
      suppress_tokens_config.get(), suppress_tokens, 3);

  litert_lm_conversation_optional_args_set_repetition_penalty_config(
      optional_args.get(), repetition_penalty_config.get());
  litert_lm_conversation_optional_args_set_no_repeat_ngram_config(
      optional_args.get(), no_repeat_ngram_config.get());
  litert_lm_conversation_optional_args_set_suppress_tokens_config(
      optional_args.get(), suppress_tokens_config.get());
  litert_lm_conversation_optional_args_set_visual_token_budget(
      optional_args.get(), 100);

  // 5. Send a message to the conversation with optional args.
  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  JsonResponsePtr response(litert_lm_conversation_send_message(
                               conversation.get(), message_json,
                               /*extra_context=*/nullptr, optional_args.get()),
                           &litert_lm_json_response_delete);
  ASSERT_NE(response, nullptr);

  const char* response_str = litert_lm_json_response_get_string(response.get());
  ASSERT_NE(response_str, nullptr);
  EXPECT_GT(strlen(response_str), 0);
}

TEST(EngineCTest, ConversationCloneNull) {
  EXPECT_EQ(litert_lm_conversation_clone(nullptr), nullptr);
}

TEST(EngineCTest, ConversationCloneSuccess) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm.litertlm");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create a Conversation.
  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(),
                                    /*conversation_config=*/nullptr),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  // 3. Clone the conversation.
  LiteRtLmConversation* cloned_raw =
      litert_lm_conversation_clone(conversation.get());
  if (cloned_raw == nullptr) {
    if (absl::IsUnimplemented(conversation->conversation->Clone().status())) {
      GTEST_SKIP() << "Clone is not supported by this engine.";
    }
  }
  ConversationPtr cloned_conversation(cloned_raw,
                                      &litert_lm_conversation_delete);
  ASSERT_NE(cloned_conversation, nullptr);
}

struct StreamCallbackData {
  std::string response;
  absl::Notification done;
  absl::Status status;
};

void StreamCallback(void* callback_data, const LiteRtLmStreamChunk* chunk) {
  auto* data = static_cast<StreamCallbackData*>(callback_data);
  const char* error_msg = litert_lm_stream_chunk_get_error(chunk);
  if (error_msg) {
    data->status = absl::InternalError(error_msg);
  }
  const char* text = litert_lm_stream_chunk_get_text(chunk);
  if (text) {
    data->response.append(text);
  }
  if (litert_lm_stream_chunk_is_final(chunk)) {
    data->done.Notify();
  }
}

TEST(EngineCTest, GenerateContentStream) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(litert_lm_engine_create_session(
                         engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt = "Hello world!";
  InputDataPtr input_data(
      litert_lm_input_data_create(kLiteRtLmInputDataTypeText, prompt,
                                  strlen(prompt)),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};
  StreamCallbackData callback_data;
  int result = litert_lm_session_generate_content_stream(
      session.get(), inputs, 1, &StreamCallback, &callback_data);
  ASSERT_EQ(result, 0);

  callback_data.done.WaitForNotification();

  // This model is too small and generate random output, so the result may be
  // either success or failure due to maximum kv-cache size reached.
  EXPECT_THAT(
      callback_data.status,
      testing::AnyOf(absl_testing::IsOk(),
                     absl_testing::StatusIs(
                         absl::StatusCode::kInternal,
                         testing::HasSubstr("Max number of tokens reached."))));
  EXPECT_GT(callback_data.response.length(), 0);
}

TEST(EngineCTest, SessionGenerateContentStreamAndCancel) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 128);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(litert_lm_engine_create_session(
                         engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt =
      "Hello world! Write a long essay about the history of Rome.";
  InputDataPtr input_data(
      litert_lm_input_data_create(kLiteRtLmInputDataTypeText, prompt,
                                  strlen(prompt)),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};
  StreamCallbackData callback_data;
  int result = litert_lm_session_generate_content_stream(
      session.get(), inputs, 1, &StreamCallback, &callback_data);
  ASSERT_EQ(result, 0);

  litert_lm_session_cancel_process(session.get());

  callback_data.done.WaitForNotification();

  EXPECT_THAT(callback_data.status,
              absl_testing::StatusIs(absl::StatusCode::kInternal,
                                     testing::HasSubstr("CANCELLED")));
}

TEST(EngineCTest, ConversationSendMessageStream) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(),
                                    /*conversation_config=*/nullptr),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  StreamCallbackData callback_data;
  int result = litert_lm_conversation_send_message_stream(
      conversation.get(), message_json, /*extra_context=*/nullptr,
      /*optional_args=*/nullptr, &StreamCallback, &callback_data);
  ASSERT_EQ(result, 0);

  callback_data.done.WaitForNotification();
  EXPECT_GT(callback_data.response.length(), 0);
}

TEST(EngineCTest, ConversationSendMessageStreamWithExtraContext) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(),
                                    /*conversation_config=*/nullptr),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  const char* extra_context = R"({"key": "value"})";
  StreamCallbackData callback_data;
  int result = litert_lm_conversation_send_message_stream(
      conversation.get(), message_json, /*extra_context=*/extra_context,
      /*optional_args=*/nullptr, &StreamCallback, &callback_data);
  ASSERT_EQ(result, 0);

  callback_data.done.WaitForNotification();
  EXPECT_GT(callback_data.response.length(), 0);
}

TEST(EngineCTest, ConversationSendMessageStreamWithOptionalArgs) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(),
                                    /*conversation_config=*/nullptr),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";

  RepetitionPenaltyConfigPtr repetition_penalty_config(
      litert_lm_repetition_penalty_config_create(),
      &litert_lm_repetition_penalty_config_delete);
  ASSERT_NE(repetition_penalty_config, nullptr);
  litert_lm_repetition_penalty_config_set_repetition_penalty(
      repetition_penalty_config.get(), 1.2f);
  litert_lm_repetition_penalty_config_set_presence_penalty(
      repetition_penalty_config.get(), 0.1f);
  litert_lm_repetition_penalty_config_set_frequency_penalty(
      repetition_penalty_config.get(), 0.2f);
  litert_lm_repetition_penalty_config_set_window_size(
      repetition_penalty_config.get(), 10);

  OptionalArgsPtr optional_args(litert_lm_conversation_optional_args_create(),
                                &litert_lm_conversation_optional_args_delete);
  ASSERT_NE(optional_args, nullptr);

  NoRepeatNgramConfigPtr no_repeat_ngram_config(
      litert_lm_no_repeat_ngram_config_create(),
      &litert_lm_no_repeat_ngram_config_delete);
  ASSERT_NE(no_repeat_ngram_config, nullptr);
  litert_lm_no_repeat_ngram_config_set_no_repeat_ngram_size(
      no_repeat_ngram_config.get(), 3);
  litert_lm_no_repeat_ngram_config_set_window_size(no_repeat_ngram_config.get(),
                                                   10);

  SuppressTokensConfigPtr suppress_tokens_config(
      litert_lm_suppress_tokens_config_create(),
      &litert_lm_suppress_tokens_config_delete);
  ASSERT_NE(suppress_tokens_config, nullptr);
  int suppress_tokens[] = {10, 20, 30};
  litert_lm_suppress_tokens_config_set_suppress_tokens(
      suppress_tokens_config.get(), suppress_tokens, 3);

  litert_lm_conversation_optional_args_set_repetition_penalty_config(
      optional_args.get(), repetition_penalty_config.get());
  litert_lm_conversation_optional_args_set_no_repeat_ngram_config(
      optional_args.get(), no_repeat_ngram_config.get());
  litert_lm_conversation_optional_args_set_suppress_tokens_config(
      optional_args.get(), suppress_tokens_config.get());
  litert_lm_conversation_optional_args_set_visual_token_budget(
      optional_args.get(), 100);

  StreamCallbackData callback_data;
  int result = litert_lm_conversation_send_message_stream(
      conversation.get(), message_json, /*extra_context=*/nullptr,
      optional_args.get(), &StreamCallback, &callback_data);
  ASSERT_EQ(result, 0);

  callback_data.done.WaitForNotification();
  EXPECT_GT(callback_data.response.length(), 0);
}

TEST(EngineCTest, ConversationSendMessageStreamAndCancel) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 512);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(),
                                    /*conversation_config=*/nullptr),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  StreamCallbackData callback_data;
  int result = litert_lm_conversation_send_message_stream(
      conversation.get(), message_json, /*extra_context=*/nullptr,
      /*optional_args=*/nullptr, &StreamCallback, &callback_data);
  ASSERT_EQ(result, 0);

  litert_lm_conversation_cancel_process(conversation.get());

  callback_data.done.WaitForNotification();
  EXPECT_THAT(callback_data.status,
              absl_testing::StatusIs(absl::StatusCode::kInternal,
                                     testing::HasSubstr("CANCELLED")));
}

using BenchmarkInfoPtr =
    std::unique_ptr<LiteRtLmBenchmarkInfo,
                    decltype(&litert_lm_benchmark_info_delete)>;

TEST(EngineCTest, Benchmark) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);
  litert_lm_engine_settings_enable_benchmark(settings.get());

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(litert_lm_engine_create_session(
                         engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt = "Hello world!";
  InputDataPtr input_data(
      litert_lm_input_data_create(kLiteRtLmInputDataTypeText, prompt,
                                  strlen(prompt)),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};
  ResponsesPtr responses(
      litert_lm_session_generate_content(session.get(), inputs, 1),
      &litert_lm_responses_delete);
  ASSERT_NE(responses, nullptr);

  BenchmarkInfoPtr benchmark_info(
      litert_lm_session_get_benchmark_info(session.get()),
      &litert_lm_benchmark_info_delete);
  ASSERT_NE(benchmark_info, nullptr);

  EXPECT_GT(
      litert_lm_benchmark_info_get_time_to_first_token(benchmark_info.get()),
      0.0);
  EXPECT_GT(litert_lm_benchmark_info_get_total_init_time_in_second(
                benchmark_info.get()),
            0.0);
  int num_prefill_turns =
      litert_lm_benchmark_info_get_num_prefill_turns(benchmark_info.get());
  EXPECT_GT(num_prefill_turns, 0);
  for (int i = 0; i < num_prefill_turns; ++i) {
    EXPECT_GT(litert_lm_benchmark_info_get_prefill_token_count_at(
                  benchmark_info.get(), i),
              0);

    EXPECT_GT(litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
                  benchmark_info.get(), i),
              0.0);
  }
  int num_decode_turns =
      litert_lm_benchmark_info_get_num_decode_turns(benchmark_info.get());
  EXPECT_GT(num_decode_turns, 0);
  for (int i = 0; i < num_decode_turns; ++i) {
    EXPECT_GT(litert_lm_benchmark_info_get_decode_token_count_at(
                  benchmark_info.get(), i),
              0);

    EXPECT_GT(litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
                  benchmark_info.get(), i),
              0.0);
  }
}

TEST(EngineCTest, RunPrefillSuccess) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(litert_lm_engine_create_session(
                         engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt = "Hello world!";
  InputDataPtr input_data(
      litert_lm_input_data_create(kLiteRtLmInputDataTypeText, prompt,
                                  strlen(prompt)),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};

  int prefill_result = litert_lm_session_run_prefill(session.get(), inputs, 1);
  EXPECT_EQ(prefill_result, 0);
}

TEST(EngineCTest, RunPrefillAndDecode) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(litert_lm_engine_create_session(
                         engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt = "Hello world!";
  InputDataPtr input_data(
      litert_lm_input_data_create(kLiteRtLmInputDataTypeText, prompt,
                                  strlen(prompt)),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};

  litert_lm_session_run_prefill(session.get(), inputs, 1);

  ResponsesPtr responses(litert_lm_session_run_decode(session.get()),
                         &litert_lm_responses_delete);
  ASSERT_NE(responses, nullptr);

  EXPECT_EQ(litert_lm_responses_get_num_candidates(responses.get()), 1);
  const char* response_text =
      litert_lm_responses_get_response_text_at(responses.get(), 0);
  ASSERT_NE(response_text, nullptr);
  EXPECT_GT(strlen(response_text), 0);
}

TEST(EngineCTest, TextScoringBasic) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(litert_lm_engine_create_session(
                         engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt = "Hello world!";
  InputDataPtr input_data(
      litert_lm_input_data_create(kLiteRtLmInputDataTypeText, prompt,
                                  strlen(prompt)),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};

  litert_lm_session_run_prefill(session.get(), inputs, 1);

  const char* target_texts[] = {"apple"};
  ResponsesPtr responses(
      litert_lm_session_run_text_scoring(session.get(), target_texts, 1,
                                         /*store_token_lengths=*/true),
      &litert_lm_responses_delete);
  ASSERT_NE(responses, nullptr);

  EXPECT_EQ(litert_lm_responses_get_num_candidates(responses.get()), 1);
}

TEST(EngineCTest, TextScoringVerifyScores) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(litert_lm_engine_create_session(
                         engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt = "Hello world!";
  InputDataPtr input_data(
      litert_lm_input_data_create(kLiteRtLmInputDataTypeText, prompt,
                                  strlen(prompt)),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};

  litert_lm_session_run_prefill(session.get(), inputs, 1);

  const char* target_texts[] = {"apple"};
  ResponsesPtr responses(
      litert_lm_session_run_text_scoring(session.get(), target_texts, 1,
                                         /*store_token_lengths=*/true),
      &litert_lm_responses_delete);
  ASSERT_NE(responses, nullptr);

  EXPECT_TRUE(litert_lm_responses_has_score_at(responses.get(), 0));
}

TEST(EngineCTest, TextScoringVerifyTokenLengths) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(litert_lm_engine_create_session(
                         engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt = "Hello world!";
  InputDataPtr input_data(
      litert_lm_input_data_create(kLiteRtLmInputDataTypeText, prompt,
                                  strlen(prompt)),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};

  litert_lm_session_run_prefill(session.get(), inputs, 1);

  const char* target_texts[] = {"apple"};
  ResponsesPtr responses(
      litert_lm_session_run_text_scoring(session.get(), target_texts, 1,
                                         /*store_token_lengths=*/true),
      &litert_lm_responses_delete);
  ASSERT_NE(responses, nullptr);

  EXPECT_TRUE(litert_lm_responses_has_token_length_at(responses.get(), 0));
  EXPECT_GT(litert_lm_responses_get_token_length_at(responses.get(), 0), 0);
}

TEST(EngineCTest, ConversationOptionalArgsTest) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm.litertlm");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu",
                                       /* vision_backend_str */ nullptr,
                                       /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(),
                                    /*conversation_config=*/nullptr),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  OptionalArgsPtr optional_args(litert_lm_conversation_optional_args_create(),
                                &litert_lm_conversation_optional_args_delete);
  ASSERT_NE(optional_args, nullptr);
  litert_lm_conversation_optional_args_set_max_output_tokens(
      optional_args.get(), 1);

  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  JsonResponsePtr response(litert_lm_conversation_send_message(
                               conversation.get(), message_json,
                               /*extra_context=*/nullptr, optional_args.get()),
                           &litert_lm_json_response_delete);
  ASSERT_NE(response, nullptr);

  const char* response_str = litert_lm_json_response_get_string(response.get());
  ASSERT_NE(response_str, nullptr);
  EXPECT_GT(strlen(response_str), 0);
  // Since max_output_tokens is 1, the response should be very short.
  auto response_json = nlohmann::ordered_json::parse(response_str);
  std::string text = response_json["content"][0]["text"];
  EXPECT_GT(text.length(), 0);
  EXPECT_LT(text.length(), 5);
  EXPECT_EQ(text, "\xE3\x81\xA9");
}

}  // namespace
