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
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "c/conversation.h"
#include "c/conversation_internal.h"
#include "c/engine_internal.h"
#include "c/error_reporter.h"
#include "c/experimental.h"
#include "c/experimental_internal.h"  // IWYU pragma: keep
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/thinking_config.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::AnyOf;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;

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

// The C API constructors return a status code and deliver the new handle
// through a trailing out-parameter. The helpers below return the handle (or
// NULL on failure) so that it can be adopted directly by a smart pointer. On
// failure they also check that a last error message was recorded.
template <typename T>
T* HandleOrNull(int status, T* handle) {
  if (status == kLiteRtLmStatusOk) {
    EXPECT_NE(handle, nullptr);
  } else {
    EXPECT_EQ(handle, nullptr);
    EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
  }
  return handle;
}

LiteRtLmEngineSettings* CreateEngineSettings(const char* model_path,
                                             const char* backend_str,
                                             const char* vision_backend_str,
                                             const char* audio_backend_str) {
  LiteRtLmEngineSettings* settings = nullptr;
  const int status = litert_lm_engine_settings_create(
      model_path, backend_str, vision_backend_str, audio_backend_str,
      &settings);
  return HandleOrNull(status, settings);
}

LiteRtLmEngineSettings* CreateEngineSettingsFromFd(
    int fd, const char* backend_str, const char* vision_backend_str,
    const char* audio_backend_str) {
  LiteRtLmEngineSettings* settings = nullptr;
  const int status = litert_lm_engine_settings_create_from_raw_file_descriptor(
      fd, backend_str, vision_backend_str, audio_backend_str, &settings);
  return HandleOrNull(status, settings);
}

LiteRtLmEngine* CreateEngine(const LiteRtLmEngineSettings* settings) {
  LiteRtLmEngine* engine = nullptr;
  const int status = litert_lm_engine_create(settings, &engine);
  return HandleOrNull(status, engine);
}

LiteRtLmSession* CreateSession(LiteRtLmEngine* engine,
                               LiteRtLmSessionConfig* config) {
  LiteRtLmSession* session = nullptr;
  const int status = litert_lm_engine_create_session(engine, config, &session);
  return HandleOrNull(status, session);
}

LiteRtLmSessionConfig* CreateSessionConfig() {
  LiteRtLmSessionConfig* config = nullptr;
  const int status = litert_lm_session_config_create(&config);
  return HandleOrNull(status, config);
}

LiteRtLmSamplerParams* CreateSamplerParams(LiteRtLmSamplerType type) {
  LiteRtLmSamplerParams* params = nullptr;
  const int status = litert_lm_sampler_params_create(type, &params);
  return HandleOrNull(status, params);
}

LiteRtLmRepetitionPenaltyConfig* CreateRepetitionPenaltyConfig() {
  LiteRtLmRepetitionPenaltyConfig* config = nullptr;
  const int status = litert_lm_repetition_penalty_config_create(&config);
  return HandleOrNull(status, config);
}

LiteRtLmNoRepeatNgramConfig* CreateNoRepeatNgramConfig() {
  LiteRtLmNoRepeatNgramConfig* config = nullptr;
  const int status = litert_lm_no_repeat_ngram_config_create(&config);
  return HandleOrNull(status, config);
}

LiteRtLmSuppressTokensConfig* CreateSuppressTokensConfig() {
  LiteRtLmSuppressTokensConfig* config = nullptr;
  const int status = litert_lm_suppress_tokens_config_create(&config);
  return HandleOrNull(status, config);
}

LiteRtLmInputData* CreateInputData(LiteRtLmInputDataType type, const void* data,
                                   size_t size) {
  LiteRtLmInputData* input_data = nullptr;
  const int status = litert_lm_input_data_create(type, data, size, &input_data);
  return HandleOrNull(status, input_data);
}

LiteRtLmResponses* GenerateContent(LiteRtLmSession* session,
                                   const LiteRtLmInputData* const* inputs,
                                   size_t num_inputs) {
  LiteRtLmResponses* responses = nullptr;
  const int status = litert_lm_session_generate_content(session, inputs,
                                                        num_inputs, &responses);
  return HandleOrNull(status, responses);
}

LiteRtLmResponses* RunDecode(LiteRtLmSession* session) {
  LiteRtLmResponses* responses = nullptr;
  const int status = litert_lm_session_run_decode(session, &responses);
  return HandleOrNull(status, responses);
}

LiteRtLmResponses* RunTextScoring(LiteRtLmSession* session,
                                  const char** target_text, size_t num_targets,
                                  bool store_token_lengths) {
  LiteRtLmResponses* responses = nullptr;
  const int status = litert_lm_session_run_text_scoring(
      session, target_text, num_targets, store_token_lengths, &responses);
  return HandleOrNull(status, responses);
}

LiteRtLmBenchmarkInfo* GetBenchmarkInfo(LiteRtLmSession* session) {
  LiteRtLmBenchmarkInfo* benchmark_info = nullptr;
  const int status =
      litert_lm_session_get_benchmark_info(session, &benchmark_info);
  return HandleOrNull(status, benchmark_info);
}

LiteRtLmTokenizeResult* Tokenize(LiteRtLmEngine* engine, const char* text) {
  LiteRtLmTokenizeResult* result = nullptr;
  const int status = litert_lm_engine_tokenize(engine, text, &result);
  return HandleOrNull(status, result);
}

LiteRtLmDetokenizeResult* Detokenize(LiteRtLmEngine* engine, const int* tokens,
                                     size_t num_tokens) {
  LiteRtLmDetokenizeResult* result = nullptr;
  const int status =
      litert_lm_engine_detokenize(engine, tokens, num_tokens, &result);
  return HandleOrNull(status, result);
}

LiteRtLmTokenUnion* GetTokenAt(const LiteRtLmTokenUnions* tokens,
                               size_t index) {
  LiteRtLmTokenUnion* token = nullptr;
  const int status = litert_lm_token_unions_get_token_at(tokens, index, &token);
  return HandleOrNull(status, token);
}

LiteRtLmConversationConfig* CreateConversationConfig() {
  LiteRtLmConversationConfig* config = nullptr;
  const int status = litert_lm_conversation_config_create(&config);
  return HandleOrNull(status, config);
}

LiteRtLmThinkingConfig* CreateThinkingConfig() {
  LiteRtLmThinkingConfig* config = nullptr;
  const int status = litert_lm_thinking_config_create(&config);
  return HandleOrNull(status, config);
}

LiteRtLmConversationOptionalArgs* CreateConversationOptionalArgs() {
  LiteRtLmConversationOptionalArgs* optional_args = nullptr;
  const int status =
      litert_lm_conversation_optional_args_create(&optional_args);
  return HandleOrNull(status, optional_args);
}

LiteRtLmConversation* CreateConversation(LiteRtLmEngine* engine,
                                         LiteRtLmConversationConfig* config) {
  LiteRtLmConversation* conversation = nullptr;
  const int status =
      litert_lm_conversation_create(engine, config, &conversation);
  return HandleOrNull(status, conversation);
}

LiteRtLmConversation* CloneConversation(LiteRtLmConversation* conversation) {
  LiteRtLmConversation* cloned = nullptr;
  const int status = litert_lm_conversation_clone(conversation, &cloned);
  return HandleOrNull(status, cloned);
}

LiteRtLmJsonResponse* ConversationSendMessage(
    LiteRtLmConversation* conversation, const char* message_json,
    const char* extra_context,
    const LiteRtLmConversationOptionalArgs* optional_args) {
  LiteRtLmJsonResponse* response = nullptr;
  const int status = litert_lm_conversation_send_message(
      conversation, message_json, extra_context, optional_args, &response);
  return HandleOrNull(status, response);
}

// Returns the response JSON string. Expects the call to succeed.
const char* GetResponseString(const LiteRtLmJsonResponse* response) {
  const char* json = nullptr;
  EXPECT_EQ(litert_lm_json_response_get_string(response, &json),
            kLiteRtLmStatusOk);
  return json;
}

// Returns the rendered preface, or NULL on failure.
const char* RenderPreface(LiteRtLmConversation* conversation) {
  const char* text = nullptr;
  const int status =
      litert_lm_conversation_render_preface_to_string(conversation, &text);
  return HandleOrNull(status, text);
}

// Returns the rendered message, or NULL on failure.
const char* RenderMessage(LiteRtLmConversation* conversation,
                          const char* message_json) {
  const char* text = nullptr;
  const int status = litert_lm_conversation_render_message_to_string(
      conversation, message_json, &text);
  return HandleOrNull(status, text);
}

// Calls a C API getter that delivers its result through a trailing
// out-parameter of type `T*`, expects it to succeed, and returns the result.
template <typename T, typename Getter, typename... Args>
T GetOk(Getter getter, Args... args) {
  T out{};
  EXPECT_EQ(getter(args..., &out), kLiteRtLmStatusOk);
  return out;
}

// Returns the engine's start token, or NULL if none is configured. Expects the
// call to succeed.
LiteRtLmTokenUnion* GetStartToken(LiteRtLmEngine* engine) {
  LiteRtLmTokenUnion* token = nullptr;
  EXPECT_EQ(litert_lm_engine_get_start_token(engine, &token),
            kLiteRtLmStatusOk);
  return token;
}

// Returns the engine's stop tokens, or NULL if none are configured. Expects
// the call to succeed.
LiteRtLmTokenUnions* GetStopTokens(LiteRtLmEngine* engine) {
  LiteRtLmTokenUnions* tokens = nullptr;
  EXPECT_EQ(litert_lm_engine_get_stop_tokens(engine, &tokens),
            kLiteRtLmStatusOk);
  return tokens;
}

TEST(EngineCTest, CreateSettingsWithNoVisionAndAudioBackend) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
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
      CreateEngineSettings(task_path.c_str(), "cpu",
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
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ "dummy_backend",
                           /* audio_backend_str */ "cpu"),
      &litert_lm_engine_settings_delete);
  ASSERT_EQ(settings, nullptr);
}

TEST(EngineCTest, SetCacheDir) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
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
      CreateEngineSettings(task_path.c_str(), "cpu",
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
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_images(settings.get(), 10);
  EXPECT_EQ(settings->settings->GetMainExecutorSettings().GetMaxNumImages(),
            10);
}

TEST(EngineCTest, SetMaxVisionTokensPerImage) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  EXPECT_FALSE(settings->settings->GetMaxVisionTokensPerImage().has_value());

  litert_lm_engine_settings_set_max_vision_tokens_per_image(settings.get(),
                                                            280);
  EXPECT_TRUE(settings->settings->GetMaxVisionTokensPerImage().has_value());
  EXPECT_EQ(settings->settings->GetMaxVisionTokensPerImage().value(), 280);
}

TEST(EngineCTest, SetPrefillChunkSize) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  int prefill_chunk_size = 128;
  litert_lm_engine_settings_set_prefill_chunk_size(settings.get(),
                                                   prefill_chunk_size);
  auto config = settings->settings->GetMainExecutorSettings()
                    .GetBackendConfig<litert::lm::CpuConfig>();
  ASSERT_OK(config);
  EXPECT_EQ(config->prefill_chunk_size, prefill_chunk_size);
}

TEST(EngineCTest, SetEnableYNNPack) {
  // Test with nullptr settings (should not crash).
  litert_lm_engine_settings_set_enable_ynnpack(nullptr, true);

  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_enable_ynnpack(settings.get(), true);
  auto config1 = settings->settings->GetMainExecutorSettings()
                     .GetBackendConfig<litert::lm::CpuConfig>();
  ASSERT_OK(config1);
  EXPECT_TRUE(config1->enable_ynnpack);  // NOLINT: config is checked above.

  litert_lm_engine_settings_set_enable_ynnpack(settings.get(), false);
  auto config2 = settings->settings->GetMainExecutorSettings()
                     .GetBackendConfig<litert::lm::CpuConfig>();
  ASSERT_OK(config2);
  EXPECT_FALSE(config2->enable_ynnpack);  // NOLINT: config is checked above.
}

TEST(EngineCTest, SetParallelFileSectionLoading) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
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

TEST(EngineCTest, SetSingleThreadedExecution) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);

  // Default should be false.
  EXPECT_FALSE(settings->settings->GetSingleThreadedExecution());

  litert_lm_engine_settings_set_single_threaded_execution(settings.get(), true);
  EXPECT_TRUE(settings->settings->GetSingleThreadedExecution());

  litert_lm_engine_settings_set_single_threaded_execution(settings.get(),
                                                          false);
  EXPECT_FALSE(settings->settings->GetSingleThreadedExecution());
}

TEST(EngineCTest, BenchmarkSettings) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
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
      CreateEngineSettings(task_path.c_str(), "cpu",
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

TEST(EngineCTest, SetUseRingbuffersLocalAttention) {
  const std::string task_path = "test_model_path_1";
  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "gpu_artisan",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);

  litert_lm_engine_settings_set_use_ringbuffers_local_attention(settings.get(),
                                                                true);
  auto config1 = settings->settings->GetMainExecutorSettings()
                     .GetBackendConfig<litert::lm::GpuArtisanConfig>();
  ASSERT_OK(config1);
  EXPECT_TRUE(config1->use_autosized_ringbuffers);

  litert_lm_engine_settings_set_use_ringbuffers_local_attention(settings.get(),
                                                                false);
  auto config2 = settings->settings->GetMainExecutorSettings()
                     .GetBackendConfig<litert::lm::GpuArtisanConfig>();
  ASSERT_OK(config2);
  EXPECT_FALSE(config2->use_autosized_ringbuffers);
}

TEST(EngineCTest, CreateSettingsFromRawFileDescriptor) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");
  int fd = open(task_path.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);
  EngineSettingsPtr settings(
      CreateEngineSettingsFromFd(fd, "cpu", /* vision_backend_str */ nullptr,
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
  SamplerParamsPtr sampler_params(CreateSamplerParams(kLiteRtLmSamplerTypeTopP),
                                  &litert_lm_sampler_params_delete);
  ASSERT_NE(sampler_params, nullptr);
  litert_lm_sampler_params_set_top_k(sampler_params.get(), 10);
  litert_lm_sampler_params_set_top_p(sampler_params.get(), 0.5f);
  litert_lm_sampler_params_set_temperature(sampler_params.get(), 0.1f);
  litert_lm_sampler_params_set_seed(sampler_params.get(), 1234);

  SessionConfigPtr config(CreateSessionConfig(),
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
  SessionConfigPtr config(CreateSessionConfig(),
                          &litert_lm_session_config_delete);
  ASSERT_NE(config, nullptr);

  // Verify that the default sampler parameters are used.
  const auto& params = config->config->GetSamplerParams();
  EXPECT_EQ(params.type(),
            litert::lm::proto::SamplerParameters::TYPE_UNSPECIFIED);
}

TEST(EngineCTest, CreateSessionConfigWithApplyPromptTemplate) {
  SessionConfigPtr config(CreateSessionConfig(),
                          &litert_lm_session_config_delete);
  ASSERT_NE(config, nullptr);

  // By default, it is true.
  EXPECT_TRUE(config->config->GetApplyPromptTemplateInSession());

  litert_lm_session_config_set_apply_prompt_template(config.get(), false);
  EXPECT_FALSE(config->config->GetApplyPromptTemplateInSession());

  litert_lm_session_config_set_apply_prompt_template(config.get(), true);
  EXPECT_TRUE(config->config->GetApplyPromptTemplateInSession());
}

TEST(EngineCTest, CreateSessionConfigWithEnableSpeculativeDecoding) {
  SessionConfigPtr config(CreateSessionConfig(),
                          &litert_lm_session_config_delete);
  ASSERT_NE(config, nullptr);

  // By default, enable_speculative_decoding is std::nullopt.
  EXPECT_FALSE(config->config->GetEnableSpeculativeDecoding().has_value());

  litert_lm_session_config_set_enable_speculative_decoding(config.get(), true);
  ASSERT_TRUE(config->config->GetEnableSpeculativeDecoding().has_value());
  EXPECT_TRUE(*config->config->GetEnableSpeculativeDecoding());

  litert_lm_session_config_set_enable_speculative_decoding(config.get(), false);
  ASSERT_TRUE(config->config->GetEnableSpeculativeDecoding().has_value());
  EXPECT_FALSE(*config->config->GetEnableSpeculativeDecoding());
}

TEST(EngineCTest, CreateConversationConfig) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create Sampler Params.
  SamplerParamsPtr sampler_params(CreateSamplerParams(kLiteRtLmSamplerTypeTopP),
                                  &litert_lm_sampler_params_delete);
  ASSERT_NE(sampler_params, nullptr);
  litert_lm_sampler_params_set_top_k(sampler_params.get(), 10);
  litert_lm_sampler_params_set_top_p(sampler_params.get(), 0.5f);
  litert_lm_sampler_params_set_temperature(sampler_params.get(), 0.1f);
  litert_lm_sampler_params_set_seed(sampler_params.get(), 1234);
  SessionConfigPtr session_config(CreateSessionConfig(),
                                  &litert_lm_session_config_delete);
  ASSERT_NE(session_config, nullptr);
  litert_lm_session_config_set_sampler_params(session_config.get(),
                                              sampler_params.get());

  // 3. Create a Conversation Config with the Engine Handle, Session Config
  // and System Message.
  const std::string system_message =
      R"({"type":"text","text":"You are a helpful assistant."})";
  ConversationConfigPtr conversation_config(
      CreateConversationConfig(), &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_session_config(conversation_config.get(),
                                                   session_config.get());
  litert_lm_conversation_config_set_system_message(conversation_config.get(),
                                                   system_message.c_str());

  // 4. Test to see if the Conversation has the Sampler Params.
  ConversationPtr conversation(
      CreateConversation(engine.get(), conversation_config.get()),
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

  litert_lm_engine_settings_set_gpu_enable_metal_residency_set(settings.get(),
                                                               true);
  EXPECT_EQ(litert_lm_experimental_engine_update_gpu_enable_metal_residency_set(
                engine.get(), true),
            kLiteRtLmStatusOk);
  EXPECT_EQ(litert_lm_experimental_engine_update_gpu_enable_metal_residency_set(
                engine.get(), false),
            kLiteRtLmStatusOk);
}

TEST(EngineCTest, CreateConversationConfigWithNoSamplerParams) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create a Conversation Config with the System Message.
  const std::string system_message =
      R"({"type":"text","text":"You are a helpful assistant."})";
  ConversationConfigPtr conversation_config(
      CreateConversationConfig(), &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_system_message(conversation_config.get(),
                                                   system_message.c_str());

  // 3. Test to see if the Conversation has the correct System Message.
  ConversationPtr conversation(
      CreateConversation(engine.get(), conversation_config.get()),
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

TEST(EngineCTest, CreateConversationConfigWithPromptTemplate) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  ConversationConfigPtr conversation_config(
      CreateConversationConfig(), &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  const std::string custom_template = "custom template content";
  litert_lm_conversation_config_set_prompt_template(conversation_config.get(),
                                                    custom_template.c_str());

  ConversationPtr conversation(
      CreateConversation(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);
}

TEST(EngineCTest, CreateConversationConfigWithNoSamplerParamsNoSystemMessage) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create a Conversation Config with the Session Config.
  SessionConfigPtr session_config(CreateSessionConfig(),
                                  &litert_lm_session_config_delete);
  ASSERT_NE(session_config, nullptr);
  ConversationConfigPtr conversation_config(
      CreateConversationConfig(), &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_session_config(conversation_config.get(),
                                                   session_config.get());

  // 4. Test to see if the Conversation has the correct System Message.
  ConversationPtr conversation(
      CreateConversation(engine.get(), conversation_config.get()),
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
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionConfigPtr session_config(CreateSessionConfig(),
                                  &litert_lm_session_config_delete);
  ASSERT_NE(session_config, nullptr);
  session_config->config->SetSamplerBackend(litert::lm::Backend::GPU);

  ConversationConfigPtr conversation_config(
      CreateConversationConfig(), &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_session_config(conversation_config.get(),
                                                   session_config.get());

  ConversationPtr conversation(
      CreateConversation(engine.get(), conversation_config.get()),
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
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
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
      CreateConversationConfig(), &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_tools(conversation_config.get(),
                                          tools_json.c_str());

  // 3. Test to see if the Conversation has the correct tools.
  ConversationPtr conversation(
      CreateConversation(engine.get(), conversation_config.get()),
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
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create a Conversation Config with an invalid tools json.
  const std::string tools_json = R"({"type": "function"})";  // Not an array

  ConversationConfigPtr conversation_config(
      CreateConversationConfig(), &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_tools(conversation_config.get(),
                                          tools_json.c_str());

  // 3. Test to see if the Conversation has no tools.
  ConversationPtr conversation(
      CreateConversation(engine.get(), conversation_config.get()),
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
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create a Conversation Config with an empty tools array.
  const std::string tools_json = R"([])";

  ConversationConfigPtr conversation_config(
      CreateConversationConfig(), &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_tools(conversation_config.get(),
                                          tools_json.c_str());

  // 3. Test to see if the Conversation has empty tools.
  ConversationPtr conversation(
      CreateConversation(engine.get(), conversation_config.get()),
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
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create a Conversation Config with malformed tools json.
  const std::string tools_json = R"([{"type": "function", ...}])";

  ConversationConfigPtr conversation_config(
      CreateConversationConfig(), &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_tools(conversation_config.get(),
                                          tools_json.c_str());

  // 3. Test to see if the Conversation has no tools.
  ConversationPtr conversation(
      CreateConversation(engine.get(), conversation_config.get()),
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
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create Sampler Params.
  SamplerParamsPtr sampler_params(CreateSamplerParams(kLiteRtLmSamplerTypeTopP),
                                  &litert_lm_sampler_params_delete);
  ASSERT_NE(sampler_params, nullptr);
  litert_lm_sampler_params_set_top_k(sampler_params.get(), 10);
  litert_lm_sampler_params_set_top_p(sampler_params.get(), 0.5f);
  litert_lm_sampler_params_set_temperature(sampler_params.get(), 0.1f);
  litert_lm_sampler_params_set_seed(sampler_params.get(), 1234);
  SessionConfigPtr session_config(CreateSessionConfig(),
                                  &litert_lm_session_config_delete);
  ASSERT_NE(session_config, nullptr);
  litert_lm_session_config_set_sampler_params(session_config.get(),
                                              sampler_params.get());

  // 3. Create a Conversation Config with the Session Config.
  ConversationConfigPtr conversation_config(
      CreateConversationConfig(), &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_session_config(conversation_config.get(),
                                                   session_config.get());

  // 4. Test to see if the Conversation has the default Sampler Params.
  ConversationPtr conversation(
      CreateConversation(engine.get(), conversation_config.get()),
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
  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu", nullptr, nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionConfigPtr session_config(CreateSessionConfig(),
                                  &litert_lm_session_config_delete);
  ASSERT_NE(session_config, nullptr);

  ConversationConfigPtr conversation_config(
      CreateConversationConfig(), &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_session_config(conversation_config.get(),
                                                   session_config.get());

  // Set thinking_config on conversation config.
  LiteRtLmThinkingConfig* thinking_config = CreateThinkingConfig();
  ASSERT_NE(thinking_config, nullptr);
  litert_lm_thinking_config_set_enable_thinking(thinking_config, true);
  litert_lm_thinking_config_set_thinking_token_budget(thinking_config, 42);
  litert_lm_conversation_config_set_thinking_config(conversation_config.get(),
                                                    thinking_config);
  litert_lm_thinking_config_delete(thinking_config);

  ConversationPtr conversation(
      CreateConversation(engine.get(), conversation_config.get()),
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
      CreateConversationOptionalArgs();
  ASSERT_NE(optional_args, nullptr);

  LiteRtLmThinkingConfig* thinking_config = CreateThinkingConfig();
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
  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu", nullptr, nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  const char* text = "hello";
  TokenizeResultPtr tokenize_result(Tokenize(engine.get(), text),
                                    &litert_lm_tokenize_result_delete);
  ASSERT_NE(tokenize_result, nullptr);
  size_t num_tokens = GetOk<size_t>(litert_lm_tokenize_result_get_num_tokens,
                                    tokenize_result.get());
  EXPECT_GT(num_tokens, 0);

  const int* tokens = GetOk<const int*>(litert_lm_tokenize_result_get_tokens,
                                        tokenize_result.get());
  DetokenizeResultPtr detokenize_result(
      Detokenize(engine.get(), tokens, num_tokens),
      &litert_lm_detokenize_result_delete);
  ASSERT_NE(detokenize_result, nullptr);
  EXPECT_STREQ(GetOk<const char*>(litert_lm_detokenize_result_get_string,
                                  detokenize_result.get()),
               text);

  TokenUnionPtr start_token(GetStartToken(engine.get()),
                            &litert_lm_token_union_delete);
  if (start_token != nullptr) {
    if (GetOk<LiteRtLmTokenUnionType>(litert_lm_token_union_get_type,
                                      start_token.get()) ==
        kLiteRtLmTokenUnionTypeIds) {
      const int* ids;
      size_t num_ids;
      EXPECT_EQ(
          litert_lm_token_union_get_ids(start_token.get(), &ids, &num_ids),
          kLiteRtLmStatusOk);
      EXPECT_GT(num_ids, 0);
    } else {
      EXPECT_NE(GetOk<const char*>(litert_lm_token_union_get_string,
                                   start_token.get()),
                nullptr);
    }
  }

  TokenUnionsPtr stop_tokens(GetStopTokens(engine.get()),
                             &litert_lm_token_unions_delete);
  if (stop_tokens != nullptr) {
    size_t num_tokens =
        GetOk<size_t>(litert_lm_token_unions_get_num_tokens, stop_tokens.get());
    for (size_t i = 0; i < num_tokens; ++i) {
      TokenUnionPtr stop_token(GetTokenAt(stop_tokens.get(), i),
                               &litert_lm_token_union_delete);
      ASSERT_NE(stop_token, nullptr);
      if (GetOk<LiteRtLmTokenUnionType>(litert_lm_token_union_get_type,
                                        stop_token.get()) ==
          kLiteRtLmTokenUnionTypeIds) {
        const int* ids;
        size_t num_ids;
        EXPECT_EQ(
            litert_lm_token_union_get_ids(stop_token.get(), &ids, &num_ids),
            kLiteRtLmStatusOk);
        EXPECT_GT(num_ids, 0);
      } else {
        EXPECT_NE(GetOk<const char*>(litert_lm_token_union_get_string,
                                     stop_token.get()),
                  nullptr);
      }
    }
  }
}

TEST(EngineCTest, GenerateContent) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(CreateSession(engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt = "Hello world!";
  InputDataPtr input_data(
      CreateInputData(kLiteRtLmInputDataTypeText, prompt, strlen(prompt)),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};
  ResponsesPtr responses(GenerateContent(session.get(), inputs, 1),
                         &litert_lm_responses_delete);
  ASSERT_NE(responses, nullptr);

  EXPECT_EQ(GetOk<int>(litert_lm_responses_get_num_candidates, responses.get()),
            1);
  const char* response_text = GetOk<const char*>(
      litert_lm_responses_get_response_text_at, responses.get(), 0);
  ASSERT_NE(response_text, nullptr);
  EXPECT_GT(strlen(response_text), 0);
}

TEST(EngineCTest, CreateSessionWithMaxOutputTokens) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // Test with max_output_tokens=1. The response length should be short (<10).
  {
    SessionConfigPtr session_config(CreateSessionConfig(),
                                    &litert_lm_session_config_delete);
    ASSERT_NE(session_config, nullptr);
    litert_lm_session_config_set_max_output_tokens(session_config.get(), 1);

    SessionPtr session(CreateSession(engine.get(), session_config.get()),
                       &litert_lm_session_delete);
    ASSERT_NE(session, nullptr);

    const char* prompt = "Hello world!";
    InputDataPtr input_data(
        CreateInputData(kLiteRtLmInputDataTypeText, prompt, strlen(prompt)),
        &litert_lm_input_data_delete);
    ASSERT_NE(input_data, nullptr);
    const LiteRtLmInputData* inputs[] = {input_data.get()};
    ResponsesPtr responses(GenerateContent(session.get(), inputs, 1),
                           &litert_lm_responses_delete);
    ASSERT_NE(responses, nullptr);

    EXPECT_EQ(
        GetOk<int>(litert_lm_responses_get_num_candidates, responses.get()), 1);
    const char* response_text = GetOk<const char*>(
        litert_lm_responses_get_response_text_at, responses.get(), 0);
    ASSERT_NE(response_text, nullptr);
    EXPECT_GT(strlen(response_text), 0);
    EXPECT_LT(strlen(response_text), 10);
  }

  // Test without max_output_tokens. The response length should be long (>=10).
  {
    SessionConfigPtr session_config(CreateSessionConfig(),
                                    &litert_lm_session_config_delete);
    ASSERT_NE(session_config, nullptr);

    SessionPtr session(CreateSession(engine.get(), session_config.get()),
                       &litert_lm_session_delete);
    ASSERT_NE(session, nullptr);

    const char* prompt = "Hello world!";
    InputDataPtr input_data(
        CreateInputData(kLiteRtLmInputDataTypeText, prompt, strlen(prompt)),
        &litert_lm_input_data_delete);
    ASSERT_NE(input_data, nullptr);
    const LiteRtLmInputData* inputs[] = {input_data.get()};
    ResponsesPtr responses(GenerateContent(session.get(), inputs, 1),
                           &litert_lm_responses_delete);
    ASSERT_NE(responses, nullptr);

    EXPECT_EQ(
        GetOk<int>(litert_lm_responses_get_num_candidates, responses.get()), 1);
    const char* response_text = GetOk<const char*>(
        litert_lm_responses_get_response_text_at, responses.get(), 0);
    ASSERT_NE(response_text, nullptr);
    EXPECT_GT(strlen(response_text), 10);
  }
}

TEST(EngineCTest, ConversationSendMessage) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  ConversationPtr conversation(CreateConversation(engine.get(),
                                                  /*config=*/nullptr),
                               &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  JsonResponsePtr response(
      ConversationSendMessage(conversation.get(), message_json,
                              /*extra_context=*/nullptr,
                              /*optional_args=*/nullptr),
      &litert_lm_json_response_delete);
  ASSERT_NE(response, nullptr);

  const char* response_str = GetResponseString(response.get());
  ASSERT_NE(response_str, nullptr);
  EXPECT_GT(strlen(response_str), 0);
}

TEST(EngineCTest, ConversationRenderPreface) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm.litertlm");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  ConversationConfigPtr conversation_config(
      CreateConversationConfig(), &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);

  const char* messages_json =
      R"([{"role": "system", "content": "You are a helpful assistant."}])";
  litert_lm_conversation_config_set_messages(conversation_config.get(),
                                             messages_json);

  ConversationPtr conversation(
      CreateConversation(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const char* rendered = RenderPreface(conversation.get());
  ASSERT_NE(rendered, nullptr);
  EXPECT_THAT(rendered, HasSubstr("You are a helpful assistant."));
}

TEST(EngineCTest, ConversationSendMessageWithConfig) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm.litertlm");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create Sampler Params.
  SamplerParamsPtr sampler_params(CreateSamplerParams(kLiteRtLmSamplerTypeTopP),
                                  &litert_lm_sampler_params_delete);
  ASSERT_NE(sampler_params, nullptr);
  litert_lm_sampler_params_set_top_k(sampler_params.get(), 10);
  litert_lm_sampler_params_set_top_p(sampler_params.get(), 0.5f);
  litert_lm_sampler_params_set_temperature(sampler_params.get(), 0.1f);
  litert_lm_sampler_params_set_seed(sampler_params.get(), 1234);
  SessionConfigPtr session_config(CreateSessionConfig(),
                                  &litert_lm_session_config_delete);
  ASSERT_NE(session_config, nullptr);
  litert_lm_session_config_set_sampler_params(session_config.get(),
                                              sampler_params.get());

  // 3. Create a Conversation Config with the Session Config
  // and System Message.
  const std::string system_message =
      R"({"type":"text","text":"You are a helpful assistant."})";
  ConversationConfigPtr conversation_config(
      CreateConversationConfig(), &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  litert_lm_conversation_config_set_session_config(conversation_config.get(),
                                                   session_config.get());
  litert_lm_conversation_config_set_system_message(conversation_config.get(),
                                                   system_message.c_str());

  // 4. Create a Conversation with the Conversation Config.
  ConversationPtr conversation(
      CreateConversation(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  // 5. Send a message to the conversation.
  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  JsonResponsePtr response(
      ConversationSendMessage(conversation.get(), message_json,
                              /*extra_context=*/nullptr,
                              /*optional_args=*/nullptr),
      &litert_lm_json_response_delete);
  ASSERT_NE(response, nullptr);

  const char* response_str = GetResponseString(response.get());
  ASSERT_NE(response_str, nullptr);
  EXPECT_GT(strlen(response_str), 0);
}

TEST(EngineCTest, ConversationSendMessageWithExtraContext) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm.litertlm");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create a Conversation Config.
  ConversationConfigPtr conversation_config(
      CreateConversationConfig(), &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);

  // 3. Create a Conversation with the Conversation Config.
  ConversationPtr conversation(
      CreateConversation(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  // 4. Send a message to the conversation with extra context.
  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  const char* extra_context = R"({"key": "value"})";
  JsonResponsePtr response(
      ConversationSendMessage(conversation.get(), message_json,
                              /*extra_context=*/extra_context,
                              /*optional_args=*/nullptr),
      &litert_lm_json_response_delete);
  ASSERT_NE(response, nullptr);

  const char* response_str = GetResponseString(response.get());
  ASSERT_NE(response_str, nullptr);
  EXPECT_GT(strlen(response_str), 0);
}

TEST(EngineCTest, ConversationSendMessageWithOptionalArgs) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm.litertlm");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create a Conversation Config.
  ConversationConfigPtr conversation_config(
      CreateConversationConfig(), &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);

  // 3. Create a Conversation with the Conversation Config.
  ConversationPtr conversation(
      CreateConversation(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  // 4. Create Optional Args.
  RepetitionPenaltyConfigPtr repetition_penalty_config(
      CreateRepetitionPenaltyConfig(),
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

  OptionalArgsPtr optional_args(CreateConversationOptionalArgs(),
                                &litert_lm_conversation_optional_args_delete);
  ASSERT_NE(optional_args, nullptr);

  NoRepeatNgramConfigPtr no_repeat_ngram_config(
      CreateNoRepeatNgramConfig(), &litert_lm_no_repeat_ngram_config_delete);
  ASSERT_NE(no_repeat_ngram_config, nullptr);
  litert_lm_no_repeat_ngram_config_set_no_repeat_ngram_size(
      no_repeat_ngram_config.get(), 3);
  litert_lm_no_repeat_ngram_config_set_window_size(no_repeat_ngram_config.get(),
                                                   10);

  SuppressTokensConfigPtr suppress_tokens_config(
      CreateSuppressTokensConfig(), &litert_lm_suppress_tokens_config_delete);
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
  JsonResponsePtr response(
      ConversationSendMessage(conversation.get(), message_json,
                              /*extra_context=*/nullptr, optional_args.get()),
      &litert_lm_json_response_delete);
  ASSERT_NE(response, nullptr);

  const char* response_str = GetResponseString(response.get());
  ASSERT_NE(response_str, nullptr);
  EXPECT_GT(strlen(response_str), 0);
}

TEST(EngineCTest, ConversationSendMessageWithLlGuidance) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm.litertlm");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create a Conversation Config with constrained decoding enabled.
  ConversationConfigPtr conversation_config(
      CreateConversationConfig(), &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  LiteRtLmConstraintProviderType provider =
      kLiteRtLmConstraintProviderTypeLlGuidance;
  litert_lm_conversation_config_set_constraint_provider(
      conversation_config.get(), &provider);
  litert_lm_conversation_config_set_constraint_provider(
      conversation_config.get(), nullptr);
  litert_lm_conversation_config_set_constraint_provider(
      conversation_config.get(), &provider);
  litert_lm_conversation_config_set_enable_constrained_decoding(
      conversation_config.get(), true);

  // 3. Create a Conversation with the Conversation Config.
  ConversationPtr conversation(
      CreateConversation(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  // 4. Create Optional Args with constraint.
  OptionalArgsPtr optional_args(CreateConversationOptionalArgs(),
                                &litert_lm_conversation_optional_args_delete);
  ASSERT_NE(optional_args, nullptr);

  litert_lm_conversation_optional_args_set_constraint(
      optional_args.get(), kLiteRtLmConstraintTypeRegex, "aiedge");

  // 5. Send a message to the conversation with optional args.
  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";

  JsonResponsePtr response(
      ConversationSendMessage(conversation.get(), message_json,
                              /* extra_context */ nullptr, optional_args.get()),
      &litert_lm_json_response_delete);
  ASSERT_NE(response, nullptr);

  const char* response_str = GetResponseString(response.get());
  ASSERT_NE(response_str, nullptr);

  auto response_json = nlohmann::ordered_json::parse(response_str);
  ASSERT_TRUE(response_json.contains("content"));
  ASSERT_TRUE(response_json["content"].is_array());
  ASSERT_GE(response_json["content"].size(), 1);
  std::string text = response_json["content"][0]["text"];
  EXPECT_EQ(text, "aiedge");
}

TEST(EngineCTest, ConversationCloneNull) {
  litert_lm_clear_last_error();
  LiteRtLmConversation* cloned = reinterpret_cast<LiteRtLmConversation*>(0x1);
  const int status = litert_lm_conversation_clone(nullptr, &cloned);
  EXPECT_EQ(status, kLiteRtLmStatusInvalidArgument);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_EQ(cloned, nullptr);
}

TEST(EngineCTest, ConversationCloneSuccess) {
  // 1. Create an engine.
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm.litertlm");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  // 2. Create a Conversation.
  ConversationPtr conversation(CreateConversation(engine.get(),
                                                  /*config=*/nullptr),
                               &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  // 3. Clone the conversation.
  LiteRtLmConversation* cloned_raw = CloneConversation(conversation.get());
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
  const char* error_msg =
      GetOk<const char*>(litert_lm_stream_chunk_get_error, chunk);
  if (error_msg) {
    data->status = absl::InternalError(error_msg);
  }
  const char* text = GetOk<const char*>(litert_lm_stream_chunk_get_text, chunk);
  if (text) {
    data->response.append(text);
  }
  if (GetOk<bool>(litert_lm_stream_chunk_is_final, chunk)) {
    data->done.Notify();
  }
}

TEST(EngineCTest, GenerateContentStream) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(CreateSession(engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt = "Hello world!";
  InputDataPtr input_data(
      CreateInputData(kLiteRtLmInputDataTypeText, prompt, strlen(prompt)),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};
  StreamCallbackData callback_data;
  int result = litert_lm_session_generate_content_stream(
      session.get(), inputs, 1, &StreamCallback, &callback_data);
  ASSERT_EQ(result, kLiteRtLmStatusOk);

  callback_data.done.WaitForNotification();

  // This model is too small and generate random output, so the result may be
  // either success or failure due to maximum kv-cache size reached.
  EXPECT_THAT(
      callback_data.status,
      AnyOf(IsOk(), StatusIs(absl::StatusCode::kInternal,
                             HasSubstr("Max number of tokens reached."))));
  EXPECT_GT(callback_data.response.length(), 0);
}

TEST(EngineCTest, SessionGenerateContentStreamAndCancel) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 512);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(CreateSession(engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt =
      "Hello world! Write a long essay about the history of Rome.";
  InputDataPtr input_data(
      CreateInputData(kLiteRtLmInputDataTypeText, prompt, strlen(prompt)),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};
  StreamCallbackData callback_data;
  int result = litert_lm_session_generate_content_stream(
      session.get(), inputs, 1, &StreamCallback, &callback_data);
  ASSERT_EQ(result, kLiteRtLmStatusOk);

  EXPECT_EQ(litert_lm_session_cancel_process(session.get()), kLiteRtLmStatusOk);

  callback_data.done.WaitForNotification();

  EXPECT_THAT(callback_data.status,
              StatusIs(absl::StatusCode::kInternal, HasSubstr("CANCELLED")));
}

TEST(EngineCTest, ConversationSendMessageStream) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  ConversationPtr conversation(CreateConversation(engine.get(),
                                                  /*config=*/nullptr),
                               &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  StreamCallbackData callback_data;
  int result = litert_lm_conversation_send_message_stream(
      conversation.get(), message_json, /*extra_context=*/nullptr,
      /*optional_args=*/nullptr, &StreamCallback, &callback_data);
  ASSERT_EQ(result, kLiteRtLmStatusOk);

  callback_data.done.WaitForNotification();
  EXPECT_GT(callback_data.response.length(), 0);
}

TEST(EngineCTest, ConversationSendMessageStreamWithExtraContext) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  ConversationPtr conversation(CreateConversation(engine.get(),
                                                  /*config=*/nullptr),
                               &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  const char* extra_context = R"({"key": "value"})";
  StreamCallbackData callback_data;
  int result = litert_lm_conversation_send_message_stream(
      conversation.get(), message_json, /*extra_context=*/extra_context,
      /*optional_args=*/nullptr, &StreamCallback, &callback_data);
  ASSERT_EQ(result, kLiteRtLmStatusOk);

  callback_data.done.WaitForNotification();
  EXPECT_GT(callback_data.response.length(), 0);
}

TEST(EngineCTest, ConversationSendMessageStreamWithOptionalArgs) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  ConversationPtr conversation(CreateConversation(engine.get(),
                                                  /*config=*/nullptr),
                               &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";

  RepetitionPenaltyConfigPtr repetition_penalty_config(
      CreateRepetitionPenaltyConfig(),
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

  OptionalArgsPtr optional_args(CreateConversationOptionalArgs(),
                                &litert_lm_conversation_optional_args_delete);
  ASSERT_NE(optional_args, nullptr);

  NoRepeatNgramConfigPtr no_repeat_ngram_config(
      CreateNoRepeatNgramConfig(), &litert_lm_no_repeat_ngram_config_delete);
  ASSERT_NE(no_repeat_ngram_config, nullptr);
  litert_lm_no_repeat_ngram_config_set_no_repeat_ngram_size(
      no_repeat_ngram_config.get(), 3);
  litert_lm_no_repeat_ngram_config_set_window_size(no_repeat_ngram_config.get(),
                                                   10);

  SuppressTokensConfigPtr suppress_tokens_config(
      CreateSuppressTokensConfig(), &litert_lm_suppress_tokens_config_delete);
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
  ASSERT_EQ(result, kLiteRtLmStatusOk);

  callback_data.done.WaitForNotification();
  EXPECT_GT(callback_data.response.length(), 0);
}

TEST(EngineCTest, ConversationSendMessageStreamAndCancel) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 512);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  ConversationPtr conversation(CreateConversation(engine.get(),
                                                  /*config=*/nullptr),
                               &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  StreamCallbackData callback_data;
  int result = litert_lm_conversation_send_message_stream(
      conversation.get(), message_json, /*extra_context=*/nullptr,
      /*optional_args=*/nullptr, &StreamCallback, &callback_data);
  ASSERT_EQ(result, kLiteRtLmStatusOk);

  EXPECT_EQ(litert_lm_conversation_cancel_process(conversation.get()),
            kLiteRtLmStatusOk);

  callback_data.done.WaitForNotification();
  EXPECT_THAT(callback_data.status,
              StatusIs(absl::StatusCode::kInternal, HasSubstr("CANCELLED")));
}

struct EngineAndConversation {
  EnginePtr engine;
  ConversationPtr conversation;
};

EngineAndConversation CreateTestEngineAndConversation(
    int max_num_tokens = 512) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  if (settings) {
    litert_lm_engine_settings_set_max_num_tokens(settings.get(),
                                                 max_num_tokens);
  }

  EnginePtr engine(settings ? CreateEngine(settings.get()) : nullptr,
                   &litert_lm_engine_delete);

  ConversationPtr conversation(engine ? CreateConversation(engine.get(),
                                                           /*config=*/nullptr)
                                      : nullptr,
                               &litert_lm_conversation_delete);

  return {std::move(engine), std::move(conversation)};
}

TEST(EngineCTest, ConversationSendMessageStreamAndWaitUntilDone) {
  auto [engine, conversation] = CreateTestEngineAndConversation();
  ASSERT_NE(engine, nullptr);
  ASSERT_NE(conversation, nullptr);

  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  StreamCallbackData callback_data;
  int result = litert_lm_conversation_send_message_stream(
      conversation.get(), message_json, /*extra_context=*/nullptr,
      /*optional_args=*/nullptr, &StreamCallback, &callback_data);
  ASSERT_EQ(result, kLiteRtLmStatusOk);

  EXPECT_EQ(litert_lm_conversation_wait_until_done(conversation.get()),
            kLiteRtLmStatusOk);

  // The stream has completed by the time the wait returns.
  EXPECT_TRUE(callback_data.done.HasBeenNotified());
  EXPECT_THAT(callback_data.response, Not(IsEmpty()));
}

TEST(EngineCTest, ConversationWaitUntilDoneWithInvalidConversationFails) {
  EXPECT_EQ(litert_lm_conversation_wait_until_done(/*conversation=*/nullptr),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_STREQ(litert_lm_get_last_error_message(), "Invalid conversation.");
}

TEST(EngineCTest, ConversationWaitUntilDonePropagatesError) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 512);

  ConversationPtr conversation(nullptr, &litert_lm_conversation_delete);
  {
    EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
    ASSERT_NE(engine, nullptr);

    conversation.reset(CreateConversation(engine.get(), /*config=*/nullptr));
    ASSERT_NE(conversation, nullptr);
  }
  // The engine has been deleted, so the underlying execution manager is no
  // longer available. WaitUntilDone() must propagate this error.
  EXPECT_EQ(litert_lm_conversation_wait_until_done(conversation.get()),
            kLiteRtLmStatusFailedPrecondition);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              HasSubstr("Execution manager is not available."));
}

TEST(EngineCTest, ConversationWaitUntilDoneOnIdleConversationSucceeds) {
  auto [engine, conversation] = CreateTestEngineAndConversation();
  ASSERT_NE(engine, nullptr);
  ASSERT_NE(conversation, nullptr);

  EXPECT_EQ(litert_lm_conversation_wait_until_done(conversation.get()),
            kLiteRtLmStatusOk);
}

using BenchmarkInfoPtr =
    std::unique_ptr<LiteRtLmBenchmarkInfo,
                    decltype(&litert_lm_benchmark_info_delete)>;

TEST(EngineCTest, Benchmark) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);
  litert_lm_engine_settings_enable_benchmark(settings.get());

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(CreateSession(engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt = "Hello world!";
  InputDataPtr input_data(
      CreateInputData(kLiteRtLmInputDataTypeText, prompt, strlen(prompt)),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};
  ResponsesPtr responses(GenerateContent(session.get(), inputs, 1),
                         &litert_lm_responses_delete);
  ASSERT_NE(responses, nullptr);

  BenchmarkInfoPtr benchmark_info(GetBenchmarkInfo(session.get()),
                                  &litert_lm_benchmark_info_delete);
  ASSERT_NE(benchmark_info, nullptr);

  EXPECT_GT(GetOk<double>(litert_lm_benchmark_info_get_time_to_first_token,
                          benchmark_info.get()),
            0.0);
  EXPECT_GT(
      GetOk<double>(litert_lm_benchmark_info_get_total_init_time_in_second,
                    benchmark_info.get()),
      0.0);
  int num_prefill_turns = GetOk<int>(
      litert_lm_benchmark_info_get_num_prefill_turns, benchmark_info.get());
  EXPECT_GT(num_prefill_turns, 0);
  for (int i = 0; i < num_prefill_turns; ++i) {
    EXPECT_GT(GetOk<int>(litert_lm_benchmark_info_get_prefill_token_count_at,
                         benchmark_info.get(), i),
              0);

    EXPECT_GT(
        GetOk<double>(litert_lm_benchmark_info_get_prefill_tokens_per_sec_at,
                      benchmark_info.get(), i),
        0.0);
  }
  int num_decode_turns = GetOk<int>(
      litert_lm_benchmark_info_get_num_decode_turns, benchmark_info.get());
  EXPECT_GT(num_decode_turns, 0);
  for (int i = 0; i < num_decode_turns; ++i) {
    EXPECT_GT(GetOk<int>(litert_lm_benchmark_info_get_decode_token_count_at,
                         benchmark_info.get(), i),
              0);

    EXPECT_GT(
        GetOk<double>(litert_lm_benchmark_info_get_decode_tokens_per_sec_at,
                      benchmark_info.get(), i),
        0.0);
  }

  // Turn indices past the end are rejected, and scalar out-parameters are left
  // untouched.
  const auto expect_out_of_range = [](const char* name, int status) {
    SCOPED_TRACE(name);
    EXPECT_EQ(status, kLiteRtLmStatusOutOfRange);
    EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
  };
  int token_count = -7;
  double tokens_per_sec = -7.0;
  litert_lm_clear_last_error();
  expect_out_of_range(
      "prefill_token_count_at",
      litert_lm_benchmark_info_get_prefill_token_count_at(
          benchmark_info.get(), num_prefill_turns, &token_count));
  litert_lm_clear_last_error();
  expect_out_of_range(
      "prefill_tokens_per_sec_at",
      litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
          benchmark_info.get(), num_prefill_turns, &tokens_per_sec));
  litert_lm_clear_last_error();
  expect_out_of_range(
      "decode_token_count_at",
      litert_lm_benchmark_info_get_decode_token_count_at(
          benchmark_info.get(), num_decode_turns, &token_count));
  litert_lm_clear_last_error();
  expect_out_of_range("decode_tokens_per_sec_at",
                      litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
                          benchmark_info.get(), -1, &tokens_per_sec));
  EXPECT_EQ(token_count, -7);
  EXPECT_EQ(tokens_per_sec, -7.0);
}

TEST(EngineCTest, RunPrefillSuccess) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(CreateSession(engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt = "Hello world!";
  InputDataPtr input_data(
      CreateInputData(kLiteRtLmInputDataTypeText, prompt, strlen(prompt)),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};

  int prefill_result = litert_lm_session_run_prefill(session.get(), inputs, 1);
  EXPECT_EQ(prefill_result, kLiteRtLmStatusOk);
}

TEST(EngineCTest, RunPrefillAndDecode) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(CreateSession(engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt = "Hello world!";
  InputDataPtr input_data(
      CreateInputData(kLiteRtLmInputDataTypeText, prompt, strlen(prompt)),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};

  litert_lm_session_run_prefill(session.get(), inputs, 1);

  ResponsesPtr responses(RunDecode(session.get()), &litert_lm_responses_delete);
  ASSERT_NE(responses, nullptr);

  EXPECT_EQ(GetOk<int>(litert_lm_responses_get_num_candidates, responses.get()),
            1);
  const char* response_text = GetOk<const char*>(
      litert_lm_responses_get_response_text_at, responses.get(), 0);
  ASSERT_NE(response_text, nullptr);
  EXPECT_GT(strlen(response_text), 0);
}

TEST(EngineCTest, TextScoringBasic) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(CreateSession(engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt = "Hello world!";
  InputDataPtr input_data(
      CreateInputData(kLiteRtLmInputDataTypeText, prompt, strlen(prompt)),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};

  litert_lm_session_run_prefill(session.get(), inputs, 1);

  const char* target_texts[] = {"apple"};
  ResponsesPtr responses(RunTextScoring(session.get(), target_texts, 1,
                                        /*store_token_lengths=*/true),
                         &litert_lm_responses_delete);
  ASSERT_NE(responses, nullptr);

  EXPECT_EQ(GetOk<int>(litert_lm_responses_get_num_candidates, responses.get()),
            1);
}

TEST(EngineCTest, TextScoringVerifyScores) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(CreateSession(engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt = "Hello world!";
  InputDataPtr input_data(
      CreateInputData(kLiteRtLmInputDataTypeText, prompt, strlen(prompt)),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};

  litert_lm_session_run_prefill(session.get(), inputs, 1);

  const char* target_texts[] = {"apple"};
  ResponsesPtr responses(RunTextScoring(session.get(), target_texts, 1,
                                        /*store_token_lengths=*/true),
                         &litert_lm_responses_delete);
  ASSERT_NE(responses, nullptr);

  EXPECT_TRUE(
      GetOk<bool>(litert_lm_responses_has_score_at, responses.get(), 0));
  float score = 0.0f;
  EXPECT_EQ(litert_lm_responses_get_score_at(responses.get(), 0, &score),
            kLiteRtLmStatusOk);
}

TEST(EngineCTest, TextScoringVerifyTokenLengths) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(CreateSession(engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt = "Hello world!";
  InputDataPtr input_data(
      CreateInputData(kLiteRtLmInputDataTypeText, prompt, strlen(prompt)),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};

  litert_lm_session_run_prefill(session.get(), inputs, 1);

  const char* target_texts[] = {"apple"};
  ResponsesPtr responses(RunTextScoring(session.get(), target_texts, 1,
                                        /*store_token_lengths=*/true),
                         &litert_lm_responses_delete);
  ASSERT_NE(responses, nullptr);

  EXPECT_TRUE(
      GetOk<bool>(litert_lm_responses_has_token_length_at, responses.get(), 0));
  EXPECT_GT(
      GetOk<int>(litert_lm_responses_get_token_length_at, responses.get(), 0),
      0);
}

TEST(EngineCTest, ConversationOptionalArgsTest) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm.litertlm");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  ConversationPtr conversation(CreateConversation(engine.get(),
                                                  /*config=*/nullptr),
                               &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  OptionalArgsPtr optional_args(CreateConversationOptionalArgs(),
                                &litert_lm_conversation_optional_args_delete);
  ASSERT_NE(optional_args, nullptr);
  litert_lm_conversation_optional_args_set_max_output_tokens(
      optional_args.get(), 1);

  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  JsonResponsePtr response(
      ConversationSendMessage(conversation.get(), message_json,
                              /*extra_context=*/nullptr, optional_args.get()),
      &litert_lm_json_response_delete);
  ASSERT_NE(response, nullptr);

  const char* response_str = GetResponseString(response.get());
  ASSERT_NE(response_str, nullptr);
  EXPECT_GT(strlen(response_str), 0);
  // Since max_output_tokens is 1, the response should be very short.
  auto response_json = nlohmann::ordered_json::parse(response_str);
  std::string text = response_json["content"][0]["text"];
  EXPECT_GT(text.length(), 0);
  EXPECT_LT(text.length(), 5);
  EXPECT_EQ(text, "\xE6\xB2\xBF");
}

TEST(EngineCErrorTest, InputDataCreateUnknownTypeSetsError) {
  litert_lm_clear_last_error();
  LiteRtLmInputData* input_data = reinterpret_cast<LiteRtLmInputData*>(0x1);
  // 7 is within the enum's value range but is not a declared enumerator.
  EXPECT_EQ(litert_lm_input_data_create(static_cast<LiteRtLmInputDataType>(7),
                                        "a", 1, &input_data),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(input_data, nullptr);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("Unknown LiteRtLmInputDataType"));
}

TEST(EngineCErrorTest, InputDataCreateNullDataWithSizeSetsError) {
  litert_lm_clear_last_error();
  LiteRtLmInputData* input_data = reinterpret_cast<LiteRtLmInputData*>(0x1);
  EXPECT_EQ(litert_lm_input_data_create(kLiteRtLmInputDataTypeText, nullptr, 4,
                                        &input_data),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(input_data, nullptr);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("data must not be NULL"));
}

TEST(EngineCErrorTest, CreateSucceedsAndReturnsHandleThroughOutParam) {
  LiteRtLmInputData* input_data = nullptr;
  ASSERT_EQ(litert_lm_input_data_create(kLiteRtLmInputDataTypeText, "a", 1,
                                        &input_data),
            kLiteRtLmStatusOk);
  EXPECT_NE(input_data, nullptr);
  litert_lm_input_data_delete(input_data);

  LiteRtLmSessionConfig* session_config = nullptr;
  ASSERT_EQ(litert_lm_session_config_create(&session_config),
            kLiteRtLmStatusOk);
  EXPECT_NE(session_config, nullptr);
  litert_lm_session_config_delete(session_config);

  LiteRtLmEngineSettings* settings = nullptr;
  ASSERT_EQ(litert_lm_engine_settings_create("test_model_path_1", "cpu",
                                             /*vision_backend_str=*/nullptr,
                                             /*audio_backend_str=*/nullptr,
                                             &settings),
            kLiteRtLmStatusOk);
  EXPECT_NE(settings, nullptr);
  litert_lm_engine_settings_delete(settings);
}

TEST(EngineCErrorTest, CreateWithNullOutParamReturnsInvalidArgument) {
  const auto expect_invalid_out_param = [](int status) {
    EXPECT_EQ(status, kLiteRtLmStatusInvalidArgument);
    EXPECT_THAT(litert_lm_get_last_error_message(),
                testing::HasSubstr("must not be NULL"));
    litert_lm_clear_last_error();
  };
  litert_lm_clear_last_error();
  expect_invalid_out_param(
      litert_lm_sampler_params_create(kLiteRtLmSamplerTypeTopK, nullptr));
  expect_invalid_out_param(litert_lm_session_config_create(nullptr));
  expect_invalid_out_param(litert_lm_repetition_penalty_config_create(nullptr));
  expect_invalid_out_param(litert_lm_no_repeat_ngram_config_create(nullptr));
  expect_invalid_out_param(litert_lm_suppress_tokens_config_create(nullptr));
  expect_invalid_out_param(litert_lm_input_data_create(
      kLiteRtLmInputDataTypeText, "a", 1, /*out_input_data=*/nullptr));
  expect_invalid_out_param(litert_lm_engine_settings_create(
      "test_model_path_1", "cpu", /*vision_backend_str=*/nullptr,
      /*audio_backend_str=*/nullptr, /*out_settings=*/nullptr));
  expect_invalid_out_param(
      litert_lm_engine_settings_create_from_raw_file_descriptor(
          /*fd=*/-1, "cpu", /*vision_backend_str=*/nullptr,
          /*audio_backend_str=*/nullptr, /*out_settings=*/nullptr));
  expect_invalid_out_param(
      litert_lm_engine_create(/*settings=*/nullptr, /*out_engine=*/nullptr));
  expect_invalid_out_param(litert_lm_engine_create_session(
      /*engine=*/nullptr, /*config=*/nullptr, /*out_session=*/nullptr));
}

TEST(EngineCErrorTest, EngineSettingsCreateFailureSetsLastError) {
  litert_lm_clear_last_error();
  LiteRtLmEngineSettings* settings =
      reinterpret_cast<LiteRtLmEngineSettings*>(0x1);
  const int status = litert_lm_engine_settings_create(
      "test_model_path_1", "not_a_backend", /*vision_backend_str=*/nullptr,
      /*audio_backend_str=*/nullptr, &settings);
  EXPECT_EQ(status, kLiteRtLmStatusInvalidArgument);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_EQ(settings, nullptr);

  litert_lm_clear_last_error();
  settings = reinterpret_cast<LiteRtLmEngineSettings*>(0x1);
  EXPECT_EQ(litert_lm_engine_settings_create_from_raw_file_descriptor(
                /*fd=*/-1, "cpu", /*vision_backend_str=*/nullptr,
                /*audio_backend_str=*/nullptr, &settings),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_EQ(settings, nullptr);
}

TEST(EngineCErrorTest, EngineCreateWithBadModelPathSetsLastError) {
  EngineSettingsPtr settings(
      CreateEngineSettings("/nonexistent/model.litertlm", "cpu",
                           /*vision_backend_str=*/nullptr,
                           /*audio_backend_str=*/nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);

  litert_lm_clear_last_error();
  LiteRtLmEngine* engine = reinterpret_cast<LiteRtLmEngine*>(0x1);
  const int status = litert_lm_engine_create(settings.get(), &engine);
  EXPECT_NE(status, kLiteRtLmStatusOk);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_EQ(engine, nullptr);
}

TEST(EngineCErrorTest, EngineCreateSessionWithNullEngineReturnsError) {
  litert_lm_clear_last_error();
  LiteRtLmSession* session = reinterpret_cast<LiteRtLmSession*>(0x1);
  EXPECT_EQ(litert_lm_engine_create_session(/*engine=*/nullptr,
                                            /*config=*/nullptr, &session),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_EQ(session, nullptr);
}

TEST(EngineCErrorTest, ResultProducersWithNullOutParamReturnInvalidArgument) {
  const auto expect_invalid_out_param = [](int status) {
    EXPECT_EQ(status, kLiteRtLmStatusInvalidArgument);
    EXPECT_THAT(litert_lm_get_last_error_message(),
                testing::HasSubstr("must not be NULL"));
    litert_lm_clear_last_error();
  };
  const char* target_text[] = {"a"};
  const int tokens[] = {1};
  LiteRtLmTokenUnions token_unions;
  token_unions.tokens.emplace_back();
  litert_lm_clear_last_error();
  expect_invalid_out_param(litert_lm_session_run_decode(
      /*session=*/nullptr, /*out_responses=*/nullptr));
  expect_invalid_out_param(litert_lm_session_run_text_scoring(
      /*session=*/nullptr, target_text, 1, /*store_token_lengths=*/false,
      /*out_responses=*/nullptr));
  expect_invalid_out_param(litert_lm_session_generate_content(
      /*session=*/nullptr, /*inputs=*/nullptr, 0, /*out_responses=*/nullptr));
  expect_invalid_out_param(litert_lm_session_get_benchmark_info(
      /*session=*/nullptr, /*out_benchmark_info=*/nullptr));
  expect_invalid_out_param(litert_lm_engine_tokenize(
      /*engine=*/nullptr, "a", /*out_result=*/nullptr));
  expect_invalid_out_param(litert_lm_engine_detokenize(
      /*engine=*/nullptr, tokens, 1, /*out_result=*/nullptr));
  expect_invalid_out_param(litert_lm_engine_get_start_token(
      /*engine=*/nullptr, /*out_token=*/nullptr));
  expect_invalid_out_param(litert_lm_engine_get_stop_tokens(
      /*engine=*/nullptr, /*out_tokens=*/nullptr));
  // A valid collection and index still fail without an out-parameter.
  expect_invalid_out_param(litert_lm_token_unions_get_token_at(
      &token_unions, 0, /*out_token=*/nullptr));
}

// Calls `call` with an out-parameter holding a non-NULL sentinel and expects
// kLiteRtLmStatusInvalidArgument, a recorded error message, and the
// out-parameter reset to NULL.
template <typename T>
void ExpectInvalidArgumentResetsOut(const char* name,
                                    const std::function<int(T**)>& call) {
  SCOPED_TRACE(name);
  litert_lm_clear_last_error();
  T* out = reinterpret_cast<T*>(0x1);
  const int status = call(&out);
  EXPECT_EQ(status, kLiteRtLmStatusInvalidArgument);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_EQ(out, nullptr);
}

TEST(EngineCErrorTest, ResultProducersWithNullHandleResetOutParam) {
  const char* target_text[] = {"a"};
  const int tokens[] = {1};
  ExpectInvalidArgumentResetsOut<LiteRtLmResponses>(
      "session_run_decode", [](LiteRtLmResponses** out) {
        return litert_lm_session_run_decode(nullptr, out);
      });
  ExpectInvalidArgumentResetsOut<LiteRtLmResponses>(
      "session_run_text_scoring", [&](LiteRtLmResponses** out) {
        return litert_lm_session_run_text_scoring(
            nullptr, target_text, 1, /*store_token_lengths=*/false, out);
      });
  ExpectInvalidArgumentResetsOut<LiteRtLmResponses>(
      "session_generate_content", [](LiteRtLmResponses** out) {
        return litert_lm_session_generate_content(nullptr, nullptr, 0, out);
      });
  ExpectInvalidArgumentResetsOut<LiteRtLmBenchmarkInfo>(
      "session_get_benchmark_info", [](LiteRtLmBenchmarkInfo** out) {
        return litert_lm_session_get_benchmark_info(nullptr, out);
      });
  ExpectInvalidArgumentResetsOut<LiteRtLmTokenizeResult>(
      "engine_tokenize", [](LiteRtLmTokenizeResult** out) {
        return litert_lm_engine_tokenize(nullptr, "a", out);
      });
  ExpectInvalidArgumentResetsOut<LiteRtLmDetokenizeResult>(
      "engine_detokenize", [&](LiteRtLmDetokenizeResult** out) {
        return litert_lm_engine_detokenize(nullptr, tokens, 1, out);
      });
  ExpectInvalidArgumentResetsOut<LiteRtLmTokenUnion>(
      "engine_get_start_token", [](LiteRtLmTokenUnion** out) {
        return litert_lm_engine_get_start_token(nullptr, out);
      });
  ExpectInvalidArgumentResetsOut<LiteRtLmTokenUnions>(
      "engine_get_stop_tokens", [](LiteRtLmTokenUnions** out) {
        return litert_lm_engine_get_stop_tokens(nullptr, out);
      });
  ExpectInvalidArgumentResetsOut<LiteRtLmTokenUnion>(
      "token_unions_get_token_at", [](LiteRtLmTokenUnion** out) {
        return litert_lm_token_unions_get_token_at(nullptr, 0, out);
      });
}

TEST(EngineCErrorTest, TokenUnionsGetTokenAt) {
  LiteRtLmTokenUnions token_unions;
  token_unions.tokens.emplace_back().mutable_token_ids()->add_ids(7);

  LiteRtLmTokenUnion* token = nullptr;
  ASSERT_EQ(litert_lm_token_unions_get_token_at(&token_unions, 0, &token),
            kLiteRtLmStatusOk);
  TokenUnionPtr token_ptr(token, &litert_lm_token_union_delete);
  ASSERT_NE(token_ptr, nullptr);
  const int* ids = nullptr;
  size_t num_ids = 0;
  ASSERT_EQ(litert_lm_token_union_get_ids(token_ptr.get(), &ids, &num_ids),
            kLiteRtLmStatusOk);
  ASSERT_EQ(num_ids, 1);
  EXPECT_EQ(ids[0], 7);

  litert_lm_clear_last_error();
  token = reinterpret_cast<LiteRtLmTokenUnion*>(0x1);
  const int status =
      litert_lm_token_unions_get_token_at(&token_unions, 1, &token);
  EXPECT_EQ(status, kLiteRtLmStatusOutOfRange);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("out of range"));
  EXPECT_EQ(token, nullptr);
}

TEST(EngineCErrorTest, SessionResultProducersFailureSetsLastError) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");
  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);
  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);
  SessionPtr session(CreateSession(engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  // Text scoring requires at least one target.
  litert_lm_clear_last_error();
  const char* target_text[] = {"a"};
  LiteRtLmResponses* responses = reinterpret_cast<LiteRtLmResponses*>(0x1);
  int status = litert_lm_session_run_text_scoring(
      session.get(), target_text, 0, /*store_token_lengths=*/false, &responses);
  EXPECT_EQ(status, kLiteRtLmStatusInvalidArgument);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_EQ(responses, nullptr);

  // Benchmarking is not enabled for this engine. The exact code is determined
  // by the runtime; it must be a failure code and record a last error message.
  litert_lm_clear_last_error();
  LiteRtLmBenchmarkInfo* benchmark_info =
      reinterpret_cast<LiteRtLmBenchmarkInfo*>(0x1);
  status = litert_lm_session_get_benchmark_info(session.get(), &benchmark_info);
  EXPECT_GT(status, kLiteRtLmStatusOk);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_EQ(benchmark_info, nullptr);

  // Tokenizing requires text.
  litert_lm_clear_last_error();
  LiteRtLmTokenizeResult* tokenize_result =
      reinterpret_cast<LiteRtLmTokenizeResult*>(0x1);
  status = litert_lm_engine_tokenize(engine.get(), /*text=*/nullptr,
                                     &tokenize_result);
  EXPECT_EQ(status, kLiteRtLmStatusInvalidArgument);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_EQ(tokenize_result, nullptr);
}

TEST(EngineCErrorTest, SuppressTokensNullWithCountSetsError) {
  SuppressTokensConfigPtr config(CreateSuppressTokensConfig(),
                                 &litert_lm_suppress_tokens_config_delete);
  ASSERT_NE(config, nullptr);
  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_suppress_tokens_config_set_suppress_tokens(config.get(),
                                                                 nullptr, 3),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("suppress_tokens must not be NULL"));
}

TEST(EngineCErrorTest, VoidSettersWithNullHandleSetError) {
  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_engine_settings_set_max_num_tokens(nullptr, 16),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("Invalid engine settings"));

  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_session_config_set_max_output_tokens(nullptr, 16),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("Invalid session config"));

  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_sampler_params_set_top_k(nullptr, 1),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("params must not be NULL"));

  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_repetition_penalty_config_set_window_size(nullptr, 1),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("config must not be NULL"));

  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_session_cancel_process(nullptr),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("Invalid session"));
}

TEST(EngineCErrorTest, SessionConfigNullSamplerParamsSetsError) {
  SessionConfigPtr config(CreateSessionConfig(),
                          &litert_lm_session_config_delete);
  ASSERT_NE(config, nullptr);
  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_session_config_set_sampler_params(config.get(), nullptr),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("sampler_params must not be NULL"));
}

// Runs every setter in `setters` against `handle` (expecting OK) and against
// NULL (expecting kLiteRtLmStatusInvalidArgument and a recorded error message).
template <typename T>
void ExpectSettersReturnStatus(
    T* handle,
    const std::vector<std::pair<std::string, std::function<int(T*)>>>&
        setters) {
  for (const auto& [name, setter] : setters) {
    SCOPED_TRACE(name);
    EXPECT_EQ(setter(handle), kLiteRtLmStatusOk);
    litert_lm_clear_last_error();
    EXPECT_EQ(setter(nullptr), kLiteRtLmStatusInvalidArgument);
    EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
  }
}

TEST(EngineCStatusTest, SamplerParamsSettersReturnStatus) {
  SamplerParamsPtr params(CreateSamplerParams(kLiteRtLmSamplerTypeTopK),
                          &litert_lm_sampler_params_delete);
  ASSERT_NE(params, nullptr);
  using P = LiteRtLmSamplerParams;
  ExpectSettersReturnStatus<P>(
      params.get(),
      {
          {"set_top_k",
           [](P* p) { return litert_lm_sampler_params_set_top_k(p, 5); }},
          {"set_top_p",
           [](P* p) { return litert_lm_sampler_params_set_top_p(p, 0.5f); }},
          {"set_temperature",
           [](P* p) {
             return litert_lm_sampler_params_set_temperature(p, 0.7f);
           }},
          {"set_seed",
           [](P* p) { return litert_lm_sampler_params_set_seed(p, 42); }},
      });
}

TEST(EngineCStatusTest, SessionConfigSettersReturnStatus) {
  SessionConfigPtr config(CreateSessionConfig(),
                          &litert_lm_session_config_delete);
  ASSERT_NE(config, nullptr);
  SamplerParamsPtr params(CreateSamplerParams(kLiteRtLmSamplerTypeGreedy),
                          &litert_lm_sampler_params_delete);
  ASSERT_NE(params, nullptr);
  const LiteRtLmSamplerParams* sampler_params = params.get();
  using C = LiteRtLmSessionConfig;
  ExpectSettersReturnStatus<C>(
      config.get(),
      {
          {"set_max_output_tokens",
           [](C* c) {
             return litert_lm_session_config_set_max_output_tokens(c, 8);
           }},
          {"set_apply_prompt_template",
           [](C* c) {
             return litert_lm_session_config_set_apply_prompt_template(c,
                                                                       false);
           }},
          {"set_enable_speculative_decoding",
           [](C* c) {
             return litert_lm_session_config_set_enable_speculative_decoding(
                 c, true);
           }},
          {"set_sampler_params",
           [sampler_params](C* c) {
             return litert_lm_session_config_set_sampler_params(c,
                                                                sampler_params);
           }},
      });
}

TEST(EngineCStatusTest, ConstraintConfigSettersReturnStatus) {
  RepetitionPenaltyConfigPtr repetition_config(
      CreateRepetitionPenaltyConfig(),
      &litert_lm_repetition_penalty_config_delete);
  ASSERT_NE(repetition_config, nullptr);
  using R = LiteRtLmRepetitionPenaltyConfig;
  ExpectSettersReturnStatus<R>(
      repetition_config.get(),
      {
          {"set_repetition_penalty",
           [](R* c) {
             return litert_lm_repetition_penalty_config_set_repetition_penalty(
                 c, 1.2f);
           }},
          {"set_presence_penalty",
           [](R* c) {
             return litert_lm_repetition_penalty_config_set_presence_penalty(
                 c, 0.5f);
           }},
          {"set_frequency_penalty",
           [](R* c) {
             return litert_lm_repetition_penalty_config_set_frequency_penalty(
                 c, 0.5f);
           }},
          {"set_window_size",
           [](R* c) {
             return litert_lm_repetition_penalty_config_set_window_size(c, 4);
           }},
      });

  NoRepeatNgramConfigPtr ngram_config(CreateNoRepeatNgramConfig(),
                                      &litert_lm_no_repeat_ngram_config_delete);
  ASSERT_NE(ngram_config, nullptr);
  using N = LiteRtLmNoRepeatNgramConfig;
  ExpectSettersReturnStatus<N>(
      ngram_config.get(),
      {
          {"set_no_repeat_ngram_size",
           [](N* c) {
             return litert_lm_no_repeat_ngram_config_set_no_repeat_ngram_size(
                 c, 3);
           }},
          {"set_window_size",
           [](N* c) {
             return litert_lm_no_repeat_ngram_config_set_window_size(c, 8);
           }},
      });

  SuppressTokensConfigPtr suppress_config(
      CreateSuppressTokensConfig(), &litert_lm_suppress_tokens_config_delete);
  ASSERT_NE(suppress_config, nullptr);
  using S = LiteRtLmSuppressTokensConfig;
  ExpectSettersReturnStatus<S>(
      suppress_config.get(),
      {
          {"set_suppress_tokens",
           [](S* c) {
             const int tokens[] = {1, 2, 3};
             return litert_lm_suppress_tokens_config_set_suppress_tokens(
                 c, tokens, 3);
           }},
          {"set_suppress_tokens_clear",
           [](S* c) {
             return litert_lm_suppress_tokens_config_set_suppress_tokens(
                 c, nullptr, 0);
           }},
      });
}

std::vector<std::pair<std::string, std::function<int(LiteRtLmEngineSettings*)>>>
AllEngineSettingsSetters() {
  using E = LiteRtLmEngineSettings;
  return {
      {"set_max_num_tokens",
       [](E* s) {
         return litert_lm_engine_settings_set_max_num_tokens(s, 16);
       }},
      {"set_num_threads",
       [](E* s) { return litert_lm_engine_settings_set_num_threads(s, 2); }},
      {"set_audio_num_threads",
       [](E* s) {
         return litert_lm_engine_settings_set_audio_num_threads(s, 2);
       }},
      {"set_parallel_file_section_loading",
       [](E* s) {
         return litert_lm_engine_settings_set_parallel_file_section_loading(
             s, false);
       }},
      {"set_single_threaded_execution",
       [](E* s) {
         return litert_lm_engine_settings_set_single_threaded_execution(s,
                                                                        true);
       }},
      {"set_max_num_images",
       [](E* s) { return litert_lm_engine_settings_set_max_num_images(s, 2); }},
      {"set_max_vision_tokens_per_image",
       [](E* s) {
         return litert_lm_engine_settings_set_max_vision_tokens_per_image(s,
                                                                          280);
       }},
      {"set_cache_dir",
       [](E* s) {
         return litert_lm_engine_settings_set_cache_dir(s, "test_cache_dir");
       }},
      {"set_litert_dispatch_lib_dir",
       [](E* s) {
         return litert_lm_engine_settings_set_litert_dispatch_lib_dir(
             s, "test_lib_dir");
       }},
      {"set_activation_data_type",
       [](E* s) {
         return litert_lm_engine_settings_set_activation_data_type(
             s, kLiteRtLmActivationDataTypeFloat16);
       }},
      {"set_prefill_chunk_size",
       [](E* s) {
         return litert_lm_engine_settings_set_prefill_chunk_size(s, 128);
       }},
      {"set_enable_ynnpack",
       [](E* s) {
         return litert_lm_engine_settings_set_enable_ynnpack(s, true);
       }},
      {"enable_benchmark",
       [](E* s) { return litert_lm_engine_settings_enable_benchmark(s); }},
      {"set_num_prefill_tokens",
       [](E* s) {
         return litert_lm_engine_settings_set_num_prefill_tokens(s, 8);
       }},
      {"set_num_decode_tokens",
       [](E* s) {
         return litert_lm_engine_settings_set_num_decode_tokens(s, 8);
       }},
      {"set_enable_speculative_decoding",
       [](E* s) {
         return litert_lm_engine_settings_set_enable_speculative_decoding(s,
                                                                          true);
       }},
      {"set_gpu_decode_steps_per_sync",
       [](E* s) {
         return litert_lm_engine_settings_set_gpu_decode_steps_per_sync(s, 4);
       }},
      {"set_gpu_wait_for_weight_uploads",
       [](E* s) {
         return litert_lm_engine_settings_set_gpu_wait_for_weight_uploads(s,
                                                                          true);
       }},
      {"set_use_ringbuffers_local_attention",
       [](E* s) {
         return litert_lm_engine_settings_set_use_ringbuffers_local_attention(
             s, true);
       }},
      {"set_lora_rank",
       [](E* s) { return litert_lm_engine_settings_set_lora_rank(s, 8); }},
      {"set_audio_lora_rank",
       [](E* s) {
         return litert_lm_engine_settings_set_audio_lora_rank(s, 8);
       }},
      {"set_gpu_enable_metal_residency_set",
       [](E* s) {
         return litert_lm_engine_settings_set_gpu_enable_metal_residency_set(
             s, true);
       }},
  };
}

TEST(EngineCStatusTest, EngineSettingsSettersReturnStatus) {
  // CPU main backend with audio and vision: GPU-only knobs are no-ops that
  // still return OK.
  EngineSettingsPtr settings(
      CreateEngineSettings("test_model_path_1", "cpu",
                           /* vision_backend_str */ "gpu",
                           /* audio_backend_str */ "cpu"),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  ExpectSettersReturnStatus<LiteRtLmEngineSettings>(settings.get(),
                                                    AllEngineSettingsSetters());
}

TEST(EngineCStatusTest, EngineSettingsNoOpSettersReturnOk) {
  // GPU main backend without audio: CPU-only and audio knobs are no-ops that
  // still return OK.
  EngineSettingsPtr settings(
      CreateEngineSettings("test_model_path_1", "gpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  ExpectSettersReturnStatus<LiteRtLmEngineSettings>(settings.get(),
                                                    AllEngineSettingsSetters());
}

TEST(EngineCStatusTest, EngineSettingsNullStringArgumentsReturnError) {
  EngineSettingsPtr settings(
      CreateEngineSettings("test_model_path_1", "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);

  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_engine_settings_set_cache_dir(settings.get(), nullptr),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("cache_dir must not be NULL"));

  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_engine_settings_set_litert_dispatch_lib_dir(
                settings.get(), nullptr),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("lib_dir must not be NULL"));
}

TEST(EngineCStatusTest, SetMinLogLevelReturnsStatus) {
  EXPECT_EQ(litert_lm_set_min_log_level(kLiteRtLmLogSeverityInfo),
            kLiteRtLmStatusOk);

  litert_lm_clear_last_error();
  // 6 is within the enum's value range but is not a declared enumerator.
  EXPECT_EQ(litert_lm_set_min_log_level(static_cast<LiteRtLmLogSeverity>(6)),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("Unknown LiteRtLmLogSeverity"));
}

TEST(EngineCErrorTest, GettersWithNullHandleSetError) {
  litert_lm_clear_last_error();
  int num_candidates = -7;
  EXPECT_EQ(litert_lm_responses_get_num_candidates(nullptr, &num_candidates),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(num_candidates, -7);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("responses must not be NULL"));

  litert_lm_clear_last_error();
  const int* tokens = reinterpret_cast<const int*>(0x1);
  EXPECT_EQ(litert_lm_tokenize_result_get_tokens(nullptr, &tokens),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(tokens, nullptr);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("result must not be NULL"));

  litert_lm_clear_last_error();
  int num_turns = -7;
  EXPECT_EQ(litert_lm_benchmark_info_get_num_decode_turns(nullptr, &num_turns),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(num_turns, -7);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("benchmark_info must not be NULL"));

  litert_lm_clear_last_error();
  LiteRtLmTokenUnion* token = reinterpret_cast<LiteRtLmTokenUnion*>(0x1);
  EXPECT_EQ(litert_lm_token_unions_get_token_at(nullptr, 0, &token),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(token, nullptr);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("tokens must not be NULL"));
}

// A named C API call that returns a LiteRtLmStatusCode.
struct NamedCall {
  const char* name;
  std::function<int()> call;
};

// Runs every call and expects it to return `expected_code`, to record a last
// error message, and (if non-empty) for that message to contain
// `expected_message`.
void ExpectAllReturn(const std::vector<NamedCall>& calls, int expected_code,
                     const std::string& expected_message = "") {
  for (const NamedCall& named_call : calls) {
    SCOPED_TRACE(named_call.name);
    litert_lm_clear_last_error();
    const int status = named_call.call();
    EXPECT_EQ(status, expected_code);
    EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
    if (!expected_message.empty()) {
      EXPECT_THAT(litert_lm_get_last_error_message(),
                  testing::HasSubstr(expected_message));
    }
  }
}

TEST(EngineCErrorTest, AccessorsWithNullOutParamReturnInvalidArgument) {
  // Handles are valid (or NULL, which must not matter): the out-parameter is
  // validated first.
  LiteRtLmResponses responses{
      litert::lm::Responses(litert::lm::TaskState::kDone, {"a"}, {0.5f})};
  LiteRtLmStreamChunk chunk;
  LiteRtLmTokenizeResult tokenize_result;
  LiteRtLmDetokenizeResult detokenize_result;
  LiteRtLmTokenUnion token_union;
  LiteRtLmTokenUnions token_unions;
  const LiteRtLmBenchmarkInfo* benchmark_info = nullptr;
  ExpectAllReturn(
      {
          {"responses_get_num_candidates",
           [&] {
             return litert_lm_responses_get_num_candidates(&responses, nullptr);
           }},
          {"responses_get_response_text_at",
           [&] {
             return litert_lm_responses_get_response_text_at(&responses, 0,
                                                             nullptr);
           }},
          {"responses_has_score_at",
           [&] {
             return litert_lm_responses_has_score_at(&responses, 0, nullptr);
           }},
          {"responses_get_score_at",
           [&] {
             return litert_lm_responses_get_score_at(&responses, 0, nullptr);
           }},
          {"responses_has_token_length_at",
           [&] {
             return litert_lm_responses_has_token_length_at(&responses, 0,
                                                            nullptr);
           }},
          {"responses_get_token_length_at",
           [&] {
             return litert_lm_responses_get_token_length_at(&responses, 0,
                                                            nullptr);
           }},
          {"responses_has_token_scores_at",
           [&] {
             return litert_lm_responses_has_token_scores_at(&responses, 0,
                                                            nullptr);
           }},
          {"responses_get_num_token_scores_at",
           [&] {
             return litert_lm_responses_get_num_token_scores_at(&responses, 0,
                                                                nullptr);
           }},
          {"responses_get_token_scores_at",
           [&] {
             return litert_lm_responses_get_token_scores_at(&responses, 0,
                                                            nullptr);
           }},
          {"benchmark_info_get_time_to_first_token",
           [&] {
             return litert_lm_benchmark_info_get_time_to_first_token(
                 benchmark_info, nullptr);
           }},
          {"benchmark_info_get_total_init_time_in_second",
           [&] {
             return litert_lm_benchmark_info_get_total_init_time_in_second(
                 benchmark_info, nullptr);
           }},
          {"benchmark_info_get_num_prefill_turns",
           [&] {
             return litert_lm_benchmark_info_get_num_prefill_turns(
                 benchmark_info, nullptr);
           }},
          {"benchmark_info_get_num_decode_turns",
           [&] {
             return litert_lm_benchmark_info_get_num_decode_turns(
                 benchmark_info, nullptr);
           }},
          {"benchmark_info_get_prefill_token_count_at",
           [&] {
             return litert_lm_benchmark_info_get_prefill_token_count_at(
                 benchmark_info, 0, nullptr);
           }},
          {"benchmark_info_get_decode_token_count_at",
           [&] {
             return litert_lm_benchmark_info_get_decode_token_count_at(
                 benchmark_info, 0, nullptr);
           }},
          {"benchmark_info_get_prefill_tokens_per_sec_at",
           [&] {
             return litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
                 benchmark_info, 0, nullptr);
           }},
          {"benchmark_info_get_decode_tokens_per_sec_at",
           [&] {
             return litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
                 benchmark_info, 0, nullptr);
           }},
          {"stream_chunk_get_text",
           [&] { return litert_lm_stream_chunk_get_text(&chunk, nullptr); }},
          {"stream_chunk_is_final",
           [&] { return litert_lm_stream_chunk_is_final(&chunk, nullptr); }},
          {"stream_chunk_get_error",
           [&] { return litert_lm_stream_chunk_get_error(&chunk, nullptr); }},
          {"tokenize_result_get_tokens",
           [&] {
             return litert_lm_tokenize_result_get_tokens(&tokenize_result,
                                                         nullptr);
           }},
          {"tokenize_result_get_num_tokens",
           [&] {
             return litert_lm_tokenize_result_get_num_tokens(&tokenize_result,
                                                             nullptr);
           }},
          {"detokenize_result_get_string",
           [&] {
             return litert_lm_detokenize_result_get_string(&detokenize_result,
                                                           nullptr);
           }},
          {"token_union_get_type",
           [&] {
             return litert_lm_token_union_get_type(&token_union, nullptr);
           }},
          {"token_union_get_string",
           [&] {
             return litert_lm_token_union_get_string(&token_union, nullptr);
           }},
          {"token_union_get_ids_null_tokens",
           [&] {
             size_t num_tokens = 0;
             return litert_lm_token_union_get_ids(&token_union, nullptr,
                                                  &num_tokens);
           }},
          {"token_union_get_ids_null_num_tokens",
           [&] {
             const int* tokens = nullptr;
             return litert_lm_token_union_get_ids(&token_union, &tokens,
                                                  nullptr);
           }},
          {"token_unions_get_num_tokens",
           [&] {
             return litert_lm_token_unions_get_num_tokens(&token_unions,
                                                          nullptr);
           }},
      },
      kLiteRtLmStatusInvalidArgument, "must not be NULL");
}

TEST(EngineCErrorTest, AccessorsWithNullHandleReturnInvalidArgument) {
  // Scalar out-parameters must be left untouched and pointer out-parameters
  // must be reset to NULL.
  constexpr int kIntSentinel = -7;
  constexpr float kFloatSentinel = -7.0f;
  constexpr double kDoubleSentinel = -7.0;
  constexpr size_t kSizeSentinel = 77;
  int out_int = kIntSentinel;
  bool out_bool = true;
  float out_float = kFloatSentinel;
  double out_double = kDoubleSentinel;
  size_t out_size = kSizeSentinel;
  LiteRtLmTokenUnionType out_type = kLiteRtLmTokenUnionTypeString;
  const char* out_text = "sentinel";
  const float* out_floats = &kFloatSentinel;
  const int* out_ints = &kIntSentinel;
  const int* out_union_ids = &kIntSentinel;
  ExpectAllReturn(
      {
          {"responses_get_num_candidates",
           [&] {
             return litert_lm_responses_get_num_candidates(nullptr, &out_int);
           }},
          {"responses_get_response_text_at",
           [&] {
             return litert_lm_responses_get_response_text_at(nullptr, 0,
                                                             &out_text);
           }},
          {"responses_has_score_at",
           [&] {
             return litert_lm_responses_has_score_at(nullptr, 0, &out_bool);
           }},
          {"responses_get_score_at",
           [&] {
             return litert_lm_responses_get_score_at(nullptr, 0, &out_float);
           }},
          {"responses_has_token_length_at",
           [&] {
             return litert_lm_responses_has_token_length_at(nullptr, 0,
                                                            &out_bool);
           }},
          {"responses_get_token_length_at",
           [&] {
             return litert_lm_responses_get_token_length_at(nullptr, 0,
                                                            &out_int);
           }},
          {"responses_has_token_scores_at",
           [&] {
             return litert_lm_responses_has_token_scores_at(nullptr, 0,
                                                            &out_bool);
           }},
          {"responses_get_num_token_scores_at",
           [&] {
             return litert_lm_responses_get_num_token_scores_at(nullptr, 0,
                                                                &out_int);
           }},
          {"responses_get_token_scores_at",
           [&] {
             return litert_lm_responses_get_token_scores_at(nullptr, 0,
                                                            &out_floats);
           }},
          {"benchmark_info_get_time_to_first_token",
           [&] {
             return litert_lm_benchmark_info_get_time_to_first_token(
                 nullptr, &out_double);
           }},
          {"benchmark_info_get_total_init_time_in_second",
           [&] {
             return litert_lm_benchmark_info_get_total_init_time_in_second(
                 nullptr, &out_double);
           }},
          {"benchmark_info_get_num_prefill_turns",
           [&] {
             return litert_lm_benchmark_info_get_num_prefill_turns(nullptr,
                                                                   &out_int);
           }},
          {"benchmark_info_get_num_decode_turns",
           [&] {
             return litert_lm_benchmark_info_get_num_decode_turns(nullptr,
                                                                  &out_int);
           }},
          {"benchmark_info_get_prefill_token_count_at",
           [&] {
             return litert_lm_benchmark_info_get_prefill_token_count_at(
                 nullptr, 0, &out_int);
           }},
          {"benchmark_info_get_decode_token_count_at",
           [&] {
             return litert_lm_benchmark_info_get_decode_token_count_at(
                 nullptr, 0, &out_int);
           }},
          {"benchmark_info_get_prefill_tokens_per_sec_at",
           [&] {
             return litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
                 nullptr, 0, &out_double);
           }},
          {"benchmark_info_get_decode_tokens_per_sec_at",
           [&] {
             return litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
                 nullptr, 0, &out_double);
           }},
          {"stream_chunk_get_text",
           [&] { return litert_lm_stream_chunk_get_text(nullptr, &out_text); }},
          {"stream_chunk_is_final",
           [&] { return litert_lm_stream_chunk_is_final(nullptr, &out_bool); }},
          {"stream_chunk_get_error",
           [&] {
             return litert_lm_stream_chunk_get_error(nullptr, &out_text);
           }},
          {"tokenize_result_get_tokens",
           [&] {
             return litert_lm_tokenize_result_get_tokens(nullptr, &out_ints);
           }},
          {"tokenize_result_get_num_tokens",
           [&] {
             return litert_lm_tokenize_result_get_num_tokens(nullptr,
                                                             &out_size);
           }},
          {"detokenize_result_get_string",
           [&] {
             return litert_lm_detokenize_result_get_string(nullptr, &out_text);
           }},
          {"token_union_get_type",
           [&] { return litert_lm_token_union_get_type(nullptr, &out_type); }},
          {"token_union_get_string",
           [&] {
             return litert_lm_token_union_get_string(nullptr, &out_text);
           }},
          {"token_union_get_ids",
           [&] {
             return litert_lm_token_union_get_ids(nullptr, &out_union_ids,
                                                  &out_size);
           }},
          {"token_unions_get_num_tokens",
           [&] {
             return litert_lm_token_unions_get_num_tokens(nullptr, &out_size);
           }},
      },
      kLiteRtLmStatusInvalidArgument, "must not be NULL");
  EXPECT_EQ(out_int, kIntSentinel);
  EXPECT_TRUE(out_bool);
  EXPECT_EQ(out_float, kFloatSentinel);
  EXPECT_EQ(out_double, kDoubleSentinel);
  EXPECT_EQ(out_size, kSizeSentinel);
  EXPECT_EQ(out_type, kLiteRtLmTokenUnionTypeString);
  EXPECT_EQ(out_text, nullptr);
  EXPECT_EQ(out_floats, nullptr);
  EXPECT_EQ(out_ints, nullptr);
  EXPECT_EQ(out_union_ids, nullptr);
}

// Returns responses with two candidates. Candidate 0 has a score, a token
// length and token scores; candidate 1 has none of them.
LiteRtLmResponses MakeTwoCandidateResponses() {
  LiteRtLmResponses responses{litert::lm::Responses(
      litert::lm::TaskState::kDone, {"first", "second"}, {0.5f}, {3})};
  responses.responses.GetMutableTokenScores() =
      std::vector<std::vector<float>>{{0.25f, 0.75f}};
  return responses;
}

TEST(EngineCErrorTest, ResponsesAccessorsReturnValuesAtValidIndex) {
  const LiteRtLmResponses responses = MakeTwoCandidateResponses();
  EXPECT_EQ(GetOk<int>(litert_lm_responses_get_num_candidates, &responses), 2);
  EXPECT_STREQ(GetOk<const char*>(litert_lm_responses_get_response_text_at,
                                  &responses, 0),
               "first");
  EXPECT_STREQ(GetOk<const char*>(litert_lm_responses_get_response_text_at,
                                  &responses, 1),
               "second");
  EXPECT_TRUE(GetOk<bool>(litert_lm_responses_has_score_at, &responses, 0));
  EXPECT_FALSE(GetOk<bool>(litert_lm_responses_has_score_at, &responses, 1));
  EXPECT_EQ(GetOk<float>(litert_lm_responses_get_score_at, &responses, 0),
            0.5f);
  EXPECT_TRUE(
      GetOk<bool>(litert_lm_responses_has_token_length_at, &responses, 0));
  EXPECT_FALSE(
      GetOk<bool>(litert_lm_responses_has_token_length_at, &responses, 1));
  EXPECT_EQ(GetOk<int>(litert_lm_responses_get_token_length_at, &responses, 0),
            3);
  EXPECT_TRUE(
      GetOk<bool>(litert_lm_responses_has_token_scores_at, &responses, 0));
  EXPECT_FALSE(
      GetOk<bool>(litert_lm_responses_has_token_scores_at, &responses, 1));
  ASSERT_EQ(
      GetOk<int>(litert_lm_responses_get_num_token_scores_at, &responses, 0),
      2);
  const float* token_scores = GetOk<const float*>(
      litert_lm_responses_get_token_scores_at, &responses, 0);
  ASSERT_NE(token_scores, nullptr);
  EXPECT_EQ(token_scores[0], 0.25f);
  EXPECT_EQ(token_scores[1], 0.75f);
}

TEST(EngineCErrorTest, ResponsesAccessorsWithIndexOutOfRangeReturnOutOfRange) {
  const LiteRtLmResponses responses = MakeTwoCandidateResponses();
  for (const int index : {-1, 2}) {
    SCOPED_TRACE(index);
    bool out_bool = true;
    int out_int = -7;
    float out_float = -7.0f;
    const char* out_text = "sentinel";
    const float* out_floats = &out_float;
    ExpectAllReturn(
        {
            {"responses_get_response_text_at",
             [&] {
               return litert_lm_responses_get_response_text_at(
                   &responses, index, &out_text);
             }},
            {"responses_has_score_at",
             [&] {
               return litert_lm_responses_has_score_at(&responses, index,
                                                       &out_bool);
             }},
            {"responses_get_score_at",
             [&] {
               return litert_lm_responses_get_score_at(&responses, index,
                                                       &out_float);
             }},
            {"responses_has_token_length_at",
             [&] {
               return litert_lm_responses_has_token_length_at(&responses, index,
                                                              &out_bool);
             }},
            {"responses_get_token_length_at",
             [&] {
               return litert_lm_responses_get_token_length_at(&responses, index,
                                                              &out_int);
             }},
            {"responses_has_token_scores_at",
             [&] {
               return litert_lm_responses_has_token_scores_at(&responses, index,
                                                              &out_bool);
             }},
            {"responses_get_num_token_scores_at",
             [&] {
               return litert_lm_responses_get_num_token_scores_at(
                   &responses, index, &out_int);
             }},
            {"responses_get_token_scores_at",
             [&] {
               return litert_lm_responses_get_token_scores_at(&responses, index,
                                                              &out_floats);
             }},
        },
        kLiteRtLmStatusOutOfRange, "out of range");
    EXPECT_TRUE(out_bool);
    EXPECT_EQ(out_int, -7);
    EXPECT_EQ(out_float, -7.0f);
    EXPECT_EQ(out_text, nullptr);
    EXPECT_EQ(out_floats, nullptr);
  }
}

TEST(EngineCErrorTest, ResponsesAccessorsWithAbsentValueReturnNotFound) {
  const LiteRtLmResponses responses = MakeTwoCandidateResponses();
  int out_int = -7;
  float out_float = -7.0f;
  const float* out_floats = &out_float;
  ExpectAllReturn(
      {
          {"responses_get_score_at",
           [&] {
             return litert_lm_responses_get_score_at(&responses, 1, &out_float);
           }},
          {"responses_get_token_length_at",
           [&] {
             return litert_lm_responses_get_token_length_at(&responses, 1,
                                                            &out_int);
           }},
          {"responses_get_num_token_scores_at",
           [&] {
             return litert_lm_responses_get_num_token_scores_at(&responses, 1,
                                                                &out_int);
           }},
          {"responses_get_token_scores_at",
           [&] {
             return litert_lm_responses_get_token_scores_at(&responses, 1,
                                                            &out_floats);
           }},
      },
      kLiteRtLmStatusNotFound, "at response index 1");
  EXPECT_EQ(out_int, -7);
  EXPECT_EQ(out_float, -7.0f);
  EXPECT_EQ(out_floats, nullptr);

  // Scoring-only responses have candidates but no response texts.
  const LiteRtLmResponses scores_only{litert::lm::Responses(
      litert::lm::TaskState::kDone, /*response_texts=*/{}, {0.1f, 0.2f})};
  EXPECT_EQ(GetOk<int>(litert_lm_responses_get_num_candidates, &scores_only),
            2);
  const char* out_text = "sentinel";
  ExpectAllReturn({{"responses_get_response_text_at",
                    [&] {
                      return litert_lm_responses_get_response_text_at(
                          &scores_only, 0, &out_text);
                    }}},
                  kLiteRtLmStatusNotFound, "No response text");
  EXPECT_EQ(out_text, nullptr);
}

TEST(EngineCErrorTest, StreamChunkAccessorsReturnAbsentValuesAsNull) {
  LiteRtLmStreamChunk final_chunk;
  final_chunk.is_final = true;
  // Absent text / error are successes with a NULL result, even if the
  // out-parameter held a stale value.
  const char* text = "sentinel";
  EXPECT_EQ(litert_lm_stream_chunk_get_text(&final_chunk, &text),
            kLiteRtLmStatusOk);
  EXPECT_EQ(text, nullptr);
  const char* error = "sentinel";
  EXPECT_EQ(litert_lm_stream_chunk_get_error(&final_chunk, &error),
            kLiteRtLmStatusOk);
  EXPECT_EQ(error, nullptr);
  EXPECT_TRUE(GetOk<bool>(litert_lm_stream_chunk_is_final, &final_chunk));

  LiteRtLmStreamChunk chunk;
  chunk.text = "hello";
  chunk.error_msg = "boom";
  EXPECT_STREQ(GetOk<const char*>(litert_lm_stream_chunk_get_text, &chunk),
               "hello");
  EXPECT_STREQ(GetOk<const char*>(litert_lm_stream_chunk_get_error, &chunk),
               "boom");
  EXPECT_FALSE(GetOk<bool>(litert_lm_stream_chunk_is_final, &chunk));
}

TEST(EngineCErrorTest, TokenResultAccessors) {
  LiteRtLmTokenizeResult tokenize_result{{5, 6, 7}};
  EXPECT_EQ(
      GetOk<size_t>(litert_lm_tokenize_result_get_num_tokens, &tokenize_result),
      3);
  const int* tokens =
      GetOk<const int*>(litert_lm_tokenize_result_get_tokens, &tokenize_result);
  ASSERT_NE(tokens, nullptr);
  EXPECT_EQ(tokens[2], 7);

  LiteRtLmDetokenizeResult detokenize_result{"text"};
  EXPECT_STREQ(GetOk<const char*>(litert_lm_detokenize_result_get_string,
                                  &detokenize_result),
               "text");

  LiteRtLmTokenUnions token_unions;
  token_unions.tokens.emplace_back().set_token_str("<eos>");
  token_unions.tokens.emplace_back().mutable_token_ids()->add_ids(1);
  EXPECT_EQ(GetOk<size_t>(litert_lm_token_unions_get_num_tokens, &token_unions),
            2);
}

TEST(EngineCErrorTest, TokenUnionAccessorsRejectWrongVariant) {
  LiteRtLmTokenUnion string_union;
  string_union.token_union.set_token_str("<eos>");
  EXPECT_EQ(GetOk<LiteRtLmTokenUnionType>(litert_lm_token_union_get_type,
                                          &string_union),
            kLiteRtLmTokenUnionTypeString);
  EXPECT_STREQ(
      GetOk<const char*>(litert_lm_token_union_get_string, &string_union),
      "<eos>");

  LiteRtLmTokenUnion ids_union;
  ids_union.token_union.mutable_token_ids()->add_ids(1);
  EXPECT_EQ(
      GetOk<LiteRtLmTokenUnionType>(litert_lm_token_union_get_type, &ids_union),
      kLiteRtLmTokenUnionTypeIds);
  constexpr int kIntSentinel = -7;
  const char* out_string = "sentinel";
  const int* out_ids = &kIntSentinel;
  size_t out_num_ids = 77;
  ExpectAllReturn(
      {
          {"token_union_get_string on ids",
           [&] {
             return litert_lm_token_union_get_string(&ids_union, &out_string);
           }},
          {"token_union_get_ids on string",
           [&] {
             return litert_lm_token_union_get_ids(&string_union, &out_ids,
                                                  &out_num_ids);
           }},
      },
      kLiteRtLmStatusInvalidArgument, "does not contain");
  EXPECT_EQ(out_string, nullptr);
  EXPECT_EQ(out_ids, nullptr);
  EXPECT_EQ(out_num_ids, 77);
}

TEST(EngineCErrorTest, ConversationNullArgumentsSetError) {
  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_conversation_config_set_system_message(nullptr, "{}"),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("config must not be NULL"));

  ConversationConfigPtr config(CreateConversationConfig(),
                               &litert_lm_conversation_config_delete);
  ASSERT_NE(config, nullptr);
  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_conversation_config_set_tools(config.get(), nullptr),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("tools_json must not be NULL"));

  litert_lm_clear_last_error();
  EXPECT_EQ(
      litert_lm_conversation_optional_args_set_max_output_tokens(nullptr, 1),
      kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("args must not be NULL"));

  litert_lm_clear_last_error();
  const char* json = "sentinel";
  EXPECT_EQ(litert_lm_json_response_get_string(nullptr, &json),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(json, nullptr);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("response must not be NULL"));

  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_conversation_cancel_process(nullptr),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("Invalid conversation"));
}

TEST(EngineCStatusTest, ConversationConfigSettersReturnStatus) {
  ConversationConfigPtr config(CreateConversationConfig(),
                               &litert_lm_conversation_config_delete);
  ASSERT_NE(config, nullptr);
  SessionConfigPtr session_config(CreateSessionConfig(),
                                  &litert_lm_session_config_delete);
  ASSERT_NE(session_config, nullptr);
  std::unique_ptr<LiteRtLmThinkingConfig,
                  decltype(&litert_lm_thinking_config_delete)>
      thinking_config(CreateThinkingConfig(),
                      &litert_lm_thinking_config_delete);
  ASSERT_NE(thinking_config, nullptr);
  const LiteRtLmSessionConfig* session = session_config.get();
  const LiteRtLmThinkingConfig* thinking = thinking_config.get();
  using C = LiteRtLmConversationConfig;
  ExpectSettersReturnStatus<C>(
      config.get(),
      {
          {"set_session_config",
           [session](C* c) {
             return litert_lm_conversation_config_set_session_config(c,
                                                                     session);
           }},
          {"set_system_message",
           [](C* c) {
             return litert_lm_conversation_config_set_system_message(c, "hi");
           }},
          {"set_tools",
           [](C* c) {
             return litert_lm_conversation_config_set_tools(c, "[]");
           }},
          {"set_messages",
           [](C* c) {
             return litert_lm_conversation_config_set_messages(c, "[]");
           }},
          {"set_extra_context",
           [](C* c) {
             return litert_lm_conversation_config_set_extra_context(c, "{}");
           }},
          {"set_prompt_template",
           [](C* c) {
             return litert_lm_conversation_config_set_prompt_template(c, "t");
           }},
          {"set_enable_constrained_decoding",
           [](C* c) {
             return litert_lm_conversation_config_set_enable_constrained_decoding(  // NOLINT
                 c, true);
           }},
          {"set_constraint_provider",
           [](C* c) {
             const LiteRtLmConstraintProviderType provider =
                 kLiteRtLmConstraintProviderTypeLlGuidance;
             return litert_lm_conversation_config_set_constraint_provider(
                 c, &provider);
           }},
          {"set_constraint_provider_unset",
           [](C* c) {
             return litert_lm_conversation_config_set_constraint_provider(
                 c, nullptr);
           }},
          {"set_filter_channel_content_from_kv_cache",
           [](C* c) {
             return litert_lm_conversation_config_set_filter_channel_content_from_kv_cache(  // NOLINT
                 c, true);
           }},
          {"set_stream_tool_calls",
           [](C* c) {
             return litert_lm_conversation_config_set_stream_tool_calls(c, true,
                                                                        "tool");
           }},
          {"set_thinking_config",
           [thinking](C* c) {
             return litert_lm_conversation_config_set_thinking_config(c,
                                                                      thinking);
           }},
      });

  using T = LiteRtLmThinkingConfig;
  ExpectSettersReturnStatus<T>(
      thinking_config.get(),
      {
          {"set_enable_thinking",
           [](T* t) {
             return litert_lm_thinking_config_set_enable_thinking(t, false);
           }},
          {"set_thinking_token_budget",
           [](T* t) {
             return litert_lm_thinking_config_set_thinking_token_budget(t, 16);
           }},
      });
}

TEST(EngineCStatusTest, ConversationOptionalArgsSettersReturnStatus) {
  OptionalArgsPtr args(CreateConversationOptionalArgs(),
                       &litert_lm_conversation_optional_args_delete);
  ASSERT_NE(args, nullptr);
  RepetitionPenaltyConfigPtr repetition_config(
      CreateRepetitionPenaltyConfig(),
      &litert_lm_repetition_penalty_config_delete);
  ASSERT_NE(repetition_config, nullptr);
  const LiteRtLmRepetitionPenaltyConfig* repetition = repetition_config.get();
  using A = LiteRtLmConversationOptionalArgs;
  ExpectSettersReturnStatus<A>(
      args.get(),
      {
          {"set_repetition_penalty_config",
           [repetition](A* a) {
             return litert_lm_conversation_optional_args_set_repetition_penalty_config(  // NOLINT
                 a, repetition);
           }},
          {"set_no_repeat_ngram_config",
           [](A* a) {
             return litert_lm_conversation_optional_args_set_no_repeat_ngram_config(  // NOLINT
                 a, nullptr);
           }},
          {"set_suppress_tokens_config",
           [](A* a) {
             return litert_lm_conversation_optional_args_set_suppress_tokens_config(  // NOLINT
                 a, nullptr);
           }},
          {"set_visual_token_budget",
           [](A* a) {
             return litert_lm_conversation_optional_args_set_visual_token_budget(  // NOLINT
                 a, 70);
           }},
          {"set_max_output_tokens",
           [](A* a) {
             return litert_lm_conversation_optional_args_set_max_output_tokens(
                 a, 8);
           }},
          {"set_thinking_config",
           [](A* a) {
             return litert_lm_conversation_optional_args_set_thinking_config(
                 a, nullptr);
           }},
          {"set_constraint",
           [](A* a) {
             return litert_lm_conversation_optional_args_set_constraint(
                 a, kLiteRtLmConstraintTypeRegex, "a+");
           }},
      });
}

TEST(EngineCStatusTest, ConversationSettersRejectUnknownEnums) {
  ConversationConfigPtr config(CreateConversationConfig(),
                               &litert_lm_conversation_config_delete);
  ASSERT_NE(config, nullptr);
  litert_lm_clear_last_error();
  // 0 is within the enum's value range but is not a declared enumerator.
  const auto provider = static_cast<LiteRtLmConstraintProviderType>(0);
  EXPECT_EQ(litert_lm_conversation_config_set_constraint_provider(config.get(),
                                                                  &provider),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("Unknown LiteRtLmConstraintProviderType"));

  OptionalArgsPtr args(CreateConversationOptionalArgs(),
                       &litert_lm_conversation_optional_args_delete);
  ASSERT_NE(args, nullptr);
  litert_lm_clear_last_error();
  // 3 is within the enum's value range but is not a declared enumerator.
  EXPECT_EQ(litert_lm_conversation_optional_args_set_constraint(
                args.get(), static_cast<LiteRtLmConstraintType>(3), "a+"),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("Unknown LiteRtLmConstraintType"));
}

TEST(EngineCErrorTest, ExperimentalNullArgumentsSetError) {
  // A non-NULL sentinel, to check that pointer out-parameters are reset to
  // NULL on failure.
  auto* const kSentinelDebugInfo =
      reinterpret_cast<LiteRtLmSessionDebugInfo*>(0x1);
  const char kSentinelString[] = "sentinel";

  litert_lm_clear_last_error();
  LiteRtLmSessionDebugInfo* debug_info = kSentinelDebugInfo;
  int status = litert_lm_experimental_session_get_debug_info(
      /*session=*/nullptr, &debug_info);
  EXPECT_EQ(status, kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("Invalid session"));
  EXPECT_EQ(debug_info, nullptr);

  litert_lm_clear_last_error();
  debug_info = kSentinelDebugInfo;
  status = litert_lm_experimental_conversation_get_session_debug_info(
      /*conversation=*/nullptr, &debug_info);
  EXPECT_EQ(status, kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("Invalid conversation"));
  EXPECT_EQ(debug_info, nullptr);

  litert_lm_clear_last_error();
  const char* capture_dir = kSentinelString;
  status = litert_lm_experimental_session_debug_info_get_capture_dir(
      /*debug_info=*/nullptr, &capture_dir);
  EXPECT_EQ(status, kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("debug_info must not be NULL"));
  EXPECT_EQ(capture_dir, nullptr);
}

TEST(EngineCErrorTest, ExperimentalNullOutParamsReturnInvalidArgument) {
  LiteRtLmSessionDebugInfo debug_info;
  const std::vector<std::pair<std::string, std::function<int()>>> calls = {
      {"experimental_is_debugger_enabled",
       [] { return litert_lm_experimental_is_debugger_enabled(nullptr); }},
      {"experimental_session_get_debug_info",
       [] {
         return litert_lm_experimental_session_get_debug_info(nullptr, nullptr);
       }},
      {"experimental_conversation_get_session_debug_info",
       [] {
         return litert_lm_experimental_conversation_get_session_debug_info(
             nullptr, nullptr);
       }},
      {"experimental_session_debug_info_get_capture_dir",
       [&debug_info] {
         return litert_lm_experimental_session_debug_info_get_capture_dir(
             &debug_info, nullptr);
       }},
  };
  for (const auto& [name, call] : calls) {
    SCOPED_TRACE(name);
    litert_lm_clear_last_error();
    const int status = call();
    EXPECT_EQ(status, kLiteRtLmStatusInvalidArgument);
    EXPECT_THAT(litert_lm_get_last_error_message(),
                testing::HasSubstr("must not be NULL"));
  }
}

TEST(EngineCTest, ExperimentalSessionDebugInfoGetCaptureDir) {
  LiteRtLmSessionDebugInfo debug_info;
  debug_info.debug_info.capture_dir = "litert_lm_debugger/7";
  const char* capture_dir = nullptr;
  EXPECT_EQ(litert_lm_experimental_session_debug_info_get_capture_dir(
                &debug_info, &capture_dir),
            kLiteRtLmStatusOk);
  EXPECT_STREQ(capture_dir, "litert_lm_debugger/7");
}

TEST(EngineCTest, ExperimentalDebugInfo) {
  bool enabled = false;
  ASSERT_EQ(litert_lm_experimental_is_debugger_enabled(&enabled),
            kLiteRtLmStatusOk);

  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");
  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);
  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  auto* const kSentinelDebugInfo =
      reinterpret_cast<LiteRtLmSessionDebugInfo*>(0x1);

  // Some engines support only one session at a time, so query the session
  // before creating the conversation.
  SessionPtr session(CreateSession(engine.get(), /*config=*/nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);
  LiteRtLmSessionDebugInfo* session_debug_info = kSentinelDebugInfo;
  ASSERT_EQ(litert_lm_experimental_session_get_debug_info(session.get(),
                                                          &session_debug_info),
            kLiteRtLmStatusOk);
  EXPECT_NE(session_debug_info, kSentinelDebugInfo);
  if (!enabled) {
    // Debug info is absent when the debugger is disabled: success + NULL.
    EXPECT_EQ(session_debug_info, nullptr);
  }
  litert_lm_experimental_session_debug_info_delete(session_debug_info);
  session.reset();

  ConversationConfigPtr conversation_config(
      CreateConversationConfig(), &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);
  ConversationPtr conversation(
      CreateConversation(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);
  LiteRtLmSessionDebugInfo* conversation_debug_info = kSentinelDebugInfo;
  ASSERT_EQ(litert_lm_experimental_conversation_get_session_debug_info(
                conversation.get(), &conversation_debug_info),
            kLiteRtLmStatusOk);
  EXPECT_NE(conversation_debug_info, kSentinelDebugInfo);
  if (!enabled) {
    EXPECT_EQ(conversation_debug_info, nullptr);
  }
  litert_lm_experimental_session_debug_info_delete(conversation_debug_info);
}

// Asserts that `status` is `expected_code` and a thread-local last error
// message was recorded.
void ExpectCanonicalFailure(int status, int expected_code) {
  EXPECT_EQ(status, expected_code);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
}

TEST(EngineCStatusTest, StatusFunctionsWithNullArgumentsReturnInvalidArgument) {
  const std::vector<std::pair<std::string, std::function<int()>>> calls = {
      {"session_config_set_lora_path",
       [] { return litert_lm_session_config_set_lora_path(nullptr, "a"); }},
      {"session_config_set_audio_lora_path",
       [] {
         return litert_lm_session_config_set_audio_lora_path(nullptr, "a");
       }},
      {"engine_settings_set_supported_lora_ranks",
       [] {
         const int ranks[] = {8};
         return litert_lm_engine_settings_set_supported_lora_ranks(nullptr,
                                                                   ranks, 1);
       }},
      {"engine_settings_set_supported_audio_lora_ranks",
       [] {
         const int ranks[] = {8};
         return litert_lm_engine_settings_set_supported_audio_lora_ranks(
             nullptr, ranks, 1);
       }},
      {"session_save_checkpoint",
       [] { return litert_lm_session_save_checkpoint(nullptr, "a"); }},
      {"session_rewind_to_checkpoint",
       [] { return litert_lm_session_rewind_to_checkpoint(nullptr, "a"); }},
      {"session_rewind_to_step",
       [] { return litert_lm_session_rewind_to_step(nullptr, 0); }},
      {"session_run_prefill",
       [] { return litert_lm_session_run_prefill(nullptr, nullptr, 0); }},
      {"session_run_decode_async",
       [] {
         return litert_lm_session_run_decode_async(nullptr, &StreamCallback,
                                                   nullptr);
       }},
      {"session_generate_content_stream",
       [] {
         return litert_lm_session_generate_content_stream(
             nullptr, nullptr, 0, &StreamCallback, nullptr);
       }},
      {"token_union_get_ids",
       [] {
         const int* ids = nullptr;
         size_t num_ids = 0;
         return litert_lm_token_union_get_ids(nullptr, &ids, &num_ids);
       }},
      {"conversation_send_message_stream",
       [] {
         return litert_lm_conversation_send_message_stream(
             nullptr, "{}", /*extra_context=*/nullptr,
             /*optional_args=*/nullptr, &StreamCallback, nullptr);
       }},
      {"experimental_engine_update_gpu_enable_metal_residency_set",
       [] {
         return litert_lm_experimental_engine_update_gpu_enable_metal_residency_set(  // NOLINT
             nullptr, true);
       }},
  };
  for (const auto& [name, call] : calls) {
    SCOPED_TRACE(name);
    litert_lm_clear_last_error();
    ExpectCanonicalFailure(call(), kLiteRtLmStatusInvalidArgument);
  }
}

TEST(EngineCStatusTest, SessionConfigSetLoraPathMissingFileReturnsCode) {
  SessionConfigPtr config(CreateSessionConfig(),
                          &litert_lm_session_config_delete);
  ASSERT_NE(config, nullptr);

  litert_lm_clear_last_error();
  ExpectCanonicalFailure(
      litert_lm_session_config_set_lora_path(config.get(), ""),
      kLiteRtLmStatusInvalidArgument);

  litert_lm_clear_last_error();
  int status = litert_lm_session_config_set_lora_path(
      config.get(), "/nonexistent/litert_lm/lora.tflite");
  EXPECT_GT(status, kLiteRtLmStatusOk);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
}

TEST(EngineCStatusTest, SupportedAudioLoraRanksWithoutAudioFailsPrecondition) {
  EngineSettingsPtr settings(
      CreateEngineSettings("test_model_path_1", "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  const int ranks[] = {8};

  litert_lm_clear_last_error();
  ExpectCanonicalFailure(
      litert_lm_engine_settings_set_supported_audio_lora_ranks(settings.get(),
                                                               ranks, 1),
      kLiteRtLmStatusFailedPrecondition);

  litert_lm_clear_last_error();
  ExpectCanonicalFailure(litert_lm_engine_settings_set_supported_lora_ranks(
                             settings.get(), ranks, 0),
                         kLiteRtLmStatusInvalidArgument);
}

TEST(EngineCStatusTest, SessionRuntimeFailuresReturnCanonicalCode) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(CreateSession(engine.get(), /* session_config */ nullptr),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  // A checkpoint that was never saved cannot be rewound to. The exact code is
  // determined by the runtime; it must be a canonical failure code and record
  // a last error message.
  litert_lm_clear_last_error();
  int status = litert_lm_session_rewind_to_checkpoint(session.get(),
                                                      "no_such_checkpoint");
  EXPECT_GT(status, kLiteRtLmStatusOk);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);

  litert_lm_clear_last_error();
  ExpectCanonicalFailure(
      litert_lm_session_save_checkpoint(session.get(), nullptr),
      kLiteRtLmStatusInvalidArgument);

  litert_lm_clear_last_error();
  ExpectCanonicalFailure(
      litert_lm_session_run_prefill(session.get(), nullptr, 0),
      kLiteRtLmStatusInvalidArgument);

  litert_lm_clear_last_error();
  ExpectCanonicalFailure(
      litert_lm_session_run_decode_async(session.get(), nullptr, nullptr),
      kLiteRtLmStatusInvalidArgument);

  litert_lm_clear_last_error();
  LiteRtLmResponses* out_responses = reinterpret_cast<LiteRtLmResponses*>(0x1);
  ExpectCanonicalFailure(litert_lm_session_generate_content(
                             session.get(), nullptr, 1, &out_responses),
                         kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(out_responses, nullptr);

  litert_lm_clear_last_error();
  ExpectCanonicalFailure(litert_lm_session_generate_content_stream(
                             session.get(), nullptr, 0, nullptr, nullptr),
                         kLiteRtLmStatusInvalidArgument);

  litert_lm_clear_last_error();
  ExpectCanonicalFailure(
      litert_lm_session_generate_content_stream(session.get(), nullptr, 1,
                                                &StreamCallback, nullptr),
      kLiteRtLmStatusInvalidArgument);

  litert_lm_clear_last_error();
  const char* null_target[] = {nullptr};
  out_responses = reinterpret_cast<LiteRtLmResponses*>(0x1);
  ExpectCanonicalFailure(
      litert_lm_session_run_text_scoring(session.get(), null_target, 1, false,
                                         &out_responses),
      kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(out_responses, nullptr);

  TokenUnionPtr start_token(GetStartToken(engine.get()),
                            &litert_lm_token_union_delete);
  if (start_token != nullptr) {
    litert_lm_clear_last_error();
    ExpectCanonicalFailure(
        litert_lm_token_union_get_ids(start_token.get(), nullptr, nullptr),
        kLiteRtLmStatusInvalidArgument);
  }
}

TEST(EngineCErrorTest, ConversationCreatorsReturnHandleThroughOutParam) {
  LiteRtLmConversationConfig* config = nullptr;
  ASSERT_EQ(litert_lm_conversation_config_create(&config), kLiteRtLmStatusOk);
  EXPECT_NE(config, nullptr);
  litert_lm_conversation_config_delete(config);

  LiteRtLmThinkingConfig* thinking_config = nullptr;
  ASSERT_EQ(litert_lm_thinking_config_create(&thinking_config),
            kLiteRtLmStatusOk);
  EXPECT_NE(thinking_config, nullptr);
  litert_lm_thinking_config_delete(thinking_config);

  LiteRtLmConversationOptionalArgs* optional_args = nullptr;
  ASSERT_EQ(litert_lm_conversation_optional_args_create(&optional_args),
            kLiteRtLmStatusOk);
  EXPECT_NE(optional_args, nullptr);
  litert_lm_conversation_optional_args_delete(optional_args);
}

TEST(EngineCErrorTest, ConversationFunctionsWithNullOutParamReturnError) {
  LiteRtLmJsonResponse response;
  ExpectAllReturn(
      {
          {"conversation_config_create",
           [] { return litert_lm_conversation_config_create(nullptr); }},
          {"thinking_config_create",
           [] { return litert_lm_thinking_config_create(nullptr); }},
          {"conversation_optional_args_create",
           [] { return litert_lm_conversation_optional_args_create(nullptr); }},
          {"conversation_create",
           [] {
             return litert_lm_conversation_create(nullptr, nullptr, nullptr);
           }},
          {"conversation_clone",
           [] { return litert_lm_conversation_clone(nullptr, nullptr); }},
          {"conversation_send_message",
           [] {
             return litert_lm_conversation_send_message(
                 nullptr, "{}", nullptr, nullptr, /*out_response=*/nullptr);
           }},
          // A valid response still fails without an out-parameter.
          {"json_response_get_string",
           [&] {
             return litert_lm_json_response_get_string(&response, nullptr);
           }},
          {"conversation_render_message_to_string",
           [] {
             return litert_lm_conversation_render_message_to_string(
                 nullptr, "{}", /*out_text=*/nullptr);
           }},
          {"conversation_render_preface_to_string",
           [] {
             return litert_lm_conversation_render_preface_to_string(nullptr,
                                                                    nullptr);
           }},
          {"conversation_get_benchmark_info",
           [] {
             return litert_lm_conversation_get_benchmark_info(nullptr, nullptr);
           }},
          {"conversation_get_token_count",
           [] {
             return litert_lm_conversation_get_token_count(nullptr, nullptr);
           }},
      },
      kLiteRtLmStatusInvalidArgument, "must not be NULL");
}

TEST(EngineCErrorTest, ConversationFunctionsWithNullHandleResetOutParam) {
  ExpectInvalidArgumentResetsOut<LiteRtLmConversation>(
      "conversation_create", [](LiteRtLmConversation** out) {
        return litert_lm_conversation_create(nullptr, nullptr, out);
      });
  ExpectInvalidArgumentResetsOut<LiteRtLmConversation>(
      "conversation_clone", [](LiteRtLmConversation** out) {
        return litert_lm_conversation_clone(nullptr, out);
      });
  ExpectInvalidArgumentResetsOut<LiteRtLmJsonResponse>(
      "conversation_send_message", [](LiteRtLmJsonResponse** out) {
        return litert_lm_conversation_send_message(nullptr, "{}", nullptr,
                                                   nullptr, out);
      });
  ExpectInvalidArgumentResetsOut<const char>(
      "json_response_get_string", [](const char** out) {
        return litert_lm_json_response_get_string(nullptr, out);
      });
  ExpectInvalidArgumentResetsOut<const char>(
      "conversation_render_message_to_string", [](const char** out) {
        return litert_lm_conversation_render_message_to_string(nullptr, "{}",
                                                               out);
      });
  ExpectInvalidArgumentResetsOut<const char>(
      "conversation_render_preface_to_string", [](const char** out) {
        return litert_lm_conversation_render_preface_to_string(nullptr, out);
      });
  ExpectInvalidArgumentResetsOut<LiteRtLmBenchmarkInfo>(
      "conversation_get_benchmark_info", [](LiteRtLmBenchmarkInfo** out) {
        return litert_lm_conversation_get_benchmark_info(nullptr, out);
      });

  // Scalar out-parameters are not written on failure.
  litert_lm_clear_last_error();
  int count = 42;
  const int status = litert_lm_conversation_get_token_count(nullptr, &count);
  EXPECT_EQ(status, kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("Invalid conversation"));
  EXPECT_EQ(count, 42);
}

TEST(EngineCErrorTest, JsonResponseGetStringReturnsOwnedString) {
  LiteRtLmJsonResponse response;
  response.json_string = R"({"role":"model"})";
  const char* json = nullptr;
  ASSERT_EQ(litert_lm_json_response_get_string(&response, &json),
            kLiteRtLmStatusOk);
  EXPECT_EQ(json, response.json_string.c_str());
}

TEST(EngineCStatusTest, ConversationResultProducers) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      CreateEngineSettings(task_path.c_str(), "cpu",
                           /* vision_backend_str */ nullptr,
                           /* audio_backend_str */ nullptr),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(CreateEngine(settings.get()), &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  ConversationPtr conversation(
      CreateConversation(engine.get(), /*config=*/nullptr),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  // Token count succeeds and is delivered through the out-parameter.
  litert_lm_clear_last_error();
  int count = -1;
  const int count_status =
      litert_lm_conversation_get_token_count(conversation.get(), &count);
  if (count_status == kLiteRtLmStatusUnimplemented) {
    GTEST_SKIP() << "Token count is not supported by this engine.";
  }
  ASSERT_EQ(count_status, kLiteRtLmStatusOk);
  EXPECT_GE(count, 0);
  EXPECT_EQ(litert_lm_get_last_error_message(), nullptr);

  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  const char* rendered = RenderMessage(conversation.get(), message_json);
  ASSERT_NE(rendered, nullptr);
  EXPECT_GT(strlen(rendered), 0);

  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_conversation_send_message_stream(
                conversation.get(), message_json, nullptr, nullptr,
                /*callback=*/nullptr, nullptr),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              testing::HasSubstr("callback must not be NULL"));

  // Malformed message JSON is rejected and leaves the out-parameters NULL.
  const char* bad_json = "{not json";
  litert_lm_clear_last_error();
  LiteRtLmJsonResponse* response = reinterpret_cast<LiteRtLmJsonResponse*>(0x1);
  int status = litert_lm_conversation_send_message(conversation.get(), bad_json,
                                                   nullptr, nullptr, &response);
  EXPECT_EQ(status, kLiteRtLmStatusInvalidArgument);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_EQ(response, nullptr);

  litert_lm_clear_last_error();
  const char* text = "sentinel";
  status = litert_lm_conversation_render_message_to_string(conversation.get(),
                                                           bad_json, &text);
  EXPECT_EQ(status, kLiteRtLmStatusInvalidArgument);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_EQ(text, nullptr);

  // Benchmarking is not enabled for this engine. The exact code is determined
  // by the runtime; it must be a failure code and record a last error message.
  litert_lm_clear_last_error();
  LiteRtLmBenchmarkInfo* benchmark_info =
      reinterpret_cast<LiteRtLmBenchmarkInfo*>(0x1);
  status = litert_lm_conversation_get_benchmark_info(conversation.get(),
                                                     &benchmark_info);
  EXPECT_GT(status, kLiteRtLmStatusOk);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_EQ(benchmark_info, nullptr);
}

}  // namespace
