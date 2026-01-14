#include "c/engine.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/conversation/conversation.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"

struct LiteRtLmEngineSettings {
  std::unique_ptr<litert::lm::EngineSettings> settings;
};

struct LiteRtLmSessionConfig {
  std::unique_ptr<litert::lm::SessionConfig> config;
};

struct LiteRtLmConversationConfig {
  std::unique_ptr<litert::lm::ConversationConfig> config;
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
using ConversationPtr =
    std::unique_ptr<LiteRtLmConversation,
                    decltype(&litert_lm_conversation_delete)>;
using JsonResponsePtr =
    std::unique_ptr<LiteRtLmJsonResponse,
                    decltype(&litert_lm_json_response_delete)>;
using SessionConfigPtr =
    std::unique_ptr<LiteRtLmSessionConfig,
                    decltype(&litert_lm_session_config_delete)>;
using ConversationConfigPtr =
    std::unique_ptr<LiteRtLmConversationConfig,
                    decltype(&litert_lm_conversation_config_delete)>;

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

TEST(EngineCTest, CreateSessionConfigWithSamplerParams) {
  LiteRtLmSamplerParams sampler_params;
  sampler_params.type = kTopP;
  sampler_params.top_k = 10;
  sampler_params.top_p = 0.5f;
  sampler_params.temperature = 0.1f;
  sampler_params.seed = 1234;

  SessionConfigPtr config(litert_lm_session_config_create(),
                          &litert_lm_session_config_delete);
  ASSERT_NE(config, nullptr);
  litert_lm_session_config_set_sampler_params(config.get(), &sampler_params);

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
  LiteRtLmSamplerParams sampler_params;
  sampler_params.type = kTopP;
  sampler_params.top_k = 10;
  sampler_params.top_p = 0.5f;
  sampler_params.temperature = 0.1f;
  sampler_params.seed = 1234;
  SessionConfigPtr session_config(litert_lm_session_config_create(),
                                  &litert_lm_session_config_delete);
  ASSERT_NE(session_config, nullptr);
  litert_lm_session_config_set_sampler_params(session_config.get(),
                                              &sampler_params);

  // 3. Create a Conversation Config with the Engine Handle, Session Config
  // and System Message.
  const std::string system_message =
      R"({"type":"text","text":"You are a helpful assistant."})";
  ConversationConfigPtr conversation_config(
      litert_lm_conversation_config_create(
          engine.get(), session_config.get(), system_message.c_str(),
          /*tools_json=*/nullptr, /*enable_constrained_decoding=*/false),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);

  // 4. Test to see if the Conversation Config has the Sampler Params.
  const auto& params =
      conversation_config->config->GetSessionConfig().GetSamplerParams();
  EXPECT_EQ(params.k(), 10);
  EXPECT_FLOAT_EQ(params.p(), 0.5f);
  EXPECT_FLOAT_EQ(params.temperature(), 0.1f);
  EXPECT_EQ(params.seed(), 1234);

  // 5. Test to see if the Conversation Config has the correct System Message.
  const auto& preface = std::get<litert::lm::JsonPreface>(
      conversation_config->config->GetPreface());
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

  // 2. Create a Conversation Config with the Engine Handle and System Message.
  const std::string system_message =
      R"({"type":"text","text":"You are a helpful assistant."})";
  SessionConfigPtr session_config(litert_lm_session_config_create(),
                                  &litert_lm_session_config_delete);
  ASSERT_NE(session_config, nullptr);
  ConversationConfigPtr conversation_config(
      litert_lm_conversation_config_create(
          engine.get(), session_config.get(), system_message.c_str(),
          /*tools_json=*/nullptr, /*enable_constrained_decoding=*/false),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);

  // 3. Test to see if the Conversation Config has the correct System Message.
  const auto& preface = std::get<litert::lm::JsonPreface>(
      conversation_config->config->GetPreface());
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

  // 2. Create a Conversation Config with the Engine Handle and System Message.
  SessionConfigPtr session_config(litert_lm_session_config_create(),
                                  &litert_lm_session_config_delete);
  ASSERT_NE(session_config, nullptr);
  ConversationConfigPtr conversation_config(
      litert_lm_conversation_config_create(
          engine.get(),
          /*session_config=*/session_config.get(),
          /*system_message_json=*/nullptr,
          /*tools_json=*/nullptr,
          /*enable_constrained_decoding=*/false),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);

  // 4. Test to see if the Conversation Config has the correct System Message.
  const auto& preface = std::get<litert::lm::JsonPreface>(
      conversation_config->config->GetPreface());
  EXPECT_EQ(preface.messages, nullptr);
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
      litert_lm_conversation_config_create(
          engine.get(), /*session_config=*/nullptr,
          /*system_message_json=*/nullptr, tools_json.c_str(),
          /*enable_constrained_decoding=*/false),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);

  // 3. Test to see if the Conversation Config has the correct tools.
  const auto& preface = std::get<litert::lm::JsonPreface>(
      conversation_config->config->GetPreface());
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
      litert_lm_conversation_config_create(
          engine.get(), /*session_config=*/nullptr,
          /*system_message_json=*/nullptr, tools_json.c_str(),
          /*enable_constrained_decoding=*/false),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);

  // 3. Test to see if the Conversation Config has no tools.
  const auto& preface = std::get<litert::lm::JsonPreface>(
      conversation_config->config->GetPreface());
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
      litert_lm_conversation_config_create(
          engine.get(), /*session_config=*/nullptr,
          /*system_message_json=*/nullptr, tools_json.c_str(),
          /*enable_constrained_decoding=*/false),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);

  // 3. Test to see if the Conversation Config has empty tools.
  const auto& preface = std::get<litert::lm::JsonPreface>(
      conversation_config->config->GetPreface());
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
      litert_lm_conversation_config_create(
          engine.get(), /*session_config=*/nullptr,
          /*system_message_json=*/nullptr, tools_json.c_str(),
          /*enable_constrained_decoding=*/false),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);

  // 3. Test to see if the Conversation Config has no tools.
  const auto& preface = std::get<litert::lm::JsonPreface>(
      conversation_config->config->GetPreface());
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
  LiteRtLmSamplerParams sampler_params;
  sampler_params.type = kTopP;
  sampler_params.top_k = 10;
  sampler_params.top_p = 0.5f;
  sampler_params.temperature = 0.1f;
  sampler_params.seed = 1234;
  SessionConfigPtr session_config(litert_lm_session_config_create(),
                                  &litert_lm_session_config_delete);
  ASSERT_NE(session_config, nullptr);
  litert_lm_session_config_set_sampler_params(session_config.get(),
                                              &sampler_params);

  // 3. Create a Conversation Config with the Engine Handle and Session Config.
  ConversationConfigPtr conversation_config(
      litert_lm_conversation_config_create(
          engine.get(), session_config.get(), /*system_message_json=*/nullptr,
          /*tools_json=*/nullptr, /*enable_constrained_decoding=*/false),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);

  // 4. Test to see if the Conversation Config has the default Sampler Params.
  const auto& params =
      conversation_config->config->GetSessionConfig().GetSamplerParams();
  EXPECT_EQ(params.k(), 10);
  EXPECT_FLOAT_EQ(params.p(), 0.5f);
  EXPECT_FLOAT_EQ(params.temperature(), 0.1f);
  EXPECT_EQ(params.seed(), 1234);

  // 5. Test to see if the Conversation Config has the correct System Message.
  const auto& preface = std::get<litert::lm::JsonPreface>(
      conversation_config->config->GetPreface());
  EXPECT_EQ(preface.messages, nullptr);
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
  InputData input_data;
  input_data.type = kInputText;
  input_data.data = prompt;
  input_data.size = strlen(prompt);
  ResponsesPtr responses(
      litert_lm_session_generate_content(session.get(), &input_data, 1),
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
    InputData input_data;
    input_data.type = kInputText;
    input_data.data = prompt;
    input_data.size = strlen(prompt);
    ResponsesPtr responses(
        litert_lm_session_generate_content(session.get(), &input_data, 1),
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
    InputData input_data;
    input_data.type = kInputText;
    input_data.data = prompt;
    input_data.size = strlen(prompt);
    ResponsesPtr responses(
        litert_lm_session_generate_content(session.get(), &input_data, 1),
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
      litert_lm_conversation_send_message(conversation.get(), message_json),
      &litert_lm_json_response_delete);
  ASSERT_NE(response, nullptr);

  const char* response_str = litert_lm_json_response_get_string(response.get());
  ASSERT_NE(response_str, nullptr);
  EXPECT_GT(strlen(response_str), 0);
}

TEST(EngineCTest, ConversationSendMessageWithConfig) {
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
  LiteRtLmSamplerParams sampler_params;
  sampler_params.type = kTopP;
  sampler_params.top_k = 10;
  sampler_params.top_p = 0.5f;
  sampler_params.temperature = 0.1f;
  sampler_params.seed = 1234;
  SessionConfigPtr session_config(litert_lm_session_config_create(),
                                  &litert_lm_session_config_delete);
  ASSERT_NE(session_config, nullptr);
  litert_lm_session_config_set_sampler_params(session_config.get(),
                                              &sampler_params);

  // 3. Create a Conversation Config with the Engine Handle, Session Config
  // and System Message.
  const std::string system_message =
      R"({"type":"text","text":"You are a helpful assistant."})";
  ConversationConfigPtr conversation_config(
      litert_lm_conversation_config_create(
          engine.get(), session_config.get(), system_message.c_str(),
          /*tools_json=*/nullptr, /*enable_constrained_decoding=*/false),
      &litert_lm_conversation_config_delete);
  ASSERT_NE(conversation_config, nullptr);

  // 4. Create a Conversation with the Conversation Config.
  ConversationPtr conversation(
      litert_lm_conversation_create(engine.get(), conversation_config.get()),
      &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  // 5. Send a message to the conversation.
  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  JsonResponsePtr response(
      litert_lm_conversation_send_message(conversation.get(), message_json),
      &litert_lm_json_response_delete);
  ASSERT_NE(response, nullptr);

  const char* response_str = litert_lm_json_response_get_string(response.get());
  ASSERT_NE(response_str, nullptr);
  EXPECT_GT(strlen(response_str), 0);
}

struct StreamCallbackData {
  std::string response;
  absl::Notification done;
  absl::Status status;
};

void StreamCallback(void* callback_data, const char* chunk, bool is_final,
                    const char* error_msg) {
  auto* data = static_cast<StreamCallbackData*>(callback_data);
  if (error_msg) {
    data->status = absl::InternalError(error_msg);
  }
  if (chunk) {
    data->response.append(chunk);
  }
  if (is_final) {
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
  InputData input_data;
  input_data.type = kInputText;
  input_data.data = prompt;
  input_data.size = strlen(prompt);
  StreamCallbackData callback_data;
  int result = litert_lm_session_generate_content_stream(
      session.get(), &input_data, 1, &StreamCallback, &callback_data);
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
      conversation.get(), message_json, &StreamCallback, &callback_data);
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
      conversation.get(), message_json, &StreamCallback, &callback_data);
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
  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.string().c_str(), "cpu",
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
  InputData input_data;
  input_data.type = kInputText;
  input_data.data = prompt;
  input_data.size = strlen(prompt);
  ResponsesPtr responses(
      litert_lm_session_generate_content(session.get(), &input_data, 1),
      &litert_lm_responses_delete);
  ASSERT_NE(responses, nullptr);

  BenchmarkInfoPtr benchmark_info(
      litert_lm_session_get_benchmark_info(session.get()),
      &litert_lm_benchmark_info_delete);
  ASSERT_NE(benchmark_info, nullptr);

  EXPECT_GT(
      litert_lm_benchmark_info_get_time_to_first_token(benchmark_info.get()),
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
}  // namespace
