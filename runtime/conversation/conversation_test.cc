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

#include "runtime/conversation/conversation.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <ios>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/external_constraint_config.h"
#include "runtime/components/prompt_template.h"
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/conversation/io_types.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

absl::string_view kTestLlmPath =
    "litert_lm/runtime/testdata/test_lm.litertlm";

constexpr char kTestTokenizerPath[] =
    "litert_lm/runtime/components/testdata/gemma3_sentencepiece.model";

constexpr char kGemma3ToolsMultiPrefillTemplatePath[] =
    "litert_lm/runtime/components/testdata/"
    "google-gemma-3n-e2b-it-tools-multi-prefill.jinja";

constexpr char kGemma3TemplatePath[] =
    "litert_lm/runtime/components/testdata/google-gemma-3-1b-it.jinja";

constexpr absl::string_view kTestJinjaPromptTemplate = R"jinja(
{%- for message in messages -%}
  {{- '<start_of_turn>' + message.role + '\n' -}}
  {%- if message.content is string -%}
    {{- message.content + '<end_of_turn>\n' -}}
  {%- else -%}
    {{- message.content[0].text + '<end_of_turn>\n' -}}
  {%- endif -%}
{%- endfor -%}
)jinja";

std::string GetTestdataPath(absl::string_view file_path) {
  return absl::StrCat(::testing::SrcDir(), "/", file_path);
}

std::string ReadFile(absl::string_view path) {
  std::ifstream ifstr(std::string(path), std::ios::binary);
  std::stringstream contents;
  contents << ifstr.rdbuf();
  return contents.str();
}

class MockSession : public Engine::Session {
 public:
  MOCK_METHOD(absl::StatusOr<Responses>, GenerateContent,
              (const std::vector<InputData>& contents), (override));
  MOCK_METHOD(
      absl::Status, GenerateContentStream,
      (const std::vector<InputData>& contents,
       absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback),
      (override));
  MOCK_METHOD(
      absl::Status, GenerateContentStream,
      (const std::vector<InputData>& contents,
       absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
       const DecodeConfig& decode_config),
      (override));
  MOCK_METHOD(absl::StatusOr<Responses>, RunTextScoring,
              (const std::vector<absl::string_view>& target_text,
               bool store_token_lengths),
              (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Engine::Session::TaskController>>,
              RunTextScoringAsync,
              (const std::vector<absl::string_view>& target_text,
               absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
               bool store_token_lengths),
              (override));

  MOCK_METHOD(absl::Status, RunPrefill,
              (const std::vector<InputData>& contents), (override));
  MOCK_METHOD(
      absl::StatusOr<std::unique_ptr<Engine::Session::TaskController>>,
      RunPrefillAsync,
      (const std::vector<InputData>& contents,
       absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback),
      (override));
  MOCK_METHOD(absl::StatusOr<Responses>, RunDecode, (), (override));
  MOCK_METHOD(absl::StatusOr<Responses>, RunDecode,
              (const DecodeConfig& decode_config), (override));
  MOCK_METHOD(
      absl::StatusOr<std::unique_ptr<Engine::Session::TaskController>>,
      RunDecodeAsync,
      (absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback),
      (override));
  MOCK_METHOD(
      absl::StatusOr<std::unique_ptr<Engine::Session::TaskController>>,
      RunDecodeAsync,
      (absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
       const DecodeConfig& decode_config),
      (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Session>>, Clone, (), (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Session>>, CloneAsync,
              (absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback),
              (override));
  MOCK_METHOD(absl::StatusOr<BenchmarkInfo>, GetBenchmarkInfo, (), (override));
  MOCK_METHOD(absl::StatusOr<BenchmarkInfo*>, GetMutableBenchmarkInfo, (),
              (override));
  MOCK_METHOD(void, CancelProcess, (), (override));
  MOCK_METHOD(absl::Status, WaitUntilDone, (), (override));
  MOCK_METHOD(const SessionConfig&, GetSessionConfig, (), (const, override));
};

class MockEngine : public Engine {
 public:
  MOCK_METHOD(const EngineSettings&, GetEngineSettings, (), (const, override));
  MOCK_METHOD(const Tokenizer&, GetTokenizer, (), (const, override));
  MOCK_METHOD(absl::StatusOr<AudioExecutorProperties>,
              GetAudioExecutorProperties, (), (const, override));
  MOCK_METHOD(absl::StatusOr<VisionExecutorProperties>,
              GetVisionExecutorProperties, (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Session>>, CreateSession,
              (const SessionConfig& session_config), (override));
  MOCK_METHOD(absl::Status, WaitUntilDone, (absl::Duration timeout),
              (override));
};

class MockTaskController : public Engine::Session::TaskController {
 public:
  MockTaskController() = default;
  ~MockTaskController() override = default;
  MOCK_METHOD(absl::Status, Cancel, (), (override));
};

absl::AnyInvocable<void(absl::StatusOr<Message>)> CreateTestMessageCallback(
    Message& expected_message, absl::Notification& done) {
  return [&expected_message, &done](absl::StatusOr<Message> message) mutable {
    // If the message is not ok, fail the test.
    if (!message.ok()) {
      FAIL() << "Message user_callback failed: " << message.status();
      return;
    }
    // If the message is null, the last callback is received.
    if (message->is_null()) {
      ASSERT_TRUE(expected_message["content"][0]["text"].is_string());
      std::string expected_string = expected_message["content"][0]["text"];
      // The expected string should be empty after the last callback.
      EXPECT_TRUE(expected_string.empty());
      done.Notify();
      return;
    }
    // Otherwise, this is a partial response.
    // Compare the message text content by prefix, and update the expected
    // message to the remaining text for the next user_callback.
    ASSERT_TRUE(expected_message["content"][0]["text"].is_string());
    ASSERT_TRUE((*message)["content"][0]["text"].is_string());
    std::string expected_string = expected_message["content"][0]["text"];
    std::string actual_string = (*message)["content"][0]["text"];
    EXPECT_TRUE(absl::StartsWith(expected_string, actual_string))
        << "Expected: " << expected_string << "\nActual: " << actual_string;
    expected_message["content"][0]["text"] =
        expected_string.substr(actual_string.size());
  };
}

absl::AnyInvocable<void(absl::StatusOr<Message>)>
CreateTestMultiMessageCallback(const std::vector<Message>& expected_messages,
                               absl::Notification& done) {
  return [&expected_messages, &done,
          current_index = 0](absl::StatusOr<Message> message) mutable {
    ASSERT_OK(message);

    // If the message is null, the message stream is complete.
    if (message->is_null()) {
      EXPECT_TRUE(current_index == expected_messages.size())
          << "Expected " << expected_messages.size()
          << " messages but only got " << current_index;
      done.Notify();
      return;
    }

    ASSERT_LT(current_index, expected_messages.size())
        << "Received more messages than expected. Expected size: "
        << expected_messages.size();
    EXPECT_THAT(*message, testing::Eq(expected_messages[current_index]));
    ++current_index;
  };
}

TEST(ConversationConfigTest, CreateDefault) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));
  ASSERT_OK_AND_ASSIGN(auto config, ConversationConfig::CreateDefault(*engine));
  EXPECT_OK(Conversation::Create(*engine, config));
}

TEST(ConversationConfigTest, CreateDefaultWithOverwritePromptTemplate) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));
  ASSERT_OK_AND_ASSIGN(auto config, ConversationConfig::Builder()
                                        .SetOverwritePromptTemplate(
                                            PromptTemplate("Hello world!"))
                                        .Build(*engine));
  EXPECT_EQ(config.GetPromptTemplate().GetTemplateSource(), "Hello world!");
  EXPECT_TRUE(
      config.GetSessionConfig().GetPromptTemplates().user().prefix().empty());
}

TEST(ConversationConfigTest, CreateWithBuilder) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));

  auto session_config = SessionConfig::CreateDefault();
  session_config.GetMutableLlmModelType().mutable_gemma3n();

  ASSERT_OK_AND_ASSIGN(
      auto config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
          .SetPreface(JsonPreface{
              .messages = {{{"role", "system"},
                            {"content", "You are a helpful assistant."}}}})
          .Build(*engine));
  EXPECT_TRUE(std::holds_alternative<JsonPreface>(config.GetPreface()));
  EXPECT_EQ(
      std::get<JsonPreface>(config.GetPreface()).messages,
      nlohmann::ordered_json(
          {{{"role", "system"}, {"content", "You are a helpful assistant."}}}));
  EXPECT_EQ(config.GetSessionConfig().GetLlmModelType().model_type_case(),
            proto::LlmModelType::kGemma3N);
  EXPECT_TRUE(
      config.GetSessionConfig().GetPromptTemplates().user().prefix().empty());
  EXPECT_OK(Conversation::Create(*engine, config));
}

TEST(ConversationConfigTest, OverwritePromptTemplate) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);

  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));
  ASSERT_OK_AND_ASSIGN(
      auto config,
      ConversationConfig::Builder()
          .SetOverwritePromptTemplate(PromptTemplate("overwrite template"))
          .Build(*engine));

  EXPECT_EQ(config.GetPromptTemplate().GetTemplateSource(),
            "overwrite template");
}

struct ConversationTestParams {
  bool enable_constrained_decoding;
  bool prefill_preface_on_init;
};

class ConversationTest : public testing::TestWithParam<ConversationTestParams> {
 public:
  static std::vector<ConversationTestParams> GetTestParams() {
    std::vector<ConversationTestParams> params;
    for (bool enable_constrained_decoding : {true, false}) {
      for (bool prefill_preface_on_init : {true, false}) {
        params.push_back(
            {enable_constrained_decoding, prefill_preface_on_init});
      }
    }
    return params;
  }

 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(
        tokenizer_,
        SentencePieceTokenizer::CreateFromFile(
            (std::filesystem::path(::testing::SrcDir()) / kTestTokenizerPath)
                .string()));
    model_assets_ = ModelAssets::Create(GetTestdataPath(kTestLlmPath));
    ASSERT_OK(model_assets_);
    engine_settings_ =
        EngineSettings::CreateDefault(*model_assets_, Backend::CPU);
    ASSERT_OK(engine_settings_);

    session_config_ = SessionConfig::CreateDefault();
    session_config_.SetStartTokenId(0);
    session_config_.GetMutableStopTokenIds().push_back({1});
    *session_config_.GetMutableLlmModelType().mutable_gemma3() = {};
  }

  std::unique_ptr<MockSession> CreateMockSession() {
    auto mock_session = std::make_unique<MockSession>();
    EXPECT_CALL(*mock_session, GetSessionConfig())
        .WillRepeatedly(testing::ReturnRef(session_config_));
    return mock_session;
  }

  std::unique_ptr<MockEngine> CreateMockEngine(
      std::unique_ptr<MockSession> mock_session) {
    auto mock_engine = std::make_unique<MockEngine>();
    EXPECT_CALL(*mock_engine, GetEngineSettings())
        .WillRepeatedly(testing::ReturnRef(*engine_settings_));
    EXPECT_CALL(*mock_engine, CreateSession(testing::_))
        .WillOnce(testing::Return(std::move(mock_session)));
    EXPECT_CALL(*mock_engine, GetTokenizer())
        .WillRepeatedly(testing::ReturnRef(*tokenizer_));
    return mock_engine;
  }

  std::unique_ptr<Tokenizer> tokenizer_;
  absl::StatusOr<ModelAssets> model_assets_;
  absl::StatusOr<EngineSettings> engine_settings_;
  SessionConfig session_config_ = SessionConfig::CreateDefault();
  bool enable_constrained_decoding_ = GetParam().enable_constrained_decoding;
  bool prefill_preface_on_init_ = GetParam().prefill_preface_on_init;
};

TEST_P(ConversationTest, SendMessage) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));

  ASSERT_OK_AND_ASSIGN(
      auto config,
      ConversationConfig::Builder()
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .Build(*engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*engine, config));
  EXPECT_THAT(conversation->GetHistory(), testing::IsEmpty());
  Message user_message = {{"role", "user"}, {"content", "Hello world!"}};
  ASSERT_OK_AND_ASSIGN(const Message message,
                       conversation->SendMessage(user_message));
  // The expected message is just some gibberish text, because the test LLM has
  // random weights.
  Message expected_message = {
      {"role", "assistant"},
      {"content",
       {{{"type", "text"}, {"text", "TarefaByte دارایेत्र investigaciónప్రదేశ"}}}}};
  EXPECT_EQ(message, expected_message);
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_message, expected_message));
}

TEST_P(ConversationTest, SendMessageGemma3Template) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(20);
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));

  std::string gemma3_prompt_template =
      ReadFile(GetTestdataPath(kGemma3TemplatePath));

  ASSERT_OK_AND_ASSIGN(
      auto config,
      ConversationConfig::Builder()
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .SetOverwritePromptTemplate(PromptTemplate(gemma3_prompt_template))
          .Build(*engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*engine, config));
  EXPECT_THAT(conversation->GetHistory(), testing::IsEmpty());
  Message user_message = {{"role", "user"}, {"content", "Hello world!"}};
  EXPECT_OK(conversation->SendMessage(user_message));
}

TEST_P(ConversationTest, SendSingleMessage) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send a single message.
  Message user_message = {{"role", "user"}, {"content", "How are you?"}};

  absl::string_view expected_input_text =
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_input_text)))))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));

  ASSERT_OK_AND_ASSIGN(const Message response,
                       conversation->SendMessage(user_message));

  Message assistant_message(nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })"));
  EXPECT_EQ(response, assistant_message);
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_message, assistant_message));
}

TEST_P(ConversationTest, SendSingleMessageWithExtraContext) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation and overwrite prompt template.
  absl::string_view prompt_template = R"jinja(
{%- if enable_thinking -%}
<start_of_turn>system
Thinking enabled.<end_of_turn>
{% else %}
<start_of_turn>system
Thinking disabled.<end_of_turn>
{%- endif -%}
{%- for message in messages -%}
  {{- '<start_of_turn>' + message.role + '\n' -}}
  {%- if message.content is string -%}
    {{- message.content + '<end_of_turn>\n' -}}
  {%- else -%}
    {{- message.content[0].text + '<end_of_turn>\n' -}}
  {%- endif -%}
{%- endfor -%}
)jinja";

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(prompt_template))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send a single message.
  Message user_message = {{"role", "user"}, {"content", "How are you?"}};
  OptionalArgs optional_args;
  optional_args.extra_context = absl::flat_hash_map<std::string, std::string>{
      {"enable_thinking", "true"}};

  absl::string_view expected_input_text =
      "<start_of_turn>system\nThinking enabled.<end_of_turn>\n"
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";

  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_input_text)))))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));

  ASSERT_OK_AND_ASSIGN(
      const Message response,
      conversation->SendMessage(user_message, std::move(optional_args)));

  Message assistant_message(nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })"));
  EXPECT_EQ(response, assistant_message);
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_message, assistant_message));
}

TEST_P(ConversationTest, SendSingleMessageWithExtraContextOverwritingPreface) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation and overwrite prompt template.
  absl::string_view prompt_template = R"jinja(
{%- if key1 -%}
Key1: {{ key1 + "\n"}}
{%- endif -%}
{%- if key2 -%}
Key2: {{ key2 + "\n"}}
{%- endif -%}
{%- if key3 -%}
Key3: {{ key3 + "\n"}}
{%- endif -%}
{%- for message in messages -%}
  {{- '<start_of_turn>' + message.role + '\n' -}}
  {%- if message.content is string -%}
    {{- message.content + '<end_of_turn>\n' -}}
  {%- else -%}
    {{- message.content[0].text + '<end_of_turn>\n' -}}
  {%- endif -%}
{%- endfor -%}
)jinja";

  JsonPreface preface;

  // This extra context will be set at the Conversation level.
  preface.extra_context = {{"key1", "val1"}, {"key2", "val2"}};

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetPreface(preface)
          .SetOverwritePromptTemplate(PromptTemplate(prompt_template))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send a single message with extra context that overwrites key1 and
  // adds key3.
  Message user_message = {{"role", "user"}, {"content", "How are you?"}};
  OptionalArgs optional_args;
  optional_args.extra_context =
      nlohmann::ordered_json{{"key1", "val1_new"}, {"key3", "val3"}};

  // key1 should be overwritten to val1_new.
  // key2 should remain val2.
  // key3 should be added as val3.
  absl::string_view expected_input_text =
      "Key1: val1_new\n"
      "Key2: val2\n"
      "Key3: val3\n"
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";

  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_input_text)))))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));

  ASSERT_OK_AND_ASSIGN(
      const Message response,
      conversation->SendMessage(user_message, std::move(optional_args)));

  Message assistant_message(nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })"));
  EXPECT_EQ(response, assistant_message);
}

TEST_P(ConversationTest, SendMultipleMessages) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send two consecutive messages.
  Message user_messages(nlohmann::ordered_json::parse(R"json(
    [
      {
        "role": "user",
        "content": "Hello world!"
      },
      {
        "role": "user",
        "content": "How are you?"
      }
    ]
  )json"));

  absl::string_view expected_input_text =
      "<start_of_turn>user\n"
      "Hello world!<end_of_turn>\n"
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_input_text)))))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));

  ASSERT_OK_AND_ASSIGN(const Message response,
                       conversation->SendMessage(user_messages));

  Message assistant_message(nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })"));
  EXPECT_EQ(response, assistant_message);
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_messages[0], user_messages[1],
                                   assistant_message));
}

TEST_P(ConversationTest, SendSingleMessageWithChannel) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .SetChannels({litert::lm::Channel{
              .channel_name = "thought",
              .start = "<|channel>thought\n",
              .end = "<channel|>",
          }})
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // Send a single message.
  Message user_message = {{"role", "user"}, {"content", "How are you?"}};

  absl::string_view expected_input_text =
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";

  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_input_text)))))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(testing::Return(
          Responses(TaskState::kProcessing,
                    {"<|channel>thought\nhmm<channel|>I am good."})));

  // Send the message.
  ASSERT_OK_AND_ASSIGN(const Message response,
                       conversation->SendMessage(user_message));

  Message assistant_message(nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ],
    "channels": {
      "thought": "hmm"
    }
  })"));
  EXPECT_THAT(response, testing::Eq(assistant_message));
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_message, assistant_message));
}

TEST_P(ConversationTest, SendSingleMessageWithChannelQwenThink) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .SetChannels({litert::lm::Channel{
              .channel_name = "thought",
              .start = "<think>\n",
              .end = "\n</think>",
          }})
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // Send a single message.
  Message user_message = {{"role", "user"}, {"content", "How are you?"}};

  absl::string_view expected_input_text =
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";

  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_input_text)))))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(testing::Return(Responses(
          TaskState::kProcessing, {"<think>\nhmm\n</think>I am good."})));

  // Send the message.
  ASSERT_OK_AND_ASSIGN(const Message response,
                       conversation->SendMessage(user_message));

  Message assistant_message(nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ],
    "channels": {
      "thought": "hmm"
    }
  })"));
  EXPECT_THAT(response, testing::Eq(assistant_message));
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_message, assistant_message));
}

TEST_P(ConversationTest, SendMultipleMessagesWithHistory) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // The first user message.
  Message user_message_1(nlohmann::ordered_json::parse(R"json(
    {
      "role": "user",
      "content": "How are you?"
    }
  )json"));
  EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
      .WillOnce(testing::Return(absl::OkStatus()));

  // The first assistant response.
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));

  // Send the first user message to fill the history.
  ASSERT_OK(conversation->SendMessage(user_message_1));
  ASSERT_THAT(conversation->GetHistory().size(), testing::Eq(2));

  // We will send two consecutive messages when the history is not empty.
  Message user_messages(nlohmann::ordered_json::parse(R"json(
    [
      {
        "role": "user",
        "content": "foo"
      },
      {
        "role": "user",
        "content": "bar"
      }
    ]
  )json"));
  absl::string_view expected_input_text =
      "<start_of_turn>user\n"
      "foo<end_of_turn>\n"
      "<start_of_turn>user\n"
      "bar<end_of_turn>\n";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_input_text)))))
      .WillOnce(testing::Return(absl::OkStatus()));

  // The second assistant response.
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(testing::Return(Responses(TaskState::kProcessing, {"baz"})));

  // Send the user messages.
  ASSERT_OK(conversation->SendMessage(user_messages));

  // Check the history.
  Message assistant_message_1(nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })"));
  Message assistant_message_2(nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "baz"
      }
    ]
  })"));
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_message_1, assistant_message_1,
                                   user_messages[0], user_messages[1],
                                   assistant_message_2));
}

TEST_P(ConversationTest, RunTextScoring) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // Test sync scoring.
  auto cloned_session_sync = std::make_unique<MockSession>();
  EXPECT_CALL(*cloned_session_sync,
              RunTextScoring(testing::ElementsAre("I am good."), true))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));
  EXPECT_CALL(*mock_session_ptr, Clone())
      .WillOnce(testing::Return(std::move(cloned_session_sync)));

  ASSERT_OK_AND_ASSIGN(const Responses response,
                       conversation->RunTextScoring({"I am good."}));
  EXPECT_EQ(response.GetTexts()[0], "I am good.");

  // Test async scoring.
  auto cloned_session_async = std::make_unique<MockSession>();
  EXPECT_CALL(
      *cloned_session_async,
      RunTextScoringAsync(testing::ElementsAre("I am good."), testing::_, true))
      .WillOnce([](const std::vector<absl::string_view>& target_text,
                   absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
                   bool store_token_lengths) {
        callback(Responses(TaskState::kProcessing, {"I am good."}));
        return nullptr;
      });
  EXPECT_CALL(*mock_session_ptr, CloneAsync(testing::_))
      .WillOnce(testing::Return(std::move(cloned_session_async)));

  absl::Notification done;
  std::string response_text;
  EXPECT_OK(conversation->RunTextScoringAsync(
      {"I am good."}, [&](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        response_text = responses->GetTexts()[0];
        done.Notify();
      }));
  done.WaitForNotificationWithTimeout(absl::Seconds(10));
  EXPECT_EQ(response_text, "I am good.");
}

TEST_P(ConversationTest, SendMessageAsync) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));
  ASSERT_OK_AND_ASSIGN(
      auto config,
      ConversationConfig::Builder()
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .Build(*engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*engine, config));

  Message user_message = {{"role", "user"}, {"content", "Hello world!"}};
  // The expected message is just some gibberish text, because the test LLM has
  // random weights.
  Message expected_message =
      Message({{"role", "assistant"},
               {"content",
                {{{"type", "text"},
                  {"text", "TarefaByte دارایेत्र investigaciónప్రదేశ"}}}}});
  Message expected_message_for_confirm = expected_message;

  absl::Notification done;
  EXPECT_OK(conversation->SendMessageAsync(
      user_message, CreateTestMessageCallback(expected_message, done)));
  // Wait for the async message to be processed.
  EXPECT_OK(engine->WaitUntilDone(absl::Seconds(100)));
  done.WaitForNotificationWithTimeout(absl::Seconds(10));
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_message, expected_message_for_confirm));
}

TEST_P(ConversationTest, SendSingleMessageAsync) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send a single message.
  Message user_message = {{"role", "user"}, {"content", "How are you?"}};

  absl::string_view expected_input_text =
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_input_text))),
                      testing::_))
      .WillOnce([](const std::vector<InputData>& contents,
                   absl::AnyInvocable<void(absl::StatusOr<Responses>)>
                       user_callback) {
        user_callback(Responses(TaskState::kDone));
        return nullptr;
      });
  EXPECT_CALL(*mock_session_ptr, RunDecodeAsync(testing::_, testing::_))
      .WillOnce(
          [](absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
             const DecodeConfig& decode_config) {
            user_callback(Responses(TaskState::kProcessing, {"I am good."}));
            user_callback(Responses(TaskState::kDone));
            return nullptr;
          });

  Message assistant_message = Message(nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })"));
  Message assistant_message_for_confirm = assistant_message;
  absl::Notification done;
  auto message_callback = CreateTestMessageCallback(assistant_message, done);
  EXPECT_OK(conversation->SendMessageAsync(user_message,
                                           std::move(message_callback)));
  done.WaitForNotificationWithTimeout(absl::Seconds(10));

  EXPECT_THAT(
      conversation->GetHistory(),
      testing::ElementsAre(user_message, assistant_message_for_confirm));
}

TEST_P(ConversationTest, SendMessageAsyncWithChannelContent) {
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  std::vector<Channel> custom_channels = {{"thought", "<think>", "</think>"}};
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetChannels(custom_channels)
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  Message user_message = {{"role", "user"}, {"content", "How are you?"}};

  absl::string_view expected_input_text =
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_input_text))),
                      testing::_))
      .WillOnce([](const std::vector<InputData>& contents,
                   absl::AnyInvocable<void(absl::StatusOr<Responses>)>
                       user_callback) {
        user_callback(Responses(TaskState::kDone));
        return nullptr;
      });

  EXPECT_CALL(*mock_session_ptr, RunDecodeAsync(testing::_, testing::_))
      .WillOnce(
          [](absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
             const DecodeConfig& decode_config) {
            user_callback(Responses(TaskState::kProcessing,
                                    {"Hello <think>hmm</think> World!"}));
            user_callback(Responses(TaskState::kDone));
            return nullptr;
          });

  absl::Notification done;

  std::vector<Message> expected_messages = {
      Message{{"role", "assistant"},
              {"content", {{{"type", "text"}, {"text", "Hello "}}}}},
      Message{{"role", "assistant"}, {"channels", {{"thought", "hmm"}}}},
      Message{{"role", "assistant"},
              {"content", {{{"type", "text"}, {"text", " World!"}}}}},
  };
  auto message_callback =
      CreateTestMultiMessageCallback(expected_messages, done);
  EXPECT_OK(conversation->SendMessageAsync(user_message,
                                           std::move(message_callback)));
  done.WaitForNotificationWithTimeout(absl::Seconds(10));

  // Verify the final message in history.
  Message expected_assistant_message(nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "Hello  World!"
      }
    ],
    "channels": {
      "thought": "hmm"
    }
  })"));

  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_message, expected_assistant_message));
}

TEST_P(ConversationTest, SendSingleMessageAsyncWithExtraContext) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation and overwrite prompt template.
  absl::string_view prompt_template = R"jinja(
{%- if enable_thinking -%}
<start_of_turn>system
Thinking enabled.<end_of_turn>
{% else %}
<start_of_turn>system
Thinking disabled.<end_of_turn>
{%- endif -%}
{%- for message in messages -%}
  {{- '<start_of_turn>' + message.role + '\n' -}}
  {%- if message.content is string -%}
    {{- message.content + '<end_of_turn>\n' -}}
  {%- else -%}
    {{- message.content[0].text + '<end_of_turn>\n' -}}
  {%- endif -%}
{%- endfor -%}
)jinja";

  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(prompt_template))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send a single message.
  Message user_message = {{"role", "user"}, {"content", "How are you?"}};
  OptionalArgs optional_args;
  optional_args.extra_context = absl::flat_hash_map<std::string, std::string>{
      {"enable_thinking", "true"}};

  absl::string_view expected_input_text =
      "<start_of_turn>system\nThinking enabled.<end_of_turn>\n"
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";

  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_input_text))),
                      testing::_))
      .WillOnce([](const std::vector<InputData>& contents,
                   absl::AnyInvocable<void(absl::StatusOr<Responses>)>
                       user_callback) {
        user_callback(Responses(TaskState::kDone));
        return nullptr;
      });
  EXPECT_CALL(*mock_session_ptr, RunDecodeAsync(testing::_, testing::_))
      .WillOnce(
          [](absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
             const DecodeConfig& decode_config) {
            user_callback(
                Responses(TaskState::kProcessing, {"I am good async."}));
            user_callback(Responses(TaskState::kDone));
            return nullptr;
          });

  Message assistant_message = Message(nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good async."
      }
    ]
  })"));
  Message assistant_message_for_confirm = assistant_message;
  absl::Notification done;
  auto message_callback = CreateTestMessageCallback(assistant_message, done);
  EXPECT_OK(conversation->SendMessageAsync(
      user_message, std::move(message_callback), std::move(optional_args)));
  done.WaitForNotificationWithTimeout(absl::Seconds(10));

  EXPECT_THAT(
      conversation->GetHistory(),
      testing::ElementsAre(user_message, assistant_message_for_confirm));
}

TEST_P(ConversationTest, SendMultipleMessagesAsync) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send two consecutive messages.
  Message user_messages(nlohmann::ordered_json::parse(R"json(
    [
      {
        "role": "user",
        "content": "Hello world!"
      },
      {
        "role": "user",
        "content": "How are you?"
      }
    ]
  )json"));

  absl::string_view expected_input_text =
      "<start_of_turn>user\n"
      "Hello world!<end_of_turn>\n"
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_input_text))),
                      testing::_))
      .WillOnce([](const std::vector<InputData>& contents,
                   absl::AnyInvocable<void(absl::StatusOr<Responses>)>
                       user_callback) {
        user_callback(Responses(TaskState::kDone));
        return nullptr;
      });
  EXPECT_CALL(*mock_session_ptr, RunDecodeAsync(testing::_, testing::_))
      .WillOnce(
          [](absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
             const DecodeConfig& decode_config) {
            user_callback(Responses(TaskState::kProcessing, {"I am good."}));
            user_callback(Responses(TaskState::kDone));
            return nullptr;
          });

  Message assistant_message = Message(nlohmann::ordered_json::parse(R"json({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })json"));
  Message assistant_message_for_confirm = assistant_message;
  absl::Notification done;
  auto message_callback = CreateTestMessageCallback(assistant_message, done);
  EXPECT_OK(conversation->SendMessageAsync(user_messages,
                                           std::move(message_callback)));
  done.WaitForNotificationWithTimeout(absl::Seconds(10));

  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_messages[0], user_messages[1],
                                   assistant_message_for_confirm));
}

TEST_P(ConversationTest, SendMultipleMessagesAsyncWithHistory) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // The first user message.
  Message user_message_1(nlohmann::ordered_json::parse(R"json(
    {
      "role": "user",
      "content": "How are you?"
    }
  )json"));
  absl::string_view expected_input_text1 =
      "<start_of_turn>user\n"
      "How are you?<end_of_turn>\n";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_input_text1))),
                      testing::_))
      .WillOnce([](const std::vector<InputData>& contents,
                   absl::AnyInvocable<void(absl::StatusOr<Responses>)>
                       user_callback) {
        user_callback(Responses(TaskState::kDone));
        return nullptr;
      });
  EXPECT_CALL(*mock_session_ptr, RunDecodeAsync(testing::_, testing::_))
      .WillOnce(
          [](absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
             const DecodeConfig& decode_config) {
            user_callback(Responses(TaskState::kProcessing, {"I am good."}));
            user_callback(Responses(TaskState::kDone));
            return nullptr;
          });

  Message assistant_message_1 = Message(nlohmann::ordered_json::parse(R"json({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })json"));
  Message assistant_message_1_for_confirm = assistant_message_1;

  absl::Notification done_1;
  EXPECT_OK(conversation->SendMessageAsync(
      user_message_1, CreateTestMessageCallback(assistant_message_1, done_1)));
  done_1.WaitForNotificationWithTimeout(absl::Seconds(10));
  ASSERT_THAT(conversation->GetHistory().size(), testing::Eq(2));

  // We will send two consecutive messages when the history is not empty.
  Message user_messages = Message(nlohmann::ordered_json::parse(R"json(
    [
      {
        "role": "user",
        "content": "foo"
      },
      {
        "role": "user",
        "content": "bar"
      }
    ]
  )json"));

  absl::string_view expected_input_text2 =
      "<start_of_turn>user\n"
      "foo<end_of_turn>\n"
      "<start_of_turn>user\n"
      "bar<end_of_turn>\n";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_input_text2))),
                      testing::_))
      .WillOnce([](const std::vector<InputData>& contents,
                   absl::AnyInvocable<void(absl::StatusOr<Responses>)>
                       user_callback) {
        user_callback(Responses(TaskState::kDone));
        return nullptr;
      });
  EXPECT_CALL(*mock_session_ptr, RunDecodeAsync(testing::_, testing::_))
      .WillOnce(
          [](absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
             const DecodeConfig& decode_config) {
            user_callback(Responses(TaskState::kProcessing, {"baz"}));
            user_callback(Responses(TaskState::kDone));
            return nullptr;
          });

  Message assistant_message_2 = Message(nlohmann::ordered_json::parse(R"json({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "baz"
      }
    ]
  })json"));
  Message assistant_message_2_for_confirm = assistant_message_2;

  absl::Notification done_2;
  auto message_callbacks_2 =
      CreateTestMessageCallback(assistant_message_2, done_2);
  EXPECT_OK(conversation->SendMessageAsync(user_messages,
                                           std::move(message_callbacks_2)));
  done_2.WaitForNotificationWithTimeout(absl::Seconds(10));

  EXPECT_THAT(
      conversation->GetHistory(),
      testing::ElementsAre(user_message_1, assistant_message_1_for_confirm,
                           user_messages[0], user_messages[1],
                           assistant_message_2_for_confirm));
}

TEST_P(ConversationTest, SendMessageWithPreface) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(15);
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));
  ASSERT_OK_AND_ASSIGN(
      auto config,
      ConversationConfig::Builder()
          .SetPreface(JsonPreface{
              .messages = {{{"role", "system"},
                            {"content", "You are a helpful assistant."}}}})
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .Build(*engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*engine, config));
  ASSERT_OK_AND_ASSIGN(const Message message,
                       conversation->SendMessage(Message{
                           {"role", "user"}, {"content", "Hello world!"}}));
  // The expected message is just some gibberish text, because the test LLM has
  // random weights.
  Message expected_message;
  if (prefill_preface_on_init_) {
    expected_message = {{"role", "assistant"},
                        {"content",
                         {{{"type", "text"},
                           {"text", " rupani rupani rupani echoes echoes"}}}}};
  } else {
    expected_message = {
        {"role", "assistant"},
        {"content",
         {{{"type", "text"},
           {"text", " noses</caption> গ্রাহ<unused5296> omp"}}}}};
  }
  EXPECT_EQ(message, expected_message);
}

TEST_P(ConversationTest, GetBenchmarkInfo) {
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(15);
  proto::BenchmarkParams benchmark_params;
  engine_settings.GetMutableBenchmarkParams() = benchmark_params;
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));
  ASSERT_OK_AND_ASSIGN(
      auto config,
      ConversationConfig::Builder()
          .SetPreface(JsonPreface{
              .messages = {{{"role", "system"},
                            {"content", "You are a helpful assistant."}}}})
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .Build(*engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*engine, config));
  ASSERT_OK_AND_ASSIGN(const Message message_1,
                       conversation->SendMessage(Message{
                           {"role", "user"}, {"content", "Hello world!"}}));
  ASSERT_OK_AND_ASSIGN(const BenchmarkInfo benchmark_info_1,
                       conversation->GetBenchmarkInfo());
  EXPECT_EQ(benchmark_info_1.GetTotalPrefillTurns(),
            prefill_preface_on_init_ ? 2 : 1);

  ASSERT_OK_AND_ASSIGN(const Message message_2,
                       conversation->SendMessage(Message{
                           {"role", "user"}, {"content", "Hello world!"}}));
  ASSERT_OK_AND_ASSIGN(const BenchmarkInfo benchmark_info_2,
                       conversation->GetBenchmarkInfo());
  EXPECT_EQ(benchmark_info_2.GetTotalPrefillTurns(),
            prefill_preface_on_init_ ? 3 : 2);
}

TEST_P(ConversationTest, CancelGroupWithSendMessageAsync) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send a single message.
  Message user_message = {{"role", "user"}, {"content", "How are you?"}};

  auto mock_task_controller1 = std::make_unique<MockTaskController>();
  // Expect Cancel() to be called on the first task controller when
  // CancelGroup("group1") is called.
  EXPECT_CALL(*mock_task_controller1, Cancel())
      .WillOnce(testing::Return(absl::OkStatus()));
  auto mock_task_controller2 = std::make_unique<MockTaskController>();
  // Expect Cancel() to be called on the second task controller when
  // CancelGroup("group1") is called.
  EXPECT_CALL(*mock_task_controller2, Cancel())
      .WillOnce(testing::Return(absl::OkStatus()));

  // Expect RunPrefillAsync to be called and return the first task controller.
  EXPECT_CALL(*mock_session_ptr, RunPrefillAsync(testing::_, testing::_))
      .WillOnce([&](const std::vector<InputData>& contents,
                    absl::AnyInvocable<void(absl::StatusOr<Responses>)>
                        user_callback) {
        user_callback(Responses(TaskState::kDone));
        return std::move(mock_task_controller1);
      });
  // Expect RunDecodeAsync to be called and return the second task controller.
  EXPECT_CALL(*mock_session_ptr, RunDecodeAsync(testing::_, testing::_))
      .WillOnce(
          [&](absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
              const DecodeConfig& decode_config) {
            return std::move(mock_task_controller2);
          });

  absl::Notification done;
  absl::Status status;
  EXPECT_OK(
      conversation->SendMessageAsync(user_message,
                                     [&](absl::StatusOr<Message> message) {
                                       status = message.status();
                                       done.Notify();
                                     },
                                     {.task_group_id = "group1"}));

  conversation->CancelGroup("group1");
}

TEST_P(ConversationTest, CancelGroupWithRunTextScoringAsync) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();

  auto cloned_session = std::make_unique<MockSession>();
  // Expect GetSessionConfig to be called on the cloned session.
  MockSession* cloned_session_ptr = cloned_session.get();
  EXPECT_CALL(*cloned_session_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config_));

  // Expect CloneAsync to be called and return the cloned session.
  EXPECT_CALL(*mock_session_ptr, CloneAsync(testing::_))
      .WillOnce(testing::Return(std::move(cloned_session)));
  auto mock_engine = CreateMockEngine(std::move(mock_session));
  EXPECT_CALL(*mock_engine, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer_));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  auto mock_task_controller = std::make_unique<MockTaskController>();
  // Expect Cancel() to be called on the task controller when
  // CancelGroup("group1") is called.
  EXPECT_CALL(*mock_task_controller, Cancel())
      .WillOnce(testing::Return(absl::OkStatus()));

  // Expect RunTextScoringAsync to be called on the cloned session and return
  // the task controller.
  EXPECT_CALL(
      *cloned_session_ptr,
      RunTextScoringAsync(testing::ElementsAre("I am good."), testing::_, true))
      .WillOnce(
          [&](const std::vector<absl::string_view>& target_text,
              absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
              bool store_token_lengths) {
            return std::move(mock_task_controller);
          });

  absl::Notification done;
  std::string response_text;
  EXPECT_OK(conversation->RunTextScoringAsync(
      {"I am good."},
      [&](absl::StatusOr<Responses> responses) {
        ASSERT_OK(responses);
        response_text = responses->GetTexts()[0];
        done.Notify();
      },
      {.task_group_id = "group1"}));

  conversation->CancelGroup("group1");
}

INSTANTIATE_TEST_SUITE_P(
    ConversationTest, ConversationTest,
    testing::ValuesIn(ConversationTest::GetTestParams()),
    [](const testing::TestParamInfo<ConversationTestParams>& info) {
      return absl::StrCat(
          info.param.enable_constrained_decoding ? "Constrained" : "Free", "_",
          info.param.prefill_preface_on_init ? "PrefillOnInit"
                                             : "NoPrefillOnInit");
    });

absl::AnyInvocable<void(absl::StatusOr<Message>)>
CreateCancelledMessageCallback(absl::Status& status, absl::Notification& done) {
  return [&status, &done](absl::StatusOr<Message> message) mutable {
    if (!message.ok()) {
      status = message.status();
      done.Notify();
      return;
    }
    if (message->is_null()) {
      status = absl::OkStatus();
      done.Notify();
      return;
    }
    // Wait for a short time to slow down the decoding process, so that the
    // cancellation can be triggered in the middle of decoding.
    absl::SleepFor(absl::Milliseconds(100));
  };
}

TEST(ConversationAccessHistoryTest, AccessHistory) {
  // Create a Conversation.
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(10);
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));
  ASSERT_OK_AND_ASSIGN(auto config, ConversationConfig::CreateDefault(*engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*engine, config));

  // Send a message to the LLM.
  Message user_message = {{"role", "user"}, {"content", "Hello world!"}};
  Message expected_assistant_message =
      Message({{"role", "assistant"},
               {"content",
                {{{"type", "text"},
                  {"text", "TarefaByte دارایेत्र investigaciónప్రదేశ"}}}}});
  Message expected_assistant_message_for_confirm = expected_assistant_message;
  absl::Notification done;
  EXPECT_OK(conversation->SendMessageAsync(
      user_message,
      CreateTestMessageCallback(expected_assistant_message, done)));
  done.WaitForNotificationWithTimeout(absl::Seconds(10));

  // Get the history copy.
  auto history = conversation->GetHistory();
  ASSERT_THAT(history.size(), 2);
  ASSERT_THAT(history.back(),
              testing::Eq(expected_assistant_message_for_confirm));

  // Access the history with visitor function, and copy the last message.
  Message last_message;
  conversation->AccessHistory(
      [&last_message](const std::vector<Message>& history_view) {
        // Copy the last message to last_message. So we don't need to
        // copy the whole history, if we only need the last message.
        last_message = history_view.back();
      });
  EXPECT_THAT(last_message,
              testing::Eq(expected_assistant_message_for_confirm));
}

class ConversationCancellationTest : public testing::TestWithParam<bool> {
 protected:
  bool use_benchmark_info_ = GetParam();
};

TEST_P(ConversationCancellationTest, CancelProcessWithBenchmarkInfo) {
  bool use_benchmark_info = use_benchmark_info_;
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  // Set a large max num tokens to ensure the decoding is not finished before
  // cancellation.
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(20);
  if (use_benchmark_info) {
    proto::BenchmarkParams benchmark_params;
    engine_settings.GetMutableBenchmarkParams() = benchmark_params;
  }
  ASSERT_OK_AND_ASSIGN(auto engine, EngineFactory::CreateAny(engine_settings));
  ASSERT_OK_AND_ASSIGN(auto config, ConversationConfig::CreateDefault(*engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*engine, config));

  absl::Status status;
  absl::Notification done_1;
  conversation
      ->SendMessageAsync(Message{{"role", "user"}, {"content", "Hello world!"}},
                         CreateCancelledMessageCallback(status, done_1))
      .IgnoreError();
  // Wait for a short time to ensure the decoding has started.
  absl::SleepFor(absl::Milliseconds(100));
  conversation->CancelProcess();
  // Wait for the callback to be done.
  done_1.WaitForNotificationWithTimeout(absl::Seconds(10));
  EXPECT_THAT(status, testing::status::StatusIs(absl::StatusCode::kCancelled));

  // The history should be empty after cancellation.
  EXPECT_THAT(conversation->GetHistory().size(), 0);

  // Re-send the message after cancellation, and it should succeed.
  status = absl::OkStatus();
  absl::Notification done_2;
  conversation
      ->SendMessageAsync(Message{{"role", "user"}, {"content", "Hello world!"}},
                         CreateCancelledMessageCallback(status, done_2))
      .IgnoreError();
  EXPECT_OK(status);
  // Wait for the callback to be done.
  done_2.WaitForNotificationWithTimeout(absl::Seconds(10));
  // Without cancellation, the history should have two messages, user and
  // assistant.
  auto history = conversation->GetHistory();
  ASSERT_EQ(history.size(), 2);
  EXPECT_THAT(history[0], testing::Eq(Message{{"role", "user"},
                                              {"content", "Hello world!"}}));
  // TODO(b/450903294) - Because the cancellation is not fully rollbacked, the
  // assistant message content depends on at which step the cancellation is
  // triggered, and that is non-deterministic. Here we only check the role is
  // assistant.
  EXPECT_EQ(history[1]["role"], "assistant");

  conversation->CancelProcess();
  // No op after cancellation again.
  EXPECT_THAT(conversation->GetHistory().size(), 2);
}

INSTANTIATE_TEST_SUITE_P(ConversationCancellationTest,
                         ConversationCancellationTest, testing::Bool(),
                         testing::PrintToStringParamName());

class MockConstraint : public Constraint {
 public:
  class MockState : public State {
   public:
    ~MockState() override = default;
  };
  MOCK_METHOD(std::unique_ptr<State>, Start, (), (const, override));
  MOCK_METHOD(bool, IsEnded, (const State& state), (const, override));
  MOCK_METHOD(int, GetVocabularySize, (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<State>>, ComputeNext,
              (const State& state, int token), (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Bitmap>>, ComputeBitmap,
              (const State& state), (const, override));
};

TEST_P(ConversationTest, SendMessageWithConstraint) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));

  // Create Conversation with ExternalConstraintConfig.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetConstraintProviderConfig(ExternalConstraintConfig())
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // Create a mock constraint.
  auto mock_constraint = std::make_unique<MockConstraint>();
  Constraint* mock_constraint_ptr = mock_constraint.get();
  ExternalConstraintArg constraint_arg;
  constraint_arg.constraint = std::move(mock_constraint);

  // Send a message with the constraint.
  Message user_message = {{"role", "user"}, {"content", "How are you?"}};

  EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
      .WillOnce(testing::Return(absl::OkStatus()));

  // Verify that the constraint is passed to RunDecode.
  EXPECT_CALL(*mock_session_ptr,
              RunDecode(testing::Property(&DecodeConfig::GetConstraint,
                                          mock_constraint_ptr)))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));

  ASSERT_OK_AND_ASSIGN(
      const Message response,
      conversation->SendMessage(
          user_message, {
                            .decoding_constraint = std::move(constraint_arg),
                        }));
}

TEST_P(ConversationTest, Clone) {
  // Set up mock Session.
  auto mock_session = CreateMockSession();
  MockSession* mock_session_ptr = mock_session.get();
  auto mock_engine = CreateMockEngine(std::move(mock_session));
  EXPECT_CALL(*mock_engine, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer_));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // Send a message to populate history.
  Message user_message = {{"role", "user"}, {"content", "Hello"}};
  EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(testing::Return(Responses(TaskState::kProcessing, {"Hi"})));
  ASSERT_OK(conversation->SendMessage(user_message));

  // Expect Session::Clone to be called.
  auto cloned_mock_session = std::make_unique<MockSession>();
  MockSession* cloned_mock_session_ptr = cloned_mock_session.get();
  EXPECT_CALL(*cloned_mock_session_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config_));
  EXPECT_CALL(*mock_session_ptr, Clone())
      .WillOnce(testing::Return(std::move(cloned_mock_session)));

  // Clone the conversation.
  ASSERT_OK_AND_ASSIGN(auto cloned_conversation, conversation->Clone());

  // Verify the history in the cloned conversation.
  auto history = cloned_conversation->GetHistory();
  EXPECT_EQ(history.size(), 2);
  EXPECT_EQ(history[0], user_message);

  // Verify that sending a message in the cloned conversation works and uses the
  // cloned session.
  Message user_message2 = {{"role", "user"}, {"content", "How are you?"}};
  EXPECT_CALL(*cloned_mock_session_ptr, RunPrefill(testing::_))
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*cloned_mock_session_ptr, RunDecode(testing::_))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));

  ASSERT_OK(cloned_conversation->SendMessage(user_message2));

  // Verify that the original conversation is unaffected by the new message in
  // the cloned one.
  EXPECT_EQ(conversation->GetHistory().size(), 2);
  EXPECT_EQ(cloned_conversation->GetHistory().size(), 4);
}

TEST_P(ConversationTest, SendMessageWithMaxOutputTokens) {
  // Set up mock Session.
  auto mock_session = std::make_unique<MockSession>();
  MockSession* mock_session_ptr = mock_session.get();
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(0);
  session_config.GetMutableStopTokenIds().push_back({1});
  *session_config.GetMutableLlmModelType().mutable_gemma3() = {};
  EXPECT_CALL(*mock_session_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config));

  // Set up mock Engine.
  auto mock_engine = std::make_unique<MockEngine>();
  EXPECT_CALL(*mock_engine, CreateSession(testing::_))
      .WillOnce(testing::Return(std::move(mock_session)));
  EXPECT_CALL(*mock_engine, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer_));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  EXPECT_CALL(*mock_engine, GetEngineSettings())
      .WillRepeatedly(testing::ReturnRef(engine_settings));

  // Create Conversation with default config.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  Message user_message = {{"role", "user"}, {"content", "How are you?"}};

  EXPECT_CALL(*mock_session_ptr, RunPrefill(testing::_))
      .WillOnce(testing::Return(absl::OkStatus()));

  // Verify that the max_output_tokens is passed to RunDecode.
  EXPECT_CALL(*mock_session_ptr,
              RunDecode(testing::Property(&DecodeConfig::GetMaxOutputTokens,
                                          std::make_optional(42))))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));

  ASSERT_OK_AND_ASSIGN(
      const Message response,
      conversation->SendMessage(user_message, {.max_output_tokens = 42}));
}

TEST(AppendMessageTest, Gemma3Sync) {
  // Set up mock Session.
  auto mock_session = std::make_unique<MockSession>();
  MockSession* mock_session_ptr = mock_session.get();
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(0);
  session_config.GetMutableStopTokenIds().push_back({1});
  *session_config.GetMutableLlmModelType().mutable_gemma3() = {};
  session_config.SetApplyPromptTemplateInSession(false);
  EXPECT_CALL(*mock_session_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config));
  ASSERT_OK_AND_ASSIGN(
      auto tokenizer,
      SentencePieceTokenizer::CreateFromFile(
          (std::filesystem::path(::testing::SrcDir()) / kTestTokenizerPath)
              .string()));

  // Set up mock Engine.
  auto mock_engine = std::make_unique<MockEngine>();
  EXPECT_CALL(*mock_engine, CreateSession(testing::_))
      .WillOnce(testing::Return(std::move(mock_session)));
  EXPECT_CALL(*mock_engine, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  EXPECT_CALL(*mock_engine, GetEngineSettings())
      .WillRepeatedly(testing::ReturnRef(engine_settings));

  std::string template_text =
      ReadFile(GetTestdataPath(kGemma3ToolsMultiPrefillTemplatePath));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
          .SetOverwritePromptTemplate(PromptTemplate(template_text))
          .SetPreface(JsonPreface{
              .messages = {{{"role", "system"},
                            {"content", "You are a helpful assistant."}}}})
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // Append the 1st message.
  absl::string_view expected_prefill_1 =
      "<start_of_turn>user\nYou are a helpful "
      "assistant.\n\n<end_of_turn>\n<start_of_turn>user\nHello world!";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_1)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  ASSERT_OK(conversation->SendMessage(
      Message{{"role", "user"}, {"content", "Hello world!"}},
      {.has_pending_message = true}));

  // Append the 2nd message.
  absl::string_view expected_prefill_2 = " This is a long message.";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_2)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  ASSERT_OK(conversation->SendMessage(
      Message{{"role", "user"}, {"content", " This is a long message."}},
      {.has_pending_message = true}));

  // Append the 3rd message.
  absl::string_view expected_prefill_3 = " continuing...";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_3)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  ASSERT_OK(conversation->SendMessage(
      Message{{"role", "user"}, {"content", " continuing..."}},
      {.has_pending_message = true}));

  // Finish appending message.
  absl::string_view expected_prefill_4 =
      " The message is ended.<end_of_turn>\n<start_of_turn>model\n";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_4)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));
  ASSERT_OK_AND_ASSIGN(
      const Message response_appending,
      conversation->SendMessage(
          Message{{"role", "user"}, {"content", " The message is ended."}}));
}

TEST(AppendMessageTest, Gemma3Async) {
  // Set up mock Session.
  auto mock_session = std::make_unique<MockSession>();
  MockSession* mock_session_ptr = mock_session.get();
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(0);
  session_config.GetMutableStopTokenIds().push_back({1});
  *session_config.GetMutableLlmModelType().mutable_gemma3() = {};
  session_config.SetApplyPromptTemplateInSession(false);
  EXPECT_CALL(*mock_session_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config));
  ASSERT_OK_AND_ASSIGN(
      auto tokenizer,
      SentencePieceTokenizer::CreateFromFile(
          (std::filesystem::path(::testing::SrcDir()) / kTestTokenizerPath)
              .string()));

  // Set up mock Engine.
  auto mock_engine = std::make_unique<MockEngine>();
  EXPECT_CALL(*mock_engine, CreateSession(testing::_))
      .WillOnce(testing::Return(std::move(mock_session)));
  EXPECT_CALL(*mock_engine, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  EXPECT_CALL(*mock_engine, GetEngineSettings())
      .WillRepeatedly(testing::ReturnRef(engine_settings));

  std::string template_text =
      ReadFile(GetTestdataPath(kGemma3ToolsMultiPrefillTemplatePath));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
          .SetOverwritePromptTemplate(PromptTemplate(template_text))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  auto test_callback =
      [](const std::vector<InputData>& contents,
         absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback) {
        user_callback(Responses(TaskState::kDone));
        return nullptr;
      };

  // Append the 1st message.
  absl::string_view expected_prefill_1 = "<start_of_turn>user\nHello world!";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_prefill_1))),
                      testing::_))
      .Times(1)
      .WillOnce(test_callback);
  absl::Notification done1;
  ASSERT_OK(conversation->SendMessageAsync(
      Message{{"role", "user"}, {"content", "Hello world!"}},
      [&done1](absl::StatusOr<Message> message) { done1.Notify(); },
      {.has_pending_message = true}));
  done1.WaitForNotificationWithTimeout(absl::Seconds(3));

  // Append the 2nd message.
  absl::string_view expected_prefill_2 = " This is a long message.";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_prefill_2))),
                      testing::_))
      .Times(1)
      .WillOnce(test_callback);
  absl::Notification done2;
  ASSERT_OK(conversation->SendMessageAsync(
      Message{{"role", "user"}, {"content", " This is a long message."}},
      [&done2](absl::StatusOr<Message> message) { done2.Notify(); },
      {.has_pending_message = true}));
  done2.WaitForNotificationWithTimeout(absl::Seconds(3));

  // Append the 3rd message.
  absl::string_view expected_prefill_3 = " continuing...";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_prefill_3))),
                      testing::_))
      .Times(1)
      .WillOnce(test_callback);
  absl::Notification done3;
  ASSERT_OK(conversation->SendMessageAsync(
      Message{{"role", "user"}, {"content", " continuing..."}},
      [&done3](absl::StatusOr<Message> message) { done3.Notify(); },
      {.has_pending_message = true}));
  done3.WaitForNotificationWithTimeout(absl::Seconds(3));

  // Append the 4th message.
  absl::string_view expected_prefill_4 = " The message is ended.";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_prefill_4))),
                      testing::_))
      .Times(1)
      .WillOnce(test_callback);
  absl::Notification done4;
  EXPECT_OK(conversation->SendMessageAsync(
      Message{{"role", "user"}, {"content", " The message is ended."}},
      [&done4](absl::StatusOr<Message> message) { done4.Notify(); },
      {.has_pending_message = true}));
  done4.WaitForNotificationWithTimeout(absl::Seconds(3));

  // The 5th message triggers the decode.
  absl::string_view expected_prefill_5 =
      "<end_of_turn>\n<start_of_turn>model\n";
  EXPECT_CALL(
      *mock_session_ptr,
      RunPrefillAsync(testing::ElementsAre(testing::VariantWith<InputText>(
                          testing::Property(&InputText::GetRawTextString,
                                            expected_prefill_5))),
                      testing::_))
      .Times(1)
      .WillOnce(test_callback);
  EXPECT_CALL(*mock_session_ptr, RunDecodeAsync(testing::_, testing::_))
      .WillOnce(
          [](absl::AnyInvocable<void(absl::StatusOr<Responses>)> user_callback,
             const DecodeConfig& decode_config) {
            user_callback(Responses(TaskState::kProcessing, {"I am good."}));
            user_callback(Responses(TaskState::kDone));
            return nullptr;
          });
  Message expected_assistant_message =
      Message({{"role", "assistant"},
               {"content", {{{"type", "text"}, {"text", "I am good."}}}}});
  absl::Notification done5;
  // Trigger the decode by sending an empty message.
  EXPECT_OK(conversation->SendMessageAsync(
      Message{{"role", "user"}, {"content", ""}},
      CreateTestMessageCallback(expected_assistant_message, done5),
      {.has_pending_message = false}));
  done5.WaitForNotificationWithTimeout(absl::Seconds(3));
}

TEST(AppendMessageTest, Gemma3SyncPrefillPrefaceOnInitAndAlternateRoles) {
  // Set up mock Session.
  auto mock_session = std::make_unique<MockSession>();
  MockSession* mock_session_ptr = mock_session.get();
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(0);
  session_config.GetMutableStopTokenIds().push_back({1});
  *session_config.GetMutableLlmModelType().mutable_gemma3() = {};
  session_config.SetApplyPromptTemplateInSession(false);
  EXPECT_CALL(*mock_session_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config));
  ASSERT_OK_AND_ASSIGN(
      auto tokenizer,
      SentencePieceTokenizer::CreateFromFile(
          (std::filesystem::path(::testing::SrcDir()) / kTestTokenizerPath)
              .string()));

  // Set up mock Engine.
  auto mock_engine = std::make_unique<MockEngine>();
  EXPECT_CALL(*mock_engine, CreateSession(testing::_))
      .WillOnce(testing::Return(std::move(mock_session)));
  EXPECT_CALL(*mock_engine, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  EXPECT_CALL(*mock_engine, GetEngineSettings())
      .WillRepeatedly(testing::ReturnRef(engine_settings));

  std::string template_text =
      ReadFile(GetTestdataPath(kGemma3ToolsMultiPrefillTemplatePath));

  // Init with preface.
  absl::string_view expected_prefill_preface = R"(<start_of_turn>system
def tool_name(
    x: int | None = None,
) -> dict:
  """
  Args:
    x  """

<end_of_turn>
<start_of_turn>user
You are a helpful assistant.

<end_of_turn>
)";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(testing::VariantWith<InputText>(
                  testing::Property(&InputText::GetRawTextString,
                                    expected_prefill_preface)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
          .SetOverwritePromptTemplate(PromptTemplate(template_text))
          .SetPreface(JsonPreface{
              .messages = {{{"role", "system"},
                            {"content", "You are a helpful assistant."}}},
              .tools = nlohmann::ordered_json::parse(
                  R"json([{
                            "name": "tool_name",
                            "parameters": {
                              "properties": {
                                "x": {
                                  "type": "integer"
                                }
                              }
                            }
                          }])json")})
          .SetPrefillPrefaceOnInit(true)
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // Append the 1st message.
  absl::string_view expected_prefill_1 = "<start_of_turn>user\nHello world!";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_1)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  ASSERT_OK(conversation->SendMessage(
      Message{{"role", "user"}, {"content", "Hello world!"}},
      {.has_pending_message = true}));

  // Append the 2nd message.
  absl::string_view expected_prefill_2 =
      "<end_of_turn>\n<start_of_turn>model\nNice to meet you.";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_2)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  ASSERT_OK(conversation->SendMessage(
      Message{{"role", "model"}, {"content", "Nice to meet you."}},
      {.has_pending_message = true}));

  // Append the 3rd message.
  absl::string_view expected_prefill_3 = " How can I help you today?";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_3)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  ASSERT_OK(conversation->SendMessage(
      Message{{"role", "model"}, {"content", " How can I help you today?"}},
      {.has_pending_message = true}));

  // Append the 4th message.
  absl::string_view expected_prefill_4 = " The message is ended.";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_4)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  ASSERT_OK(conversation->SendMessage(
      Message{{"role", "model"}, {"content", " The message is ended."}},
      {.has_pending_message = true}));

  // Append the 5th message.
  absl::string_view expected_prefill_5 = R"(<end_of_turn>
<start_of_turn>user
```tool_outputs
{"location": "Paris", "temperature": 20, "unit": "C", "weather": "Sunny"})";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_5)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  ASSERT_OK(conversation->SendMessage(Message{{"role", "tool"},
                                              {"content",
                                               {
                                                   {"type", "tool_response"},
                                                   {"tool_response",
                                                    {
                                                        {"location", "Paris"},
                                                        {"temperature", 20},
                                                        {"unit", "C"},
                                                        {"weather", "Sunny"},
                                                    }},
                                               }}},
                                      {.has_pending_message = true}));

  // Append the 6th message.
  absl::string_view expected_prefill_6 =
      R"({"location": "London", "temperature": 15, "unit": "C", "weather": "Cloudy"}
```<end_of_turn>
<start_of_turn>model
)";
  EXPECT_CALL(*mock_session_ptr,
              RunPrefill(testing::ElementsAre(
                  testing::VariantWith<InputText>(testing::Property(
                      &InputText::GetRawTextString, expected_prefill_6)))))
      .Times(1)
      .WillOnce(testing::Return(absl::OkStatus()));
  EXPECT_CALL(*mock_session_ptr, RunDecode(testing::_))
      .WillOnce(
          testing::Return(Responses(TaskState::kProcessing, {"I am good."})));
  ASSERT_OK(conversation->SendMessage(Message{{"role", "tool"},
                                              {"content",
                                               {
                                                   {"type", "tool_response"},
                                                   {"tool_response",
                                                    {
                                                        {"location", "London"},
                                                        {"temperature", 15},
                                                        {"unit", "C"},
                                                        {"weather", "Cloudy"},
                                                    }},
                                               }}},
                                      {.has_pending_message = false}));
}

}  // namespace
}  // namespace litert::lm
