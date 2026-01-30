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
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
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
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

absl::string_view kTestLlmPath =
    "litert_lm/runtime/testdata/test_lm.litertlm";

constexpr char kTestTokenizerPath[] =
    "litert_lm/runtime/components/testdata/gemma3_sentencepiece.model";

constexpr char kGemma3ToolsMultiPrefillTemplatePath[] =
    "litert_lm/runtime/components/testdata/"
    "google-gemma-3n-e2b-it-tools-multi-prefill.jinja";

constexpr absl::string_view kTestJinjaPromptTemplate = R"jinja(
{%- for message in messages -%}
  {{ '<start_of_turn>' + message.role }}
  {{ message.content + '<end_of_turn>\n' }}
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
  MOCK_METHOD(absl::StatusOr<BenchmarkInfo>, GetBenchmarkInfo, (), (override));
  MOCK_METHOD(absl::StatusOr<BenchmarkInfo*>, GetMutableBenchmarkInfo, (),
              (override));
  MOCK_METHOD(void, CancelProcess, (), (override));
  MOCK_METHOD(absl::Status, WaitUntilDone, (), (override));
  MOCK_METHOD(const SessionConfig&, GetSessionConfig, (), (const, override));
  MOCK_METHOD(const Tokenizer&, GetTokenizer, (), (const, override));
};

class MockEngine : public Engine {
 public:
  MOCK_METHOD(const EngineSettings&, GetEngineSettings, (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Session>>, CreateSession,
              (const SessionConfig& session_config), (override));
  MOCK_METHOD(absl::Status, WaitUntilDone, (absl::Duration timeout),
              (override));
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
    if (auto json_message = std::get_if<JsonMessage>(&message.value());
        json_message->is_null()) {
      JsonMessage& expected_json_message =
          std::get<JsonMessage>(expected_message);
      ASSERT_TRUE(expected_json_message["content"][0]["text"].is_string());
      std::string expected_string = expected_json_message["content"][0]["text"];
      // The expected string should be empty after the last callback.
      EXPECT_TRUE(expected_string.empty());
      done.Notify();
      return;
    }
    // Otherwise, this is a partial response.
    if (auto json_message = std::get_if<JsonMessage>(&message.value())) {
      JsonMessage& expected_json_message =
          std::get<JsonMessage>(expected_message);
      // Compare the message text content by prefix, and update the expected
      // message to the remaining text for the next user_callback.
      ASSERT_TRUE(expected_json_message["content"][0]["text"].is_string());
      ASSERT_TRUE((*json_message)["content"][0]["text"].is_string());
      std::string expected_string = expected_json_message["content"][0]["text"];
      std::string actual_string = (*json_message)["content"][0]["text"];
      EXPECT_TRUE(absl::StartsWith(expected_string, actual_string))
          << "Expected: " << expected_string << "\nActual: " << actual_string;
      expected_json_message["content"][0]["text"] =
          expected_string.substr(actual_string.size());
    }
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
  EXPECT_TRUE(config.GetSessionConfig().GetLlmModelType().has_gemma3());
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
    auto tokenizer = SentencePieceTokenizer::CreateFromFile(
        (std::filesystem::path(::testing::SrcDir()) / kTestTokenizerPath)
            .string());
    ASSERT_OK(tokenizer);
    tokenizer_ = std::move(*tokenizer);
  }

  std::unique_ptr<Tokenizer> tokenizer_;
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
  JsonMessage user_message = {{"role", "user"}, {"content", "Hello world!"}};
  ASSERT_OK_AND_ASSIGN(const Message message,
                       conversation->SendMessage(user_message));
  // The expected message is just some gibberish text, because the test LLM has
  // random weights.
  JsonMessage expected_message = {
      {"role", "assistant"},
      {"content",
       {{{"type", "text"}, {"text", "TarefaByte دارایेत्र investigaciónప్రదేశ"}}}}};
  const JsonMessage& json_message = std::get<JsonMessage>(message);
  EXPECT_EQ(json_message, expected_message);
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_message, expected_message));
}

TEST_P(ConversationTest, SendSingleMessage) {
  // Set up mock Session.
  auto mock_session = std::make_unique<MockSession>();
  MockSession* mock_session_ptr = mock_session.get();
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(0);
  session_config.GetMutableStopTokenIds().push_back({1});
  *session_config.GetMutableLlmModelType().mutable_gemma3() = {};
  EXPECT_CALL(*mock_session_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config));
  EXPECT_CALL(*mock_session_ptr, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer_));

  // Set up mock Engine.
  auto mock_engine = std::make_unique<MockEngine>();
  EXPECT_CALL(*mock_engine, CreateSession(testing::_))
      .WillOnce(testing::Return(std::move(mock_session)));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  EXPECT_CALL(*mock_engine, GetEngineSettings())
      .WillRepeatedly(testing::ReturnRef(engine_settings));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send a single message.
  JsonMessage user_message = {{"role", "user"}, {"content", "How are you?"}};

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

  JsonMessage assistant_message = nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })");
  EXPECT_EQ(std::get<JsonMessage>(response), assistant_message);
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_message, assistant_message));
}

TEST_P(ConversationTest, SendMultipleMessages) {
  // Set up mock Session.
  auto mock_session = std::make_unique<MockSession>();
  MockSession* mock_session_ptr = mock_session.get();
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(0);
  session_config.GetMutableStopTokenIds().push_back({1});
  *session_config.GetMutableLlmModelType().mutable_gemma3() = {};
  EXPECT_CALL(*mock_session_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config));
  EXPECT_CALL(*mock_session_ptr, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer_));

  // Set up mock Engine.
  auto mock_engine = std::make_unique<MockEngine>();
  EXPECT_CALL(*mock_engine, CreateSession(testing::_))
      .WillOnce(testing::Return(std::move(mock_session)));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  EXPECT_CALL(*mock_engine, GetEngineSettings())
      .WillRepeatedly(testing::ReturnRef(engine_settings));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send two consecutive messages.
  JsonMessage user_messages = nlohmann::ordered_json::parse(R"json(
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
  )json");

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

  JsonMessage assistant_message = nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })");
  EXPECT_EQ(std::get<JsonMessage>(response), assistant_message);
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_messages[0], user_messages[1],
                                   assistant_message));
}

TEST_P(ConversationTest, SendMultipleMessagesWithHistory) {
  // Set up mock Session.
  auto mock_session = std::make_unique<MockSession>();
  MockSession* mock_session_ptr = mock_session.get();
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(0);
  session_config.GetMutableStopTokenIds().push_back({1});
  *session_config.GetMutableLlmModelType().mutable_gemma3() = {};
  EXPECT_CALL(*mock_session_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config));
  EXPECT_CALL(*mock_session_ptr, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer_));

  // Set up mock Engine.
  auto mock_engine = std::make_unique<MockEngine>();
  EXPECT_CALL(*mock_engine, CreateSession(testing::_))
      .WillOnce(testing::Return(std::move(mock_session)));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  EXPECT_CALL(*mock_engine, GetEngineSettings())
      .WillRepeatedly(testing::ReturnRef(engine_settings));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // The first user message.
  JsonMessage user_message_1 = nlohmann::ordered_json::parse(R"json(
    {
      "role": "user",
      "content": "How are you?"
    }
  )json");
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
  JsonMessage user_messages = nlohmann::ordered_json::parse(R"json(
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
  )json");
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
  JsonMessage assistant_message_1 = nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "I am good."
      }
    ]
  })");
  JsonMessage assistant_message_2 = nlohmann::ordered_json::parse(R"({
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "baz"
      }
    ]
  })");
  EXPECT_THAT(conversation->GetHistory(),
              testing::ElementsAre(user_message_1, assistant_message_1,
                                   user_messages[0], user_messages[1],
                                   assistant_message_2));
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

  JsonMessage user_message = {{"role", "user"}, {"content", "Hello world!"}};
  // The expected message is just some gibberish text, because the test LLM has
  // random weights.
  Message expected_message =
      JsonMessage({{"role", "assistant"},
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
  auto mock_session = std::make_unique<MockSession>();
  MockSession* mock_session_ptr = mock_session.get();
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(0);
  session_config.GetMutableStopTokenIds().push_back({1});
  *session_config.GetMutableLlmModelType().mutable_gemma3() = {};
  EXPECT_CALL(*mock_session_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config));
  EXPECT_CALL(*mock_session_ptr, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer_));

  // Set up mock Engine.
  auto mock_engine = std::make_unique<MockEngine>();
  EXPECT_CALL(*mock_engine, CreateSession(testing::_))
      .WillOnce(testing::Return(std::move(mock_session)));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  EXPECT_CALL(*mock_engine, GetEngineSettings())
      .WillRepeatedly(testing::ReturnRef(engine_settings));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send a single message.
  JsonMessage user_message = {{"role", "user"}, {"content", "How are you?"}};

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

  Message assistant_message = JsonMessage(nlohmann::ordered_json::parse(R"({
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

TEST_P(ConversationTest, SendMultipleMessagesAsync) {
  // Set up mock Session.
  auto mock_session = std::make_unique<MockSession>();
  MockSession* mock_session_ptr = mock_session.get();
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(0);
  session_config.GetMutableStopTokenIds().push_back({1});
  *session_config.GetMutableLlmModelType().mutable_gemma3() = {};
  EXPECT_CALL(*mock_session_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config));
  EXPECT_CALL(*mock_session_ptr, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer_));

  // Set up mock Engine.
  auto mock_engine = std::make_unique<MockEngine>();
  EXPECT_CALL(*mock_engine, CreateSession(testing::_))
      .WillOnce(testing::Return(std::move(mock_session)));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  EXPECT_CALL(*mock_engine, GetEngineSettings())
      .WillRepeatedly(testing::ReturnRef(engine_settings));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetPrefillPrefaceOnInit(prefill_preface_on_init_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // We will send two consecutive messages.
  JsonMessage user_messages = nlohmann::ordered_json::parse(R"json(
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
  )json");

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

  Message assistant_message = JsonMessage(nlohmann::ordered_json::parse(R"json({
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
  auto mock_session = std::make_unique<MockSession>();
  MockSession* mock_session_ptr = mock_session.get();
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(0);
  session_config.GetMutableStopTokenIds().push_back({1});
  *session_config.GetMutableLlmModelType().mutable_gemma3() = {};
  EXPECT_CALL(*mock_session_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config));
  EXPECT_CALL(*mock_session_ptr, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer_));

  // Set up mock Engine.
  auto mock_engine = std::make_unique<MockEngine>();
  EXPECT_CALL(*mock_engine, CreateSession(testing::_))
      .WillOnce(testing::Return(std::move(mock_session)));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  EXPECT_CALL(*mock_engine, GetEngineSettings())
      .WillRepeatedly(testing::ReturnRef(engine_settings));

  // Create Conversation.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
          .SetEnableConstrainedDecoding(enable_constrained_decoding_)
          .SetOverwritePromptTemplate(PromptTemplate(kTestJinjaPromptTemplate))
          .Build(*mock_engine));
  ASSERT_OK_AND_ASSIGN(auto conversation,
                       Conversation::Create(*mock_engine, conversation_config));

  // The first user message.
  JsonMessage user_message_1 = nlohmann::ordered_json::parse(R"json(
    {
      "role": "user",
      "content": "How are you?"
    }
  )json");
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

  Message assistant_message_1 =
      JsonMessage(nlohmann::ordered_json::parse(R"json({
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
  JsonMessage user_messages = nlohmann::ordered_json::parse(R"json(
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
  )json");

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

  Message assistant_message_2 =
      JsonMessage(nlohmann::ordered_json::parse(R"json({
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
                       conversation->SendMessage(JsonMessage{
                           {"role", "user"}, {"content", "Hello world!"}}));
  // The expected message is just some gibberish text, because the test LLM has
  // random weights.
  JsonMessage expected_message;
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
  const JsonMessage& json_message = std::get<JsonMessage>(message);
  EXPECT_EQ(json_message, expected_message);
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
                       conversation->SendMessage(JsonMessage{
                           {"role", "user"}, {"content", "Hello world!"}}));
  ASSERT_OK_AND_ASSIGN(const BenchmarkInfo benchmark_info_1,
                       conversation->GetBenchmarkInfo());
  EXPECT_EQ(benchmark_info_1.GetTotalPrefillTurns(),
            prefill_preface_on_init_ ? 2 : 1);

  ASSERT_OK_AND_ASSIGN(const Message message_2,
                       conversation->SendMessage(JsonMessage{
                           {"role", "user"}, {"content", "Hello world!"}}));
  ASSERT_OK_AND_ASSIGN(const BenchmarkInfo benchmark_info_2,
                       conversation->GetBenchmarkInfo());
  EXPECT_EQ(benchmark_info_2.GetTotalPrefillTurns(),
            prefill_preface_on_init_ ? 3 : 2);
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
    if (auto json_message = std::get_if<JsonMessage>(&message.value());
        json_message->is_null()) {
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
  JsonMessage user_message = {{"role", "user"}, {"content", "Hello world!"}};
  Message expected_assistant_message =
      JsonMessage({{"role", "assistant"},
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
              testing::VariantWith<JsonMessage>(std::get<JsonMessage>(
                  expected_assistant_message_for_confirm)));

  // Access the history with visitor function, and copy the last message.
  Message last_message;
  conversation->AccessHistory(
      [&last_message](const std::vector<Message>& history_view) {
        // Copy the last message to last_message. So we don't need to
        // copy the whole history, if we only need the last message.
        last_message = history_view.back();
      });
  EXPECT_THAT(last_message,
              testing::VariantWith<JsonMessage>(std::get<JsonMessage>(
                  expected_assistant_message_for_confirm)));
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
      ->SendMessageAsync(
          JsonMessage{{"role", "user"}, {"content", "Hello world!"}},
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
      ->SendMessageAsync(
          JsonMessage{{"role", "user"}, {"content", "Hello world!"}},
          CreateCancelledMessageCallback(status, done_2))
      .IgnoreError();
  EXPECT_OK(status);
  // Wait for the callback to be done.
  done_2.WaitForNotificationWithTimeout(absl::Seconds(10));
  // Without cancellation, the history should have two messages, user and
  // assistant.
  auto history = conversation->GetHistory();
  ASSERT_EQ(history.size(), 2);
  EXPECT_THAT(history[0], testing::VariantWith<JsonMessage>(JsonMessage{
                              {"role", "user"}, {"content", "Hello world!"}}));
  // TODO(b/450903294) - Because the cancellation is not fully rollbacked, the
  // assistant message content depends on at which step the cancellation is
  // triggered, and that is non-deterministic. Here we only check the role is
  // assistant.
  EXPECT_THAT(std::holds_alternative<JsonMessage>(history[1]),
              testing::IsTrue());
  EXPECT_EQ(std::get<JsonMessage>(history[1])["role"], "assistant");

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
  auto mock_session = std::make_unique<MockSession>();
  MockSession* mock_session_ptr = mock_session.get();
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(0);
  session_config.GetMutableStopTokenIds().push_back({1});
  *session_config.GetMutableLlmModelType().mutable_gemma3() = {};
  EXPECT_CALL(*mock_session_ptr, GetSessionConfig())
      .WillRepeatedly(testing::ReturnRef(session_config));
  EXPECT_CALL(*mock_session_ptr, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(*tokenizer_));

  // Set up mock Engine.
  auto mock_engine = std::make_unique<MockEngine>();
  EXPECT_CALL(*mock_engine, CreateSession(testing::_))
      .WillOnce(testing::Return(std::move(mock_session)));
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(GetTestdataPath(kTestLlmPath)));
  ASSERT_OK_AND_ASSIGN(auto engine_settings, EngineSettings::CreateDefault(
                                                 model_assets, Backend::CPU));
  EXPECT_CALL(*mock_engine, GetEngineSettings())
      .WillRepeatedly(testing::ReturnRef(engine_settings));

  // Create Conversation with ExternalConstraintConfig.
  ASSERT_OK_AND_ASSIGN(
      auto conversation_config,
      ConversationConfig::Builder()
          .SetSessionConfig(session_config)
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
  JsonMessage user_message = {{"role", "user"}, {"content", "How are you?"}};

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
  auto tokenizer = SentencePieceTokenizer::CreateFromFile(
      (std::filesystem::path(::testing::SrcDir()) / kTestTokenizerPath)
          .string());
  ASSERT_OK(tokenizer);
  EXPECT_CALL(*mock_session_ptr, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(**tokenizer));

  // Set up mock Engine.
  auto mock_engine = std::make_unique<MockEngine>();
  EXPECT_CALL(*mock_engine, CreateSession(testing::_))
      .WillOnce(testing::Return(std::move(mock_session)));
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
      JsonMessage{{"role", "user"}, {"content", "Hello world!"}},
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
      JsonMessage{{"role", "user"}, {"content", " This is a long message."}},
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
      JsonMessage{{"role", "user"}, {"content", " continuing..."}},
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
      conversation->SendMessage(JsonMessage{
          {"role", "user"}, {"content", " The message is ended."}}));
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
  auto tokenizer = SentencePieceTokenizer::CreateFromFile(
      (std::filesystem::path(::testing::SrcDir()) / kTestTokenizerPath)
          .string());
  ASSERT_OK(tokenizer);
  EXPECT_CALL(*mock_session_ptr, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(**tokenizer));

  // Set up mock Engine.
  auto mock_engine = std::make_unique<MockEngine>();
  EXPECT_CALL(*mock_engine, CreateSession(testing::_))
      .WillOnce(testing::Return(std::move(mock_session)));
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
      JsonMessage{{"role", "user"}, {"content", "Hello world!"}},
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
      JsonMessage{{"role", "user"}, {"content", " This is a long message."}},
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
      JsonMessage{{"role", "user"}, {"content", " continuing..."}},
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
      JsonMessage{{"role", "user"}, {"content", " The message is ended."}},
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
      JsonMessage({{"role", "assistant"},
                   {"content", {{{"type", "text"}, {"text", "I am good."}}}}});
  absl::Notification done5;
  // Trigger the decode by sending an empty message.
  EXPECT_OK(conversation->SendMessageAsync(
      JsonMessage{{"role", "user"}, {"content", ""}},
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
  auto tokenizer = SentencePieceTokenizer::CreateFromFile(
      (std::filesystem::path(::testing::SrcDir()) / kTestTokenizerPath)
          .string());
  ASSERT_OK(tokenizer);
  EXPECT_CALL(*mock_session_ptr, GetTokenizer())
      .WillRepeatedly(testing::ReturnRef(**tokenizer));

  // Set up mock Engine.
  auto mock_engine = std::make_unique<MockEngine>();
  EXPECT_CALL(*mock_engine, CreateSession(testing::_))
      .WillOnce(testing::Return(std::move(mock_session)));
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
      JsonMessage{{"role", "user"}, {"content", "Hello world!"}},
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
      JsonMessage{{"role", "model"}, {"content", "Nice to meet you."}},
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
      JsonMessage{{"role", "model"}, {"content", " How can I help you today?"}},
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
      JsonMessage{{"role", "model"}, {"content", " The message is ended."}},
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
  ASSERT_OK(
      conversation->SendMessage(JsonMessage{{"role", "tool"},
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
  ASSERT_OK(
      conversation->SendMessage(JsonMessage{{"role", "tool"},
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
