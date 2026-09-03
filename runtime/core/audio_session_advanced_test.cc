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

#include "runtime/core/audio_session_advanced.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/core/session_advanced.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/audio/audio_executor.h"
#include "runtime/executor/audio/audio_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/framework/resource_management/threaded_execution_manager.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

constexpr int kValidTokensInAudioData = 4;
constexpr int kVocabSize = 100;

class MockTokenizer : public support::Tokenizer {
 public:
  MOCK_METHOD(absl::StatusOr<std::vector<int>>, TextToTokenIds,
              (absl::string_view text), (override));
  MOCK_METHOD(absl::StatusOr<int>, TokenToId, (absl::string_view token),
              (override));
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText,
              (absl::Span<const int> token_ids, bool skip_special_tokens),
              (override));
  MOCK_METHOD(support::TokenizerType, GetTokenizerType, (), (const, override));
  MOCK_METHOD(std::vector<std::string>, GetTokens, (), (const, override));
  MOCK_METHOD(int, GetVocabSize, (), (const, override));
};

class FakeAudioExecutor : public AudioExecutor {
 public:
  absl::StatusOr<ExecutorAudioData> Encode(
      const TensorBuffer& spectrogram_tensor) override {
    ExecutorAudioData data;
    data.SetValidTokens(kValidTokensInAudioData);
    return data;
  }

  absl::Status Reset() override {
    reset_called_ = true;
    return absl::OkStatus();
  }

  absl::StatusOr<ExecutorAudioData> Flush() override {
    flush_called_ = true;
    ExecutorAudioData data;
    data.SetValidTokens(kValidTokensInAudioData);
    return data;
  }

  bool reset_called_ = false;
  bool flush_called_ = false;
};

class AudioSessionAdvancedTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tokenizer_ = std::make_unique<MockTokenizer>();
    EXPECT_CALL(*tokenizer_, GetVocabSize())
        .WillRepeatedly(testing::Return(kVocabSize));

    auto fake_llm_executor = std::make_unique<FakeLlmExecutor>(
        kVocabSize,
        /*prefill_tokens=*/std::vector<std::vector<int>>{{1, 2, 3}},
        /*decode_tokens=*/std::vector<std::vector<int>>{{4}, {5}, {6}});
    ASSERT_OK_AND_ASSIGN(auto* settings,
                         fake_llm_executor->GetMutableExecutorSettings());
    EXPECT_OK(settings->SetBackend(Backend::GPU_ARTISAN));

    ASSERT_OK_AND_ASSIGN(auto model_assets,
                         ModelAssets::Create("test_model_path_audio"));
    ASSERT_OK_AND_ASSIGN(auto audio_settings,
                         AudioExecutorSettings::CreateDefault(
                             model_assets, 128, Backend::GPU_ARTISAN));

    auto fake_audio_executor = std::make_unique<FakeAudioExecutor>();
    fake_audio_executor_ = fake_audio_executor.get();

    ASSERT_OK_AND_ASSIGN(
        execution_manager_,
        ThreadedExecutionManager::Create(
            tokenizer_.get(), /*model_resources=*/nullptr,
            std::move(fake_llm_executor),
            /*vision_executor_settings=*/nullptr,
            std::make_unique<AudioExecutorSettings>(std::move(audio_settings)),
            /*litert_env=*/nullptr, std::move(fake_audio_executor)));
  }

  std::unique_ptr<MockTokenizer> tokenizer_;
  FakeAudioExecutor* fake_audio_executor_ = nullptr;
  std::shared_ptr<ThreadedExecutionManager> execution_manager_;

  SessionConfig CreateAudioSessionConfig() {
    SessionConfig config = SessionConfig::CreateDefault();
    config.SetAudioModalityEnabled(true);
    config.SetEnableAudioSessionAdvanced(true);
    return config;
  }
};

TEST_F(AudioSessionAdvancedTest, FromSessionNullptrReturnsError) {
  EXPECT_THAT(AudioSessionAdvanced::FromSession(
                  static_cast<std::unique_ptr<SessionInterface>>(nullptr)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(AudioSessionAdvancedTest, FromSessionNonAudioSessionReturnsError) {
  SessionConfig config = SessionConfig::CreateDefault();
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SessionAdvanced::Create(execution_manager_, tokenizer_.get(), config,
                              /*benchmark_info=*/std::nullopt));
  EXPECT_THAT(AudioSessionAdvanced::FromSession(std::move(session)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(AudioSessionAdvancedTest, CreateAndFromSessionSuccess) {
  SessionConfig config = CreateAudioSessionConfig();
  ASSERT_OK_AND_ASSIGN(
      auto audio_session,
      AudioSessionAdvanced::Create(execution_manager_, tokenizer_.get(), config,
                                   /*benchmark_info=*/std::nullopt));

  std::unique_ptr<SessionInterface> session_interface =
      std::move(audio_session);
  ASSERT_OK_AND_ASSIGN(
      auto unique_audio_session,
      AudioSessionAdvanced::FromSession(std::move(session_interface)));
  EXPECT_NE(unique_audio_session, nullptr);
}

TEST_F(AudioSessionAdvancedTest, EncodeAudioSucceeds) {
  SessionConfig config = CreateAudioSessionConfig();
  ASSERT_OK_AND_ASSIGN(
      auto audio_session,
      AudioSessionAdvanced::Create(execution_manager_, tokenizer_.get(), config,
                                   /*benchmark_info=*/std::nullopt));

  const std::vector<float> kSpectrogramData = {1.0f, 2.0f, 3.0f, 4.0f};
  auto tensor_or = CopyToTensorBuffer<float>(kSpectrogramData, {1, 4});
  ASSERT_TRUE(tensor_or.HasValue());

  ASSERT_OK_AND_ASSIGN(auto audio_data, audio_session->EncodeAudio(*tensor_or));
  EXPECT_EQ(audio_data.GetValidTokens(), kValidTokensInAudioData);
}

TEST_F(AudioSessionAdvancedTest, ResetAudioSucceeds) {
  SessionConfig config = CreateAudioSessionConfig();
  ASSERT_OK_AND_ASSIGN(
      auto audio_session,
      AudioSessionAdvanced::Create(execution_manager_, tokenizer_.get(), config,
                                   /*benchmark_info=*/std::nullopt));

  EXPECT_OK(audio_session->ResetAudio());
  EXPECT_TRUE(fake_audio_executor_->reset_called_);
}

TEST_F(AudioSessionAdvancedTest, FlushAudioSucceeds) {
  SessionConfig config = CreateAudioSessionConfig();
  ASSERT_OK_AND_ASSIGN(
      auto audio_session,
      AudioSessionAdvanced::Create(execution_manager_, tokenizer_.get(), config,
                                   /*benchmark_info=*/std::nullopt));

  ASSERT_OK_AND_ASSIGN(auto audio_data, audio_session->FlushAudio());
  EXPECT_EQ(audio_data.GetValidTokens(), kValidTokensInAudioData);
  EXPECT_TRUE(fake_audio_executor_->flush_called_);
}

TEST_F(AudioSessionAdvancedTest, CloneReturnsAudioSessionAdvanced) {
  SessionConfig config = CreateAudioSessionConfig();
  ASSERT_OK_AND_ASSIGN(
      auto audio_session,
      AudioSessionAdvanced::Create(execution_manager_, tokenizer_.get(), config,
                                   /*benchmark_info=*/std::nullopt));

  ASSERT_OK_AND_ASSIGN(auto cloned_session, audio_session->Clone());
  ASSERT_OK_AND_ASSIGN(
      auto cloned_audio_session,
      AudioSessionAdvanced::FromSession(std::move(cloned_session)));
  EXPECT_NE(cloned_audio_session, nullptr);
}

}  // namespace
}  // namespace litert::lm
