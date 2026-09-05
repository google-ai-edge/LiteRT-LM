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

#include "omni/base/litert_lm_engine_runner.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"

namespace litert::omni {
namespace {

using ::absl_testing::IsOk;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Return;

class MockSession : public lm::SessionInterface {
 public:
  MockSession() {
    ON_CALL(*this, GetSessionConfig).WillByDefault(testing::ReturnRef(config_));
  }
  MOCK_METHOD(absl::StatusOr<lm::Responses>, GenerateContent,
              (const std::vector<lm::InputData>& contents), (override));
  MOCK_METHOD(
      absl::Status, GenerateContentStream,
      (const std::vector<lm::InputData>& contents,
       absl::AnyInvocable<void(absl::StatusOr<lm::Responses>)> callback),
      (override));
  MOCK_METHOD(absl::Status, GenerateContentStream,
              (const std::vector<lm::InputData>& contents,
               absl::AnyInvocable<void(absl::StatusOr<lm::Responses>)> callback,
               const lm::DecodeConfig& decode_config),
              (override));
  MOCK_METHOD(absl::StatusOr<lm::Responses>, RunTextScoring,
              (const std::vector<absl::string_view>& target_text,
               bool store_token_lengths),
              (override));
  MOCK_METHOD(absl::Status, RunPrefill,
              (const std::vector<lm::InputData>& contents), (override));
  MOCK_METHOD(absl::StatusOr<lm::Responses>, RunDecode, (), (override));
  MOCK_METHOD(absl::StatusOr<lm::Responses>, RunDecode,
              (const lm::DecodeConfig& decode_config), (override));
  MOCK_METHOD(absl::StatusOr<lm::BenchmarkInfo>, GetBenchmarkInfo, (),
              (override));
  MOCK_METHOD(absl::StatusOr<lm::BenchmarkInfo*>, GetMutableBenchmarkInfo, (),
              (override));
  MOCK_METHOD(absl::Status, WaitUntilDone, (), (override));
  MOCK_METHOD(const lm::SessionConfig&, GetSessionConfig, (),
              (const, override));
  MOCK_METHOD(absl::Status, RewindToStep, (int step), (override));

 private:
  lm::SessionConfig config_ = lm::SessionConfig::CreateDefault();
};

TEST(LiteRtLmEngineRunnerImplTest, PrefillPassesInputsToSession) {
  auto mock_session = std::make_unique<MockSession>();
  auto* session_ptr = mock_session.get();

  EXPECT_CALL(*session_ptr, RunPrefill(_))
      .WillOnce([](const std::vector<lm::InputData>& contents) {
        EXPECT_EQ(contents.size(), 2);
        return absl::OkStatus();
      });

  lm::SessionConfig config = lm::SessionConfig::CreateDefault();
  LiteRtLmEngineRunnerImpl runner(
      /*engine=*/nullptr, std::move(mock_session), config,
      /*model_resources=*/nullptr);

  std::vector<lm::InputData> inputs;
  inputs.emplace_back(lm::InputText("prompt"));
  inputs.emplace_back(lm::InputAudioEnd());

  EXPECT_THAT(runner.Prefill(std::move(inputs)), IsOk());
}

TEST(LiteRtLmEngineRunnerImplTest, PrefillFailsOnEmptyInputs) {
  auto mock_session = std::make_unique<MockSession>();
  lm::SessionConfig config = lm::SessionConfig::CreateDefault();
  LiteRtLmEngineRunnerImpl runner(
      /*engine=*/nullptr, std::move(mock_session), config,
      /*model_resources=*/nullptr);

  EXPECT_FALSE(runner.Prefill({}).ok());
}

TEST(LiteRtLmEngineRunnerImplTest, DecodeCallsSessionRunDecode) {
  auto mock_session = std::make_unique<MockSession>();
  auto* session_ptr = mock_session.get();

  lm::Responses responses(lm::TaskState::kDone,
                          /*response_texts=*/{"test"},
                          /*scores=*/{1.0f},
                          /*token_lengths=*/{},
                          /*token_ids=*/{{10, 20, 30}});

  EXPECT_CALL(*session_ptr, RunDecode(testing::_)).WillOnce(Return(responses));

  lm::SessionConfig config = lm::SessionConfig::CreateDefault();
  LiteRtLmEngineRunnerImpl runner(
      /*engine=*/nullptr, std::move(mock_session), config,
      /*model_resources=*/nullptr);

  auto responses_or = runner.Decode(lm::DecodeConfig::CreateDefault());
  ASSERT_THAT(responses_or, IsOk());
  ASSERT_FALSE(responses_or->GetTokenIds().empty());
  EXPECT_THAT(responses_or->GetTokenIds()[0], ElementsAre(10, 20, 30));
}

TEST(LiteRtLmEngineRunnerImplTest, ResetRewindsSession) {
  auto mock_session = std::make_unique<MockSession>();
  auto* session_ptr = mock_session.get();

  EXPECT_CALL(*session_ptr, RewindToStep(0)).WillOnce(Return(absl::OkStatus()));

  lm::SessionConfig config = lm::SessionConfig::CreateDefault();
  LiteRtLmEngineRunnerImpl runner(
      /*engine=*/nullptr, std::move(mock_session), config,
      /*model_resources=*/nullptr);

  EXPECT_THAT(runner.Reset(), IsOk());
}

}  // namespace
}  // namespace litert::omni
