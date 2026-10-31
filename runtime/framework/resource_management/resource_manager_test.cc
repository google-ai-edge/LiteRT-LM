// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/framework/resource_management/resource_manager.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/executor/executor_stats.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

// A fake executor that reports executor stats and counts stat resets.
class StatsFakeLlmExecutor : public FakeLlmExecutor {
 public:
  explicit StatsFakeLlmExecutor(int* num_resets)
      : FakeLlmExecutor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                        /*decode_tokens_set=*/{}),
        num_resets_(num_resets) {
    stats_.AccumulatePlainStep(absl::Milliseconds(10));
  }

  std::optional<ExecutorStats> GetExecutorStats() const override {
    return stats_;
  }
  void ResetExecutorStats() override { ++*num_resets_; }

 private:
  ExecutorStats stats_{.module_name = std::string(kLlmModuleName)};
  int* num_resets_;
};

// Tasks::Decode receives the `LockedLlmExecutor` wrapper rather than the
// underlying executor, so the stats APIs must be forwarded through it.
TEST(ResourceManagerTest, AcquiredExecutorForwardsExecutorStats) {
  int num_resets = 0;
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<StatsFakeLlmExecutor>(&num_resets),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmExecutor> executor,
                       resource_manager->AcquireExecutor());

  std::optional<ExecutorStats> stats = executor->GetExecutorStats();
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(stats->module_name, kLlmModuleName);
  EXPECT_EQ(stats->plain_steps(), 1);

  executor->ResetExecutorStats();
  EXPECT_EQ(num_resets, 1);
}

// A fake executor that reports a given draft verification support and accepts
// every draft that it verifies.
class DraftVerificationFakeLlmExecutor : public FakeLlmExecutor {
 public:
  explicit DraftVerificationFakeLlmExecutor(absl::Status support)
      : FakeLlmExecutor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                        /*decode_tokens_set=*/{}),
        support_(std::move(support)) {}

  absl::Status CheckDraftVerificationSupport() const override {
    return support_;
  }

  absl::StatusOr<DraftVerificationResult> VerifyDraft(
      absl::Span<const int> draft_token_ids,
      const ExecutorDecodeParams& decode_params) override {
    return DraftVerificationResult{
        .tokens =
            std::vector<int>(draft_token_ids.begin(), draft_token_ids.end()),
        .num_accepted = static_cast<int>(draft_token_ids.size())};
  }

 private:
  absl::Status support_;
};

TEST(ResourceManagerTest, AcquiredExecutorForwardsDraftVerificationSupport) {
  for (const absl::Status& support :
       {absl::OkStatus(), absl::FailedPreconditionError("No verify.")}) {
    SCOPED_TRACE(support.ToString());
    ASSERT_OK_AND_ASSIGN(
        auto resource_manager,
        ResourceManager::Create(
            /*model_resources=*/nullptr,
            std::make_unique<DraftVerificationFakeLlmExecutor>(support),
            /*vision_executor_settings=*/nullptr,
            /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmExecutor> executor,
                         resource_manager->AcquireExecutor());
    EXPECT_EQ(executor->CheckDraftVerificationSupport(), support);
  }
}

TEST(ResourceManagerTest, AcquiredExecutorForwardsDraftVerification) {
  ASSERT_OK_AND_ASSIGN(
      auto resource_manager,
      ResourceManager::Create(
          /*model_resources=*/nullptr,
          std::make_unique<DraftVerificationFakeLlmExecutor>(absl::OkStatus()),
          /*vision_executor_settings=*/nullptr,
          /*audio_executor_settings=*/nullptr, /*litert_env=*/nullptr));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmExecutor> executor,
                       resource_manager->AcquireExecutor());
  EXPECT_THAT(executor->VerifyDraft({3, 1, 4}, ExecutorDecodeParams()),
              IsOkAndHolds(FieldsAre(ElementsAre(3, 1, 4), 3)));
}

TEST(ResourceManagerTest, ExecutorsDontSupportDraftVerificationByDefault) {
  FakeLlmExecutor executor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                           /*decode_tokens_set=*/{});
  EXPECT_THAT(executor.CheckDraftVerificationSupport(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(executor.VerifyDraft({3, 1, 4}, ExecutorDecodeParams()),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

// A fake executor that claims draft verification support, but doesn't
// implement it.
class SupportOnlyFakeLlmExecutor : public FakeLlmExecutor {
 public:
  SupportOnlyFakeLlmExecutor()
      : FakeLlmExecutor(/*vocab_size=*/16, /*prefill_tokens_set=*/{},
                        /*decode_tokens_set=*/{}) {}

  absl::Status CheckDraftVerificationSupport() const override {
    return absl::OkStatus();
  }
};

TEST(ResourceManagerTest, ExecutorsDontVerifyDraftsByDefault) {
  SupportOnlyFakeLlmExecutor executor;
  EXPECT_THAT(executor.VerifyDraft({3, 1, 4}, ExecutorDecodeParams()),
              StatusIs(absl::StatusCode::kUnimplemented));
}

}  // namespace
}  // namespace litert::lm
