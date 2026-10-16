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

#include "runtime/components/stop_token_detector.h"

#include <cstddef>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::status::IsOkAndHolds;

TEST(StopTokenDetectorTest, AddStopSequence) {
  StopTokenDetector detector(1);
  EXPECT_TRUE(detector.AddStopTokenSequence({1, 2, 3}).ok());

  // Adding an empty sequence should fail
  EXPECT_EQ(absl::StatusCode::kInvalidArgument,
            detector.AddStopTokenSequence({}).code());

  // Adding a repeated sequence should be a no-op.
  EXPECT_EQ(absl::StatusCode::kOk,
            detector.AddStopTokenSequence({1, 2, 3}).code());

  EXPECT_TRUE(detector.AddStopTokenSequence({9}).ok());
}

TEST(StopTokenDetectorTest, ProcessTokensSingleStopToken) {
  StopTokenDetector detector(2);  // Batch size 2
  EXPECT_OK(detector.AddStopTokenSequence({5}));

  std::vector<int> tokens_item0 = {3, 4, 5, 6, 7};
  std::vector<int> tokens_item1 = {1, 0, 6, 5, 99};

  // Simulate processing token by token
  size_t i;
  for (i = 0; i < tokens_item0.size(); ++i) {
    std::vector<int> current_batch_tokens = {tokens_item0[i], tokens_item1[i]};
    EXPECT_TRUE(
        detector.ProcessTokens(absl::MakeSpan(current_batch_tokens)).ok());
    if (detector.AllDone().value()) {
      break;
    }
  }
  // Stop token, 5, is found for all batch items at step 3.
  EXPECT_EQ(i, 3);

  const auto& steps_before_stop_tokens = detector.GetStepsBeforeStopTokens();
  EXPECT_EQ(2, steps_before_stop_tokens.size());
  // Batch item 0: stop token found at step 2, the current step is 3. So the
  // steps before stop token is 2 = (3 - 2 + 1(# stop tokens)).
  EXPECT_EQ(2, steps_before_stop_tokens[0]);

  // Batch item 1: stop token found at step 3, the current step is 3. So the
  // steps before stop token is  = (3 - 3 + 1(# stop tokens)).
  EXPECT_EQ(1, steps_before_stop_tokens[1]);
}


TEST(StopTokenDetectorTest, ProcessTokensMultipleStopTokens) {
  StopTokenDetector detector(2);  // Batch size 2
  EXPECT_OK(detector.AddStopTokenSequence({5}));
  EXPECT_OK(detector.AddStopTokenSequence({7, 8, 9}));

  std::vector<int> tokens_item0 = {3, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> tokens_item1 = {1, 0, 0, 0, 0, 6, 5, 99};

  // Simulate processing token by token
  size_t i;
  for (i = 0; i < tokens_item0.size(); ++i) {
    std::vector<int> current_batch_tokens = {tokens_item0[i], tokens_item1[i]};
    EXPECT_TRUE(
        detector.ProcessTokens(absl::MakeSpan(current_batch_tokens)).ok());
    if (detector.AllDone().value()) {
      break;
    }
    if (i < 2) {
      EXPECT_EQ(0, detector.MaxPartialStopTokenLength(0));
    } else if (i < 4) {
      EXPECT_EQ(i - 1, detector.MaxPartialStopTokenLength(0));
    } else {
      EXPECT_EQ(3, detector.MaxPartialStopTokenLength(0));
    }
    EXPECT_EQ(0, detector.MaxPartialStopTokenLength(1));
  }
  // Stop tokens are found for all batch items at step 6.
  EXPECT_EQ(i, 6);

  const auto& steps_before_stop_tokens = detector.GetStepsBeforeStopTokens();
  EXPECT_EQ(2, steps_before_stop_tokens.size());

  // Batch item 0: stop token found at step 5, the current step is 6. So the
  // steps before stop token is 4 = (6 - 5 + 3(# stop tokens)).
  EXPECT_EQ(5, steps_before_stop_tokens[0]);

  // Batch item 1: stop token found at step 6, the current step is 6. So the
  // steps before stop token is 1 = (6 - 6 + 1(# stop tokens)).
  EXPECT_EQ(1, steps_before_stop_tokens[1]);
}

TEST(StopTokenDetectorTest, ResetBatch) {
  StopTokenDetector detector(1);
  EXPECT_OK(detector.AddStopTokenSequence({1}));
  std::vector<int> tokens1 = {0, 2, 3, 1, 5};
  size_t i;
  for (i = 0; i < tokens1.size(); ++i) {
    std::vector<int> current_batch_tokens = {tokens1[i]};
    EXPECT_TRUE(
        detector.ProcessTokens(absl::MakeSpan(current_batch_tokens)).ok());
    if (detector.AllDone().value()) {
      break;
    }
  }
  EXPECT_EQ(i, 3);
  detector.ResetBatch();
  // Batch is not done after reset.
  EXPECT_FALSE(detector.AllDone().value());
  EXPECT_EQ(0, detector.GetStepsBeforeStopTokens()[0]);
}

TEST(StopTokenDetectorTest, MarkForRollback) {
  StopTokenDetector detector(1);
  EXPECT_OK(detector.AddStopTokenSequence({5}));
  EXPECT_OK(detector.AddStopTokenSequence({7}));
  EXPECT_OK(detector.AddStopTokenSequence({8, 9}));
  EXPECT_TRUE(detector.MarkForRollback(7));
  // Only single token stop sequences that were added can be marked.
  EXPECT_FALSE(detector.MarkForRollback(8));
  EXPECT_FALSE(detector.MarkForRollback(3));

  std::vector<int> marked_token = {7};
  EXPECT_OK(detector.ProcessTokens(absl::MakeSpan(marked_token)));
  EXPECT_THAT(detector.AllDone(), IsOkAndHolds(true));
  EXPECT_TRUE(detector.EndsWithRollbackToken(0));

  // A token after the stop, such as in benchmark mode, ends the match.
  std::vector<int> next_token = {1};
  EXPECT_OK(detector.ProcessTokens(absl::MakeSpan(next_token)));
  EXPECT_THAT(detector.AllDone(), IsOkAndHolds(true));
  EXPECT_FALSE(detector.EndsWithRollbackToken(0));

  // ResetBatch() clears the stop state but keeps the mark.
  detector.ResetBatch();
  EXPECT_FALSE(detector.EndsWithRollbackToken(0));
  std::vector<int> unmarked_token = {5};
  EXPECT_OK(detector.ProcessTokens(absl::MakeSpan(unmarked_token)));
  EXPECT_THAT(detector.AllDone(), IsOkAndHolds(true));
  EXPECT_FALSE(detector.EndsWithRollbackToken(0));

  detector.ResetBatch();
  EXPECT_OK(detector.ProcessTokens(absl::MakeSpan(marked_token)));
  EXPECT_TRUE(detector.EndsWithRollbackToken(0));
}

TEST(StopTokenDetectorTest, EndsWithRollbackTokenPerBatchItem) {
  StopTokenDetector detector(2);  // Batch size 2
  EXPECT_OK(detector.AddStopTokenSequence({5}));
  EXPECT_OK(detector.AddStopTokenSequence({7}));
  EXPECT_TRUE(detector.MarkForRollback(7));

  std::vector<int> tokens = {7, 5};
  EXPECT_OK(detector.ProcessTokens(absl::MakeSpan(tokens)));
  EXPECT_THAT(detector.AllDone(), IsOkAndHolds(true));
  EXPECT_TRUE(detector.EndsWithRollbackToken(0));
  EXPECT_FALSE(detector.EndsWithRollbackToken(1));
}

}  // namespace
}  // namespace litert::lm
