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

#include "runtime/components/top_k_distribution_buffer.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/util/convert_tensor_buffer.h"

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

TEST(TopKDistributionBufferTest, GetSizeReturnsLastDimension) {
  auto probs =
      CopyToTensorBuffer<float>(std::vector<float>(2 * 3 * 4), {2, 3, 4});
  auto ids = CopyToTensorBuffer<int>(std::vector<int>(2 * 3 * 4), {2, 3, 4});
  ASSERT_TRUE(probs.HasValue());
  ASSERT_TRUE(ids.HasValue());
  ASSERT_OK_AND_ASSIGN(
      auto k, GetTopKDistributionBufferSize(*probs, *ids, /*batch_size=*/2,
                                            /*sequence_size=*/3));
  EXPECT_EQ(k, 4);
}

TEST(TopKDistributionBufferTest, GetSizeRejectsMalformedBuffers) {
  auto probs = CopyToTensorBuffer<float>(std::vector<float>(8), {1, 2, 4});
  auto ids = CopyToTensorBuffer<int>(std::vector<int>(8), {1, 2, 4});
  auto ids_k2 = CopyToTensorBuffer<int>(std::vector<int>(4), {1, 2, 2});
  auto float_ids = CopyToTensorBuffer<float>(std::vector<float>(8), {1, 2, 4});
  ASSERT_TRUE(probs.HasValue() && ids.HasValue() && ids_k2.HasValue() &&
              float_ids.HasValue());

  // Wrong number of rows.
  EXPECT_EQ(GetTopKDistributionBufferSize(*probs, *ids, /*batch_size=*/1,
                                          /*sequence_size=*/3)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  // Mismatched K.
  EXPECT_THAT(GetTopKDistributionBufferSize(*probs, *ids_k2, /*batch_size=*/1,
                                            /*sequence_size=*/2)
                  .status()
                  .message(),
              HasSubstr("same K"));
  // Wrong element type.
  EXPECT_THAT(GetTopKDistributionBufferSize(*probs, *float_ids,
                                            /*batch_size=*/1,
                                            /*sequence_size=*/2)
                  .status()
                  .message(),
              HasSubstr("element type"));
}

TEST(TopKDistributionBufferTest, WritesValidEntriesAndPads) {
  // Two rows with src_k = 3, of which the first 2 entries are valid.
  const std::vector<float> src_probs = {0.7f, 0.3f, 9.0f, 0.6f, 0.4f, 9.0f};
  const std::vector<int> src_ids = {5, 2, 99, 1, 7, 99};
  auto probs = CopyToTensorBuffer<float>(std::vector<float>(8), {1, 2, 4});
  auto ids = CopyToTensorBuffer<int>(std::vector<int>(8), {1, 2, 4});
  ASSERT_TRUE(probs.HasValue() && ids.HasValue());

  ASSERT_TRUE(WriteTopKDistribution(src_probs, src_ids, /*src_k=*/3,
                                    /*num_valid=*/2, /*batch_size=*/1,
                                    /*sequence_size=*/2, *probs, *ids)
                  .ok());

  auto out_probs = CopyFromTensorBuffer<float>(*probs);
  auto out_ids = CopyFromTensorBuffer<int>(*ids);
  ASSERT_TRUE(out_probs.HasValue() && out_ids.HasValue());
  EXPECT_THAT(*out_probs,
              ElementsAre(0.7f, 0.3f, 0.0f, 0.0f, 0.6f, 0.4f, 0.0f, 0.0f));
  EXPECT_THAT(*out_ids, ElementsAre(5, 2, -1, -1, 1, 7, -1, -1));
}

TEST(TopKDistributionBufferTest, RejectsOutputSmallerThanDistribution) {
  const std::vector<float> src_probs = {0.5f, 0.3f, 0.2f};
  const std::vector<int> src_ids = {0, 1, 2};
  auto probs = CopyToTensorBuffer<float>(std::vector<float>(2), {1, 1, 2});
  auto ids = CopyToTensorBuffer<int>(std::vector<int>(2), {1, 1, 2});
  ASSERT_TRUE(probs.HasValue() && ids.HasValue());

  const absl::Status status = WriteTopKDistribution(
      src_probs, src_ids, /*src_k=*/3, /*num_valid=*/3, /*batch_size=*/1,
      /*sequence_size=*/1, *probs, *ids);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("less than"));
}

TEST(TopKDistributionBufferTest, RejectsInconsistentSource) {
  auto probs = CopyToTensorBuffer<float>(std::vector<float>(4), {1, 1, 4});
  auto ids = CopyToTensorBuffer<int>(std::vector<int>(4), {1, 1, 4});
  ASSERT_TRUE(probs.HasValue() && ids.HasValue());
  const std::vector<float> src_probs = {0.5f, 0.5f};
  const std::vector<int> src_ids = {0};
  EXPECT_EQ(WriteTopKDistribution(src_probs, src_ids, /*src_k=*/2,
                                  /*num_valid=*/2, /*batch_size=*/1,
                                  /*sequence_size=*/1, *probs, *ids)
                .code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace litert::lm
