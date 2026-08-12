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

#include "runtime/components/logits_processor/logits_processor_pipeline.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/logits_processor/logits_processor.h"
#include "runtime/components/logits_processor/mock_logits_processor.h"
#include "runtime/components/logits_processor/repetition_penalty_config.h"
#include "runtime/components/logits_processor/suppress_tokens_config.h"
#include "runtime/util/convert_tensor_buffer.h"

namespace litert::lm {
namespace {

using ::testing::_;
using ::testing::Return;

TEST(LogitsProcessorPipelineTest, EmptyPipelineDoesNothing) {
  LogitsProcessorPipeline pipeline;
  EXPECT_TRUE(pipeline.empty());
  EXPECT_EQ(pipeline.size(), 0);

  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CreateTensorBuffer<float>({1, 1, 10}));
  EXPECT_OK(pipeline.ProcessLogits(tensor_buffer));
}

TEST(LogitsProcessorPipelineTest, ProcessorsExecution) {
  auto mock_processor1 = std::make_unique<MockLogitsProcessor>();
  auto mock_processor2 = std::make_unique<MockLogitsProcessor>();

  EXPECT_CALL(*mock_processor1,
              ProcessLogits(testing::An<::litert::TensorBuffer&>()))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_processor2,
              ProcessLogits(testing::An<::litert::TensorBuffer&>()))
      .WillOnce(Return(absl::OkStatus()));

  std::vector<std::unique_ptr<LogitsProcessor>> processors;
  processors.push_back(std::move(mock_processor1));
  processors.push_back(std::move(mock_processor2));
  LogitsProcessorPipeline pipeline(std::move(processors));
  EXPECT_FALSE(pipeline.empty());
  EXPECT_EQ(pipeline.size(), 2);

  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_buffer,
                              CreateTensorBuffer<float>({1, 1, 10}));
  EXPECT_OK(pipeline.ProcessLogits(tensor_buffer));
}

TEST(LogitsProcessorPipelineTest, UpdateStatePropagatesToAllProcessors) {
  auto mock_processor1 = std::make_unique<MockLogitsProcessor>();
  auto mock_processor2 = std::make_unique<MockLogitsProcessor>();

  EXPECT_CALL(*mock_processor1,
              UpdateState(testing::An<absl::Span<const int>>()))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*mock_processor2,
              UpdateState(testing::An<absl::Span<const int>>()))
      .WillOnce(Return(absl::OkStatus()));

  LogitsProcessorPipeline pipeline;
  pipeline.AddProcessor(std::move(mock_processor1));
  pipeline.AddProcessor(std::move(mock_processor2));

  std::vector<int> tokens = {42};
  EXPECT_OK(pipeline.UpdateState(absl::MakeSpan(tokens)));
}

TEST(LogitsProcessorPipelineTest, ConstructionFromConfig) {
  LogitsProcessorPipelineConfig config;
  config.repetition_penalty_config =
      RepetitionPenaltyConfig(1.2f, 0.0f, 0.0f, 10);
  config.suppress_tokens_config = SuppressTokensConfig({1, 2, 3});

  LogitsProcessorPipeline pipeline(/*batch_size=*/1, /*vocab_size=*/100,
                                   std::move(config));
  EXPECT_FALSE(pipeline.empty());
  EXPECT_EQ(pipeline.size(), 2);
}

}  // namespace
}  // namespace litert::lm
