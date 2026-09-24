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

#include "runtime/engine/embedding_litert_lm_lib.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl

namespace litert::lm {
namespace {

using ::testing::Eq;
using ::testing::Optional;

TEST(EmbeddingLiteRtLmLibTest, SetEmbeddingFlagSuccess) {
  EmbeddingLiteRtLmSettings settings;

  EXPECT_TRUE(SetEmbeddingFlag(&settings, "backend", "cpu").ok());
  EXPECT_EQ(settings.backend, "cpu");

  EXPECT_TRUE(SetEmbeddingFlag(&settings, "model_path", "/path/to/model").ok());
  EXPECT_EQ(settings.model_path, "/path/to/model");

  EXPECT_TRUE(SetEmbeddingFlag(&settings, "input_prompt", "Hello world").ok());
  EXPECT_EQ(settings.input_prompt, "Hello world");

  EXPECT_TRUE(SetEmbeddingFlag(&settings, "vision_backend", "npu").ok());
  EXPECT_THAT(settings.vision_backend, Optional(Eq("npu")));

  EXPECT_TRUE(SetEmbeddingFlag(&settings, "audio_backend", "cpu").ok());
  EXPECT_THAT(settings.audio_backend, Optional(Eq("cpu")));

  EXPECT_TRUE(SetEmbeddingFlag(&settings, "visual_token_budget", "70").ok());
  EXPECT_EQ(settings.visual_token_budget, 70);

  EXPECT_TRUE(
      SetEmbeddingFlag(&settings, "report_peak_memory_footprint", "true").ok());
  EXPECT_TRUE(settings.report_peak_memory_footprint);

  EXPECT_TRUE(SetEmbeddingFlag(&settings, "compare_embedding_path",
                               "/path/to/golden.json")
                  .ok());
  EXPECT_EQ(settings.compare_embedding_path, "/path/to/golden.json");

  EXPECT_TRUE(SetEmbeddingFlag(&settings, "output_embedding_path",
                               "/path/to/output.json")
                  .ok());
  EXPECT_EQ(settings.output_embedding_path, "/path/to/output.json");

  EXPECT_TRUE(SetEmbeddingFlag(&settings, "benchmark", "1").ok());
  EXPECT_TRUE(settings.benchmark);

  EXPECT_TRUE(SetEmbeddingFlag(&settings, "num_warmup", "3").ok());
  EXPECT_EQ(settings.num_warmup, 3);

  EXPECT_TRUE(SetEmbeddingFlag(&settings, "num_iterations", "5").ok());
  EXPECT_EQ(settings.num_iterations, 5);
}

TEST(EmbeddingLiteRtLmLibTest, SetEmbeddingFlagInvalidInteger) {
  EmbeddingLiteRtLmSettings settings;
  EXPECT_FALSE(
      SetEmbeddingFlag(&settings, "visual_token_budget", "invalid").ok());
  EXPECT_FALSE(SetEmbeddingFlag(&settings, "num_iterations", "abc").ok());
}

TEST(EmbeddingLiteRtLmLibTest, SetEmbeddingFlagUnknownFlag) {
  EmbeddingLiteRtLmSettings settings;
  absl::Status status = SetEmbeddingFlag(&settings, "non_existent_flag", "foo");
  EXPECT_TRUE(absl::IsNotFound(status));
}

TEST(EmbeddingLiteRtLmLibTest, SetEmbeddingFlagNullptr) {
  absl::Status status = SetEmbeddingFlag(nullptr, "backend", "cpu");
  EXPECT_TRUE(absl::IsInvalidArgument(status));
}

TEST(EmbeddingLiteRtLmLibTest, GlobalSetEmbeddingFlag) {
  EXPECT_TRUE(SetEmbeddingFlag("backend", "cpu").ok());
  EXPECT_TRUE(SetEmbeddingFlag("report_peak_memory_footprint", "false").ok());
  EXPECT_TRUE(SetEmbeddingFlag("visual_token_budget", "50").ok());
  EXPECT_TRUE(
      absl::IsNotFound(SetEmbeddingFlag("completely_unknown_flag", "val")));
}

}  // namespace
}  // namespace litert::lm
