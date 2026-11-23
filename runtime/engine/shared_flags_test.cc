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

#include "runtime/engine/shared_flags.h"

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/speculative_decoding_mode.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

TEST(ParseSpeculativeDecodingFlagTest, UnsetUsesRuntimeDefault) {
  ASSERT_OK_AND_ASSIGN(const std::optional<SpeculativeDecodingMode> mode,
                       ParseSpeculativeDecodingFlag(""));
  EXPECT_EQ(mode, std::nullopt);
}

TEST(ParseSpeculativeDecodingFlagTest, ParsesModes) {
  EXPECT_THAT(ParseSpeculativeDecodingFlag("off"),
              IsOkAndHolds(SpeculativeDecodingMode::kOff));
  EXPECT_THAT(ParseSpeculativeDecodingFlag("on"),
              IsOkAndHolds(SpeculativeDecodingMode::kOn));
  EXPECT_THAT(ParseSpeculativeDecodingFlag("adaptive"),
              IsOkAndHolds(SpeculativeDecodingMode::kAdaptive));
}

TEST(ParseSpeculativeDecodingFlagTest, IsCaseInsensitive) {
  EXPECT_THAT(ParseSpeculativeDecodingFlag("OFF"),
              IsOkAndHolds(SpeculativeDecodingMode::kOff));
  EXPECT_THAT(ParseSpeculativeDecodingFlag("On"),
              IsOkAndHolds(SpeculativeDecodingMode::kOn));
  EXPECT_THAT(ParseSpeculativeDecodingFlag("ADAPTIVE"),
              IsOkAndHolds(SpeculativeDecodingMode::kAdaptive));
}

TEST(ParseSpeculativeDecodingFlagTest, RejectsOtherValues) {
  for (const char* value :
       {"true", "false", "always", "enabled", "disabled", "1", "0", "on "}) {
    EXPECT_THAT(ParseSpeculativeDecodingFlag(value),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << value;
  }
}

TEST(BenchmarkAppendChunksFlagTest, DefaultsToZeroAndSupportsOverride) {
  EXPECT_EQ(absl::GetFlag(FLAGS_benchmark_append_chunks), 0);
  absl::SetFlag(&FLAGS_benchmark_append_chunks, 4);
  EXPECT_EQ(absl::GetFlag(FLAGS_benchmark_append_chunks), 4);
  absl::SetFlag(&FLAGS_benchmark_append_chunks, 0);
}

TEST(ApplyLiteRtLibDirFlagsTest, LeavesUnsetWhenFlagsEmpty) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create("/model"));
  ASSERT_OK_AND_ASSIGN(LlmExecutorSettings settings,
                       LlmExecutorSettings::CreateDefault(
                           std::move(model_assets), Backend::NPU));

  absl::SetFlag(&FLAGS_litert_dispatch_lib_dir, "");
  absl::SetFlag(&FLAGS_litert_compiler_lib_dir, "");
  ApplyLiteRtLibDirFlags(settings);

  EXPECT_EQ(settings.GetLitertDispatchLibDir(), "");
  EXPECT_EQ(settings.GetLitertCompilerLibDir(), "");
}

TEST(ApplyLiteRtLibDirFlagsTest, AppliesNonEmptyDispatchAndCompilerDirs) {
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets, ModelAssets::Create("/model"));
  ASSERT_OK_AND_ASSIGN(LlmExecutorSettings settings,
                       LlmExecutorSettings::CreateDefault(
                           std::move(model_assets), Backend::NPU));

  absl::SetFlag(&FLAGS_litert_dispatch_lib_dir, "/custom/dispatch");
  absl::SetFlag(&FLAGS_litert_compiler_lib_dir, "/custom/compiler");
  ApplyLiteRtLibDirFlags(settings);

  EXPECT_EQ(settings.GetLitertDispatchLibDir(), "/custom/dispatch");
  EXPECT_EQ(settings.GetLitertCompilerLibDir(), "/custom/compiler");
}

}  // namespace
}  // namespace litert::lm
