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
#include "absl/status/status.h"  // from @com_google_absl
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

}  // namespace
}  // namespace litert::lm
