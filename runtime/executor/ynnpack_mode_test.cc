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

#include "runtime/executor/ynnpack_mode.h"

#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"  // from @com_google_absl

namespace litert::lm {
namespace {

TEST(YnnpackModeTest, DefaultIsOff) {
  YnnpackMode mode;
  EXPECT_TRUE(mode.is_off());
  EXPECT_FALSE(mode.is_on());
  EXPECT_FALSE(mode.is_auto());
  EXPECT_EQ(mode, YnnpackMode::kOff);
  EXPECT_EQ(mode, false);
}

TEST(YnnpackModeTest, ParseFlag) {
  YnnpackMode mode;
  std::string error;

  EXPECT_TRUE(AbslParseFlag("true", &mode, &error));
  EXPECT_TRUE(mode.is_on());

  EXPECT_TRUE(AbslParseFlag("1", &mode, &error));
  EXPECT_TRUE(mode.is_on());

  EXPECT_TRUE(AbslParseFlag("on", &mode, &error));
  EXPECT_TRUE(mode.is_on());

  EXPECT_TRUE(AbslParseFlag("false", &mode, &error));
  EXPECT_TRUE(mode.is_off());

  EXPECT_TRUE(AbslParseFlag("0", &mode, &error));
  EXPECT_TRUE(mode.is_off());

  EXPECT_TRUE(AbslParseFlag("off", &mode, &error));
  EXPECT_TRUE(mode.is_off());

  EXPECT_TRUE(AbslParseFlag("auto", &mode, &error));
  EXPECT_TRUE(mode.is_auto());

  EXPECT_FALSE(AbslParseFlag("invalid", &mode, &error));
  EXPECT_THAT(error, testing::HasSubstr("Invalid --enable_ynnpack value"));
}

TEST(YnnpackModeTest, UnparseFlagAndStringify) {
  EXPECT_EQ(AbslUnparseFlag(YnnpackMode::kOff), "false");
  EXPECT_EQ(AbslUnparseFlag(YnnpackMode::kOn), "true");
  EXPECT_EQ(AbslUnparseFlag(YnnpackMode::kAuto), "auto");

  EXPECT_EQ(absl::StrCat(YnnpackMode(YnnpackMode::kOff)), "false");
  EXPECT_EQ(absl::StrCat(YnnpackMode(YnnpackMode::kOn)), "true");
  EXPECT_EQ(absl::StrCat(YnnpackMode(YnnpackMode::kAuto)), "auto");

  std::stringstream ss;
  ss << YnnpackMode(YnnpackMode::kOff) << " " << YnnpackMode(YnnpackMode::kOn)
     << " " << YnnpackMode(YnnpackMode::kAuto);
  EXPECT_EQ(ss.str(), "0 1 auto");
}

}  // namespace
}  // namespace litert::lm
