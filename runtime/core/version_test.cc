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

#include "runtime/core/version.h"

#include <gtest/gtest.h>

namespace litert::lm {
namespace {

TEST(VersionTest, CompareVersions_Equal) {
  EXPECT_EQ(CompareVersions("0.17.0", "0.17.0"), 0);
  EXPECT_EQ(CompareVersions("1.2.3.4", "1.2.3.4"), 0);
}

TEST(VersionTest, CompareVersions_Older) {
  EXPECT_LT(CompareVersions("0.11.0", "0.17.0"), 0);
  EXPECT_LT(CompareVersions("0.17.0", "0.17.1"), 0);
  EXPECT_LT(CompareVersions("0.16.9", "0.17.0"), 0);
  EXPECT_LT(CompareVersions("0.9.0", "0.10.0"), 0);
}

TEST(VersionTest, CompareVersions_Newer) {
  EXPECT_GT(CompareVersions("0.17.1", "0.17.0"), 0);
  EXPECT_GT(CompareVersions("0.18.0", "0.17.0"), 0);
  EXPECT_GT(CompareVersions("1.0.0", "0.17.0"), 0);
}

TEST(VersionTest, CompareVersions_FewerPartsPaddedWithZeros) {
  EXPECT_EQ(CompareVersions("0.17", "0.17.0"), 0);
  EXPECT_EQ(CompareVersions("1", "1.0.0"), 0);
  EXPECT_GT(CompareVersions("1.1", "1.0.0"), 0);
  EXPECT_LT(CompareVersions("1", "1.0.1"), 0);
}

TEST(VersionTest, CompareVersions_StopsAtNonDigits) {
  // Prerelease labels should be ignored and base version compared.
  EXPECT_EQ(CompareVersions("0.17.0.dev1", "0.17.0"), 0);
  EXPECT_EQ(CompareVersions("0.17.0-beta", "0.17.0"), 0);
  EXPECT_GT(CompareVersions("0.17.1-alpha", "0.17.0"), 0);
}

}  // namespace
}  // namespace litert::lm
