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

#include "runtime/components/constrained_decoding/suppress_tokens_config.h"

#include <gtest/gtest.h>

namespace litert::lm {
namespace {

TEST(SuppressTokensConfigTest, RetainsValidValues) {
  // Config with token ids {1, 2, 3}.
  SuppressTokensConfig config({1, 2, 3});

  EXPECT_TRUE(config.enabled());
  EXPECT_EQ(config.suppress_tokens().size(), 3);
  EXPECT_TRUE(config.suppress_tokens().contains(1));
  EXPECT_TRUE(config.suppress_tokens().contains(2));
  EXPECT_TRUE(config.suppress_tokens().contains(3));
}

TEST(SuppressTokensConfigTest, DefaultIsDisabled) {
  SuppressTokensConfig config = SuppressTokensConfig::Default();

  EXPECT_FALSE(config.enabled());
  EXPECT_TRUE(config.suppress_tokens().empty());
}

}  // namespace
}  // namespace litert::lm
