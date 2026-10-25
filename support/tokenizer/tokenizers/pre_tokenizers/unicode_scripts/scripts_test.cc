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
//
// C++ port of the tests of `src/pre_tokenizers/unicode_scripts/scripts.rs`
// from the HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team).

#include "support/tokenizer/tokenizers/pre_tokenizers/unicode_scripts/scripts.h"

#include <gtest/gtest.h>

namespace tokenizers {
namespace pre_tokenizers {
namespace unicode_scripts {
namespace {

TEST(ScriptsTest, TestUnicodeScript) {
  EXPECT_EQ(Script::kHan, GetScript(U'京'));
  EXPECT_EQ(Script::kHan, GetScript(U'太'));
  EXPECT_EQ(Script::kHiragana, GetScript(U'い'));
  EXPECT_EQ(Script::kKatakana, GetScript(U'グ'));
  EXPECT_EQ(Script::kCommon, GetScript(U'ー'));
  EXPECT_EQ(Script::kLatin, GetScript(U'a'));
  EXPECT_EQ(Script::kLatin, GetScript(U'A'));
  EXPECT_EQ(Script::kCommon, GetScript(U'0'));
  EXPECT_EQ(Script::kCommon, GetScript(U'$'));
  EXPECT_EQ(Script::kCommon, GetScript(U'@'));
  EXPECT_EQ(Script::kCommon, GetScript(U'-'));
  EXPECT_EQ(Script::kCommon, GetScript(U' '));
  EXPECT_EQ(Script::kCommon, GetScript(U'\uFFFD'));
}

// Extra (not in the Rust tests): spot checks of the table boundaries and of
// the default arm.
TEST(ScriptsTest, Boundaries) {
  EXPECT_EQ(Script::kCommon, GetScript(U'\u0000'));
  EXPECT_EQ(Script::kHiragana, GetScript(U'\u3041'));
  EXPECT_EQ(Script::kHiragana, GetScript(U'\u3096'));
  EXPECT_EQ(Script::kAny, GetScript(U'\u3097'));
  EXPECT_EQ(Script::kKatakana, GetScript(U'\u30A1'));
  EXPECT_EQ(Script::kKatakana, GetScript(U'\u30FA'));
  EXPECT_EQ(Script::kTangut, GetScript(U'\U00018AF2'));
  EXPECT_EQ(Script::kAny, GetScript(U'\U00018AF3'));
  EXPECT_EQ(Script::kAny, GetScript(U'\U0010FFFF'));
}

}  // namespace
}  // namespace unicode_scripts
}  // namespace pre_tokenizers
}  // namespace tokenizers
