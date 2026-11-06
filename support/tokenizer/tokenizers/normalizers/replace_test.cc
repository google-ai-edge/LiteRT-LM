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
// C++ port of the tests of `src/normalizers/replace.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team).

#include "support/tokenizer/tokenizers/normalizers/replace.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"  // IWYU pragma: keep
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers::normalizers {
namespace {

using ::testing::ElementsAre;
using ::testing::status::IsOkAndHolds;

TEST(ReplaceTest, TestReplace) {
  const std::string original = "This is a ''test''";
  const std::string normalized = "This is a \"test\"";

  NormalizedString n = NormalizedString::From(original);
  ASSERT_OK_AND_ASSIGN(Replace replace, Replace::New("''", "\""));
  ASSERT_OK(replace.Normalize(n));

  EXPECT_EQ(n.Get(), normalized);
}

TEST(ReplaceTest, TestReplaceRegex) {
  const std::string original = "This     is   a         test";
  const std::string normalized = "This is a test";

  NormalizedString n = NormalizedString::From(original);
  ASSERT_OK_AND_ASSIGN(Replace replace,
                       Replace::New(ReplacePattern::Regex(R"(\s+)"), " "));
  ASSERT_OK(replace.Normalize(n));

  EXPECT_EQ(n.Get(), normalized);
}

TEST(ReplaceTest, Serialization) {
  ASSERT_OK_AND_ASSIGN(Replace replace, Replace::New("Hello", "Hey"));
  absl::string_view replace_s =
      R"({"type":"Replace","pattern":{"String":"Hello"},"content":"Hey"})";
  EXPECT_EQ(replace.ToJson().dump(), replace_s);
  ASSERT_OK_AND_ASSIGN(Json parsed, ParseJson(replace_s));
  EXPECT_THAT(Replace::FromJson(parsed), IsOkAndHolds(replace));

  ASSERT_OK_AND_ASSIGN(replace,
                       Replace::New(ReplacePattern::Regex(R"(\s+)"), " "));
  replace_s = R"({"type":"Replace","pattern":{"Regex":"\\s+"},"content":" "})";
  EXPECT_EQ(replace.ToJson().dump(), replace_s);
  ASSERT_OK_AND_ASSIGN(parsed, ParseJson(replace_s));
  EXPECT_THAT(Replace::FromJson(parsed), IsOkAndHolds(replace));
}

TEST(ReplaceTest, TestReplaceDecode) {
  std::vector<std::string> original = {"hello", "_hello"};
  ASSERT_OK_AND_ASSIGN(Replace replace, Replace::New("_", " "));
  ASSERT_OK_AND_ASSIGN(std::vector<std::string> decoded,
                       replace.DecodeChain(original));
  EXPECT_THAT(decoded, ElementsAre("hello", " hello"));
}

}  // namespace
}  // namespace tokenizers::normalizers
