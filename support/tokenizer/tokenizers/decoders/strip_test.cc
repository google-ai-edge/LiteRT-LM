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
// C++ port of `src/decoders/strip.rs` (tests) from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/decoders/strip.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace decoders {
namespace {

using ::testing::ElementsAre;

Json J(absl::string_view s) {
  return Json::parse(s, /*cb=*/nullptr, /*allow_exceptions=*/false);
}

TEST(StripTest, Decode) {
  Strip decoder(U'H', 1, 0);
  ASSERT_OK_AND_ASSIGN(std::vector<std::string> res,
                       decoder.DecodeChain({"Hey", " friend!", "HHH"}));
  EXPECT_THAT(res, ElementsAre("ey", " friend!", "HH"));

  decoder = Strip(U'y', 0, 1);
  ASSERT_OK_AND_ASSIGN(res, decoder.DecodeChain({"Hey", " friend!"}));
  EXPECT_THAT(res, ElementsAre("He", " friend!"));
}

// Additional (not in the Rust file): serde round trip, including a non-ASCII
// `content` char, and the PORT NOTE edge cases of `DecodeChain`.
TEST(StripTest, Serde) {
  Strip decoder(U'▁', 1, 2);
  const std::string json =
      R"({"type":"Strip","content":"▁","start":1,"stop":2})";
  EXPECT_EQ(decoder.ToJson().dump(), json);
  ASSERT_OK_AND_ASSIGN(Strip parsed, Strip::FromJson(J(json)));
  EXPECT_EQ(parsed.content(), U'▁');
  EXPECT_EQ(parsed.start(), 1);
  EXPECT_EQ(parsed.stop(), 2);
  // A `char` must be exactly one character.
  EXPECT_FALSE(Strip::FromJson(
                   J(R"({"type":"Strip","content":"ab","start":1,"stop":2})"))
                   .ok());
}

TEST(StripTest, DecodeShortTokens) {
  // Stripping more trailing chars than the token has.
  Strip decoder(U'H', 0, 3);
  ASSERT_OK_AND_ASSIGN(std::vector<std::string> res,
                       decoder.DecodeChain({"HH", "aH"}));
  EXPECT_THAT(res, ElementsAre("", "a"));
  // Cuts crossing each other.
  decoder = Strip(U'H', 1, 1);
  ASSERT_OK_AND_ASSIGN(res, decoder.DecodeChain({"H", "HaH"}));
  EXPECT_THAT(res, ElementsAre("", "a"));
}

}  // namespace
}  // namespace decoders
}  // namespace tokenizers
