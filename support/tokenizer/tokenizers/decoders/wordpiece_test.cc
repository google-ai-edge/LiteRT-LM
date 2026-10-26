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
// C++ port of `src/decoders/wordpiece.rs` (tests) from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/decoders/wordpiece.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace decoders {
namespace {

Json J(absl::string_view s) {
  return Json::parse(s, /*cb=*/nullptr, /*allow_exceptions=*/false);
}

TEST(WordPieceTest, WordpieceDecoder) {
  WordPiece decoder("##", false);

  ASSERT_OK_AND_ASSIGN(
      std::string decoded,
      decoder.Decode({"##uelo", "Ara", "##új", "##o", "No", "##guera"}));
  EXPECT_EQ(decoded, "##uelo Araújo Noguera");
}

// Additional (not in the Rust file): `cleanup`, serde and the Rust
// `str::replace` semantics of `StrReplace`.
TEST(WordPieceTest, Cleanup) {
  WordPiece decoder;
  ASSERT_OK_AND_ASSIGN(std::string decoded,
                       decoder.Decode({"I", "do", "not", "know", "##!", ",",
                                       "do", "n't", "you", "?"}));
  EXPECT_EQ(decoded, "I do not know!, don't you?");
  EXPECT_EQ(Cleanup(" ' s"), "'s");
  EXPECT_EQ(Cleanup("it ' s"), "it's");
  EXPECT_EQ(Cleanup("i do not"), "i don't");
}

TEST(WordPieceTest, Serde) {
  WordPiece decoder;
  const std::string json =
      R"({"type":"WordPiece","prefix":"##","cleanup":true})";
  EXPECT_EQ(decoder.ToJson().dump(), json);
  ASSERT_OK_AND_ASSIGN(WordPiece parsed, WordPiece::FromJson(J(json)));
  EXPECT_EQ(parsed.prefix(), "##");
  EXPECT_TRUE(parsed.cleanup());
  // The `type` tag of a `#[serde(tag = "type")]` struct is not validated.
  EXPECT_OK(WordPiece::FromJson(J(R"({"prefix":"##","cleanup":false})")));
  EXPECT_FALSE(WordPiece::FromJson(J(R"({"prefix":"##"})")).ok());
}

TEST(WordPieceTest, StrReplace) {
  EXPECT_EQ(StrReplace("aaa", "a", "b"), "bbb");
  EXPECT_EQ(StrReplace("aaaa", "aa", "b"), "bb");
  EXPECT_EQ(StrReplace("abc", "", "-"), "-a-b-c-");
  EXPECT_EQ(StrReplace("", "", "-"), "-");
  EXPECT_EQ(StrReplace("héllo", "", "."), ".h.é.l.l.o.");
  EXPECT_EQ(StrReplace("x</w>y</w>", "</w>", " "), "x y ");
}

}  // namespace
}  // namespace decoders
}  // namespace tokenizers
