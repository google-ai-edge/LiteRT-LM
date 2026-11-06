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
// Tests for the C++ port of `src/decoders/bpe.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). The Rust file has
// no tests of its own; these cover the decoding and the serde emulation.

#include "support/tokenizer/tokenizers/decoders/bpe.h"

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
using ::testing::IsEmpty;

Json J(absl::string_view s) {
  return Json::parse(s, /*cb=*/nullptr, /*allow_exceptions=*/false);
}

TEST(BPEDecoderTest, Decode) {
  BPEDecoder decoder;
  EXPECT_EQ(decoder.suffix(), "</w>");
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::string> res,
      decoder.DecodeChain({"Hel", "lo</w>", "my</w>", "fri", "end</w>"}));
  // The suffix of the last token is removed, the others become spaces.
  EXPECT_THAT(res, ElementsAre("Hel", "lo ", "my ", "fri", "end"));
  ASSERT_OK_AND_ASSIGN(
      std::string decoded,
      decoder.Decode({"Hel", "lo</w>", "my</w>", "fri", "end</w>"}));
  EXPECT_EQ(decoded, "Hello my friend");

  ASSERT_OK_AND_ASSIGN(res, decoder.DecodeChain({}));
  EXPECT_THAT(res, IsEmpty());
}

TEST(BPEDecoderTest, Serde) {
  BPEDecoder decoder;
  const std::string json = R"({"type":"BPEDecoder","suffix":"</w>"})";
  EXPECT_EQ(decoder.ToJson().dump(), json);
  ASSERT_OK_AND_ASSIGN(BPEDecoder parsed, BPEDecoder::FromJson(J(json)));
  EXPECT_EQ(parsed.suffix(), "</w>");
  // The `type` tag of a `#[serde(tag = "type")]` struct is not validated.
  ASSERT_OK_AND_ASSIGN(parsed, BPEDecoder::FromJson(J(R"({"suffix":"x"})")));
  EXPECT_EQ(parsed.suffix(), "x");
  EXPECT_FALSE(BPEDecoder::FromJson(J(R"({"type":"BPEDecoder"})")).ok());
  EXPECT_FALSE(BPEDecoder::FromJson(J(R"({"suffix":1})")).ok());
}

}  // namespace
}  // namespace decoders
}  // namespace tokenizers
