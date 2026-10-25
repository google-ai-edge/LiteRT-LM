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
// C++ port of `src/decoders/ctc.rs` (tests) from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License
// 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line close to the
// Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/decoders/ctc.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace decoders {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

// `"...".split(' ').map(|s| s.to_string()).collect()`.
std::vector<std::string> SplitOnSpace(absl::string_view s) {
  return absl::StrSplit(s, ' ');
}

TEST(CTCTest, HandmadeSample) {
  CTC ctc_decoder;
  std::vector<std::string> id_to_string_result =
      SplitOnSpace("<pad> <pad> h e e l l <pad> l o o o <pad>");
  ASSERT_OK_AND_ASSIGN(std::vector<std::string> res,
                       ctc_decoder.DecodeChain(id_to_string_result));
  EXPECT_THAT(res, ElementsAre("h", "e", "l", "l", "o"));
}

TEST(CTCTest, HandmadeWithDelimiterSample) {
  CTC ctc_decoder;
  std::vector<std::string> id_to_string_result = SplitOnSpace(
      "<pad> <pad> h e e l l <pad> l o o o <pad> <pad> | <pad> w o o o r "
      "<pad> <pad> l l d <pad> <pad> <pad> <pad>");
  ASSERT_OK_AND_ASSIGN(std::vector<std::string> res,
                       ctc_decoder.DecodeChain(id_to_string_result));
  EXPECT_THAT(
      res, ElementsAre("h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"));
}

TEST(CTCTest, LibrispeechSample) {
  CTC ctc_decoder;
  std::vector<std::string> id_to_string_result = SplitOnSpace(
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> <pad> A | | <pad> M <pad> <pad> <pad> "
      "<pad> A <pad> <pad> N <pad> <pad> <pad> | | | <pad> <pad> <pad> <pad> S "
      "<pad> <pad> <pad> A I <pad> D D | | T T <pad> O <pad> | | T H E E | | | "
      "<pad> U U <pad> N N <pad> I <pad> <pad> V <pad> <pad> <pad> E R R <pad> "
      "<pad> <pad> S E E | | <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> <pad> S S <pad> <pad> <pad> <pad> I <pad> "
      "R R <pad> <pad> | | | <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> I "
      "<pad> <pad> <pad> | <pad> <pad> <pad> E X <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> I <pad> S <pad> <pad> T <pad> <pad> | | "
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad>");
  ASSERT_OK_AND_ASSIGN(std::vector<std::string> res,
                       ctc_decoder.DecodeChain(id_to_string_result));
  EXPECT_THAT(res, ElementsAreArray(std::vector<std::string>{
                       "A", " ", "M", "A", "N", " ", "S", "A", "I", "D",
                       " ", "T", "O", " ", "T", "H", "E", " ", "U", "N",
                       "I", "V", "E", "R", "S", "E", " ", "S", "I", "R",
                       " ", "I", " ", "E", "X", "I", "S", "T", " "}));
}

TEST(CTCTest, AnotherLibrispeechSample) {
  CTC ctc_decoder;
  std::vector<std::string> id_to_string_result = SplitOnSpace(
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> H <pad> I <pad> S S | | "
      "<pad> <pad> <pad> I N <pad> <pad> S <pad> T T <pad> <pad> A N C C T "
      "<pad> | | | | | <pad> <pad> <pad> <pad> P <pad> <pad> <pad> <pad> A "
      "<pad> <pad> N N N <pad> <pad> I <pad> C <pad> <pad> | | <pad> W <pad> "
      "<pad> A S <pad> | | <pad> <pad> <pad> F <pad> <pad> O L <pad> <pad> L L "
      "O O W E E D | | <pad> B <pad> <pad> <pad> Y <pad> | | | A | | <pad> S S "
      "S <pad> M M <pad> <pad> <pad> A L L <pad> <pad> <pad> <pad> L <pad> | | "
      "| <pad> <pad> <pad> <pad> S H H <pad> <pad> <pad> <pad> A R R <pad> "
      "<pad> P <pad> <pad> | <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> B <pad> <pad> L L <pad> <pad> <pad> <pad> <pad> O W W <pad> <pad> "
      "| | | <pad> <pad> <pad> <pad> <pad> <pad> <pad> H <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> I G H H | | <pad> <pad> O N <pad> | | H <pad> I "
      "S S | | <pad> <pad> C H H <pad> <pad> <pad> E <pad> S S <pad> T T <pad> "
      "<pad> | | | <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> "
      "<pad> <pad> <pad> <pad> <pad> <pad>");
  ASSERT_OK_AND_ASSIGN(std::vector<std::string> res,
                       ctc_decoder.DecodeChain(id_to_string_result));
  EXPECT_THAT(res,
              ElementsAreArray(std::vector<std::string>{
                  "H", "I", "S", " ", "I", "N", "S", "T", "A", "N", "C", "T",
                  " ", "P", "A", "N", "I", "C", " ", "W", "A", "S", " ", "F",
                  "O", "L", "L", "O", "W", "E", "D", " ", "B", "Y", " ", "A",
                  " ", "S", "M", "A", "L", "L", " ", "S", "H", "A", "R", "P",
                  " ", "B", "L", "O", "W", " ", "H", "I", "G", "H", " ", "O",
                  "N", " ", "H", "I", "S", " ", "C", "H", "E", "S", "T", " "}));
}

}  // namespace
}  // namespace decoders
}  // namespace tokenizers
