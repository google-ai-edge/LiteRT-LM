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
// C++ port of the tests of `src/pre_tokenizers/whitespace.rs` from the
// HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team).

#include "support/tokenizer/tokenizers/pre_tokenizers/whitespace.h"

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace pre_tokenizers {
namespace {

using SplitList = std::vector<std::pair<std::string, Offsets>>;

// `pretokenized.get_splits(ref, type).into_iter().map(|(s, o, _)| (s, o))`
SplitList Splits(const PreTokenizedString& pretokenized,
                 OffsetReferential offset_ref, OffsetType offset_type) {
  SplitList result;
  for (const auto& split : pretokenized.GetSplits(offset_ref, offset_type)) {
    result.emplace_back(std::string(std::get<0>(split)), std::get<1>(split));
  }
  return result;
}

TEST(WhitespaceTest, Basic) {
  const std::vector<std::pair<absl::string_view, SplitList>> tests = {
      {"Hey man!", {{"Hey", {0, 3}}, {"man", {4, 7}}, {"!", {7, 8}}}},
      {"How are you doing?",
       {
           {"How", {0, 3}},
           {"are", {4, 7}},
           {"you", {8, 11}},
           {"doing", {12, 17}},
           {"?", {17, 18}},
       }},
      {"\n", {}},
      {"\xf0\x9f\x91\xa8\xe2\x80\x8d\xf0\x9f\x91\xa9",
       {
           {"\xf0\x9f\x91\xa8", {0, 4}},
           {"\xe2\x80\x8d", {4, 7}},
           {"\xf0\x9f\x91\xa9", {7, 11}},
       }},
  };
  const Whitespace pretok;
  for (const auto& [s, res] : tests) {
    PreTokenizedString pretokenized = PreTokenizedString::From(s);
    ASSERT_OK(pretok.PreTokenize(pretokenized));
    EXPECT_EQ(
        Splits(pretokenized, OffsetReferential::kOriginal, OffsetType::kByte),
        res);
  }
}

TEST(WhitespaceTest, WhitespaceSplit) {
  const std::vector<std::pair<absl::string_view, SplitList>> tests = {
      {"Hey man!", {{"Hey", {0, 3}}, {"man!", {4, 8}}}},
      {"Hey, man, Good?",
       {{"Hey,", {0, 4}}, {"man,", {5, 9}}, {"Good?", {10, 15}}}},
  };
  const WhitespaceSplit pretok;
  for (const auto& [s, res] : tests) {
    PreTokenizedString pretokenized = PreTokenizedString::From(s);
    ASSERT_OK(pretok.PreTokenize(pretokenized));
    EXPECT_EQ(
        Splits(pretokenized, OffsetReferential::kOriginal, OffsetType::kByte),
        res);
  }
}

}  // namespace
}  // namespace pre_tokenizers
}  // namespace tokenizers
