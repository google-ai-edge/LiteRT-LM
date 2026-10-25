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
// C++ port of the tests of `src/pre_tokenizers/mod.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team).

#include "support/tokenizer/tokenizers/pre_tokenizers/pre_tokenizers.h"

#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/pre_tokenizers/delimiter.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/metaspace.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/sequence.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/whitespace.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace pre_tokenizers {
namespace {

using ::testing::status::StatusIs;

// `serde_json::from_str::<PreTokenizerWrapper>(s)`
absl::StatusOr<PreTokenizerWrapper> FromStr(absl::string_view s) {
  absl::StatusOr<Json> json = ParseJson(s);
  if (!json.ok()) return json.status();
  return PreTokenizerWrapper::FromJson(*json);
}

PreTokenizerWrapper SequenceOf(std::vector<PreTokenizerWrapper> pretokenizers) {
  return Sequence(std::move(pretokenizers));
}

TEST(PreTokenizerWrapperTest, TestDeserialize) {
  ASSERT_OK_AND_ASSIGN(
      PreTokenizerWrapper pre_tokenizer,
      FromStr(
          R"({"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"},{"type":"Metaspace","replacement":"▁","str_rep":"▁","add_prefix_space":true}]})"));

  EXPECT_EQ(pre_tokenizer,
            SequenceOf({PreTokenizerWrapper(WhitespaceSplit()),
                        PreTokenizerWrapper(
                            Metaspace(U'▁', PrependScheme::kAlways, true))}));

  ASSERT_OK_AND_ASSIGN(
      pre_tokenizer,
      FromStr(
          R"({"type":"Metaspace","replacement":"▁","add_prefix_space":true})"));

  EXPECT_EQ(pre_tokenizer,
            PreTokenizerWrapper(Metaspace(U'▁', PrependScheme::kAlways, true)));

  ASSERT_OK_AND_ASSIGN(
      pre_tokenizer,
      FromStr(
          R"({"type":"Sequence","pretokenizers":[{"type":"WhitespaceSplit"},{"type":"Metaspace","replacement":"▁","add_prefix_space":true}]})"));

  EXPECT_EQ(pre_tokenizer,
            SequenceOf({PreTokenizerWrapper(WhitespaceSplit()),
                        PreTokenizerWrapper(
                            Metaspace(U'▁', PrependScheme::kAlways, true))}));

  ASSERT_OK_AND_ASSIGN(
      pre_tokenizer,
      FromStr(
          R"({"type":"Metaspace","replacement":"▁","add_prefix_space":true, "prepend_scheme":"first"})"));

  EXPECT_EQ(pre_tokenizer,
            PreTokenizerWrapper(Metaspace(U'▁', PrependScheme::kFirst, true)));

  ASSERT_OK_AND_ASSIGN(
      pre_tokenizer,
      FromStr(
          R"({"type":"Metaspace","replacement":"▁","add_prefix_space":true, "prepend_scheme":"always"})"));

  EXPECT_EQ(pre_tokenizer,
            PreTokenizerWrapper(Metaspace(U'▁', PrependScheme::kAlways, true)));
}

TEST(PreTokenizerWrapperTest, TestDeserializeWhitespaceSplit) {
  ASSERT_OK_AND_ASSIGN(const PreTokenizerWrapper pre_tokenizer,
                       FromStr(R"({"type":"WhitespaceSplit"})"));
  EXPECT_EQ(pre_tokenizer, PreTokenizerWrapper(WhitespaceSplit()));
}

TEST(PreTokenizerWrapperTest, PreTokenizerDeserializationNoType) {
  // PORT NOTE: the Rust test's JSON has a stray trailing `}` which serde_json
  // never reaches (the untagged-enum error is reported first); here parsing
  // and deserialization are separate steps, so the typo is dropped.
  absl::StatusOr<PreTokenizerWrapper> reconstructed = FromStr(
      R"({"replacement":"▁","add_prefix_space":true, "prepend_scheme":"always"})");
  EXPECT_THAT(reconstructed,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "data did not match any variant of untagged enum "
                       "PreTokenizerUntagged"));

  ASSERT_OK_AND_ASSIGN(reconstructed,
                       FromStr(R"({"type":"Metaspace", "replacement":"▁" })"));
  EXPECT_EQ(*reconstructed, PreTokenizerWrapper(Metaspace()));

  reconstructed = FromStr(R"({"type":"Metaspace", "add_prefix_space":true })");
  EXPECT_THAT(reconstructed, StatusIs(absl::StatusCode::kInvalidArgument,
                                      "missing field `replacement`"));

  reconstructed = FromStr(R"({"behavior":"default_split"})");
  EXPECT_THAT(reconstructed,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "data did not match any variant of untagged enum "
                       "PreTokenizerUntagged"));
}

// Extra (not in the Rust tests): the untagged fallback accepts the struct
// name of the `Delimiter` variant, which is how `CharDelimiterSplit`
// serializes itself.
TEST(PreTokenizerWrapperTest, SerializationRoundTrip) {
  const std::vector<PreTokenizerWrapper> wrappers = {
      PreTokenizerWrapper(CharDelimiterSplit(U'x')),
      PreTokenizerWrapper(WhitespaceSplit()),
      PreTokenizerWrapper(Metaspace(U'▁', PrependScheme::kFirst, false)),
  };
  for (const PreTokenizerWrapper& wrapper : wrappers) {
    ASSERT_OK_AND_ASSIGN(const PreTokenizerWrapper parsed,
                         PreTokenizerWrapper::FromJson(wrapper.ToJson()));
    EXPECT_EQ(parsed, wrapper) << DumpJson(wrapper.ToJson());
  }
  EXPECT_EQ(DumpJson(PreTokenizerWrapper(CharDelimiterSplit(U'x')).ToJson()),
            R"({"type":"CharDelimiterSplit","delimiter":"x"})");
}

}  // namespace
}  // namespace pre_tokenizers
}  // namespace tokenizers
