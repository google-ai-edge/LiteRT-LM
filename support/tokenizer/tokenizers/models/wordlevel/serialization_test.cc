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
// C++ port of `src/models/wordlevel/serialization.rs` (`#[cfg(test)] mod
// tests`) from the HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/models/wordlevel/serialization.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/wordlevel/wordlevel.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers::models::wordlevel {
namespace {

TEST(WordLevelSerializationTest, Serde) {
  const WordLevel wl;
  constexpr const char* kWlS =
      R"({"type":"WordLevel","vocab":{},"unk_token":"<unk>"})";

  EXPECT_EQ(DumpJson(wl.ToJson()), kWlS);
  ASSERT_OK_AND_ASSIGN(Json wl_json, ParseJson(kWlS));
  ASSERT_OK_AND_ASSIGN(WordLevel parsed, WordLevel::FromJson(wl_json));
  EXPECT_EQ(parsed, wl);
}

TEST(WordLevelSerializationTest, IncompleteVocab) {
  const Vocab vocab = {{"<unk>", 0}, {"b", 2}};
  ASSERT_OK_AND_ASSIGN(
      WordLevel wordlevel,
      WordLevelBuilder().Vocab(vocab).UnkToken("<unk>").Build());
  constexpr const char* kWlS =
      R"({"type":"WordLevel","vocab":{"<unk>":0,"b":2},"unk_token":"<unk>"})";
  EXPECT_EQ(DumpJson(wordlevel.ToJson()), kWlS);
  ASSERT_OK_AND_ASSIGN(Json wl_json, ParseJson(kWlS));
  ASSERT_OK_AND_ASSIGN(WordLevel parsed, WordLevel::FromJson(wl_json));
  EXPECT_EQ(parsed, wordlevel);
}

TEST(WordLevelSerializationTest, DeserializationShouldFail) {
  constexpr const char* kMissingUnk = R"({"type":"WordLevel","vocab":{}})";
  ASSERT_OK_AND_ASSIGN(Json missing_unk_json, ParseJson(kMissingUnk));
  const absl::StatusOr<WordLevel> missing_unk_res =
      WordLevel::FromJson(missing_unk_json);
  ASSERT_FALSE(missing_unk_res.ok());
  EXPECT_THAT(missing_unk_res.status().message(),
              testing::StartsWith("missing field `unk_token`"));

  constexpr const char* kWrongType = R"({"type":"WordPiece","vocab":{}})";
  ASSERT_OK_AND_ASSIGN(Json wrong_type_json, ParseJson(kWrongType));
  const absl::StatusOr<WordLevel> wrong_type_res =
      WordLevel::FromJson(wrong_type_json);
  ASSERT_FALSE(wrong_type_res.ok());
  EXPECT_THAT(wrong_type_res.status().message(),
              testing::StartsWith(
                  "invalid value: string \"WordPiece\", expected WordLevel"));
}

}  // namespace
}  // namespace tokenizers::models::wordlevel
