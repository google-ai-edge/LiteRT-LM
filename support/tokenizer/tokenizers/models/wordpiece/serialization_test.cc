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
// C++ port of `src/models/wordpiece/serialization.rs` (`#[cfg(test)] mod
// tests`) from the HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/models/wordpiece/serialization.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/wordpiece/wordpiece.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers::models::wordpiece {
namespace {

TEST(WordPieceSerializationTest, Serde) {
  const WordPiece wp;
  constexpr const char* kWpS =
      "{"
      "\"type\":\"WordPiece\","
      "\"unk_token\":\"[UNK]\","
      "\"continuing_subword_prefix\":\"##\","
      "\"max_input_chars_per_word\":100,"
      "\"vocab\":{}"
      "}";

  EXPECT_EQ(DumpJson(wp.ToJson()), kWpS);
  ASSERT_OK_AND_ASSIGN(Json wp_json, ParseJson(kWpS));
  ASSERT_OK_AND_ASSIGN(WordPiece parsed, WordPiece::FromJson(wp_json));
  EXPECT_EQ(parsed, wp);
}

TEST(WordPieceSerializationTest, DeserializationShouldFail) {
  constexpr const char* kMissingUnk =
      "{"
      "\"type\":\"WordPiece\","
      "\"continuing_subword_prefix\":\"##\","
      "\"max_input_chars_per_word\":100,"
      "\"vocab\":{}"
      "}";
  ASSERT_OK_AND_ASSIGN(Json missing_unk_json, ParseJson(kMissingUnk));
  const absl::StatusOr<WordPiece> missing_unk_res =
      WordPiece::FromJson(missing_unk_json);
  ASSERT_FALSE(missing_unk_res.ok());
  EXPECT_THAT(missing_unk_res.status().message(),
              testing::StartsWith("missing field `unk_token`"));

  constexpr const char* kWrongType =
      "{"
      "\"type\":\"WordLevel\","
      "\"unk_token\":\"[UNK]\","
      "\"vocab\":{}"
      "}";
  ASSERT_OK_AND_ASSIGN(Json wrong_type_json, ParseJson(kWrongType));
  const absl::StatusOr<WordPiece> wrong_type_res =
      WordPiece::FromJson(wrong_type_json);
  ASSERT_FALSE(wrong_type_res.ok());
  EXPECT_THAT(wrong_type_res.status().message(),
              testing::StartsWith(
                  "invalid value: string \"WordLevel\", expected WordPiece"));
}

}  // namespace
}  // namespace tokenizers::models::wordpiece
