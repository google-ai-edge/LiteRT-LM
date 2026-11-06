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
// C++ port of `src/models/unigram/serialization.rs` (`#[cfg(test)] mod test`)
// from the HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/models/unigram/serialization.h"

#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "support/tokenizer/tokenizers/models/unigram/model.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers::models::unigram {
namespace {

TEST(UnigramSerializationTest, TestSerialization) {
  const Vocab vocab = {{"<unk>", 0.0}, {"a", -0.5}};
  ASSERT_OK_AND_ASSIGN(Unigram model, Unigram::From(vocab, 0, false));

  const std::string data = DumpJson(model.ToJson());
  ASSERT_OK_AND_ASSIGN(Json json, ParseJson(data));
  ASSERT_OK_AND_ASSIGN(Unigram reconstructed, Unigram::FromJson(json));

  EXPECT_EQ(model, reconstructed);
}

TEST(UnigramSerializationTest, TestSerializationUnkIdNotZero) {
  const Vocab vocab = {{"a", -0.5}, {"<unk>", 0.0}};
  ASSERT_OK_AND_ASSIGN(Unigram model, Unigram::From(vocab, 1, false));

  const std::string data = DumpJson(model.ToJson());
  ASSERT_OK_AND_ASSIGN(Json json, ParseJson(data));
  ASSERT_OK_AND_ASSIGN(Unigram reconstructed, Unigram::FromJson(json));

  EXPECT_EQ(model, reconstructed);
}

TEST(UnigramSerializationTest, TestSerializationNoUnkId) {
  const Vocab vocab = {{"a", -0.5}};
  ASSERT_OK_AND_ASSIGN(Unigram model,
                       Unigram::From(vocab, std::nullopt, false));

  const std::string data = DumpJson(model.ToJson());
  ASSERT_OK_AND_ASSIGN(Json json, ParseJson(data));
  ASSERT_OK_AND_ASSIGN(Unigram reconstructed, Unigram::FromJson(json));

  EXPECT_EQ(model, reconstructed);
}

}  // namespace
}  // namespace tokenizers::models::unigram
