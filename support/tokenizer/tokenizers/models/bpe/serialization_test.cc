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
// C++ port of `src/models/bpe/serialization.rs` (`#[cfg(test)] mod test`) from
// the HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/models/bpe/serialization.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "support/tokenizer/tokenizers/models/bpe/bpe.h"
#include "support/tokenizer/tokenizers/models/bpe/model.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers::models::bpe {
namespace {

TEST(BpeSerializationTest, TestSerialization) {
  const Vocab vocab = {
      {"<unk>", 0},
      {"a", 1},
      {"b", 2},
      {"ab", 3},
  };
  ASSERT_OK_AND_ASSIGN(BPE bpe, BpeBuilder()
                                    .VocabAndMerges(vocab, {{"a", "b"}})
                                    .UnkToken("<unk>")
                                    .IgnoreMerges(true)
                                    .Build());

  constexpr const char* kLegacy =
      R"({"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":["a b"]})";
  ASSERT_OK_AND_ASSIGN(Json legacy_json, ParseJson(kLegacy));
  ASSERT_OK_AND_ASSIGN(BPE legacy, BPE::FromJson(legacy_json));
  EXPECT_EQ(bpe, legacy);

  const std::string data = DumpJson(bpe.ToJson());
  EXPECT_EQ(
      data,
      R"({"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":[["a","b"]]})");
  ASSERT_OK_AND_ASSIGN(Json data_json, ParseJson(data));
  ASSERT_OK_AND_ASSIGN(BPE reconstructed, BPE::FromJson(data_json));
  EXPECT_EQ(bpe, reconstructed);

  // With a space in the token
  const Vocab vocab_space = {
      {"<unk>", 0},
      {"a", 1},
      {"b c d", 2},
      {"ab c d", 3},
  };
  ASSERT_OK_AND_ASSIGN(BPE bpe_space,
                       BpeBuilder()
                           .VocabAndMerges(vocab_space, {{"a", "b c d"}})
                           .UnkToken("<unk>")
                           .IgnoreMerges(true)
                           .Build());
  const std::string data_space = DumpJson(bpe_space.ToJson());
  EXPECT_EQ(
      data_space,
      R"({"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b c d":2,"ab c d":3},"merges":[["a","b c d"]]})");
  ASSERT_OK_AND_ASSIGN(Json data_space_json, ParseJson(data_space));
  ASSERT_OK_AND_ASSIGN(BPE reconstructed_space, BPE::FromJson(data_space_json));
  EXPECT_EQ(bpe_space, reconstructed_space);
}

TEST(BpeSerializationTest, TestSerializationIgnoreMerges) {
  const Vocab vocab = {{"<unk>", 0}, {"a", 1}, {"b", 2}};
  ASSERT_OK_AND_ASSIGN(BPE bpe, BpeBuilder()
                                    .VocabAndMerges(vocab, {})
                                    .UnkToken("<unk>")
                                    .IgnoreMerges(true)
                                    .Build());

  constexpr const char* kBpeString1 =
      R"({"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2},"merges":[]})";
  ASSERT_OK_AND_ASSIGN(Json json1, ParseJson(kBpeString1));
  ASSERT_OK_AND_ASSIGN(BPE parsed1, BPE::FromJson(json1));
  EXPECT_EQ(parsed1, bpe);

  bpe.set_ignore_merges(false);
  constexpr const char* kBpeString2 =
      R"({"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"vocab":{"<unk>":0,"a":1,"b":2},"merges":[]})";
  ASSERT_OK_AND_ASSIGN(Json json2, ParseJson(kBpeString2));
  ASSERT_OK_AND_ASSIGN(BPE parsed2, BPE::FromJson(json2));
  EXPECT_EQ(parsed2, bpe);
}

}  // namespace
}  // namespace tokenizers::models::bpe
