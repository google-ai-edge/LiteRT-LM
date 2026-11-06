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
// C++ port of `src/models/mod.rs` (`#[cfg(test)] mod tests`) from the
// HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/models/models.h"

#include <cstdint>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/bpe/model.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers::models {
namespace {

// PORT NOTE: `trainer_wrapper_train_model_wrapper` is omitted because trainers
// (`TrainerWrapper`) are out of scope for the inference port.

TEST(ModelsTest, IncompleteOrderedVocab) {
  const absl::flat_hash_map<uint32_t, std::string> vocab_r = {{0, "Hi"},
                                                              {2, "There"}};

  const OrderedVocabIter ordered(vocab_r);

  const std::string serialized = DumpJson(ordered.Serialize());
  EXPECT_EQ(serialized, R"({"Hi":0,"There":2})");
}

TEST(ModelsTest, Serialization) {
  const bpe::Vocab vocab = {
      {"<unk>", 0},
      {"a", 1},
      {"b", 2},
      {"ab", 3},
  };
  ASSERT_OK_AND_ASSIGN(bpe::BPE bpe, bpe::BpeBuilder()
                                         .VocabAndMerges(vocab, {{"a", "b"}})
                                         .UnkToken("<unk>")
                                         .IgnoreMerges(true)
                                         .Build());

  const ModelWrapper model(std::move(bpe));
  constexpr const char* kLegacyWithType =
      R"({"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":["a b"]})";
  ASSERT_OK_AND_ASSIGN(Json legacy_json, ParseJson(kLegacyWithType));
  ASSERT_OK_AND_ASSIGN(ModelWrapper legacy,
                       ModelWrapper::FromJson(legacy_json));
  EXPECT_EQ(model, legacy);

  const std::string data = DumpJson(model.ToJson());
  EXPECT_EQ(
      data,
      R"({"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":[["a","b"]]})");
  ASSERT_OK_AND_ASSIGN(Json data_json, ParseJson(data));
  ASSERT_OK_AND_ASSIGN(ModelWrapper reconstructed,
                       ModelWrapper::FromJson(data_json));
  EXPECT_EQ(model, reconstructed);

  // Legacy check, type is not necessary.
  constexpr const char* kLegacyWithoutType =
      R"({"dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":["a b"]})";
  ASSERT_OK_AND_ASSIGN(Json legacy_untagged_json,
                       ParseJson(kLegacyWithoutType));
  ASSERT_OK_AND_ASSIGN(ModelWrapper reconstructed_legacy,
                       ModelWrapper::FromJson(legacy_untagged_json));
  EXPECT_EQ(model, reconstructed_legacy);

  constexpr const char* kInvalid =
      R"({"type":"BPE","dropout":null,"unk_token":"<unk>","continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":true,"vocab":{"<unk>":0,"a":1,"b":2,"ab":3},"merges":["a b c"]})";
  ASSERT_OK_AND_ASSIGN(Json invalid_json, ParseJson(kInvalid));
  const absl::StatusOr<ModelWrapper> invalid_reconstructed =
      ModelWrapper::FromJson(invalid_json);
  ASSERT_FALSE(invalid_reconstructed.ok());
  EXPECT_EQ(invalid_reconstructed.status().message(),
            "Merges text file invalid at line 1");
}

}  // namespace
}  // namespace tokenizers::models
