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
// C++ port of `src/models/bpe/model.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_BPE_MODEL_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_BPE_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/bpe/bpe.h"
#include "support/tokenizer/tokenizers/models/bpe/word.h"
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/utils/cache.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers::models::bpe {

using Vocab = absl::flat_hash_map<std::string, uint32_t>;
using VocabR = absl::flat_hash_map<uint32_t, std::string>;
using MergeMap = absl::flat_hash_map<Pair, std::pair<uint32_t, uint32_t>>;
using Merges = std::vector<std::pair<std::string, std::string>>;

class BPE;

// A `BpeBuilder` can be used to create a `BPE` model with a custom
// configuration.
class BpeBuilder {
 public:
  // Constructs a new `BpeBuilder`.
  BpeBuilder();

  // Set the input files.
  BpeBuilder& Files(std::string vocab, std::string merges);

  // Set the vocab (token -> ID) and merges mappings.
  BpeBuilder& VocabAndMerges(Vocab vocab, Merges merges);

  // Set the cache's capacity. Set to 0 if you want to disable caching.
  BpeBuilder& CacheCapacity(size_t capacity);

  // Use [dropout](https://arxiv.org/abs/1910.13267) with the model.
  BpeBuilder& Dropout(float dropout);

  // Set the `UNK` token for the vocab.
  BpeBuilder& UnkToken(std::string unk_token);

  // Set the `continuing_subword_prefix` option.
  BpeBuilder& ContinuingSubwordPrefix(std::string prefix);

  // Set the `end_of_word_suffix` option.
  BpeBuilder& EndOfWordSuffix(std::string prefix);

  // Set the `fuse_unk` option.
  BpeBuilder& FuseUnk(bool fuse_unk);

  // Set the `byte_fallback` option.
  BpeBuilder& ByteFallback(bool byte_fallback);

  // Set the `ignore_merges` option.
  BpeBuilder& IgnoreMerges(bool ignore_merges);

  // Returns a `BPE` model that uses the `BpeBuilder`'s configuration.
  //
  // PORT NOTE: the Rust method consumes the builder; here the configuration is
  // moved out, so the builder must not be reused after `Build()`.
  absl::StatusOr<BPE> Build();

 private:
  struct Config {
    std::optional<std::pair<std::string, std::string>> files;
    Vocab vocab;
    Merges merges;
    size_t cache_capacity;
    std::optional<float> dropout;
    std::optional<std::string> unk_token;
    std::optional<std::string> continuing_subword_prefix;
    std::optional<std::string> end_of_word_suffix;
    bool fuse_unk;
    bool byte_fallback;
    bool ignore_merges;
  };

  Config config_;
};

// Converts the merges strings (for example from `merges.txt` file) with the
// format "{pair_a} {pair_b}" into the format expected by the BPE struct
absl::StatusOr<Merges> ConvertMergesToHashmap(
    const std::vector<std::string>& lines, const Vocab& vocab);

// A [Byte Pair Encoding](https://www.aclweb.org/anthology/P16-1162/) model.
class BPE : public Model {
 public:
  // `Default`: an empty model.
  BPE();

  // `Clone` can't be derive because it's not implemented for `BpeCache`.
  // To keep things simple when we clone, the new BPE will start with a fresh
  // cache.
  BPE(const BPE& other);
  BPE& operator=(const BPE& other);
  BPE(BPE&& other) = default;
  BPE& operator=(BPE&& other) = default;
  ~BPE() override = default;

  // Initialize a `BpeBuilder`.
  static BpeBuilder Builder();

  // Create a new BPE model with the given vocab and merges.
  //
  // PORT NOTE: the Rust constructor `unwrap()`s the builder result; invalid
  // merges are reported as a status instead of a panic.
  static absl::StatusOr<BPE> New(Vocab vocab, Merges merges);

  // Initialize a BpeBuilder model from vocab and merges files
  static BpeBuilder FromFile(absl::string_view vocab, absl::string_view merges);

  // Read the given files to extract the vocab and merges
  static absl::StatusOr<std::pair<Vocab, Merges>> ReadFile(
      absl::string_view vocab, absl::string_view merges);

  // Reset the cache.
  void ClearCache() const;

  // Resize the cache
  void ResizeCache(size_t capacity);

  const std::optional<std::string>& GetUnkToken() const { return unk_token_; }

  const std::optional<std::string>& GetContinuingSubwordPrefix() const {
    return continuing_subword_prefix_;
  }

  // `impl Model for BPE`
  absl::flat_hash_map<std::string, uint32_t> GetVocab() const override;
  size_t GetVocabSize() const override;
  absl::StatusOr<std::vector<Token>> Tokenize(
      absl::string_view sequence) const override;
  std::optional<uint32_t> TokenToId(absl::string_view token) const override;
  std::optional<std::string> IdToToken(uint32_t id) const override;

  // serde (`bpe/serialization.rs`).
  static absl::StatusOr<BPE> FromJson(const Json& json);
  Json ToJson() const;

  // `PartialEq` (the cache never takes part in the comparison).
  bool operator==(const BPE& other) const;
  bool operator!=(const BPE& other) const { return !(*this == other); }

  // The vocabulary assigns a number to each token.
  const Vocab& vocab() const { return vocab_; }
  // Reversed vocabulary, to rebuild sentences.
  const VocabR& vocab_r() const { return vocab_r_; }
  // Contains the mapping between Pairs and their (rank, new_id).
  const MergeMap& merges() const { return merges_; }

  // Dropout probability for merges. 0.0 = no dropout is the default. At 1.0,
  // tokenization will perform no merges, so the result will just be characters.
  std::optional<float> dropout() const { return dropout_; }
  void set_dropout(std::optional<float> dropout) { dropout_ = dropout; }
  // The unknown token to be used when we encounter an unknown char
  const std::optional<std::string>& unk_token() const { return unk_token_; }
  void set_unk_token(std::optional<std::string> unk_token) {
    unk_token_ = std::move(unk_token);
  }
  // An optional prefix to use on any subword that exist only behind another one
  const std::optional<std::string>& continuing_subword_prefix() const {
    return continuing_subword_prefix_;
  }
  void set_continuing_subword_prefix(std::optional<std::string> prefix) {
    continuing_subword_prefix_ = std::move(prefix);
  }
  // An optional suffix to characterize and end-of-word subword
  const std::optional<std::string>& end_of_word_suffix() const {
    return end_of_word_suffix_;
  }
  void set_end_of_word_suffix(std::optional<std::string> suffix) {
    end_of_word_suffix_ = std::move(suffix);
  }
  // Do multiple unk tokens get fused
  bool fuse_unk() const { return fuse_unk_; }
  void set_fuse_unk(bool fuse_unk) { fuse_unk_ = fuse_unk; }
  // Byte fallback from sentence pieces, instead of UNK, uses `"<0x00>"`
  // for each byte in the unk token
  bool byte_fallback() const { return byte_fallback_; }
  void set_byte_fallback(bool byte_fallback) { byte_fallback_ = byte_fallback; }
  // Whether or not to direct output words if they are part of the vocab.
  bool ignore_merges() const { return ignore_merges_; }
  void set_ignore_merges(bool ignore_merges) { ignore_merges_ = ignore_merges; }

 private:
  friend class BpeBuilder;

  absl::StatusOr<Word> MergeWord(absl::string_view w) const;
  std::vector<Token> WordToTokens(const Word& word) const;
  absl::StatusOr<std::vector<Token>> TokenizeWithCache(
      absl::string_view sequence) const;

  Vocab vocab_;
  VocabR vocab_r_;
  MergeMap merges_;
  // Contains the cache for optimizing the encoding step.
  //
  // PORT NOTE: the Rust crate keeps a per-thread `BPE_LOCAL_CACHE` keyed by a
  // per-instance `BpeCache` id. This port uses the shared, mutex-protected
  // `Cache` from utils/cache.h (the crate's previous design) which has the
  // same observable semantics: a best-effort memo of merged `Word`s per BPE
  // instance, with `ClearCache()` / `ResizeCache()` keeping their meaning. A
  // null pointer is `Option::None` (cache disabled with capacity 0).
  std::unique_ptr<Cache<std::string, Word>> cache_;
  std::optional<float> dropout_;
  std::optional<std::string> unk_token_;
  std::optional<std::string> continuing_subword_prefix_;
  std::optional<std::string> end_of_word_suffix_;
  bool fuse_unk_ = false;
  bool byte_fallback_ = false;
  bool ignore_merges_ = false;
};

}  // namespace tokenizers::models::bpe

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_BPE_MODEL_H_
