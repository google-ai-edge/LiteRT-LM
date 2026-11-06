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
// C++ port of `src/models/unigram/model.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_UNIGRAM_MODEL_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_UNIGRAM_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/unigram/lattice.h"
#include "support/tokenizer/tokenizers/models/unigram/trie.h"
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/utils/cache.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers::models::unigram {

using TokenMap = absl::flat_hash_map<std::string, uint32_t>;
using Vocab = std::vector<std::pair<std::string, double>>;

// PORT NOTE: the Rust `thiserror` enum becomes `absl::Status` factories.
struct UnigramError {
  // "The vocabulary is empty but at least <unk> is needed"
  static absl::Status EmptyVocabulary();
  // "The `unk_id` is larger than vocabulary size"
  static absl::Status UnkIdNotInVocabulary();
  // "Encountered an unknown token but `unk_id` is missing"
  static absl::Status MissingUnkId();
};

class UnigramIterator;

// A `Unigram` model to encode sentences.
class Unigram : public Model {
 public:
  // `Default`: a model whose only token is `<unk>`.
  Unigram();

  // `Clone` can't be derive because it's not implemented for `Cache`.
  // To keep things simple when we clone, the new Unigram will start with a
  // fresh cache.
  Unigram(const Unigram& other);
  Unigram& operator=(const Unigram& other);
  Unigram(Unigram&& other) = default;
  Unigram& operator=(Unigram&& other) = default;
  ~Unigram() override = default;

  // Create a `Unigram` model from a given vocabulary.
  // Vocabulary are the various tokens and their associated score which is a
  // sort of a logprob of their frequency, which will enable tokenization and
  // sampling. unk_id, is the index within the vocabulary. For now `Unigram`
  // *requires* at least `unk` because we might find a never seen char. Further
  // versions might allow that part to be hidden.
  static absl::StatusOr<Unigram> From(Vocab vocab, std::optional<size_t> unk_id,
                                      bool byte_fallback);

  // `#[cfg(test)]` helpers of the Rust model, public here for the tests.
  void SetFuseUnk(bool fuse_unk);
  void SetOptimized(bool is_optimized);

  bool byte_fallback() const { return byte_fallback_; }

  size_t Len() const { return vocab_.size(); }

  void PopulateNodes(Lattice& lattice) const;

  // This functions take a String, and will encode it in a Vec of Strings,
  // of the best tokenization available to the current model.
  // ```
  // use tokenizers::models::unigram::Unigram;
  //
  // let pieces = vec![
  //     ("<unk>".to_string(), 0.0),
  //     ("a".to_string(), 0.0),
  //     ("b".to_string(), 0.0),
  //     ("c".to_string(), 0.0),
  //     ("d".to_string(), 0.0),
  //     ("cd".to_string(), 1.0),
  //     ("ab".to_string(), 2.0),
  //     ("abc".to_string(), 5.0),
  //     ("abcd".to_string(), 10.0),
  // ];
  // let model = Unigram::from(pieces, Some(0), false).unwrap();
  // let result = model.encode("abcdacdxx").unwrap();
  // assert_eq!(result, vec!["abcd", "a", "cd", "xx"]);
  // ```
  absl::StatusOr<std::vector<std::string>> Encode(
      absl::string_view sentence) const;

  // Iterate of vocabulary of the model as a pair of `(token, score)`.
  UnigramIterator Iter() const;

  // Loads a SentencePiece output model after being trained by tokenizers.
  // After that you can use the model with tokenizers library.
  // ```no_run
  // use tokenizers::models::unigram::Unigram;
  // use std::path::Path;
  //
  // let model = Unigram::load("mymodel-unigram.json").unwrap();
  // ```
  static absl::StatusOr<Unigram> Load(absl::string_view path);

  // Clears the internal cache
  void ClearCache();

  // Resize the cache
  void ResizeCache(size_t capacity);

  // `impl Model for Unigram`
  absl::flat_hash_map<std::string, uint32_t> GetVocab() const override;
  size_t GetVocabSize() const override;
  absl::StatusOr<std::vector<Token>> Tokenize(
      absl::string_view sentence) const override;
  std::optional<uint32_t> TokenToId(absl::string_view token) const override;
  std::optional<std::string> IdToToken(uint32_t id) const override;

  // serde (`unigram/serialization.rs`).
  static absl::StatusOr<Unigram> FromJson(const Json& json);
  Json ToJson() const;

  // `PartialEq`
  bool operator==(const Unigram& other) const;
  bool operator!=(const Unigram& other) const { return !(*this == other); }

  // `pub(crate)` / `pub(super)` / `pub` fields.
  const Vocab& vocab() const { return vocab_; }
  double min_score() const { return min_score_; }
  std::optional<size_t> unk_id() const { return unk_id_; }
  size_t bos_id() const { return bos_id_; }
  size_t eos_id() const { return eos_id_; }
  std::optional<double> alpha() const { return alpha_; }
  void set_alpha(std::optional<double> alpha) { alpha_ = alpha; }
  std::optional<size_t> nbest_size() const { return nbest_size_; }
  void set_nbest_size(std::optional<size_t> nbest_size) {
    nbest_size_ = nbest_size;
  }

 private:
  Unigram(Vocab vocab, TokenMap token_to_ids, Trie<uint8_t> trie,
          double min_score, std::optional<size_t> unk_id, size_t bos_id,
          size_t eos_id, bool fuse_unk, bool is_optimized, bool byte_fallback);

  absl::StatusOr<std::vector<std::string>> EncodeOptimized(
      absl::string_view sentence) const;
  absl::StatusOr<std::vector<std::string>> EncodeUnoptimized(
      absl::string_view sentence) const;

  TokenMap token_to_ids_;
  Vocab vocab_;
  std::unique_ptr<Cache<std::string, std::vector<std::string>>> cache_;
  Trie<uint8_t> trie_;
  double min_score_;
  std::optional<size_t> unk_id_;
  size_t bos_id_;
  size_t eos_id_;

  bool fuse_unk_;
  bool is_optimized_;
  bool byte_fallback_;

  std::optional<double> alpha_;
  std::optional<size_t> nbest_size_;
};

// Iterator to iterate of vocabulary of the model, and their relative score.
class UnigramIterator {
 public:
  explicit UnigramIterator(const Unigram* model) : model_(model), i_(0) {}

  // Returns the next `(token, score)` pair, or nullptr at the end.
  const std::pair<std::string, double>* Next();

 private:
  const Unigram* model_;
  size_t i_;
};

}  // namespace tokenizers::models::unigram

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_UNIGRAM_MODEL_H_
