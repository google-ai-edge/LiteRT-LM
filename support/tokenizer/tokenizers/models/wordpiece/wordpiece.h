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
// C++ port of `src/models/wordpiece/mod.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_WORDPIECE_WORDPIECE_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_WORDPIECE_WORDPIECE_H_

// [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
// model.
//
// PORT NOTE: `trainer.rs` is not ported (training is out of scope).

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/bpe/model.h"
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers::models::wordpiece {

// PORT NOTE: the Rust `thiserror` enum becomes `absl::Status` factories.
struct Error {
  // "WordPiece error: Missing [UNK] token from the vocabulary"
  static absl::Status MissingUnkToken();
};

class WordPiece;

// A `WordPieceBuilder` can be used to create a `WordPiece` model with a custom
// configuration.
class WordPieceBuilder {
 public:
  // Construct a new `WordPieceBuilder`.
  WordPieceBuilder();

  // Set the input files.
  WordPieceBuilder& Files(std::string vocab);

  // Set the vocab (token -> ID) mapping.
  WordPieceBuilder& Vocab(absl::flat_hash_map<std::string, uint32_t> vocab);

  // The the `UNK` token for the vocab.
  WordPieceBuilder& UnkToken(std::string unk_token);

  // Set the prefix for continuing subwords.
  WordPieceBuilder& ContinuingSubwordPrefix(
      std::string continuing_subword_prefix);

  // Set the maximum number of input characters per word.
  WordPieceBuilder& MaxInputCharsPerWord(size_t max_input_chars_per_word);

  // Constructs a `WordPiece` model that uses the `WordPieceBuilder`'s
  // configuration.
  //
  // PORT NOTE: the Rust method consumes the builder; here the configuration is
  // moved out, so the builder must not be reused after `Build()`.
  absl::StatusOr<WordPiece> Build();

 private:
  struct Config {
    std::optional<std::string> files;
    absl::flat_hash_map<std::string, uint32_t> vocab;
    std::string unk_token;
    std::string continuing_subword_prefix;
    size_t max_input_chars_per_word;
  };

  Config config_;
};

// A
// [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
// model.
class WordPiece : public Model {
 public:
  // `Default`.
  WordPiece();
  ~WordPiece() override = default;

  // Get a `WordPieceBuilder`.
  static WordPieceBuilder Builder();

  // Read the given files to extract the vocab
  static absl::StatusOr<absl::flat_hash_map<std::string, uint32_t>> ReadFile(
      absl::string_view vocab);

  static absl::StatusOr<absl::flat_hash_map<std::string, uint32_t>> ReadBytes(
      absl::string_view vocab);

  static absl::StatusOr<WordPiece> FromBytes(absl::string_view bytes);

  // Initialize a `WordPiece` model from a vocab mapping file.
  static WordPieceBuilder FromFile(absl::string_view vocab);

  // Create a `WordPiece` model from a `BPE` model.
  static WordPiece FromBpe(const bpe::BPE& bpe);

  // `impl Model for WordPiece`
  absl::flat_hash_map<std::string, uint32_t> GetVocab() const override;
  size_t GetVocabSize() const override;
  absl::StatusOr<std::vector<Token>> Tokenize(
      absl::string_view sequence) const override;
  std::optional<uint32_t> TokenToId(absl::string_view token) const override;
  std::optional<std::string> IdToToken(uint32_t id) const override;

  // serde (`wordpiece/serialization.rs`).
  static absl::StatusOr<WordPiece> FromJson(const Json& json);
  Json ToJson() const;

  // `PartialEq`
  bool operator==(const WordPiece& other) const;
  bool operator!=(const WordPiece& other) const { return !(*this == other); }

  const std::string& unk_token() const { return unk_token_; }
  void set_unk_token(std::string unk_token) {
    unk_token_ = std::move(unk_token);
  }
  const std::string& continuing_subword_prefix() const {
    return continuing_subword_prefix_;
  }
  void set_continuing_subword_prefix(std::string prefix) {
    continuing_subword_prefix_ = std::move(prefix);
  }
  size_t max_input_chars_per_word() const { return max_input_chars_per_word_; }
  void set_max_input_chars_per_word(size_t max_input_chars_per_word) {
    max_input_chars_per_word_ = max_input_chars_per_word;
  }

 private:
  friend class WordPieceBuilder;

  absl::flat_hash_map<std::string, uint32_t> vocab_;
  absl::flat_hash_map<uint32_t, std::string> vocab_r_;
  std::string unk_token_;
  std::string continuing_subword_prefix_;
  size_t max_input_chars_per_word_;
};

}  // namespace tokenizers::models::wordpiece

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_WORDPIECE_WORDPIECE_H_
