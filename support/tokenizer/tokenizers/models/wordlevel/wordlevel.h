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
// C++ port of `src/models/wordlevel/mod.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_WORDLEVEL_WORDLEVEL_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_WORDLEVEL_WORDLEVEL_H_

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
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers::models::wordlevel {

using Vocab = absl::flat_hash_map<std::string, uint32_t>;

// PORT NOTE: the Rust `thiserror` enum becomes `absl::Status` factories.
struct Error {
  // "WordLevel error: Missing [UNK] token from the vocabulary"
  static absl::Status MissingUnkToken();
  // "Bad vocabulary json file"
  static absl::Status BadVocabulary();
};

class WordLevel;

// A `WordLevelBuilder` can be used to create a `WordLevel`
// model with a custom configuration.
class WordLevelBuilder {
 public:
  // Construct a new `WordLevelBuilder`.
  WordLevelBuilder();

  // Set the input files.
  WordLevelBuilder& Files(std::string vocab);

  // Set the vocab (token -> ID) mapping.
  WordLevelBuilder& Vocab(absl::flat_hash_map<std::string, uint32_t> vocab);

  // The the `UNK` token for the vocab.
  WordLevelBuilder& UnkToken(std::string unk_token);

  // Constructs a `WordLevel` model that uses the `WordLevelBuilder`'s
  // configuration.
  //
  // PORT NOTE: the Rust method consumes the builder; here the configuration is
  // moved out, so the builder must not be reused after `Build()`.
  absl::StatusOr<WordLevel> Build();

 private:
  struct Config {
    std::optional<std::string> files;
    absl::flat_hash_map<std::string, uint32_t> vocab;
    std::string unk_token;
  };

  Config config_;
};

class WordLevel : public Model {
 public:
  // `Default`.
  WordLevel();
  ~WordLevel() override = default;

  static WordLevelBuilder Builder();

  static absl::StatusOr<absl::flat_hash_map<std::string, uint32_t>> ReadFile(
      absl::string_view vocab_path);

  // Initialize a WordLevel model from vocab and merges file.
  static absl::StatusOr<WordLevel> FromFile(absl::string_view vocab_path,
                                            std::string unk_token);

  // `impl Model for WordLevel`
  absl::StatusOr<std::vector<Token>> Tokenize(
      absl::string_view token) const override;
  std::optional<uint32_t> TokenToId(absl::string_view token) const override;
  std::optional<std::string> IdToToken(uint32_t id) const override;
  absl::flat_hash_map<std::string, uint32_t> GetVocab() const override;
  size_t GetVocabSize() const override;

  // serde (`wordlevel/serialization.rs`).
  static absl::StatusOr<WordLevel> FromJson(const Json& json);
  Json ToJson() const;

  // `PartialEq`
  bool operator==(const WordLevel& other) const;
  bool operator!=(const WordLevel& other) const { return !(*this == other); }

  const std::string& unk_token() const { return unk_token_; }
  void set_unk_token(std::string unk_token) {
    unk_token_ = std::move(unk_token);
  }

 private:
  friend class WordLevelBuilder;

  absl::flat_hash_map<std::string, uint32_t> vocab_;
  absl::flat_hash_map<uint32_t, std::string> vocab_r_;
  std::string unk_token_;
};

}  // namespace tokenizers::models::wordlevel

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_WORDLEVEL_WORDLEVEL_H_
