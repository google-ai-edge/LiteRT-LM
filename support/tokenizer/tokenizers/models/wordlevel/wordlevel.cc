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

#include "support/tokenizer/tokenizers/models/wordlevel/wordlevel.h"

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
#include "support/tokenizer/tokenizers/models/models.h"
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers::models::wordlevel {

absl::Status Error::MissingUnkToken() {
  return absl::InvalidArgumentError(
      "WordLevel error: Missing [UNK] token from the vocabulary");
}

absl::Status Error::BadVocabulary() {
  return absl::InvalidArgumentError("Bad vocabulary json file");
}

WordLevelBuilder::WordLevelBuilder()
    : config_{
          /*files=*/std::nullopt,
          /*vocab=*/{},
          /*unk_token=*/"<unk>",
      } {}

WordLevelBuilder& WordLevelBuilder::Files(std::string vocab) {
  config_.files = std::move(vocab);
  return *this;
}

WordLevelBuilder& WordLevelBuilder::Vocab(
    absl::flat_hash_map<std::string, uint32_t> vocab) {
  config_.vocab = std::move(vocab);
  return *this;
}

WordLevelBuilder& WordLevelBuilder::UnkToken(std::string unk_token) {
  config_.unk_token = std::move(unk_token);
  return *this;
}

absl::StatusOr<WordLevel> WordLevelBuilder::Build() {
  if (config_.files.has_value()) {
    ASSIGN_OR_RETURN(config_.vocab, WordLevel::ReadFile(*config_.files));
  }

  absl::flat_hash_map<uint32_t, std::string> vocab_r;
  vocab_r.reserve(config_.vocab.size());
  for (const auto& [key, val] : config_.vocab) {
    vocab_r[val] = key;
  }

  WordLevel wl;
  wl.vocab_ = std::move(config_.vocab);
  wl.vocab_r_ = std::move(vocab_r);
  wl.unk_token_ = std::move(config_.unk_token);
  return wl;
}

WordLevel::WordLevel() : unk_token_("<unk>") {}

WordLevelBuilder WordLevel::Builder() { return WordLevelBuilder(); }

absl::StatusOr<absl::flat_hash_map<std::string, uint32_t>> WordLevel::ReadFile(
    absl::string_view vocab_path) {
  ASSIGN_OR_RETURN(const std::string buffer, ReadToString(vocab_path));
  absl::flat_hash_map<std::string, uint32_t> vocab;

  ASSIGN_OR_RETURN(const Json json, ParseJson(buffer));

  if (json.is_object()) {
    for (const auto& [token, id] : json.items()) {
      if (id.is_number()) {
        if (!id.is_number_unsigned()) {
          return Error::BadVocabulary();
        }
        vocab[token] = static_cast<uint32_t>(id.template get<uint64_t>());
      }
    }
  } else {
    return Error::BadVocabulary();
  }
  return vocab;
}

absl::StatusOr<WordLevel> WordLevel::FromFile(absl::string_view vocab_path,
                                              std::string unk_token) {
  ASSIGN_OR_RETURN(auto vocab, WordLevel::ReadFile(vocab_path));
  return Builder()
      .Vocab(std::move(vocab))
      .UnkToken(std::move(unk_token))
      .Build();
}

absl::StatusOr<std::vector<Token>> WordLevel::Tokenize(
    absl::string_view token) const {
  if (auto id = vocab_.find(token); id != vocab_.end()) {
    return std::vector<Token>{
        Token{id->second, std::string(token), Offsets(0, token.size())}};
  } else if (auto unk_id = vocab_.find(unk_token_); unk_id != vocab_.end()) {
    return std::vector<Token>{
        Token{unk_id->second, unk_token_, Offsets(0, token.size())}};
  } else {
    return Error::MissingUnkToken();
  }
}

std::optional<uint32_t> WordLevel::TokenToId(absl::string_view token) const {
  auto it = vocab_.find(token);
  if (it == vocab_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<std::string> WordLevel::IdToToken(uint32_t id) const {
  auto it = vocab_r_.find(id);
  if (it == vocab_r_.end()) {
    return std::nullopt;
  }
  return it->second;
}

absl::flat_hash_map<std::string, uint32_t> WordLevel::GetVocab() const {
  return vocab_;
}

size_t WordLevel::GetVocabSize() const { return vocab_.size(); }

bool WordLevel::operator==(const WordLevel& other) const {
  return vocab_ == other.vocab_ && vocab_r_ == other.vocab_r_ &&
         unk_token_ == other.unk_token_;
}

}  // namespace tokenizers::models::wordlevel
