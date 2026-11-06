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

#include "support/tokenizer/tokenizers/models/wordpiece/wordpiece.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/bpe/model.h"
#include "support/tokenizer/tokenizers/models/models.h"
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/unicode/categories.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/status_macros.h"

namespace tokenizers::models::wordpiece {

namespace {

// `str::trim_end()`: strips trailing Unicode `White_Space` characters.
absl::string_view TrimEnd(absl::string_view s) {
  size_t end = 0;
  for (const auto [offset, c] : CharIndices(s)) {
    if (!unicode::IsWhitespace(c)) {
      end = offset + Utf8Len(c);
    }
  }
  return s.substr(0, end);
}

// `substr.chars().last().map_or(1, |c| c.len_utf8())`: the byte length of the
// last character of a (valid UTF-8) string.
size_t LastCharLen(absl::string_view s) {
  if (s.empty()) {
    return 1;
  }
  size_t i = s.size() - 1;
  while (i > 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80 &&
         s.size() - i < 4) {
    --i;
  }
  return s.size() - i;
}

}  // namespace

absl::Status Error::MissingUnkToken() {
  return absl::InvalidArgumentError(
      "WordPiece error: Missing [UNK] token from the vocabulary");
}

WordPieceBuilder::WordPieceBuilder()
    : config_{
          /*files=*/std::nullopt,
          /*vocab=*/{},
          /*unk_token=*/"[UNK]",
          /*continuing_subword_prefix=*/"##",
          /*max_input_chars_per_word=*/100,
      } {}

WordPieceBuilder& WordPieceBuilder::Files(std::string vocab) {
  config_.files = std::move(vocab);
  return *this;
}

WordPieceBuilder& WordPieceBuilder::Vocab(
    absl::flat_hash_map<std::string, uint32_t> vocab) {
  config_.vocab = std::move(vocab);
  return *this;
}

WordPieceBuilder& WordPieceBuilder::UnkToken(std::string unk_token) {
  config_.unk_token = std::move(unk_token);
  return *this;
}

WordPieceBuilder& WordPieceBuilder::ContinuingSubwordPrefix(
    std::string continuing_subword_prefix) {
  config_.continuing_subword_prefix = std::move(continuing_subword_prefix);
  return *this;
}

WordPieceBuilder& WordPieceBuilder::MaxInputCharsPerWord(
    size_t max_input_chars_per_word) {
  config_.max_input_chars_per_word = max_input_chars_per_word;
  return *this;
}

absl::StatusOr<WordPiece> WordPieceBuilder::Build() {
  if (config_.files.has_value()) {
    ASSIGN_OR_RETURN(config_.vocab, WordPiece::ReadFile(*config_.files));
  }

  absl::flat_hash_map<uint32_t, std::string> vocab_r;
  vocab_r.reserve(config_.vocab.size());
  for (const auto& [key, val] : config_.vocab) {
    vocab_r[val] = key;
  }

  WordPiece wp;
  wp.vocab_ = std::move(config_.vocab);
  wp.vocab_r_ = std::move(vocab_r);
  wp.unk_token_ = std::move(config_.unk_token);
  wp.continuing_subword_prefix_ = std::move(config_.continuing_subword_prefix);
  wp.max_input_chars_per_word_ = config_.max_input_chars_per_word;
  return wp;
}

WordPiece::WordPiece()
    : unk_token_("[UNK]"),
      continuing_subword_prefix_("##"),
      max_input_chars_per_word_(100) {}

WordPieceBuilder WordPiece::Builder() { return WordPieceBuilder(); }

absl::StatusOr<absl::flat_hash_map<std::string, uint32_t>> WordPiece::ReadFile(
    absl::string_view vocab) {
  ASSIGN_OR_RETURN(const std::string file, ReadToString(vocab));
  return ReadBytes(file);
}

absl::StatusOr<absl::flat_hash_map<std::string, uint32_t>> WordPiece::ReadBytes(
    absl::string_view vocab) {
  ASSIGN_OR_RETURN(const std::vector<std::string> lines, BufReadLines(vocab));

  absl::flat_hash_map<std::string, uint32_t> result;
  for (size_t index = 0; index < lines.size(); ++index) {
    result[std::string(TrimEnd(lines[index]))] = static_cast<uint32_t>(index);
  }

  return result;
}

absl::StatusOr<WordPiece> WordPiece::FromBytes(absl::string_view bytes) {
  ASSIGN_OR_RETURN(const Json json, ParseJson(bytes));
  return FromJson(json);
}

WordPieceBuilder WordPiece::FromFile(absl::string_view vocab) {
  return WordPiece::Builder().Files(std::string(vocab));
}

WordPiece WordPiece::FromBpe(const bpe::BPE& bpe) {
  // Cannot fail: no files are involved.
  WordPiece wp = Builder().Vocab(bpe.GetVocab()).Build().value();
  if (bpe.GetUnkToken().has_value()) {
    wp.unk_token_ = *bpe.GetUnkToken();
  }
  if (bpe.GetContinuingSubwordPrefix().has_value()) {
    wp.continuing_subword_prefix_ = *bpe.GetContinuingSubwordPrefix();
  }
  return wp;
}

absl::flat_hash_map<std::string, uint32_t> WordPiece::GetVocab() const {
  return vocab_;
}

size_t WordPiece::GetVocabSize() const { return vocab_.size(); }

absl::StatusOr<std::vector<Token>> WordPiece::Tokenize(
    absl::string_view sequence) const {
  const size_t char_len = CountChars(sequence);
  if (char_len > max_input_chars_per_word_) {
    auto unk = vocab_.find(unk_token_);
    if (unk == vocab_.end()) {
      return Error::MissingUnkToken();
    }
    return std::vector<Token>{
        Token{unk->second, unk_token_, Offsets(0, sequence.size())}};
  }

  bool is_bad = false;
  size_t start = 0;
  std::vector<Token> sub_tokens;

  while (start < sequence.size()) {
    size_t end = sequence.size();
    std::optional<Token> cur_str;

    while (start < end) {
      std::string substr(sequence.substr(start, end - start));

      if (start > 0) {
        substr = absl::StrCat(continuing_subword_prefix_, substr);
      }
      auto id = vocab_.find(substr);
      if (id != vocab_.end()) {
        cur_str = Token{id->second, substr, Offsets(start, end)};
        break;
      }
      end -= LastCharLen(substr);
    }

    if (!cur_str.has_value()) {
      is_bad = true;
      break;
    }

    sub_tokens.push_back(*std::move(cur_str));
    start = end;
  }

  if (is_bad) {
    auto unk = vocab_.find(unk_token_);
    if (unk == vocab_.end()) {
      return Error::MissingUnkToken();
    }
    return std::vector<Token>{
        Token{unk->second, unk_token_, Offsets(0, sequence.size())}};
  } else {
    return sub_tokens;
  }
}

std::optional<uint32_t> WordPiece::TokenToId(absl::string_view token) const {
  auto it = vocab_.find(token);
  if (it == vocab_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<std::string> WordPiece::IdToToken(uint32_t id) const {
  auto it = vocab_r_.find(id);
  if (it == vocab_r_.end()) {
    return std::nullopt;
  }
  return it->second;
}

bool WordPiece::operator==(const WordPiece& other) const {
  return vocab_ == other.vocab_ && vocab_r_ == other.vocab_r_ &&
         unk_token_ == other.unk_token_ &&
         continuing_subword_prefix_ == other.continuing_subword_prefix_ &&
         max_input_chars_per_word_ == other.max_input_chars_per_word_;
}

}  // namespace tokenizers::models::wordpiece
