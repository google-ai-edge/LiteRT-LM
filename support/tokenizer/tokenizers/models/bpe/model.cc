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

#include "support/tokenizer/tokenizers/models/bpe/model.h"

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
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/bpe/bpe.h"
#include "support/tokenizer/tokenizers/models/bpe/word.h"
#include "support/tokenizer/tokenizers/models/models.h"
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/utils/cache.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/status_macros.h"

namespace tokenizers::models::bpe {

BpeBuilder::BpeBuilder()
    : config_{
          /*files=*/std::nullopt,
          /*vocab=*/{},
          /*merges=*/{},
          /*cache_capacity=*/kDefaultCacheCapacity,
          /*dropout=*/std::nullopt,
          /*unk_token=*/std::nullopt,
          /*continuing_subword_prefix=*/std::nullopt,
          /*end_of_word_suffix=*/std::nullopt,
          /*fuse_unk=*/false,
          /*byte_fallback=*/false,
          /*ignore_merges=*/false,
      } {}

BpeBuilder& BpeBuilder::Files(std::string vocab, std::string merges) {
  config_.files = std::make_pair(std::move(vocab), std::move(merges));
  return *this;
}

BpeBuilder& BpeBuilder::VocabAndMerges(Vocab vocab, Merges merges) {
  config_.vocab = std::move(vocab);
  config_.merges = std::move(merges);
  return *this;
}

BpeBuilder& BpeBuilder::CacheCapacity(size_t capacity) {
  config_.cache_capacity = capacity;
  return *this;
}

BpeBuilder& BpeBuilder::Dropout(float dropout) {
  config_.dropout = dropout;
  return *this;
}

BpeBuilder& BpeBuilder::UnkToken(std::string unk_token) {
  config_.unk_token = std::move(unk_token);
  return *this;
}

BpeBuilder& BpeBuilder::ContinuingSubwordPrefix(std::string prefix) {
  config_.continuing_subword_prefix = std::move(prefix);
  return *this;
}

BpeBuilder& BpeBuilder::EndOfWordSuffix(std::string prefix) {
  config_.end_of_word_suffix = std::move(prefix);
  return *this;
}

BpeBuilder& BpeBuilder::FuseUnk(bool fuse_unk) {
  config_.fuse_unk = fuse_unk;
  return *this;
}

BpeBuilder& BpeBuilder::ByteFallback(bool byte_fallback) {
  config_.byte_fallback = byte_fallback;
  return *this;
}

BpeBuilder& BpeBuilder::IgnoreMerges(bool ignore_merges) {
  config_.ignore_merges = ignore_merges;
  return *this;
}

absl::StatusOr<BPE> BpeBuilder::Build() {
  // Validate dropout.
  if (config_.dropout.has_value()) {
    const float p = *config_.dropout;
    if (!(p >= 0.0f && p <= 1.0f)) {
      return Error::InvalidDropout();
    }
  }

  // Read files if necessary
  if (config_.files.has_value()) {
    ASSIGN_OR_RETURN(
        auto read, BPE::ReadFile(config_.files->first, config_.files->second));
    config_.vocab = std::move(read.first);
    config_.merges = std::move(read.second);
  }

  VocabR vocab_r;
  vocab_r.reserve(config_.vocab.size());
  for (const auto& [key, val] : config_.vocab) {
    vocab_r[val] = key;
  }
  std::unique_ptr<Cache<std::string, Word>> cache;
  if (config_.cache_capacity != 0) {
    cache = std::make_unique<Cache<std::string, Word>>(config_.cache_capacity);
  }

  Vocab vocab = std::move(config_.vocab);
  const size_t prefix_len = config_.continuing_subword_prefix.has_value()
                                ? config_.continuing_subword_prefix->size()
                                : 0;
  MergeMap merge_map;
  merge_map.reserve(config_.merges.size());
  for (size_t i = 0; i < config_.merges.size(); ++i) {
    const std::string& a = config_.merges[i].first;
    const std::string& b = config_.merges[i].second;
    auto a_id = vocab.find(a);
    if (a_id == vocab.end()) {
      return Error::MergeTokenOutOfVocabulary(a);
    }
    auto b_id = vocab.find(b);
    if (b_id == vocab.end()) {
      return Error::MergeTokenOutOfVocabulary(b);
    }
    // PORT NOTE: the Rust code slices `b` past `prefix_len` bytes into a
    // fixed buffer and panics when `b` is shorter than the prefix (or the
    // merged token is longer than any vocab entry); both become errors here.
    if (b.size() < prefix_len) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Token `", b, "` is shorter than the continuing_subword_prefix"));
    }
    const std::string new_token =
        absl::StrCat(a, absl::string_view(b).substr(prefix_len));
    auto new_id = vocab.find(new_token);
    if (new_id == vocab.end()) {
      return Error::MergeTokenOutOfVocabulary(new_token);
    }
    merge_map[Pair(a_id->second, b_id->second)] =
        std::make_pair(static_cast<uint32_t>(i), new_id->second);
  }

  // merges.insert(pair, (rank as u32, *new_id));

  BPE bpe;
  bpe.vocab_ = std::move(vocab);
  bpe.vocab_r_ = std::move(vocab_r);
  bpe.merges_ = std::move(merge_map);
  bpe.cache_ = std::move(cache);
  bpe.dropout_ = config_.dropout;
  bpe.unk_token_ = std::move(config_.unk_token);
  bpe.continuing_subword_prefix_ = std::move(config_.continuing_subword_prefix);
  bpe.end_of_word_suffix_ = std::move(config_.end_of_word_suffix);
  bpe.fuse_unk_ = config_.fuse_unk;
  bpe.byte_fallback_ = config_.byte_fallback;
  bpe.ignore_merges_ = config_.ignore_merges;
  return bpe;
}

BPE::BPE()
    : cache_(
          std::make_unique<Cache<std::string, Word>>(kDefaultCacheCapacity)) {}

BPE::BPE(const BPE& other)
    : vocab_(other.vocab_),
      vocab_r_(other.vocab_r_),
      merges_(other.merges_),
      cache_(other.cache_ == nullptr
                 ? nullptr
                 : std::make_unique<Cache<std::string, Word>>(
                       other.cache_->capacity())),
      dropout_(other.dropout_),
      unk_token_(other.unk_token_),
      continuing_subword_prefix_(other.continuing_subword_prefix_),
      end_of_word_suffix_(other.end_of_word_suffix_),
      fuse_unk_(other.fuse_unk_),
      byte_fallback_(other.byte_fallback_),
      ignore_merges_(other.ignore_merges_) {}

BPE& BPE::operator=(const BPE& other) {
  if (this != &other) {
    *this = BPE(other);
  }
  return *this;
}

absl::StatusOr<Merges> ConvertMergesToHashmap(
    const std::vector<std::string>& lines, const Vocab& vocab) {
  Merges merges;

  size_t rank = 0;
  for (const std::string& line : lines) {
    if (absl::StartsWith(line, "#version")) {
      continue;
    }
    const std::vector<absl::string_view> parts = absl::StrSplit(line, ' ');
    if (parts.size() != 2) {
      return Error::BadMerges(rank + 1);
    }

    merges.push_back({std::string(parts[0]), std::string(parts[1])});
    ++rank;
  }

  return merges;
}

BpeBuilder BPE::Builder() { return BpeBuilder(); }

absl::StatusOr<BPE> BPE::New(Vocab vocab, Merges merges) {
  return Builder().VocabAndMerges(std::move(vocab), std::move(merges)).Build();
}

BpeBuilder BPE::FromFile(absl::string_view vocab, absl::string_view merges) {
  return Builder().Files(std::string(vocab), std::string(merges));
}

absl::StatusOr<std::pair<Vocab, Merges>> BPE::ReadFile(
    absl::string_view vocab, absl::string_view merges) {
  // Read vocab.json
  absl::StatusOr<std::string> buffer = ReadToString(vocab);
  if (!buffer.ok()) {
    return Error::Io(buffer.status().message());
  }
  absl::StatusOr<Json> json = ParseJson(*buffer);
  if (!json.ok()) {
    return Error::JsonError(json.status().message());
  }
  Vocab vocab_map;
  if (json->is_object()) {
    for (const auto& [token, id] : json->items()) {
      if (id.is_number()) {
        if (!id.is_number_unsigned()) {
          return Error::BadVocabulary();
        }
        vocab_map[token] = static_cast<uint32_t>(id.template get<uint64_t>());
      }
    }
  } else {
    return Error::BadVocabulary();
  }

  // Read merges file
  absl::StatusOr<std::string> merge_file = ReadToString(merges);
  if (!merge_file.ok()) {
    return Error::Io(merge_file.status().message());
  }
  absl::StatusOr<std::vector<std::string>> lines = BufReadLines(*merge_file);
  if (!lines.ok()) {
    return Error::Io(lines.status().message());
  }
  ASSIGN_OR_RETURN(Merges merge_list,
                   ConvertMergesToHashmap(*lines, vocab_map));

  return std::make_pair(std::move(vocab_map), std::move(merge_list));
}

void BPE::ClearCache() const {
  if (cache_ != nullptr) {
    cache_->Clear();
  }
}

void BPE::ResizeCache(size_t capacity) {
  if (cache_ != nullptr) {
    cache_->Resize(capacity);
  }
}

absl::flat_hash_map<std::string, uint32_t> BPE::GetVocab() const {
  return vocab_;
}

size_t BPE::GetVocabSize() const { return vocab_.size(); }

absl::StatusOr<Word> BPE::MergeWord(absl::string_view w) const {
  Word word = Word::WithCapacity(w.size());
  std::optional<std::pair<uint32_t, size_t>> unk;
  size_t i = 0;
  while (i < w.size()) {
    // `char_indices().peekable()`: `end` is the byte offset of the next char.
    size_t end = i;
    DecodeUtf8(w, &end);
    const bool is_first = i == 0;
    const bool is_last = end >= w.size();

    std::string s(w.substr(i, end - i));
    const size_t byte_len = s.size();

    // Add the `continuing_subword_prefix` if relevant
    if (!is_first) {
      if (continuing_subword_prefix_.has_value()) {
        s = absl::StrCat(*continuing_subword_prefix_, s);
      }
    }
    // Add the `end_of_word_suffix` if relevant
    if (is_last) {
      if (end_of_word_suffix_.has_value()) {
        s = absl::StrCat(s, *end_of_word_suffix_);
      }
    }

    auto id = vocab_.find(s);
    if (id != vocab_.end()) {
      if (unk.has_value()) {
        word.Add(unk->first, unk->second);
        unk.reset();
      }
      word.Add(id->second, byte_len);
    } else {
      if (byte_fallback_) {
        std::optional<std::vector<uint32_t>> tokens = std::vector<uint32_t>();
        for (const unsigned char b : s) {
          const std::string code = absl::StrFormat("<0x%02X>", b);

          auto byte_id = vocab_.find(code);
          if (byte_id == vocab_.end()) {
            tokens.reset();
            break;
          }
          tokens->push_back(byte_id->second);
        }
        if (tokens.has_value()) {
          for (const uint32_t t : *tokens) {
            word.Add(t, 1);
          }
          i = end;
          continue;
        }
      }
      if (unk_token_.has_value()) {
        if (unk.has_value() && fuse_unk_) {
          // Fuse unk
          unk = std::make_pair(unk->first, unk->second + byte_len);
        } else if (unk.has_value() && !fuse_unk_) {
          // Do not fuse unk, add the previous one
          word.Add(unk->first, unk->second);
          auto unk_id = vocab_.find(*unk_token_);
          if (unk_id == vocab_.end()) {
            return Error::UnkTokenOutOfVocabulary(*unk_token_);
          }
          unk = std::make_pair(unk_id->second, byte_len);
        } else {
          auto unk_id = vocab_.find(*unk_token_);
          if (unk_id == vocab_.end()) {
            return Error::UnkTokenOutOfVocabulary(*unk_token_);
          }
          unk = std::make_pair(unk_id->second, byte_len);
        }
      }
    }
    i = end;
  }
  if (unk.has_value()) {
    word.Add(unk->first, unk->second);
  }

  word.MergeAll(merges_, dropout_);

  return word;
}

std::vector<Token> BPE::WordToTokens(const Word& word) const {
  std::vector<Token> tokens;
  tokens.reserve(word.symbols().size());
  size_t pos = 0;
  for (const Symbol& symbol : word.symbols()) {
    const size_t new_pos = pos + symbol.len;
    // Every id in a `Word` comes from `vocab_`, so it is always in `vocab_r_`.
    auto value = vocab_r_.find(symbol.c);
    tokens.push_back(
        Token{symbol.c, value == vocab_r_.end() ? std::string() : value->second,
              Offsets(pos, new_pos)});
    pos = new_pos;
  }
  return tokens;
}

absl::StatusOr<std::vector<Token>> BPE::TokenizeWithCache(
    absl::string_view sequence) const {
  if (ignore_merges_) {
    auto id = vocab_.find(sequence);
    if (id != vocab_.end()) {
      return std::vector<Token>{Token{id->second, std::string(sequence),
                                      Offsets(0, sequence.size())}};
    }
  }
  if (cache_ == nullptr) {
    // Cache disabled (capacity 0): fall back to the uncached path.
    ASSIGN_OR_RETURN(Word word, MergeWord(sequence));
    return WordToTokens(word);
  }
  if (std::optional<Word> hit = cache_->Get(std::string(sequence));
      hit.has_value()) {
    return WordToTokens(*hit);
  }
  ASSIGN_OR_RETURN(Word word, MergeWord(sequence));
  std::vector<Token> ret = WordToTokens(word);
  if (sequence.size() < kMaxLength) {
    cache_->Set(std::string(sequence), std::move(word));
  }
  return ret;
}

absl::StatusOr<std::vector<Token>> BPE::Tokenize(
    absl::string_view sequence) const {
  if (sequence.empty()) {
    return std::vector<Token>();
  }

  if (!dropout_.has_value() || *dropout_ == 0.0f) {
    return TokenizeWithCache(sequence);
  } else {
    ASSIGN_OR_RETURN(Word word, MergeWord(sequence));
    return WordToTokens(word);
  }
}

std::optional<uint32_t> BPE::TokenToId(absl::string_view token) const {
  auto it = vocab_.find(token);
  if (it == vocab_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<std::string> BPE::IdToToken(uint32_t id) const {
  auto it = vocab_r_.find(id);
  if (it == vocab_r_.end()) {
    return std::nullopt;
  }
  return it->second;
}

bool BPE::operator==(const BPE& other) const {
  return vocab_ == other.vocab_ && vocab_r_ == other.vocab_r_ &&
         merges_ == other.merges_ && dropout_ == other.dropout_ &&
         unk_token_ == other.unk_token_ &&
         continuing_subword_prefix_ == other.continuing_subword_prefix_ &&
         end_of_word_suffix_ == other.end_of_word_suffix_ &&
         fuse_unk_ == other.fuse_unk_ &&
         byte_fallback_ == other.byte_fallback_ &&
         ignore_merges_ == other.ignore_merges_;
}

}  // namespace tokenizers::models::bpe
