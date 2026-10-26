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

#include "support/tokenizer/tokenizers/models/unigram/model.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
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
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/models.h"
#include "support/tokenizer/tokenizers/models/unigram/lattice.h"
#include "support/tokenizer/tokenizers/models/unigram/trie.h"
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/utils/cache.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/status_macros.h"

namespace tokenizers::models::unigram {

namespace {

constexpr double kUnkPenalty = 10.0;

}  // namespace

absl::Status UnigramError::EmptyVocabulary() {
  return absl::InvalidArgumentError(
      "The vocabulary is empty but at least <unk> is needed");
}

absl::Status UnigramError::UnkIdNotInVocabulary() {
  return absl::InvalidArgumentError(
      "The `unk_id` is larger than vocabulary size");
}

absl::Status UnigramError::MissingUnkId() {
  return absl::InvalidArgumentError(
      "Encountered an unknown token but `unk_id` is missing");
}

Unigram::Unigram()
    : cache_(std::make_unique<Cache<std::string, std::vector<std::string>>>()),
      min_score_(0.0),
      unk_id_(0),
      bos_id_(2),
      eos_id_(3),
      fuse_unk_(true),
      is_optimized_(true),
      byte_fallback_(false) {
  absl::StatusOr<Unigram> model = From({{"<unk>", 0.0}}, 0, false);
  if (model.ok()) {
    *this = *std::move(model);
  }
}

Unigram::Unigram(const Unigram& other)
    : token_to_ids_(other.token_to_ids_),
      vocab_(other.vocab_),
      cache_(std::make_unique<Cache<std::string, std::vector<std::string>>>(
          other.cache_ == nullptr ? kDefaultCacheCapacity
                                  : other.cache_->capacity())),
      trie_(other.trie_),
      min_score_(other.min_score_),
      unk_id_(other.unk_id_),
      bos_id_(other.bos_id_),
      eos_id_(other.eos_id_),
      fuse_unk_(other.fuse_unk_),
      is_optimized_(other.is_optimized_),
      byte_fallback_(other.byte_fallback_),
      alpha_(other.alpha_),
      nbest_size_(other.nbest_size_) {}

Unigram& Unigram::operator=(const Unigram& other) {
  if (this != &other) {
    *this = Unigram(other);
  }
  return *this;
}

Unigram::Unigram(Vocab vocab, TokenMap token_to_ids, Trie<uint8_t> trie,
                 double min_score, std::optional<size_t> unk_id, size_t bos_id,
                 size_t eos_id, bool fuse_unk, bool is_optimized,
                 bool byte_fallback)
    : token_to_ids_(std::move(token_to_ids)),
      vocab_(std::move(vocab)),
      cache_(std::make_unique<Cache<std::string, std::vector<std::string>>>()),
      trie_(std::move(trie)),
      min_score_(min_score),
      unk_id_(unk_id),
      bos_id_(bos_id),
      eos_id_(eos_id),
      fuse_unk_(fuse_unk),
      is_optimized_(is_optimized),
      byte_fallback_(byte_fallback),
      alpha_(std::nullopt),
      nbest_size_(std::nullopt) {}

bool Unigram::operator==(const Unigram& other) const {
  return unk_id_ == other.unk_id_ && vocab_ == other.vocab_ &&
         alpha_ == other.alpha_ && nbest_size_ == other.nbest_size_;
}

absl::StatusOr<Unigram> Unigram::From(Vocab vocab, std::optional<size_t> unk_id,
                                      bool byte_fallback) {
  const size_t n = vocab.size();
  TokenMap token_to_ids;
  TrieBuilder<uint8_t> builder;

  if (unk_id.has_value()) {
    if (vocab.empty()) {
      return UnigramError::EmptyVocabulary();
    }
    if (*unk_id >= vocab.size()) {
      return UnigramError::UnkIdNotInVocabulary();
    }
  }
  const size_t bos_id = n + 1;
  const size_t eos_id = n + 2;

  double min_score = std::numeric_limits<double>::infinity();
  for (size_t id = 0; id < vocab.size(); ++id) {
    const std::string& token = vocab[id].first;
    const double score = vocab[id].second;
    token_to_ids[token] = static_cast<uint32_t>(id);
    builder.Push(absl::Span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(token.data()), token.size()));
    if (score < min_score) {
      min_score = score;
    }
  }
  Trie<uint8_t> trie = builder.Build();
  const bool fuse_unk = true;
  const bool is_optimized = true;

  return Unigram(std::move(vocab), std::move(token_to_ids), std::move(trie),
                 min_score, unk_id, bos_id, eos_id, fuse_unk, is_optimized,
                 byte_fallback);
}

void Unigram::SetFuseUnk(bool fuse_unk) {
  fuse_unk_ = fuse_unk;
  cache_ = std::make_unique<Cache<std::string, std::vector<std::string>>>(
      cache_->capacity());
}

void Unigram::SetOptimized(bool is_optimized) { is_optimized_ = is_optimized; }

void Unigram::PopulateNodes(Lattice& lattice) const {
  const double unk_score = min_score_ - kUnkPenalty;

  const size_t len = lattice.Len();

  size_t begin_pos = 0;
  while (begin_pos < len) {
    size_t next_pos = begin_pos;
    DecodeUtf8(lattice.sentence(), &next_pos);
    const size_t mblen = next_pos - begin_pos;

    bool has_single_node = false;

    const absl::string_view suffix = lattice.sentence().substr(begin_pos);
    const absl::Span<const uint8_t> suffix_bytes(
        reinterpret_cast<const uint8_t*>(suffix.data()), suffix.size());
    TrieIterator<uint8_t> iter = trie_.CommonPrefixSearch(suffix_bytes);
    while (iter.Next()) {
      const std::vector<uint8_t>& bytes = iter.prefix();
      const size_t n = bytes.size();
      const std::string tok(reinterpret_cast<const char*>(bytes.data()), n);
      auto id_it = token_to_ids_.find(tok);
      if (id_it == token_to_ids_.end()) {
        continue;
      }
      const uint32_t id = id_it->second;

      const auto& item = vocab_[id];
      const double score = item.second;
      lattice.Insert(begin_pos, n, score, static_cast<size_t>(id));
      if (!has_single_node && n == mblen) {
        has_single_node = true;
      }
    }

    if (!has_single_node) {
      if (unk_id_.has_value()) {
        lattice.Insert(begin_pos, mblen, unk_score, *unk_id_);
      }
    }
    begin_pos += mblen;
  }
}

absl::StatusOr<std::vector<std::string>> Unigram::Encode(
    absl::string_view sentence) const {
  if (sentence.empty()) {
    return std::vector<std::string>();
  }
  if (!alpha_.has_value() || *alpha_ == 0.0) {
    if (std::optional<std::vector<std::string>> result = cache_->Get(sentence);
        result.has_value()) {
      return *std::move(result);
    } else {
      std::vector<std::string> encoded;
      if (is_optimized_) {
        ASSIGN_OR_RETURN(encoded, EncodeOptimized(sentence));
      } else {
        ASSIGN_OR_RETURN(encoded, EncodeUnoptimized(sentence));
      }
      if (sentence.size() < kMaxLength) {
        cache_->Set(std::string(sentence), encoded);
      }
      return encoded;
    }
  } else {
    ASSIGN_OR_RETURN(std::vector<std::string> result,
                     EncodeUnoptimized(sentence));
    return result;
  }
}

absl::StatusOr<std::vector<std::string>> Unigram::EncodeOptimized(
    absl::string_view sentence) const {
  // https://github.com/google/sentencepiece/blob/d48247191a6d50e469ed1a4a36e877befffd1851/src/unigram_model.cc#L600
  struct BestPathNode {
    // The vocab id. (maybe UNK)
    size_t id = 0;
    // The total score of the best path ending at this node.
    double best_path_score = 0.0;
    // The starting position (in utf-8) of this node. The entire best
    // path can be constructed by backtracking along this link.
    std::optional<size_t> starts_at = std::nullopt;
  };
  const size_t size = sentence.size();
  const double unk_score = min_score_ - kUnkPenalty;

  std::vector<BestPathNode> best_path_ends_at(size + 1);
  size_t starts_at = 0;
  while (starts_at < size) {
    const double best_path_score_till_here =
        best_path_ends_at[starts_at].best_path_score;
    bool has_single_node = false;
    size_t next_pos = starts_at;
    DecodeUtf8(sentence, &next_pos);
    const size_t mblen = next_pos - starts_at;
    const absl::string_view suffix = sentence.substr(starts_at);
    const absl::Span<const uint8_t> suffix_bytes(
        reinterpret_cast<const uint8_t*>(suffix.data()), suffix.size());
    TrieIterator<uint8_t> iter = trie_.CommonPrefixSearch(suffix_bytes);
    while (iter.Next()) {
      const std::vector<uint8_t>& tok_bytes = iter.prefix();
      const size_t key_pos = starts_at + tok_bytes.size();
      const std::string token(reinterpret_cast<const char*>(tok_bytes.data()),
                              tok_bytes.size());
      BestPathNode& target_node = best_path_ends_at[key_pos];
      const size_t length = key_pos - starts_at;
      auto id_it = token_to_ids_.find(token);
      if (id_it == token_to_ids_.end()) {
        continue;
      }
      const uint32_t id = id_it->second;
      const double score = vocab_[id].second;
      const double candidate_best_path_score =
          score + best_path_score_till_here;
      if (!target_node.starts_at.has_value() ||
          candidate_best_path_score > target_node.best_path_score) {
        target_node.best_path_score = candidate_best_path_score;
        target_node.starts_at = starts_at;
        target_node.id = static_cast<size_t>(id);
      }
      if (!has_single_node && length == mblen) {
        has_single_node = true;
      }
    }
    if (!has_single_node) {
      BestPathNode& target_node = best_path_ends_at[starts_at + mblen];
      const double candidate_best_path_score =
          unk_score + best_path_score_till_here;
      if (!target_node.starts_at.has_value() ||
          candidate_best_path_score > target_node.best_path_score) {
        target_node.best_path_score = candidate_best_path_score;
        target_node.starts_at = starts_at;
        if (!unk_id_.has_value()) {
          return UnigramError::MissingUnkId();
        }
        target_node.id = *unk_id_;
      }
    }
    starts_at += mblen;
  }
  size_t ends_at = size;
  std::vector<std::string> results;
  std::vector<std::string> token;
  while (ends_at > 0) {
    const BestPathNode& node = best_path_ends_at[ends_at];
    if (!node.starts_at.has_value()) {
      return UnigramError::MissingUnkId();
    }
    const size_t node_starts_at = *node.starts_at;
    if (fuse_unk_ && unk_id_.has_value() && node.id == *unk_id_) {
      token.push_back(std::string(
          sentence.substr(node_starts_at, ends_at - node_starts_at)));
    } else {
      if (!token.empty()) {
        std::reverse(token.begin(), token.end());
        results.push_back(absl::StrJoin(token, ""));
        token.clear();
      }
      results.push_back(std::string(
          sentence.substr(node_starts_at, ends_at - node_starts_at)));
    }
    ends_at = node_starts_at;
  }
  if (!token.empty()) {
    std::reverse(token.begin(), token.end());
    results.push_back(absl::StrJoin(token, ""));
  }
  std::reverse(results.begin(), results.end());
  return results;
}

absl::StatusOr<std::vector<std::string>> Unigram::EncodeUnoptimized(
    absl::string_view sentence) const {
  Lattice lattice = Lattice::From(sentence, bos_id_, eos_id_);
  PopulateNodes(lattice);
  std::vector<NodeRef> path;
  if (nbest_size_.has_value() && alpha_.has_value() && *nbest_size_ > 0) {
    path = lattice.SampleNbest(*nbest_size_, *alpha_);
  } else if (alpha_.has_value()) {
    ASSIGN_OR_RETURN(path, lattice.Sample(*alpha_));
  } else {
    path = lattice.Viterbi();
  }
  if (fuse_unk_) {
    std::vector<std::string> results;
    std::string token;
    for (NodeRef node : path) {
      std::string item = lattice.Piece(*node);
      if (!unk_id_.has_value()) {
        return UnigramError::MissingUnkId();
      }
      if (node->id == *unk_id_) {
        absl::StrAppend(&token, item);
      } else {
        if (!token.empty()) {
          results.push_back(std::move(token));
          token.clear();
        }
        results.push_back(std::move(item));
      }
    }
    if (!token.empty()) {
      results.push_back(std::move(token));
    }
    return results;
  } else {
    std::vector<std::string> results;
    results.reserve(path.size());
    for (NodeRef node : path) {
      results.push_back(lattice.Piece(*node));
    }
    return results;
  }
}

UnigramIterator Unigram::Iter() const { return UnigramIterator(this); }

absl::StatusOr<Unigram> Unigram::Load(absl::string_view path) {
  ASSIGN_OR_RETURN(std::string string, ReadToString(path));
  ASSIGN_OR_RETURN(Json json, ParseJson(string));
  return FromJson(json);
}

void Unigram::ClearCache() { cache_->Clear(); }

void Unigram::ResizeCache(size_t capacity) { cache_->Resize(capacity); }

const std::pair<std::string, double>* UnigramIterator::Next() {
  const size_t i = i_;
  if (i < model_->Len()) {
    const auto* r = &model_->vocab()[i];
    ++i_;
    return r;
  } else {
    return nullptr;
  }
}

absl::flat_hash_map<std::string, uint32_t> Unigram::GetVocab() const {
  return token_to_ids_;
}

size_t Unigram::GetVocabSize() const { return vocab_.size(); }

absl::StatusOr<std::vector<Token>> Unigram::Tokenize(
    absl::string_view sentence) const {
  ASSIGN_OR_RETURN(std::vector<std::string> str_tokens, Encode(sentence));
  size_t offset = 0;
  std::vector<Token> tokens;
  tokens.reserve(str_tokens.size());
  for (std::string& string : str_tokens) {
    const size_t len = string.size();
    const Offsets offsets(offset, offset + len);
    uint32_t id = 0;
    auto it = token_to_ids_.find(string);
    if (it != token_to_ids_.end()) {
      id = it->second;
    } else {
      if (byte_fallback_) {
        std::optional<std::vector<Token>> byte_tokens = std::vector<Token>();
        for (const unsigned char byte : string) {
          std::string byte_string = absl::StrFormat("<0x%02X>", byte);
          auto byte_it = token_to_ids_.find(byte_string);
          if (byte_it == token_to_ids_.end()) {
            byte_tokens.reset();
            break;
          }
          byte_tokens->push_back(Token{byte_it->second, std::move(byte_string),
                                       Offsets(offset, offset + len)});
        }
        if (byte_tokens.has_value()) {
          for (Token& token : *byte_tokens) {
            tokens.push_back(std::move(token));
          }
          offset += len;
          continue;
        }
      }
      if (!unk_id_.has_value()) {
        return UnigramError::MissingUnkId();
      }
      id = static_cast<uint32_t>(*unk_id_);
    }
    offset += len;
    tokens.push_back(Token{id, std::move(string), offsets});
  }
  return tokens;
}

std::optional<uint32_t> Unigram::TokenToId(absl::string_view token) const {
  auto it = token_to_ids_.find(token);
  if (it == token_to_ids_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<std::string> Unigram::IdToToken(uint32_t id) const {
  if (static_cast<size_t>(id) >= vocab_.size()) {
    return std::nullopt;
  }
  return vocab_[id].first;
}

}  // namespace tokenizers::models::unigram
