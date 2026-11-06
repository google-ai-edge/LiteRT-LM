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
// C++ port of `src/tokenizer/added_vocabulary.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/tokenizer/added_vocabulary.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/aho_corasick.h"
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/regex.h"
#include "support/util/status_macros.h"

namespace tokenizers {

AddedToken AddedToken::From(absl::string_view content, bool special) {
  AddedToken token;
  token.content = std::string(content);
  token.normalized = !special;
  token.special = special;
  return token;
}

absl::StatusOr<AddedToken> AddedToken::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireObject(json, "struct AddedToken"));
  AddedToken token;
  ASSIGN_OR_RETURN(token.content, GetField<std::string>(json, "content"));
  ASSIGN_OR_RETURN(token.single_word, GetField<bool>(json, "single_word"));
  ASSIGN_OR_RETURN(token.lstrip, GetField<bool>(json, "lstrip"));
  ASSIGN_OR_RETURN(token.rstrip, GetField<bool>(json, "rstrip"));
  ASSIGN_OR_RETURN(token.normalized, GetField<bool>(json, "normalized"));
  ASSIGN_OR_RETURN(token.special, GetField<bool>(json, "special"));
  return token;
}

Json AddedToken::ToJson() const {
  Json json = Json::object();
  json["content"] = content;
  json["single_word"] = single_word;
  json["lstrip"] = lstrip;
  json["rstrip"] = rstrip;
  json["normalized"] = normalized;
  json["special"] = special;
  return json;
}

namespace {

// PORT NOTE: the Rust code uses `regex::Regex::new(r"^\w")` etc., where `^` and
// `$` anchor to the beginning and end of the text unless `(?m)` is enabled, and
// `\w` (`regex_syntax::PERL_WORD`, UTS #18 Annex C) includes `\p{Join_Control}`
// (U+200C ZWNJ and U+200D ZWJ). Oniguruma's default syntax treats `^` and `$`
// as line anchors and excludes `\p{Join_Control}` from `\w`, so `\A` / `\z` and
// `[\w\p{Join_Control}]` are used here to match the Rust `regex` semantics.
const SysRegex& StartsWithWordRegex() {
  static const SysRegex* const kRegex =
      new SysRegex(*SysRegex::New(R"(\A[\w\p{Join_Control}])"));
  return *kRegex;
}

const SysRegex& EndsWithWordRegex() {
  static const SysRegex* const kRegex =
      new SysRegex(*SysRegex::New(R"([\w\p{Join_Control}]\z)"));
  return *kRegex;
}

const SysRegex& RightmostSpaceAtStartRegex() {
  static const SysRegex* const kRegex =
      new SysRegex(*SysRegex::New(R"(\A\s*)"));
  return *kRegex;
}

const SysRegex& LeftmostSpaceAtEndRegex() {
  static const SysRegex* const kRegex =
      new SysRegex(*SysRegex::New(R"(\s*\z)"));
  return *kRegex;
}

bool EndsWithWord(absl::string_view sentence) {
  return !EndsWithWordRegex().FindIter(sentence).empty();
}

bool StartsWithWord(absl::string_view sentence) {
  return !StartsWithWordRegex().FindIter(sentence).empty();
}

size_t SpaceLeftmostAtEnd(absl::string_view sentence) {
  const std::vector<std::pair<size_t, size_t>> matches =
      LeftmostSpaceAtEndRegex().FindIter(sentence);
  if (!matches.empty()) {
    return matches.front().first;
  } else {
    return sentence.size();
  }
}

size_t SpaceRightmostAtStart(absl::string_view sentence) {
  const std::vector<std::pair<size_t, size_t>> matches =
      RightmostSpaceAtStartRegex().FindIter(sentence);
  if (!matches.empty()) {
    return matches.front().second;
  } else {
    return 0;
  }
}

}  // namespace

std::optional<uint32_t> AddedVocabulary::TokenToId(absl::string_view token,
                                                   const Model& model) const {
  auto it = added_tokens_map_.find(token);
  if (it != added_tokens_map_.end()) {
    return it->second;
  }
  return model.TokenToId(token);
}

std::optional<std::string> AddedVocabulary::IdToToken(
    uint32_t id, const Model& model) const {
  auto it = added_tokens_map_r_.find(id);
  if (it != added_tokens_map_r_.end()) {
    return it->second.content;
  }
  return model.IdToToken(id);
}

std::optional<std::string> AddedVocabulary::SimpleIdToToken(uint32_t id) const {
  auto it = added_tokens_map_r_.find(id);
  if (it == added_tokens_map_r_.end()) {
    return std::nullopt;
  }
  auto cache_it = normalized_cache_.find(id);
  if (cache_it != normalized_cache_.end()) {
    return cache_it->second;
  }
  return it->second.content;
}

absl::StatusOr<size_t> AddedVocabulary::AddSpecialTokens(
    std::vector<AddedToken> tokens, const Model& model,
    const Normalizer* normalizer) {
  return AddTokens(std::move(tokens), model, normalizer);
}

absl::StatusOr<size_t> AddedVocabulary::AddTokens(
    std::vector<AddedToken> tokens, const Model& model,
    const Normalizer* normalizer) {
  size_t ignored = 0;
  size_t total = 0;

  const uint32_t vocab_size = static_cast<uint32_t>(model.GetVocabSize());
  uint32_t next_id = vocab_size;
  if (!added_tokens_map_r_.empty()) {
    uint32_t max_id = 0;
    for (const auto& [id, _] : added_tokens_map_r_) {
      max_id = std::max(max_id, id);
    }
    if (max_id >= vocab_size || model.GetVocabSize() == 0) {
      next_id = max_id + 1;
    } else {
      next_id = vocab_size;
    }
  }

  for (AddedToken& token : tokens) {
    total += 1;
    if (token.content.empty()) {
      ignored += 1;
      continue;
    }
    // Fast path: skip if this content is already in the map with identical
    // properties.
    if (auto id_it = added_tokens_map_.find(token.content);
        id_it != added_tokens_map_.end()) {
      auto r_it = added_tokens_map_r_.find(id_it->second);
      if (r_it != added_tokens_map_r_.end() && r_it->second == token) {
        ignored += 1;
        continue;
      }
    }

    uint32_t new_id;
    if (std::optional<uint32_t> existing_id = TokenToId(token.content, model);
        existing_id.has_value()) {
      new_id = *existing_id;
    } else {
      new_id = next_id;
      next_id += 1;
    }

    if (token.normalized && normalizer != nullptr) {
      NormalizedString normalized = NormalizedString::From(token.content);
      RETURN_IF_ERROR(normalizer->Normalize(normalized));
      std::string normed(normalized.Get());
      if (normed != token.content) {
        normalized_cache_.insert_or_assign(new_id, std::move(normed));
      }
    }

    added_tokens_map_.insert_or_assign(token.content, new_id);

    const bool is_new_special = token.special && !token.content.empty() &&
                                !special_tokens_set_.contains(token.content);
    if (is_new_special) {
      special_tokens_set_.insert(token.content);
    }
    added_tokens_map_r_.insert_or_assign(new_id, std::move(token));
  }

  RETURN_IF_ERROR(RefreshAddedTokens());

  // Return the number of added tokens
  return total - ignored;
}

absl::Status AddedVocabulary::RefreshNormalizedTokens(
    const Normalizer* normalizer) {
  normalized_cache_.clear();
  for (const auto& [id, token] : added_tokens_map_r_) {
    if (token.normalized && normalizer != nullptr) {
      NormalizedString normalized = NormalizedString::From(token.content);
      RETURN_IF_ERROR(normalizer->Normalize(normalized));
      std::string normed(normalized.Get());
      if (normed != token.content) {
        normalized_cache_.insert_or_assign(id, std::move(normed));
      }
    }
  }
  return RefreshAddedTokens();
}

absl::Status AddedVocabulary::RefreshAddedTokens() {
  std::vector<std::pair<std::string, uint32_t>> normalized_pairs;
  std::vector<std::pair<std::string, uint32_t>> non_normalized_pairs;
  for (const auto& [id, token] : added_tokens_map_r_) {
    if (token.normalized) {
      auto cache_it = normalized_cache_.find(id);
      const std::string& pattern = cache_it != normalized_cache_.end()
                                       ? cache_it->second
                                       : token.content;
      normalized_pairs.emplace_back(pattern, id);
    } else {
      non_normalized_pairs.emplace_back(token.content, id);
    }
  }

  if (non_normalized_pairs.empty()) {
    split_trie_ = std::nullopt;
  } else {
    ASSIGN_OR_RETURN(AhoCorasick trie,
                     AhoCorasick::BuildWithValues(non_normalized_pairs));
    split_trie_ = std::move(trie);
  }

  if (normalized_pairs.empty()) {
    split_normalized_trie_ = std::nullopt;
  } else {
    ASSIGN_OR_RETURN(AhoCorasick trie,
                     AhoCorasick::BuildWithValues(normalized_pairs));
    split_normalized_trie_ = std::move(trie);
  }

  return absl::OkStatus();
}

std::vector<std::pair<std::optional<uint32_t>, Offsets>>
AddedVocabulary::FindMatches(absl::string_view sentence,
                             const MatchingSet& split_re) const {
  if (sentence.empty()) {
    return {{std::nullopt, Offsets{0, 0}}};
  }

  size_t start_offset = 0;
  std::vector<std::pair<std::optional<uint32_t>, Offsets>> splits;

  if (!split_re.has_value()) {
    return {{std::nullopt, Offsets{0, sentence.size()}}};
  }
  const AhoCorasick& trie = *split_re;
  for (const AhoCorasickMatch& mat : trie.LeftmostFindIter(sentence)) {
    size_t start = mat.start;
    size_t stop = mat.end;
    const uint32_t id = mat.value;
    auto it = added_tokens_map_r_.find(id);
    if (it == added_tokens_map_r_.end()) {
      continue;
    }
    const AddedToken& added_token = it->second;

    if (encode_special_tokens_ &&
        special_tokens_set_.contains(added_token.content)) {
      continue;
    }

    if (added_token.single_word) {
      const bool start_space =
          start == 0 || !EndsWithWord(sentence.substr(0, start));
      const bool stop_space =
          stop == sentence.size() || !StartsWithWord(sentence.substr(stop));

      if (!stop_space || !start_space) {
        // Discard not single word
        continue;
      }
    }
    if (added_token.lstrip) {
      // This will be strictly inferior to start and in correct sentence offset
      const size_t newstart = SpaceLeftmostAtEnd(sentence.substr(0, start));

      // The previous match could have already matched those spaces
      // Ignore them if it's already matched
      start = std::max(newstart, start_offset);
    }
    if (added_token.rstrip) {
      // This will starting a the stop+1 character, so we need
      // to add the previous stop value
      stop += SpaceRightmostAtStart(sentence.substr(stop));
    }
    if (start_offset < start) {
      splits.emplace_back(std::nullopt, Offsets{start_offset, start});
    }
    splits.emplace_back(id, Offsets{start, stop});
    start_offset = stop;
  }

  const size_t total_byte_len = sentence.size();
  if (start_offset != total_byte_len) {
    splits.emplace_back(std::nullopt, Offsets{start_offset, total_byte_len});
  }

  return splits;
}

absl::StatusOr<std::vector<Split>> AddedVocabulary::SplitWithIndices(
    NormalizedString sentence, const MatchingSet& split_re) const {
  std::vector<std::pair<std::optional<uint32_t>, Offsets>> matches =
      FindMatches(sentence.Get(), split_re);
  std::vector<Split> out;
  out.reserve(matches.size());
  for (const auto& [id, byte_offsets] : matches) {
    std::optional<NormalizedString> slice = sentence.Slice(
        Range::Normalized(byte_offsets.first, byte_offsets.second));
    if (!slice.has_value()) {
      return absl::InternalError("AddedVocabulary bad split");
    }
    if (id.has_value()) {
      std::string value(slice->Get());
      const size_t len = value.size();
      out.emplace_back(
          *std::move(slice),
          std::vector<Token>{Token(*id, std::move(value), Offsets{0, len})});
    } else {
      out.emplace_back(*std::move(slice), std::nullopt);
    }
  }
  return out;
}

absl::StatusOr<PreTokenizedString> AddedVocabulary::ExtractAndNormalize(
    const Normalizer* normalizer, absl::string_view sequence) const {
  PreTokenizedString pretokenized = PreTokenizedString::From(sequence);

  // 1. We extract all the non-normalized tokens from the non-normalized string
  RETURN_IF_ERROR(pretokenized.Split([this](size_t, NormalizedString seq)
                                         -> absl::StatusOr<std::vector<Split>> {
    return SplitWithIndices(std::move(seq), split_trie_);
  }));

  // 2. Then extract the normalized tokens from the normalized pieces of the
  // string
  RETURN_IF_ERROR(pretokenized.Split(
      [this, normalizer](
          size_t, NormalizedString seq) -> absl::StatusOr<std::vector<Split>> {
        if (normalizer != nullptr) {
          normalizer->Normalize(seq).IgnoreError();
        }
        return SplitWithIndices(std::move(seq), split_normalized_trie_);
      }));

  return pretokenized;
}

Json AddedVocabulary::ToJson() const {
  std::vector<AddedTokenWithId> added_tokens;
  added_tokens.reserve(added_tokens_map_r_.size());
  for (const auto& [id, token] : added_tokens_map_r_) {
    added_tokens.push_back(AddedTokenWithId{id, token});
  }
  // We need to have these added tokens ordered by ascending ID
  std::sort(added_tokens.begin(), added_tokens.end(),
            [](const AddedTokenWithId& a, const AddedTokenWithId& b) {
              return a.id < b.id;
            });

  Json vocabulary = Json::array();
  for (const AddedTokenWithId& token : added_tokens) {
    vocabulary.push_back(token.ToJson());
  }
  return vocabulary;
}

absl::StatusOr<AddedTokenWithId> AddedTokenWithId::FromJson(const Json& json) {
  RETURN_IF_ERROR(RequireObject(json, "struct AddedTokenWithId"));
  AddedTokenWithId item;
  ASSIGN_OR_RETURN(item.id, GetField<uint32_t>(json, "id"));
  ASSIGN_OR_RETURN(item.token, AddedToken::FromJson(json));
  return item;
}

Json AddedTokenWithId::ToJson() const {
  Json json = Json::object();
  json["id"] = id;
  json["content"] = token.content;
  json["single_word"] = token.single_word;
  json["lstrip"] = token.lstrip;
  json["rstrip"] = token.rstrip;
  json["normalized"] = token.normalized;
  json["special"] = token.special;
  return json;
}

}  // namespace tokenizers
