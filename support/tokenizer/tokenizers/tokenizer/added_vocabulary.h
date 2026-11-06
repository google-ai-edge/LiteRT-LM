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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_ADDED_VOCABULARY_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_ADDED_VOCABULARY_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/aho_corasick.h"
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {

// Represent a token added by the user on top of the existing Model vocabulary.
// AddedToken can be configured to specify the behavior they should have in
// various situations like:
//   - Whether they should only match single words
//   - Whether to include any whitespace on its left or right
//
// `#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]`
struct AddedToken {
  // The content of the added token (original, as provided by the user)
  std::string content;
  // Whether this token must be a single word or can break words
  bool single_word = false;
  // Whether this token should strip whitespaces on its left
  bool lstrip = false;
  // Whether this token should strip whitespaces on its right
  bool rstrip = false;
  // Whether this token should be normalized
  bool normalized = true;
  // Whether this token is special
  bool special = false;

  // `impl Default for AddedToken`
  AddedToken() = default;

  // Build this token from the given content, specifying if it is intended to
  // be a special token. Special tokens are not normalized by default.
  //
  // `AddedToken::from(content, special)`
  static AddedToken From(absl::string_view content, bool special);

  // Builder-style setters (`#[must_use] pub fn single_word(mut self, v) ->
  // Self` etc.). Both an rvalue overload (moves) and an lvalue overload
  // (returns a modified copy) are provided.

  // Specify whether this token should only match on whole single words, and
  // never part of a word.
  AddedToken SingleWord(bool single_word) && {
    this->single_word = single_word;
    return std::move(*this);
  }
  AddedToken SingleWord(bool single_word) const& {
    return AddedToken(*this).SingleWord(single_word);
  }
  // Specify whether this token should include all the whitespaces on its
  // left, in order to strip them out.
  AddedToken Lstrip(bool lstrip) && {
    this->lstrip = lstrip;
    return std::move(*this);
  }
  AddedToken Lstrip(bool lstrip) const& {
    return AddedToken(*this).Lstrip(lstrip);
  }
  // Specify whether this token should include all the whitespaces on its
  // right, in order to strip them out.
  AddedToken Rstrip(bool rstrip) && {
    this->rstrip = rstrip;
    return std::move(*this);
  }
  AddedToken Rstrip(bool rstrip) const& {
    return AddedToken(*this).Rstrip(rstrip);
  }
  // Specify whether this token should be normalized and match against its
  // normalized version in the input text.
  AddedToken Normalized(bool normalized) && {
    this->normalized = normalized;
    return std::move(*this);
  }
  AddedToken Normalized(bool normalized) const& {
    return AddedToken(*this).Normalized(normalized);
  }
  // Specify whether this token is special, meaning if it should be skipped
  // when decoding
  AddedToken Special(bool special) && {
    this->special = special;
    return std::move(*this);
  }
  AddedToken Special(bool special) const& {
    return AddedToken(*this).Special(special);
  }

  // `#[derive(Serialize, Deserialize)]`: all six fields are required.
  static absl::StatusOr<AddedToken> FromJson(const Json& json);
  Json ToJson() const;

  // AddedTokens can be updated if value changed
  // (`impl Hash for AddedToken` hashes `content` only.)
  template <typename H>
  friend H AbslHashValue(H h, const AddedToken& token) {
    return H::combine(std::move(h), token.content);
  }

  friend bool operator==(const AddedToken& a, const AddedToken& b) {
    return a.content == b.content && a.single_word == b.single_word &&
           a.lstrip == b.lstrip && a.rstrip == b.rstrip &&
           a.normalized == b.normalized && a.special == b.special;
  }
  friend bool operator!=(const AddedToken& a, const AddedToken& b) {
    return !(a == b);
  }
};

// `type MatchingSet = Option<DoubleArrayAhoCorasick<u32>>;`
using MatchingSet = std::optional<AhoCorasick>;

//
// A vocabulary built on top of the Model
//
// This provides a way to add new vocabulary to a Tokenizer that has already
// been trained, in a previous process, maybe by someone else. This is
// especially interesting in the case of fine-tunings, where we want to finetune
// a model while adding some new functionalities using some new special tokens,
// or maybe add some tokens in the case of unknown tokens, etc.
//
// One of the reasons we need to handle these tokens outside of the model is
// simply that for many models, it is not possible to add new tokens after the
// training process. For example, using BPE, the training process generates
// merges pairs along the vocabulary, and any token in the vocabulary can be
// decomposed in other tokens, down to the original alphabet. If we were to add
// new tokens after this training process, we couldn't make sure the merges
// pairs exist as required.
//
// `#[derive(Clone)]`
class AddedVocabulary {
 public:
  // `AddedVocabulary::new()` / `impl Default`
  AddedVocabulary() = default;

  // Size of the additional vocabulary
  size_t Len() const { return added_tokens_map_.size(); }

  // Whether or not this vocabulary is empty
  bool IsEmpty() const { return added_tokens_map_.empty(); }

  // Get the additional vocabulary
  const absl::flat_hash_map<std::string, uint32_t>& GetVocab() const {
    return added_tokens_map_;
  }

  // Get the additional vocabulary with the AddedTokens
  const absl::flat_hash_map<uint32_t, AddedToken>& GetAddedTokensDecoder()
      const {
    return added_tokens_map_r_;
  }

  // Get the id matching one of our token if it exists
  std::optional<uint32_t> TokenToId(absl::string_view token,
                                    const Model& model) const;

  // Get the token matching the given id if it exists
  //
  // `#[deprecated(since = "0.19.0", note = "please use
  // `added_vocabulary.simple_id_to_token(id).or_else(|| model.id_to_token(id)`
  // instead")]`
  std::optional<std::string> IdToToken(uint32_t id, const Model& model) const;

  // Return the string form of an added token used during **decoding**.
  //
  // For tokens that were normalized on the way *in* (e.g. byte-level encoding),
  // this returns the cached normalized form so that the configured `Decoder`
  // can invert the transformation correctly. For all other tokens, the
  // original `content` is returned.
  std::optional<std::string> SimpleIdToToken(uint32_t id) const;

  //
  void SetEncodeSpecialTokens(bool value) { encode_special_tokens_ = value; }

  bool GetEncodeSpecialTokens() const { return encode_special_tokens_; }

  // Check if a token is a special token
  bool IsSpecialToken(absl::string_view token) const {
    return special_tokens_set_.contains(token);
  }

  // Add some special tokens to the vocabulary
  absl::StatusOr<size_t> AddSpecialTokens(std::vector<AddedToken> tokens,
                                          const Model& model,
                                          const Normalizer* normalizer);

  // Add some tokens to the vocabulary
  absl::StatusOr<size_t> AddTokens(std::vector<AddedToken> tokens,
                                   const Model& model,
                                   const Normalizer* normalizer);

  // Re-apply normalization to every added token that has `normalized = true`,
  // then rebuild the matching tries.
  //
  // This is called automatically by `TokenizerImpl::with_normalizer` when the
  // normalizer is replaced. For tokenizers with many added tokens the trie
  // rebuild can be slow; prefer setting the normalizer *before* calling
  // `add_tokens` when constructing a tokenizer programmatically. During
  // deserialization this is never triggered because the normalizer is set via
  // the builder before tokens are added.
  absl::Status RefreshNormalizedTokens(const Normalizer* normalizer);

  // Extract the additional vocabulary from the given sentence, normalizing it
  // along the way.
  //
  // Some tokens should match against their normalized representation, as well
  // as the non-normalized one. For example, when we expect to extract the
  // token `yesterday` in the input sentence `I read a book Yesterday`, if the
  // normalizer is supposed to lowercase everything, we expect a match.
  //
  // PORT NOTE: the Rust version `.expect()`s on an impossible bad split; this
  // version returns the error instead. Errors returned by the normalizer are
  // ignored, exactly like the Rust version (`normalizer.map(|n|
  // n.normalize(&mut sequence));` drops the `Result`).
  absl::StatusOr<PreTokenizedString> ExtractAndNormalize(
      const Normalizer* normalizer, absl::string_view sequence) const;

  // `impl Serialize for AddedVocabulary`: the list of `AddedTokenWithId`
  // ordered by ascending ID.
  Json ToJson() const;

  // Find any AddedToken in the given sentence, using the provided MatchingSet.
  // This method returns a list "splits", each of them being a pair of Offsets
  // and an optional ID if it is an AddedToken.
  // The list of splits cover the entire input string.
  //
  // (Rust: private; public here because the Rust unit tests call it.)
  std::vector<std::pair<std::optional<uint32_t>, Offsets>> FindMatches(
      absl::string_view sentence, const MatchingSet& split_re) const;

  // The `split_trie` / `split_normalized_trie` fields (exposed for tests).
  const MatchingSet& SplitTrie() const { return split_trie_; }
  const MatchingSet& SplitNormalizedTrie() const {
    return split_normalized_trie_;
  }

 private:
  // Reconstruct our internal RegexSet when new tokens are added to the
  // vocabulary.
  absl::Status RefreshAddedTokens();

  // Split the input sentence to extract anything we found from the
  // `MatchingSet`, as well as the list of corresponding IDs
  // The list of IDs have the exact same number of elements than the Iterator.
  absl::StatusOr<std::vector<Split>> SplitWithIndices(
      NormalizedString sentence, const MatchingSet& split_re) const;

  // Contains the mapping from String (token content) to ID. This map contains
  // both special tokens and classic added tokens that were added to the this
  // vocabulary.
  absl::flat_hash_map<std::string, uint32_t> added_tokens_map_;
  // Contains the mapping from ID to AddedToken for all the added tokens, both
  // special and classic.
  absl::flat_hash_map<uint32_t, AddedToken> added_tokens_map_r_;

  // A Set, containing all the special token for easy access while decoding.
  // This let's us remove them easily with an O(1) complexity.
  absl::flat_hash_set<std::string> special_tokens_set_;

  // Cache of the normalizer output for tokens that have `normalized = true`.
  // Keyed by token ID. Not serialized — rebuilt by `add_tokens` and
  // `refresh_normalized_tokens` whenever the normalizer changes.
  // Kept separate from `AddedToken` so the token struct stays lean.
  absl::flat_hash_map<uint32_t, std::string> normalized_cache_;

  // A RegexSet containing all the non-normalized patterns used to split on
  // AddedTokens
  MatchingSet split_trie_;
  // A RegexSet containing all the normalized patterns used to split on
  // AddedTokens
  MatchingSet split_normalized_trie_;

  // Whether or not special tokens should be splitted when encoding. This is
  // equivalent to ignoring them
  bool encode_special_tokens_ = false;
};

// `#[derive(Debug, Serialize, Deserialize)] pub(super) struct AddedTokenWithId`
// (used by tokenizer/serialization for the `added_tokens` list).
struct AddedTokenWithId {
  // The id assigned to this token
  uint32_t id = 0;
  // The target AddedToken
  // `#[serde(flatten)]`
  AddedToken token;

  // `{"id": .., "content": .., "single_word": .., "lstrip": .., "rstrip": ..,
  // "normalized": .., "special": ..}`
  static absl::StatusOr<AddedTokenWithId> FromJson(const Json& json);
  Json ToJson() const;
};

}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_ADDED_VOCABULARY_H_
