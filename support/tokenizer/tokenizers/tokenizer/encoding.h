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
// C++ port of `src/tokenizer/encoding.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_ENCODING_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_ENCODING_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {

// PORT NOTE: `PaddingDirection` (utils/padding.rs) and `TruncationDirection`
// (utils/truncation.rs) are defined here instead of padding.h / truncation.h:
// `Encoding::Pad` / `Encoding::Truncate` need them while `PadEncodings` /
// `TruncateEncodings` need `Encoding`, which would be a dependency cycle
// between Bazel targets. padding.h / truncation.h include this header, so
// including either of them still provides the enum.

// The various possible padding directions.
//
// `#[derive(Debug, Clone, Copy, Serialize, Deserialize)] pub enum
// PaddingDirection { Left, Right }`
enum class PaddingDirection { kLeft, kRight };

// `#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Eq,
// Default)] pub enum TruncationDirection { Left, #[default] Right }`
enum class TruncationDirection { kLeft, kRight };

// `std::ops::Range<usize>` (half-open `[first, second)`).
using SequenceRange = std::pair<size_t, size_t>;

// Represents the output of a `Tokenizer`.
//
// `#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]`
class Encoding {
 public:
  // `(u32, String, (usize, usize), Option<u32>, u32)` =
  // (id, token, offsets, word, type_id), see `FromIter`.
  using Item = std::tuple<uint32_t, std::string, Offsets,
                          std::optional<uint32_t>, uint32_t>;

  // `Encoding::default()`
  Encoding() = default;

  // `Encoding::new(...)`
  Encoding(std::vector<uint32_t> ids, std::vector<uint32_t> type_ids,
           std::vector<std::string> tokens,
           std::vector<std::optional<uint32_t>> words,
           std::vector<Offsets> offsets,
           std::vector<uint32_t> special_tokens_mask,
           std::vector<uint32_t> attention_mask,
           std::vector<Encoding> overflowing,
           absl::flat_hash_map<size_t, SequenceRange> sequence_ranges);

  // `Encoding::with_capacity(len)`
  static Encoding WithCapacity(size_t len);

  // `Encoding::from_tokens(tokens, type_id)`
  static Encoding FromTokens(std::vector<Token> tokens, uint32_t type_id);

  // `impl FromIterator<(u32, String, (usize, usize), Option<u32>, u32)> for
  // Encoding`
  static Encoding FromIter(std::vector<Item> items);

  // Whether this Encoding is empty
  bool IsEmpty() const { return ids_.empty(); }

  // Return the total length of this Encoding
  size_t Len() const { return ids_.size(); }

  // Return the number of sequences combined in this Encoding
  size_t NSequences() const;

  // Set the given sequence id for the whole range of tokens contained in this
  // Encoding
  void SetSequenceId(size_t sequence_id);

  const std::vector<std::string>& GetTokens() const { return tokens_; }

  const std::vector<std::optional<uint32_t>>& GetWordIds() const {
    return words_;
  }

  std::vector<std::optional<uint32_t>>& GetWordIdsMut() { return words_; }

  std::vector<std::optional<size_t>> GetSequenceIds() const;

  const std::vector<uint32_t>& GetIds() const { return ids_; }

  const std::vector<uint32_t>& GetTypeIds() const { return type_ids_; }

  void SetTypeIds(std::vector<uint32_t> type_ids) {
    type_ids_ = std::move(type_ids);
  }

  const std::vector<Offsets>& GetOffsets() const { return offsets_; }

  std::vector<Offsets>& GetOffsetsMut() { return offsets_; }

  const std::vector<uint32_t>& GetSpecialTokensMask() const {
    return special_tokens_mask_;
  }

  const std::vector<uint32_t>& GetAttentionMask() const {
    return attention_mask_;
  }

  const std::vector<Encoding>& GetOverflowing() const { return overflowing_; }

  void SetOverflowing(std::vector<Encoding> overflowing) {
    overflowing_ = std::move(overflowing);
  }

  std::vector<Encoding>& GetOverflowingMut() { return overflowing_; }

  std::vector<Encoding> TakeOverflowing() {
    return std::exchange(overflowing_, {});
  }

  // `pub(crate) fn process_tokens_with_offsets_mut<F>(&mut self, func: F)`:
  // calls `func(i, token, offsets)` for every token (offsets are mutable).
  void ProcessTokensWithOffsets(
      absl::FunctionRef<void(size_t, const std::string&, Offsets&)> func);

  // The sequence ranges (`sequence_ranges` field; exposed for tests).
  const absl::flat_hash_map<size_t, SequenceRange>& GetSequenceRanges() const {
    return sequence_ranges_;
  }

  // Returns the index of the sequence containing the given token
  std::optional<size_t> TokenToSequence(size_t token) const;

  // Get the encoded tokens corresponding to the word at the given index in the
  // input sequence, with the form (start_token, end_token + 1)
  std::optional<std::pair<size_t, size_t>> WordToTokens(
      uint32_t word, size_t sequence_id) const;

  // Get the offsets of the word at the given index in the input sequence.
  std::optional<Offsets> WordToChars(uint32_t word, size_t sequence_id) const;

  // Get the offsets of the token at the given index.
  std::optional<std::pair<size_t, Offsets>> TokenToChars(size_t token) const;

  // Get the word that contains the token at the given index.
  std::optional<std::pair<size_t, uint32_t>> TokenToWord(size_t token) const;

  // Get the token that contains the given char.
  std::optional<size_t> CharToToken(size_t pos, size_t sequence_id) const;

  // Get the word that contains the given char.
  std::optional<uint32_t> CharToWord(size_t pos, size_t sequence_id) const;

  // Truncate the current `Encoding`.
  //
  // PORT NOTE: the Rust version panics if `stride >= max_len`; this returns an
  // `InvalidArgumentError` with the same message instead.
  absl::Status Truncate(size_t max_len, size_t stride,
                        TruncationDirection direction);

  // Merge all Encodings together
  static Encoding Merge(std::vector<Encoding> encodings, bool growing_offsets);

  // Merge ourself with the given `Encoding`. Happens in place.
  void MergeWith(Encoding pair, bool growing_offsets);

  void Pad(size_t target_length, uint32_t pad_id, uint32_t pad_type_id,
           absl::string_view pad_token, PaddingDirection direction);

  // `#[derive(Serialize, Deserialize)]`
  static absl::StatusOr<Encoding> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const Encoding& a, const Encoding& b);
  friend bool operator!=(const Encoding& a, const Encoding& b) {
    return !(a == b);
  }

 private:
  // Returns the range to target to retrieve something (word_id, offsets, ..)
  // related to the given sequence id
  SequenceRange SequenceRangeOf(size_t sequence_id) const;

  // IDs produced by the `Tokenizer`
  std::vector<uint32_t> ids_;
  // Type of the IDs
  std::vector<uint32_t> type_ids_;
  // Tokens associated to each ID
  std::vector<std::string> tokens_;
  // Indice of the word associated to each token/ID
  std::vector<std::optional<uint32_t>> words_;
  // Offsets of the token/ID from the NormalizedString
  std::vector<Offsets> offsets_;
  // Mask identifying special tokens
  std::vector<uint32_t> special_tokens_mask_;
  // Mask identifying padding tokens for the attention mechanism
  std::vector<uint32_t> attention_mask_;
  // A list of overflowing Encoding generated when we got truncated
  std::vector<Encoding> overflowing_;
  // Ranges of tokens covered by each sequence. If this is empty we consider
  // there is only one sequence in this Encoding, and that it covers the entire
  // range.
  absl::flat_hash_map<size_t, SequenceRange> sequence_ranges_;
};

}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_ENCODING_H_
