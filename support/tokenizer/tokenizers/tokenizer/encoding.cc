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

#include "support/tokenizer/tokenizers/tokenizer/encoding.h"

#include <algorithm>
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
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers {

Encoding::Encoding(std::vector<uint32_t> ids, std::vector<uint32_t> type_ids,
                   std::vector<std::string> tokens,
                   std::vector<std::optional<uint32_t>> words,
                   std::vector<Offsets> offsets,
                   std::vector<uint32_t> special_tokens_mask,
                   std::vector<uint32_t> attention_mask,
                   std::vector<Encoding> overflowing,
                   absl::flat_hash_map<size_t, SequenceRange> sequence_ranges)
    : ids_(std::move(ids)),
      type_ids_(std::move(type_ids)),
      tokens_(std::move(tokens)),
      words_(std::move(words)),
      offsets_(std::move(offsets)),
      special_tokens_mask_(std::move(special_tokens_mask)),
      attention_mask_(std::move(attention_mask)),
      overflowing_(std::move(overflowing)),
      sequence_ranges_(std::move(sequence_ranges)) {}

Encoding Encoding::WithCapacity(size_t len) {
  Encoding encoding;
  encoding.ids_.reserve(len);
  encoding.type_ids_.reserve(len);
  encoding.tokens_.reserve(len);
  encoding.words_.reserve(len);
  encoding.offsets_.reserve(len);
  encoding.special_tokens_mask_.reserve(len);
  encoding.attention_mask_.reserve(len);
  return encoding;
}

Encoding Encoding::FromTokens(std::vector<Token> tokens, uint32_t type_id) {
  const size_t length = tokens.size();
  std::vector<uint32_t> ids;
  std::vector<std::string> values;
  std::vector<Offsets> offsets;
  ids.reserve(length);
  values.reserve(length);
  offsets.reserve(length);
  for (Token& t : tokens) {
    ids.push_back(t.id);
    values.push_back(std::move(t.value));
    offsets.push_back(t.offsets);
  }

  Encoding encoding;
  encoding.ids_ = std::move(ids);
  encoding.tokens_ = std::move(values);
  encoding.offsets_ = std::move(offsets);
  encoding.words_ = std::vector<std::optional<uint32_t>>(length, std::nullopt);
  encoding.type_ids_ = std::vector<uint32_t>(length, type_id);
  encoding.attention_mask_ = std::vector<uint32_t>(length, 1);
  encoding.special_tokens_mask_ = std::vector<uint32_t>(length, 0);
  return encoding;
}

Encoding Encoding::FromIter(std::vector<Item> items) {
  const size_t length = items.size();
  Encoding encoding = WithCapacity(length);

  for (Item& item : items) {
    auto& [id, token, offsets, word, type_id] = item;
    encoding.ids_.push_back(id);
    encoding.tokens_.push_back(std::move(token));
    encoding.offsets_.push_back(offsets);
    encoding.type_ids_.push_back(type_id);
    encoding.words_.push_back(word);
    encoding.special_tokens_mask_.push_back(0);
    encoding.attention_mask_.push_back(1);
  }

  return encoding;
}

size_t Encoding::NSequences() const {
  if (sequence_ranges_.empty()) {
    return 1;
  } else {
    return sequence_ranges_.size();
  }
}

void Encoding::SetSequenceId(size_t sequence_id) {
  sequence_ranges_.insert_or_assign(sequence_id, SequenceRange{0, Len()});
}

std::vector<std::optional<size_t>> Encoding::GetSequenceIds() const {
  std::vector<std::optional<size_t>> sequences(Len(), std::nullopt);
  for (size_t seq_id = 0; seq_id < NSequences(); ++seq_id) {
    const SequenceRange range = SequenceRangeOf(seq_id);
    // PORT NOTE: `Vec::splice` panics on an out-of-bounds range; clamp instead.
    for (size_t i = range.first; i < range.second && i < sequences.size();
         ++i) {
      sequences[i] = seq_id;
    }
  }
  return sequences;
}

void Encoding::ProcessTokensWithOffsets(
    absl::FunctionRef<void(size_t, const std::string&, Offsets&)> func) {
  const size_t n = std::min(tokens_.size(), offsets_.size());
  for (size_t i = 0; i < n; ++i) {
    func(i, tokens_[i], offsets_[i]);
  }
}

SequenceRange Encoding::SequenceRangeOf(size_t sequence_id) const {
  auto it = sequence_ranges_.find(sequence_id);
  if (it != sequence_ranges_.end()) return it->second;
  return SequenceRange{0, Len()};
}

std::optional<size_t> Encoding::TokenToSequence(size_t token) const {
  if (token > Len()) {
    return std::nullopt;
  } else if (sequence_ranges_.empty()) {
    return 0;
  } else {
    for (const auto& [seq_id, range] : sequence_ranges_) {
      if (range.first <= token && token < range.second) {
        return seq_id;
      }
    }
    return std::nullopt;
  }
}

namespace {

// `slice.get(range)`: `None` if the range is not within the slice.
bool RangeInBounds(SequenceRange range, size_t len) {
  return range.first <= range.second && range.second <= len;
}

}  // namespace

std::optional<std::pair<size_t, size_t>> Encoding::WordToTokens(
    uint32_t word, size_t sequence_id) const {
  std::optional<size_t> start;
  std::optional<size_t> end;
  const SequenceRange sequence_range = SequenceRangeOf(sequence_id);

  if (!RangeInBounds(sequence_range, words_.size())) return std::nullopt;
  for (size_t i = 0; i + sequence_range.first < sequence_range.second; ++i) {
    const std::optional<uint32_t>& w = words_[sequence_range.first + i];
    // `.take_while(|(_, w)| **w <= Some(word))`
    if (!(w <= std::optional<uint32_t>(word))) break;
    // `.filter(|(_, w)| **w == Some(word))`
    if (w != std::optional<uint32_t>(word)) continue;
    if (!start.has_value() || std::optional<size_t>(i) < start) {
      start = i;
    }
    if (!end.has_value() || std::optional<size_t>(i) >= end) {
      end = i + 1;
    }
  }

  if (start.has_value() && end.has_value()) {
    return std::make_pair(sequence_range.first + *start,
                          sequence_range.first + *end);
  } else {
    return std::nullopt;
  }
}

std::optional<Offsets> Encoding::WordToChars(uint32_t word,
                                             size_t sequence_id) const {
  std::optional<std::pair<size_t, size_t>> tokens =
      WordToTokens(word, sequence_id);
  if (!tokens.has_value()) return std::nullopt;
  const auto [start, end] = *tokens;
  if (end == 0) {
    return std::nullopt;
  } else {
    return Offsets{offsets_[start].first, offsets_[end - 1].second};
  }
}

std::optional<std::pair<size_t, Offsets>> Encoding::TokenToChars(
    size_t token) const {
  std::optional<size_t> sequence = TokenToSequence(token);
  if (!sequence.has_value()) return std::nullopt;
  if (token >= offsets_.size()) return std::nullopt;
  return std::make_pair(*sequence, offsets_[token]);
}

std::optional<std::pair<size_t, uint32_t>> Encoding::TokenToWord(
    size_t token) const {
  std::optional<size_t> sequence = TokenToSequence(token);
  if (!sequence.has_value()) return std::nullopt;
  if (token >= words_.size() || !words_[token].has_value()) {
    return std::nullopt;
  }
  return std::make_pair(*sequence, *words_[token]);
}

std::optional<size_t> Encoding::CharToToken(size_t pos,
                                            size_t sequence_id) const {
  const SequenceRange sequence_range = SequenceRangeOf(sequence_id);

  if (!RangeInBounds(sequence_range, offsets_.size())) return std::nullopt;
  for (size_t i = sequence_range.first; i < sequence_range.second; ++i) {
    const auto [start, end] = offsets_[i];
    if (pos >= start && pos < end) {
      return i;
    }
  }
  return std::nullopt;
}

std::optional<uint32_t> Encoding::CharToWord(size_t pos,
                                             size_t sequence_id) const {
  std::optional<size_t> token = CharToToken(pos, sequence_id);
  if (!token.has_value()) return std::nullopt;
  std::optional<std::pair<size_t, uint32_t>> word = TokenToWord(*token);
  if (!word.has_value()) return std::nullopt;
  return word->second;
}

namespace {

template <typename T>
std::vector<T> SliceOf(const std::vector<T>& v, size_t start, size_t stop) {
  return std::vector<T>(v.begin() + start, v.begin() + stop);
}

}  // namespace

absl::Status Encoding::Truncate(size_t max_len, size_t stride,
                                TruncationDirection direction) {
  const size_t encoding_len = ids_.size();
  if (max_len >= encoding_len) {
    return absl::OkStatus();
  }

  if (max_len == 0) {
    Encoding o = std::exchange(*this, Encoding::WithCapacity(0));
    overflowing_.push_back(std::move(o));
    return absl::OkStatus();
  }

  if (stride >= max_len) {
    return absl::InvalidArgumentError(absl::StrCat(
        "`stride` must be strictly less than `max_len=", max_len,
        "` (note that `max_len` may be shorter than the max length of the "
        "original model, as it subtracts the number of special characters"));
  }

  // When truncating, we lose the `sequence_ranges` information.
  sequence_ranges_.clear();

  const size_t offset = max_len - stride;
  bool end = false;
  std::vector<std::pair<size_t, size_t>> parts_ranges;
  switch (direction) {
    case TruncationDirection::kRight:
      // `(0..encoding_len).step_by(offset).filter_map(...)`
      for (size_t start = 0; start < encoding_len; start += offset) {
        if (!end) {
          const size_t stop = std::min(start + max_len, encoding_len);
          end = stop == encoding_len;
          parts_ranges.emplace_back(start, stop);
        }
      }
      break;
    case TruncationDirection::kLeft:
      // `(0..encoding_len).rev().step_by(offset).filter_map(...)`
      for (size_t i = encoding_len; i > 0;) {
        const size_t stop = i;  // `let stop = stop + 1;`
        const size_t start = stop >= max_len ? stop - max_len : 0;
        if (start < stop && !end) {
          end = start == 0;
          parts_ranges.emplace_back(start, stop);
        }
        if (i - 1 < offset) break;
        i -= offset;
      }
      break;
  }

  size_t i = 0;
  auto make_part = [this](std::pair<size_t, size_t> part) {
    const auto [start, stop] = part;
    Encoding e;
    e.ids_ = SliceOf(ids_, start, stop);
    e.type_ids_ = SliceOf(type_ids_, start, stop);
    e.tokens_ = SliceOf(tokens_, start, stop);
    e.words_ = SliceOf(words_, start, stop);
    e.offsets_ = SliceOf(offsets_, start, stop);
    e.special_tokens_mask_ = SliceOf(special_tokens_mask_, start, stop);
    e.attention_mask_ = SliceOf(attention_mask_, start, stop);
    return e;
  };
  Encoding new_encoding = make_part(parts_ranges[i]);

  while (true) {
    if (i == parts_ranges.size() - 1) {
      break;
    }
    i += 1;
    new_encoding.overflowing_.push_back(make_part(parts_ranges[i]));
  }
  *this = std::move(new_encoding);
  return absl::OkStatus();
}

Encoding Encoding::Merge(std::vector<Encoding> encodings,
                         bool growing_offsets) {
  Encoding encoding;

  // TODO this is suboptimal as we're doing this iteratively instead of
  // preallocating all the encodings sizes all at once and only copying into
  // this preallocated vector
  // https://github.com/huggingface/tokenizers/pull/1049

  // In order to fix, we just need to preallocate all vectors, then copy
  // everything into it (and deal with overlowings correctly)
  for (Encoding& sub : encodings) {
    encoding.MergeWith(std::move(sub), growing_offsets);
  }

  return encoding;
}

void Encoding::MergeWith(Encoding pair, bool growing_offsets) {
  // Handle merging the overflowing parts too: Combine them all
  // In most of the cases, we expect `pair.overflowing.len() == 0`
  std::vector<Encoding> overflowings;

  // 1. All our overflowings with all the others
  for (const Encoding& self_o : overflowing_) {
    // 1. The pair itself
    Encoding n_encoding = self_o;
    n_encoding.MergeWith(pair, growing_offsets);
    overflowings.push_back(std::move(n_encoding));

    // 2. Its overflowings (this should rarely happen...)
    for (const Encoding& other_o : pair.overflowing_) {
      Encoding n_encoding = self_o;
      n_encoding.MergeWith(other_o, growing_offsets);
      overflowings.push_back(std::move(n_encoding));
    }
  }
  // 2. Ourself with all the other overflowings (this should rarely happen
  // too...)
  for (const Encoding& other_o : pair.overflowing_) {
    Encoding n_encoding = *this;
    n_encoding.MergeWith(other_o, growing_offsets);
    overflowings.push_back(std::move(n_encoding));
  }

  // Finish by merging ourself with the other encoding
  const size_t original_self_len =
      Len();  // Must be before any modification to self.ids

  for (const auto& [seq_id, range] : pair.sequence_ranges_) {
    sequence_ranges_.insert_or_assign(
        seq_id, SequenceRange{original_self_len + range.first,
                              original_self_len + range.second});
  }
  ids_.insert(ids_.end(), pair.ids_.begin(), pair.ids_.end());
  type_ids_.insert(type_ids_.end(), pair.type_ids_.begin(),
                   pair.type_ids_.end());
  tokens_.insert(tokens_.end(), std::make_move_iterator(pair.tokens_.begin()),
                 std::make_move_iterator(pair.tokens_.end()));
  words_.insert(words_.end(), pair.words_.begin(), pair.words_.end());

  const size_t starting_offset =
      growing_offsets ? (offsets_.empty() ? 0 : offsets_.back().second) : 0;
  for (const auto& [start, end] : pair.offsets_) {
    offsets_.emplace_back(start + starting_offset, end + starting_offset);
  }
  special_tokens_mask_.insert(special_tokens_mask_.end(),
                              pair.special_tokens_mask_.begin(),
                              pair.special_tokens_mask_.end());
  attention_mask_.insert(attention_mask_.end(), pair.attention_mask_.begin(),
                         pair.attention_mask_.end());
  overflowing_ = std::move(overflowings);
}

void Encoding::Pad(size_t target_length, uint32_t pad_id, uint32_t pad_type_id,
                   absl::string_view pad_token, PaddingDirection direction) {
  // Dispatch call to all the overflowings first
  for (Encoding& encoding : overflowing_) {
    encoding.Pad(target_length, pad_id, pad_type_id, pad_token, direction);
  }

  // Then check if we should pad ourself
  if (ids_.size() >= target_length) {
    // We just do nothing if the wanted padding length is smaller than us
    return;
  }
  const size_t pad_length = target_length - ids_.size();

  switch (direction) {
    case PaddingDirection::kLeft: {
      ids_.insert(ids_.begin(), pad_length, pad_id);
      type_ids_.insert(type_ids_.begin(), pad_length, pad_type_id);
      tokens_.insert(tokens_.begin(), pad_length, std::string(pad_token));
      words_.insert(words_.begin(), pad_length, std::nullopt);
      attention_mask_.insert(attention_mask_.begin(), pad_length, 0);
      special_tokens_mask_.insert(special_tokens_mask_.begin(), pad_length, 1);
      offsets_.insert(offsets_.begin(), pad_length, Offsets{0, 0});
      for (auto& [seq_id, range] : sequence_ranges_) {
        range =
            SequenceRange{range.first + pad_length, range.second + pad_length};
      }
      break;
    }
    case PaddingDirection::kRight: {
      ids_.insert(ids_.end(), pad_length, pad_id);
      type_ids_.insert(type_ids_.end(), pad_length, pad_type_id);
      tokens_.insert(tokens_.end(), pad_length, std::string(pad_token));
      words_.insert(words_.end(), pad_length, std::nullopt);
      attention_mask_.insert(attention_mask_.end(), pad_length, 0);
      special_tokens_mask_.insert(special_tokens_mask_.end(), pad_length, 1);
      offsets_.insert(offsets_.end(), pad_length, Offsets{0, 0});
      break;
    }
  }
}

bool operator==(const Encoding& a, const Encoding& b) {
  return a.ids_ == b.ids_ && a.type_ids_ == b.type_ids_ &&
         a.tokens_ == b.tokens_ && a.words_ == b.words_ &&
         a.offsets_ == b.offsets_ &&
         a.special_tokens_mask_ == b.special_tokens_mask_ &&
         a.attention_mask_ == b.attention_mask_ &&
         a.overflowing_ == b.overflowing_ &&
         a.sequence_ranges_ == b.sequence_ranges_;
}

// ---------------------------------------------------------------------------
// `#[derive(Serialize, Deserialize)]`

namespace {

absl::Status MissingField(absl::string_view name) {
  return absl::InvalidArgumentError(absl::StrCat("missing field `", name, "`"));
}

absl::Status InvalidType(const Json& json, absl::string_view expected) {
  return absl::InvalidArgumentError(
      absl::StrCat("invalid type: ", json.dump(), ", expected ", expected));
}

absl::StatusOr<const Json*> RequiredField(const Json& json,
                                          absl::string_view name) {
  auto it = json.find(std::string(name));
  if (it == json.end()) return MissingField(name);
  return &*it;
}

absl::StatusOr<uint32_t> ParseU32(const Json& json) {
  if (!json.is_number_unsigned() || json.get<uint64_t>() > UINT32_MAX) {
    return InvalidType(json, "u32");
  }
  return json.get<uint32_t>();
}

absl::StatusOr<size_t> ParseUsize(const Json& json) {
  if (!json.is_number_unsigned()) return InvalidType(json, "usize");
  return static_cast<size_t>(json.get<uint64_t>());
}

absl::StatusOr<std::vector<uint32_t>> ParseU32Array(const Json& json) {
  if (!json.is_array()) return InvalidType(json, "a sequence");
  std::vector<uint32_t> out;
  out.reserve(json.size());
  for (const Json& item : json) {
    ASSIGN_OR_RETURN(uint32_t v, ParseU32(item));
    out.push_back(v);
  }
  return out;
}

Json ToJsonArray(const std::vector<uint32_t>& values) {
  Json out = Json::array();
  for (uint32_t v : values) out.push_back(v);
  return out;
}

}  // namespace

absl::StatusOr<Encoding> Encoding::FromJson(const Json& json) {
  if (!json.is_object()) return InvalidType(json, "struct Encoding");
  Encoding encoding;

  ASSIGN_OR_RETURN(const Json* ids, RequiredField(json, "ids"));
  ASSIGN_OR_RETURN(encoding.ids_, ParseU32Array(*ids));

  ASSIGN_OR_RETURN(const Json* type_ids, RequiredField(json, "type_ids"));
  ASSIGN_OR_RETURN(encoding.type_ids_, ParseU32Array(*type_ids));

  ASSIGN_OR_RETURN(const Json* tokens, RequiredField(json, "tokens"));
  if (!tokens->is_array()) return InvalidType(*tokens, "a sequence");
  for (const Json& token : *tokens) {
    if (!token.is_string()) return InvalidType(token, "a string");
    encoding.tokens_.push_back(token.get<std::string>());
  }

  ASSIGN_OR_RETURN(const Json* words, RequiredField(json, "words"));
  if (!words->is_array()) return InvalidType(*words, "a sequence");
  for (const Json& word : *words) {
    if (word.is_null()) {
      encoding.words_.push_back(std::nullopt);
    } else {
      ASSIGN_OR_RETURN(uint32_t w, ParseU32(word));
      encoding.words_.push_back(w);
    }
  }

  ASSIGN_OR_RETURN(const Json* offsets, RequiredField(json, "offsets"));
  if (!offsets->is_array()) return InvalidType(*offsets, "a sequence");
  for (const Json& offset : *offsets) {
    if (!offset.is_array() || offset.size() != 2) {
      return InvalidType(offset, "a tuple of size 2");
    }
    ASSIGN_OR_RETURN(size_t start, ParseUsize(offset[0]));
    ASSIGN_OR_RETURN(size_t end, ParseUsize(offset[1]));
    encoding.offsets_.emplace_back(start, end);
  }

  ASSIGN_OR_RETURN(const Json* special_tokens_mask,
                   RequiredField(json, "special_tokens_mask"));
  ASSIGN_OR_RETURN(encoding.special_tokens_mask_,
                   ParseU32Array(*special_tokens_mask));

  ASSIGN_OR_RETURN(const Json* attention_mask,
                   RequiredField(json, "attention_mask"));
  ASSIGN_OR_RETURN(encoding.attention_mask_, ParseU32Array(*attention_mask));

  ASSIGN_OR_RETURN(const Json* overflowing, RequiredField(json, "overflowing"));
  if (!overflowing->is_array()) return InvalidType(*overflowing, "a sequence");
  for (const Json& o : *overflowing) {
    ASSIGN_OR_RETURN(Encoding e, Encoding::FromJson(o));
    encoding.overflowing_.push_back(std::move(e));
  }

  ASSIGN_OR_RETURN(const Json* sequence_ranges,
                   RequiredField(json, "sequence_ranges"));
  if (!sequence_ranges->is_object()) {
    return InvalidType(*sequence_ranges, "a map");
  }
  for (const auto& [key, value] : sequence_ranges->items()) {
    size_t seq_id;
    if (!absl::SimpleAtoi(key, &seq_id)) {
      return absl::InvalidArgumentError(
          absl::StrCat("invalid value: string \"", key, "\", expected usize"));
    }
    if (!value.is_object()) return InvalidType(value, "struct Range");
    ASSIGN_OR_RETURN(const Json* start, RequiredField(value, "start"));
    ASSIGN_OR_RETURN(const Json* end, RequiredField(value, "end"));
    ASSIGN_OR_RETURN(size_t s, ParseUsize(*start));
    ASSIGN_OR_RETURN(size_t e, ParseUsize(*end));
    encoding.sequence_ranges_.insert_or_assign(seq_id, SequenceRange{s, e});
  }

  return encoding;
}

Json Encoding::ToJson() const {
  Json json = Json::object();
  json["ids"] = ToJsonArray(ids_);
  json["type_ids"] = ToJsonArray(type_ids_);
  json["tokens"] = tokens_;
  Json words = Json::array();
  for (const std::optional<uint32_t>& word : words_) {
    if (word.has_value()) {
      words.push_back(*word);
    } else {
      words.push_back(nullptr);
    }
  }
  json["words"] = std::move(words);
  Json offsets = Json::array();
  for (const Offsets& offset : offsets_) {
    offsets.push_back(Json::array({offset.first, offset.second}));
  }
  json["offsets"] = std::move(offsets);
  json["special_tokens_mask"] = ToJsonArray(special_tokens_mask_);
  json["attention_mask"] = ToJsonArray(attention_mask_);
  Json overflowing = Json::array();
  for (const Encoding& o : overflowing_) {
    overflowing.push_back(o.ToJson());
  }
  json["overflowing"] = std::move(overflowing);
  // PORT NOTE: hash map iteration order is unspecified in Rust too; the keys
  // are emitted in ascending order to make the output deterministic.
  std::vector<std::pair<size_t, SequenceRange>> ranges(sequence_ranges_.begin(),
                                                       sequence_ranges_.end());
  std::sort(ranges.begin(), ranges.end());
  Json sequence_ranges = Json::object();
  for (const auto& [seq_id, range] : ranges) {
    Json r = Json::object();
    r["start"] = range.first;
    r["end"] = range.second;
    sequence_ranges[absl::StrCat(seq_id)] = std::move(r);
  }
  json["sequence_ranges"] = std::move(sequence_ranges);
  return json;
}

}  // namespace tokenizers
