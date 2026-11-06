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
// C++ port of `src/tokenizer/normalizer.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/pattern.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/unicode/categories.h"
#include "support/tokenizer/tokenizers/unicode/normalization.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/utf8.h"
#include "support/util/status_macros.h"

namespace tokenizers {

namespace {

using Alignment = std::pair<size_t, size_t>;
using ByteRange = std::pair<size_t, size_t>;

// `str::is_char_boundary(index)`
bool IsCharBoundary(absl::string_view s, size_t index) {
  if (index == 0 || index == s.size()) return true;
  if (index > s.size()) return false;
  return (static_cast<unsigned char>(s[index]) & 0xC0) != 0x80;
}

// `str::get(start..end)`: `None` if the range is out of bounds or not on char
// boundaries.
std::optional<absl::string_view> StrGet(absl::string_view s, size_t start,
                                        size_t end) {
  if (start > end || end > s.size()) return std::nullopt;
  if (!IsCharBoundary(s, start) || !IsCharBoundary(s, end)) {
    return std::nullopt;
  }
  return s.substr(start, end - start);
}

std::optional<absl::string_view> StrGet(absl::string_view s, ByteRange range) {
  return StrGet(s, range.first, range.second);
}

// `s.chars()` together with the byte length of every char (which equals
// `c.len_utf8()` for valid UTF-8).
struct DecodedChar {
  char32_t c;
  size_t len;
};

std::vector<DecodedChar> DecodeChars(absl::string_view s) {
  std::vector<DecodedChar> chars;
  size_t pos = 0;
  while (pos < s.size()) {
    const size_t start = pos;
    const char32_t c = DecodeUtf8(s, &pos);
    chars.push_back({c, pos - start});
  }
  return chars;
}

std::vector<char32_t> Chars(absl::string_view s) {
  std::vector<char32_t> chars;
  size_t pos = 0;
  while (pos < s.size()) {
    chars.push_back(DecodeUtf8(s, &pos));
  }
  return chars;
}

// Returns the range covered by a slice of alignments
std::optional<ByteRange> ExpandAlignments(const Alignment* alignments,
                                          size_t len) {
  if (len == 0) {
    return std::nullopt;
  } else {
    const size_t start = alignments[0].first;
    const size_t end = alignments[len - 1].second;
    return ByteRange{start, end};
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Range

std::optional<size_t> Range::Len() const {
  // `let end = match range.end_bound() { Unbounded => None, ... }?;`
  if (!end.has_value()) return std::nullopt;
  // `match range.start_bound() { Unbounded => Some(end), Included(i) =>
  // Some(end - *i), ... }`
  if (!start.has_value()) return *end;
  return *end - *start;
}

std::pair<size_t, size_t> Range::IntoFullRange(size_t max_len) const {
  return {start.value_or(0), end.value_or(max_len)};
}

// ---------------------------------------------------------------------------
// SplitDelimiterBehavior

absl::string_view SplitDelimiterBehaviorToString(
    SplitDelimiterBehavior behavior) {
  switch (behavior) {
    case SplitDelimiterBehavior::kRemoved:
      return "Removed";
    case SplitDelimiterBehavior::kIsolated:
      return "Isolated";
    case SplitDelimiterBehavior::kMergedWithPrevious:
      return "MergedWithPrevious";
    case SplitDelimiterBehavior::kMergedWithNext:
      return "MergedWithNext";
    case SplitDelimiterBehavior::kContiguous:
      return "Contiguous";
  }
  return "Removed";
}

absl::StatusOr<SplitDelimiterBehavior> SplitDelimiterBehaviorFromJson(
    const Json& json) {
  if (!json.is_string()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "invalid type: ", json.dump(), ", expected variant identifier"));
  }
  const std::string s = json.get<std::string>();
  if (s == "Removed") return SplitDelimiterBehavior::kRemoved;
  if (s == "Isolated") return SplitDelimiterBehavior::kIsolated;
  if (s == "MergedWithPrevious") {
    return SplitDelimiterBehavior::kMergedWithPrevious;
  }
  if (s == "MergedWithNext") return SplitDelimiterBehavior::kMergedWithNext;
  if (s == "Contiguous") return SplitDelimiterBehavior::kContiguous;
  return absl::InvalidArgumentError(absl::StrCat(
      "unknown variant `", s,
      "`, expected one of `Removed`, `Isolated`, `MergedWithPrevious`, "
      "`MergedWithNext`, `Contiguous`"));
}

Json SplitDelimiterBehaviorToJson(SplitDelimiterBehavior behavior) {
  return std::string(SplitDelimiterBehaviorToString(behavior));
}

// ---------------------------------------------------------------------------
// NormalizedString

NormalizedString NormalizedString::From(absl::string_view s) {
  // `impl From<String> for NormalizedString`
  std::vector<Alignment> alignments;
  alignments.reserve(s.size());
  size_t pos = 0;
  while (pos < s.size()) {
    const size_t b = pos;
    DecodeUtf8(s, &pos);
    const size_t len = pos - b;
    for (size_t i = 0; i < len; ++i) {
      alignments.emplace_back(b, b + len);
    }
  }
  return NormalizedString(std::string(s), std::string(s), std::move(alignments),
                          0);
}

NormalizedString NormalizedString::New(std::string original,
                                       std::string normalized,
                                       std::vector<Alignment> alignments,
                                       size_t original_shift) {
  return NormalizedString(std::move(original), std::move(normalized),
                          std::move(alignments), original_shift);
}

std::optional<ByteRange> NormalizedString::ConvertOffsets(
    const Range& range) const {
  const size_t len_original = LenOriginal();
  const size_t len_normalized = Len();

  const bool original = range.IsOriginal();
  const ByteRange target = original ? range.IntoFullRange(len_original)
                                    : range.IntoFullRange(len_normalized);

  // If we target an empty range, let's return the same
  if (target.first == target.second) {
    return target;
  }
  // If the target goes reverse, return None
  if (target.first > target.second) {
    return std::nullopt;
  }

  // If we target 0..0 on an empty string, we want to expand to the entire
  // equivalent
  if (original && original_.empty() && target == ByteRange{0, 0}) {
    return ByteRange{0, len_normalized};
  }
  if (!original && normalized_.empty() && target == ByteRange{0, 0}) {
    return ByteRange{0, len_original};
  }

  if (original) {
    std::optional<size_t> start;
    std::optional<size_t> end;
    for (size_t i = 0; i < alignments_.size(); ++i) {
      const Alignment& alignment = alignments_[i];
      // `.take_while(|(_, alignment)| target.end >= alignment.1)`
      if (!(target.second >= alignment.second)) break;
      if (!start.has_value() && target.first <= alignment.first) {
        // For now, don't update if width == 0
        if (alignment.first != alignment.second) {
          start = i;
        }
      }
      if (target.second >= alignment.second) {
        end = i + 1;
      }
    }

    if (start.has_value() && !end.has_value()) {
      // Targeting inexistent beginning
      return ByteRange{*start, *start};
    } else if (!start.has_value() && end.has_value()) {
      // Targeting inexistent end
      return ByteRange{*end, *end};
    } else if (start.has_value() && end.has_value()) {
      // Found the range
      return ByteRange{*start, *end};
    } else {
      return std::nullopt;
    }
  } else {
    // `self.alignments.get(target).and_then(expand_alignments)`
    if (target.second > alignments_.size()) return std::nullopt;
    return ExpandAlignments(alignments_.data() + target.first,
                            target.second - target.first);
  }
}

std::optional<absl::string_view> NormalizedString::GetRange(
    const Range& range) const {
  if (range.IsOriginal()) {
    std::optional<ByteRange> converted = ConvertOffsets(range);
    if (!converted.has_value()) return std::nullopt;
    return StrGet(normalized_, *converted);
  } else {
    return StrGet(normalized_, range.IntoFullRange(Len()));
  }
}

std::optional<absl::string_view> NormalizedString::GetRangeOriginal(
    const Range& range) const {
  if (range.IsOriginal()) {
    return StrGet(original_, range.IntoFullRange(LenOriginal()));
  } else {
    std::optional<ByteRange> converted = ConvertOffsets(range);
    if (!converted.has_value()) return std::nullopt;
    return StrGet(original_, *converted);
  }
}

std::optional<Range> NormalizedString::ValidateRange(const Range& range) const {
  if (range.IsOriginal()) {
    const ByteRange r = range.IntoFullRange(original_.size());
    if (!(IsCharBoundary(original_, r.first) &&
          IsCharBoundary(original_, r.second))) {
      return std::nullopt;
    } else {
      return Range::Original(r.first, r.second);
    }
  } else {
    const ByteRange r = range.IntoFullRange(normalized_.size());
    if (!(IsCharBoundary(normalized_, r.first) &&
          IsCharBoundary(normalized_, r.second))) {
      return std::nullopt;
    } else {
      return Range::Normalized(r.first, r.second);
    }
  }
}

std::optional<NormalizedString> NormalizedString::Slice(
    const Range& range) const {
  std::optional<Range> full_range = ValidateRange(range);
  if (!full_range.has_value()) return std::nullopt;
  ByteRange normalized_range;
  ByteRange original_range;
  if (full_range->IsOriginal()) {
    std::optional<ByteRange> converted = ConvertOffsets(*full_range);
    if (!converted.has_value()) return std::nullopt;
    normalized_range = *converted;
    original_range = {*full_range->start, *full_range->end};
  } else {
    normalized_range = {*full_range->start, *full_range->end};
    std::optional<ByteRange> converted = ConvertOffsets(*full_range);
    if (!converted.has_value()) return std::nullopt;
    original_range = *converted;
  }

  const size_t n_shift = original_range.first;

  // `self.alignments.get(normalized_range)?`
  if (normalized_range.first > normalized_range.second ||
      normalized_range.second > alignments_.size()) {
    return std::nullopt;
  }
  std::vector<Alignment> alignments;
  alignments.reserve(normalized_range.second - normalized_range.first);
  for (size_t i = normalized_range.first; i < normalized_range.second; ++i) {
    alignments.emplace_back(alignments_[i].first - n_shift,
                            alignments_[i].second - n_shift);
  }

  return NormalizedString(
      std::string(GetRangeOriginal(*full_range).value_or("")),
      std::string(GetRange(*full_range).value_or("")), std::move(alignments),
      original_shift_ + original_range.first);
}

void NormalizedString::TransformRange(
    const Range& range, const std::vector<std::pair<char32_t, int64_t>>& dest,
    size_t initial_offset) {
  ByteRange n_range;
  if (range.IsNormalized()) {
    n_range = range.IntoFullRange(Len());
  } else {
    std::optional<ByteRange> converted = ConvertOffsets(range);
    if (!converted.has_value()) return;
    n_range = *converted;
  }
  // PORT NOTE: `self.normalized[n_range.clone()]` panics on an invalid range;
  // leave the string untouched instead.
  if (n_range.first > n_range.second || n_range.second > normalized_.size() ||
      !IsCharBoundary(normalized_, n_range.first) ||
      !IsCharBoundary(normalized_, n_range.second)) {
    return;
  }

  // Retrieve the original characters that are being replaced. This let us
  // compute the change in byte sizes along the way.
  const std::vector<DecodedChar> replaced_normalized =
      DecodeChars(absl::string_view(normalized_)
                      .substr(n_range.first, n_range.second - n_range.first));
  size_t replaced_next = 0;  // iterator position in `replaced_normalized`
  size_t initial_removed = 0;
  for (size_t i = 0;
       i < initial_offset && replaced_next < replaced_normalized.size(); ++i) {
    initial_removed += replaced_normalized[replaced_next++].len;
  }

  size_t offset = initial_removed + n_range.first;
  std::vector<Alignment> alignments;
  alignments.reserve(n_range.second - n_range.first);
  std::string normalized;
  // PORT NOTE: the Rust code indexes `self.alignments[idx]`, which panics if
  // `dest` replaces more chars than the range contains; clamp to the last
  // alignment instead.
  auto alignment_at = [this](size_t idx) -> Alignment {
    if (idx < alignments_.size()) return alignments_[idx];
    if (alignments_.empty()) return {0, 0};
    return alignments_.back();
  };
  for (const auto& [c, changes] : dest) {
    const size_t idx = offset;
    Alignment align;
    if (changes > 0) {
      if (idx < 1) {
        align = {0, 0};
      } else {
        // This is a newly inserted character, so it shares the same alignment
        // than the previous one
        align = alignment_at(idx - 1);
      }
    } else {
      align = alignment_at(idx);
    }

    // If we are replacing a character, find it and compute the change in size
    std::optional<DecodedChar> replaced_char;
    if (changes <= 0 && replaced_next < replaced_normalized.size()) {
      replaced_char = replaced_normalized[replaced_next++];
    }
    const size_t replaced_char_size =
        replaced_char.has_value() ? replaced_char->len : 0;

    // If we are removing some characters, find them too
    size_t total_bytes_to_remove = 0;
    if (changes < 0) {
      for (int64_t i = 0;
           i < -changes && replaced_next < replaced_normalized.size(); ++i) {
        total_bytes_to_remove += replaced_normalized[replaced_next++].len;
      }
    }

    // Keep track of the changes for next offsets
    offset += replaced_char_size;
    offset += total_bytes_to_remove;

    const size_t len_utf8 = Utf8Len(c);
    for (size_t i = 0; i < len_utf8; ++i) {
      alignments.push_back(align);
    }

    // Then we keep only the char for string reconstruction
    AppendUtf8(c, &normalized);
  }

  // `self.alignments.splice(n_range.clone(), alignments);`
  alignments_.erase(alignments_.begin() + n_range.first,
                    alignments_.begin() + n_range.second);
  alignments_.insert(alignments_.begin() + n_range.first, alignments.begin(),
                     alignments.end());

  // `self.normalized.as_mut_vec().splice(n_range, normalized.bytes());`
  normalized_.replace(n_range.first, n_range.second - n_range.first,
                      normalized);
}

void NormalizedString::Transform(
    const std::vector<std::pair<char32_t, int64_t>>& dest,
    size_t initial_offset) {
  TransformRange(Range::OriginalAll(), dest, initial_offset);
}

NormalizedString& NormalizedString::Nfd() {
  Transform(unicode::Nfd(normalized_), 0);
  return *this;
}

NormalizedString& NormalizedString::Nfkd() {
  Transform(unicode::Nfkd(normalized_), 0);
  return *this;
}

NormalizedString& NormalizedString::Nfc() {
  Transform(unicode::Nfc(normalized_), 0);
  return *this;
}

NormalizedString& NormalizedString::Nfkc() {
  Transform(unicode::Nfkc(normalized_), 0);
  return *this;
}

NormalizedString& NormalizedString::Filter(
    absl::FunctionRef<bool(char32_t)> keep) {
  int64_t removed = 0;
  size_t removed_start = 0;

  std::vector<std::pair<char32_t, int64_t>> transforms;
  transforms.reserve(normalized_.size());
  std::optional<char32_t> last_c;
  for (const char32_t c : Chars(normalized_)) {
    if (keep(c)) {
      if (last_c.has_value()) {
        transforms.emplace_back(*last_c, -removed);
      } else {
        removed_start = static_cast<size_t>(removed);
      }
      last_c = c;
      removed = 0;
    } else {
      removed += 1;
    }
  }
  if (last_c.has_value()) {
    transforms.emplace_back(*last_c, -removed);
  }
  Transform(transforms, removed_start);
  return *this;
}

NormalizedString& NormalizedString::Prepend(absl::string_view s) {
  if (!normalized_.empty()) {
    size_t pos = 0;
    const char32_t next = DecodeUtf8(normalized_, &pos);
    std::vector<std::pair<char32_t, int64_t>> transformations;
    size_t i = 0;
    for (const char32_t c : Chars(s)) {
      transformations.emplace_back(c, i != 0 ? 1 : 0);
      ++i;
    }
    transformations.emplace_back(next, 1);

    TransformRange(Range::Normalized(0, pos), transformations, 0);
  }
  return *this;
}

NormalizedString& NormalizedString::Append(absl::string_view s) {
  if (!normalized_.empty()) {
    // `self.normalized.char_indices().last()`
    size_t b = 0;
    char32_t prev = 0;
    for (const auto [offset, c] : CharIndices(normalized_)) {
      b = offset;
      prev = c;
    }
    std::vector<std::pair<char32_t, int64_t>> transformations;
    transformations.emplace_back(prev, 0);
    for (const char32_t c : Chars(s)) {
      transformations.emplace_back(c, 1);
    }
    TransformRange(Range::NormalizedFrom(b), transformations, 0);
  } else {
    std::vector<std::pair<char32_t, int64_t>> transformations;
    for (const char32_t c : Chars(s)) {
      transformations.emplace_back(c, 1);
    }
    TransformRange(Range::NormalizedAll(), transformations, 0);
  }
  return *this;
}

NormalizedString& NormalizedString::Map(
    absl::FunctionRef<char32_t(char32_t)> map) {
  std::vector<std::pair<char32_t, int64_t>> transformations;
  for (const char32_t c : Chars(normalized_)) {
    transformations.emplace_back(map(c), 0);
  }
  Transform(transformations, 0);
  return *this;
}

const NormalizedString& NormalizedString::ForEach(
    absl::FunctionRef<void(char32_t)> foreach) const {
  size_t pos = 0;
  while (pos < normalized_.size()) {
    foreach (DecodeUtf8(normalized_, &pos));
  }
  return *this;
}

NormalizedString& NormalizedString::Lowercase() {
  std::vector<std::pair<char32_t, int64_t>> new_chars;
  ForEach([&new_chars](char32_t c) {
    size_t index = 0;
    for (const char32_t lc : unicode::ToLowercase(c)) {
      new_chars.emplace_back(lc, index > 0 ? 1 : 0);
      ++index;
    }
  });
  Transform(new_chars, 0);
  return *this;
}

NormalizedString& NormalizedString::Uppercase() {
  std::vector<std::pair<char32_t, int64_t>> new_chars;
  ForEach([&new_chars](char32_t c) {
    size_t index = 0;
    for (const char32_t uc : unicode::ToUppercase(c)) {
      new_chars.emplace_back(uc, index > 0 ? 1 : 0);
      ++index;
    }
  });
  Transform(new_chars, 0);
  return *this;
}

absl::Status NormalizedString::Replace(const Pattern& pattern,
                                       absl::string_view content) {
  std::string new_normalized;
  new_normalized.reserve(
      normalized_.size());  // Initially allocate for the input size
  std::vector<Alignment> new_alignments;
  new_alignments.reserve(alignments_.size());
  size_t last_end = 0;  // Keep track of the last end position

  ASSIGN_OR_RETURN(const auto matches, pattern.FindMatches(normalized_));
  const std::vector<char32_t> content_chars = Chars(content);
  for (const auto& [offsets, is_match] : matches) {
    if (is_match) {
      const auto [start, end] = offsets;
      // PORT NOTE: the Rust code panics (slice index) if the pattern violates
      // the `find_matches` contract (contiguous, ordered slices).
      if (start < last_end || start > end || end > normalized_.size()) {
        return absl::InternalError(
            "NormalizedString::Replace: pattern matches are not contiguous "
            "ordered slices");
      }

      const size_t removed_chars =
          CountChars(absl::string_view(normalized_).substr(start, end - start));

      /* The following code is equivalent to this call, but computationally
      much more efficient
      self.transform_range(
          Range::Normalized(range),
          content.chars().map(|c| {
              new_len += c.len_utf8();
              (c, 1)
          }),
          removed_chars,
      ); */

      // Copy the part of the string that is before the match
      new_normalized.append(normalized_, last_end, start - last_end);
      new_alignments.insert(new_alignments.end(),
                            alignments_.begin() + last_end,
                            alignments_.begin() + start);

      const ByteRange n_range =
          Range::Normalized(start, end).IntoFullRange(Len());

      // Retrieve the original characters that are being replaced. This let us
      // compute the change in byte sizes along the way.
      const std::vector<DecodedChar> replaced_normalized = DecodeChars(
          absl::string_view(normalized_)
              .substr(n_range.first, n_range.second - n_range.first));
      size_t replaced_next = 0;
      size_t initial_removed = 0;
      for (size_t i = 0;
           i < removed_chars && replaced_next < replaced_normalized.size();
           ++i) {
        initial_removed += replaced_normalized[replaced_next++].len;
      }

      size_t offset = initial_removed + n_range.first;
      for (const char32_t c : content_chars) {
        // `dest = content.chars().map(|c| (c, 1))`: `changes` is always 1.
        const size_t idx = offset;
        Alignment align;
        if (idx < 1) {
          align = {0, 0};
        } else {
          // This is a newly inserted character, so it shares the same
          // alignment than the previous one
          align = alignments_[idx - 1];
        }

        // If we are replacing a character, find it and compute the change in
        // size: never, since `changes` is positive.
        const size_t replaced_char_size = 0;

        // If we are removing some characters, find them too: never.
        const size_t total_bytes_to_remove = 0;

        // Keep track of the changes for next offsets
        offset += replaced_char_size;
        offset += total_bytes_to_remove;

        const size_t len_utf8 = Utf8Len(c);
        for (size_t i = 0; i < len_utf8; ++i) {
          new_alignments.push_back(align);
        }

        // Then we keep only the char for string reconstruction
        AppendUtf8(c, &new_normalized);
      }

      last_end = end;
    }
  }

  // Copy the remaining part of the input
  new_normalized.append(normalized_, last_end, std::string::npos);
  new_alignments.insert(new_alignments.end(), alignments_.begin() + last_end,
                        alignments_.end());

  normalized_ = std::move(new_normalized);
  alignments_ = std::move(new_alignments);
  return absl::OkStatus();
}

size_t NormalizedString::Clear() {
  const size_t len = Len();
  Transform({}, len);
  return len;
}

absl::StatusOr<std::vector<NormalizedString>> NormalizedString::Split(
    const Pattern& pattern, SplitDelimiterBehavior behavior) const {
  ASSIGN_OR_RETURN(auto matches, pattern.FindMatches(normalized_));

  // Process the matches according to the selected behavior: Vec<(Offsets,
  // should_remove)>
  std::vector<std::pair<Offsets, bool>> splits;
  switch (behavior) {
    case SplitDelimiterBehavior::kIsolated:
      splits.reserve(matches.size());
      for (const auto& [offsets, is_match] : matches) {
        splits.emplace_back(offsets, false);
      }
      break;
    case SplitDelimiterBehavior::kRemoved:
      splits = std::move(matches);
      break;
    case SplitDelimiterBehavior::kContiguous: {
      bool previous_match = false;
      for (const auto& [offsets, is_match] : matches) {
        if (is_match == previous_match) {
          if (!splits.empty()) {
            splits.back().first.second = offsets.second;
          } else {
            splits.emplace_back(offsets, false);
          }
        } else {
          splits.emplace_back(offsets, false);
        }
        previous_match = is_match;
      }
      break;
    }
    case SplitDelimiterBehavior::kMergedWithPrevious: {
      bool previous_match = false;
      for (const auto& [offsets, is_match] : matches) {
        if (is_match && !previous_match) {
          if (!splits.empty()) {
            splits.back().first.second = offsets.second;
          } else {
            splits.emplace_back(offsets, false);
          }
        } else {
          splits.emplace_back(offsets, false);
        }
        previous_match = is_match;
      }
      break;
    }
    case SplitDelimiterBehavior::kMergedWithNext: {
      bool previous_match = false;
      for (auto it = matches.rbegin(); it != matches.rend(); ++it) {
        const auto& [offsets, is_match] = *it;
        if (is_match && !previous_match) {
          if (!splits.empty()) {
            splits.back().first.first = offsets.first;
          } else {
            splits.emplace_back(offsets, false);
          }
        } else {
          splits.emplace_back(offsets, false);
        }
        previous_match = is_match;
      }
      std::reverse(splits.begin(), splits.end());
      break;
    }
  }

  // Then we split according to the computed splits
  std::vector<NormalizedString> result;
  for (const auto& [offsets, remove] : splits) {
    if (!remove) {
      std::optional<NormalizedString> slice =
          Slice(Range::Normalized(offsets.first, offsets.second));
      if (!slice.has_value()) {
        // `.expect("NormalizedString bad split")`
        return absl::InternalError("NormalizedString bad split");
      }
      result.push_back(*std::move(slice));
    }
  }
  return result;
}

NormalizedString& NormalizedString::Lstrip() { return Lrstrip(true, false); }

NormalizedString& NormalizedString::Rstrip() { return Lrstrip(false, true); }

NormalizedString& NormalizedString::Strip() { return Lrstrip(true, true); }

NormalizedString& NormalizedString::Lrstrip(bool left, bool right) {
  const std::vector<char32_t> chars = Chars(normalized_);
  size_t leading_spaces = 0;
  if (left) {
    for (const char32_t c : chars) {
      if (!unicode::IsWhitespace(c)) break;
      ++leading_spaces;
    }
  }
  size_t trailing_spaces = 0;
  if (right) {
    for (auto it = chars.rbegin(); it != chars.rend(); ++it) {
      if (!unicode::IsWhitespace(*it)) break;
      ++trailing_spaces;
    }
  }

  if (leading_spaces > 0 || trailing_spaces > 0) {
    const size_t count = chars.size();
    std::vector<std::pair<char32_t, int64_t>> transformation;
    for (size_t i = 0; i < chars.size(); ++i) {
      const char32_t c = chars[i];
      if (i < leading_spaces || i >= count - trailing_spaces) {
        continue;
      } else if (i == Len() - trailing_spaces - 1) {
        // PORT NOTE: `self.len()` is the *byte* length while `i` is a char
        // index; kept exactly like the Rust code (harmless: see tests).
        transformation.emplace_back(c, -static_cast<int64_t>(trailing_spaces));
      } else {
        transformation.emplace_back(c, 0);
      }
    }
    Transform(transformation, leading_spaces);
  }
  return *this;
}

std::vector<Alignment> NormalizedString::AlignmentsOriginal() const {
  // Start, end are in alignments
  // offset, length are in alignments_original
  std::vector<Alignment> alignments_original;
  alignments_original.reserve(original_.size());

  // PORT NOTE: the Rust code indexes `self.alignments[0]` (panics when empty).
  if (alignments_.empty()) {
    alignments_original.assign(original_.size(), Alignment{0, 0});
    return alignments_original;
  }

  // Eventual gap before first group
  const size_t start = alignments_[0].first;
  if (start != 0) {
    alignments_original.insert(alignments_original.end(), start,
                               Alignment{0, 0});
  }

  Alignment last = alignments_[0];
  size_t offset = 0;
  size_t length = 0;
  for (const Alignment& alignment : alignments_) {
    const auto [start, end] = alignment;
    if (last == alignment) {
      // This is the same group
      length += 1;
    } else {
      // This is a new group
      // PORT NOTE: `panic!("We can't have overlapping ranges.")` when
      // `start < last.1`; the gap is clamped to zero instead.

      // Add the old group
      alignments_original.insert(alignments_original.end(),
                                 last.second - last.first,
                                 Alignment{offset, offset + length});
      offset += length;
      length = 1;

      // Eventual gap between the 2 groups
      alignments_original.insert(alignments_original.end(),
                                 start > last.second ? start - last.second : 0,
                                 Alignment{offset, offset});
    }

    last = alignment;
  }
  // Add the last group
  alignments_original.insert(alignments_original.end(),
                             last.second - last.first,
                             Alignment{offset, offset + length});

  // Add eventual last gap
  offset += length;
  if (original_.size() > alignments_original.size()) {
    alignments_original.insert(alignments_original.end(),
                               original_.size() - alignments_original.size(),
                               Alignment{offset, offset});
  }

  // assert_eq!(alignments_original.len(), self.original.len());
  return alignments_original;
}

// ---------------------------------------------------------------------------
// Free functions

std::optional<absl::string_view> GetRangeOf(absl::string_view s, size_t start,
                                            std::optional<size_t> end_bound) {
  const size_t len = CountChars(s);
  const size_t end = end_bound.value_or(len);

  if (start == 0 && end == 0) {
    return s.substr(0, 0);
  } else if (start >= len || end > len || start >= end) {
    return std::nullopt;
  } else {
    // `s.char_indices().map(|(i, _)| i).nth(start).unwrap_or(0)`
    size_t start_b = 0;
    size_t end_b = s.size();
    size_t i = 0;
    for (const auto [b, c] : CharIndices(s)) {
      if (i == start) start_b = b;
      if (i == end) end_b = b;
      ++i;
    }
    return s.substr(start_b, end_b - start_b);
  }
}

std::optional<ByteRange> BytesToChar(absl::string_view s, ByteRange range) {
  std::optional<size_t> start;
  std::optional<size_t> end;
  if (range == ByteRange{0, 0}) {
    start = 0;
    end = 0;
  }

  size_t i = 0;
  for (const auto [b, c] : CharIndices(s)) {
    // `.take_while(|(_, (b, _))| *b <= range.end)`
    if (!(b <= range.second)) break;
    // `.filter(|(_, (b, _))| *b >= range.start)`
    if (b >= range.first) {
      if (b == range.first) {
        start = i;
      }
      if (b == range.second) {
        end = i;
      }
      if (b + Utf8Len(c) == range.second) {
        end = i + 1;
      }
    }
    ++i;
  }

  if (!start.has_value() || !end.has_value()) return std::nullopt;
  return ByteRange{*start, *end};
}

std::optional<ByteRange> CharToBytes(absl::string_view s, ByteRange range) {
  std::optional<size_t> start;
  std::optional<size_t> end;
  if (range == ByteRange{0, 0}) {
    start = 0;
    end = 0;
  }

  if (range.first == range.second) {
    // `.skip(range.start).take(1)`
    size_t i = 0;
    for (const auto [b, c] : CharIndices(s)) {
      if (i == range.first) {
        start = b;
        end = b;
        break;
      }
      ++i;
    }
  } else {
    // PORT NOTE: `range.end - range.start` underflows (panics) in Rust when
    // the range is reversed; return `None` instead.
    if (range.second < range.first) return std::nullopt;
    // `.skip(range.start).take(range.end - range.start)`
    size_t i = 0;
    for (const auto [b, c] : CharIndices(s)) {
      if (i >= range.first) {
        if (i >= range.second) break;
        if (!start.has_value()) {
          start = b;
        }
        end = b + Utf8Len(c);
      }
      ++i;
    }
  }

  if (!start.has_value() || !end.has_value()) return std::nullopt;
  return ByteRange{*start, *end};
}

}  // namespace tokenizers
