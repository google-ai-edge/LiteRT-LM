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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_NORMALIZER_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_NORMALIZER_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/pattern.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {

// The possible offsets referential
//
// `#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub enum OffsetReferential`
enum class OffsetReferential { kOriginal, kNormalized };

// Represents a Range usable by the NormalizedString to index its content.
// A Range can use indices relative to either the `Original` or the
// `Normalized` string
//
// `pub enum Range<T: RangeBounds<usize> + Clone> { Original(T), Normalized(T)
// }`
//
// PORT NOTE: Rust's generic `RangeBounds<usize>` (`a..b`, `a..`, `..b`, `..`,
// `a..=b`) is represented by two optional half-open bounds; `std::nullopt`
// means "unbounded" on that side. Inclusive ranges are expressed as
// `Original(a, b + 1)`.
struct Range {
  OffsetReferential referential = OffsetReferential::kOriginal;
  std::optional<size_t> start;  // `Bound::Included(start)` or `Unbounded`
  std::optional<size_t> end;    // `Bound::Excluded(end)` or `Unbounded`

  // `Range::Original(s..e)`
  static Range Original(size_t s, size_t e) {
    return {OffsetReferential::kOriginal, s, e};
  }
  // `Range::Normalized(s..e)`
  static Range Normalized(size_t s, size_t e) {
    return {OffsetReferential::kNormalized, s, e};
  }
  // `Range::Original(s..)`
  static Range OriginalFrom(size_t s) {
    return {OffsetReferential::kOriginal, s, std::nullopt};
  }
  // `Range::Normalized(s..)`
  static Range NormalizedFrom(size_t s) {
    return {OffsetReferential::kNormalized, s, std::nullopt};
  }
  // `Range::Original(..e)`
  static Range OriginalTo(size_t e) {
    return {OffsetReferential::kOriginal, std::nullopt, e};
  }
  // `Range::Normalized(..e)`
  static Range NormalizedTo(size_t e) {
    return {OffsetReferential::kNormalized, std::nullopt, e};
  }
  // `Range::Original(..)`
  static Range OriginalAll() {
    return {OffsetReferential::kOriginal, std::nullopt, std::nullopt};
  }
  // `Range::Normalized(..)`
  static Range NormalizedAll() {
    return {OffsetReferential::kNormalized, std::nullopt, std::nullopt};
  }

  bool IsOriginal() const {
    return referential == OffsetReferential::kOriginal;
  }
  bool IsNormalized() const {
    return referential == OffsetReferential::kNormalized;
  }

  // Return the length of the current Range if not Unbounded
  std::optional<size_t> Len() const;

  // Converts the current Range to a `std::ops::Range<usize>`. This requires the
  // `max_len` of the represented string (in chars, not bytes) in order to
  // cover the case where the original provided range was unbounded
  std::pair<size_t, size_t> IntoFullRange(size_t max_len) const;

  friend bool operator==(const Range& a, const Range& b) {
    return a.referential == b.referential && a.start == b.start &&
           a.end == b.end;
  }
  friend bool operator!=(const Range& a, const Range& b) { return !(a == b); }
};

// Defines the expected behavior for the delimiter of a Split Pattern
// When splitting on `'-'` for example, with input `the-final--countdown`:
//  - Removed => `[ "the", "final", "countdown" ]`
//  - Isolated => `[ "the", "-", "final", "-", "-", "countdown" ]`
//  - MergedWithPrevious => `[ "the-", "final-", "-", "countdown" ]`
//  - MergedWithNext => `[ "the", "-final", "-", "-countdown" ]`
//  - Contiguous => `[ "the", "-", "final", "--", "countdown" ]`
//
// `#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Eq)]`
enum class SplitDelimiterBehavior {
  kRemoved,
  kIsolated,
  kMergedWithPrevious,
  kMergedWithNext,
  kContiguous,
};

// `impl std::fmt::Display for SplitDelimiterBehavior` (the serde variant
// name, e.g. "MergedWithPrevious").
absl::string_view SplitDelimiterBehaviorToString(
    SplitDelimiterBehavior behavior);
// `#[derive(Serialize, Deserialize)]`: the variant name as a JSON string.
absl::StatusOr<SplitDelimiterBehavior> SplitDelimiterBehaviorFromJson(
    const Json& json);
Json SplitDelimiterBehaviorToJson(SplitDelimiterBehavior behavior);

// A `NormalizedString` takes care of processing an "original" string to modify
// it and obtain a "normalized" string. It keeps both version of the string,
// alignments information between both and provides an interface to retrieve
// ranges of each string, using offsets from any of them.
//
// It is possible to retrieve a part of the original string, by indexing it with
// offsets from the normalized one, and the other way around too. It is also
// possible to convert offsets from one referential to the other one easily.
//
// `#[derive(Default, Debug, Clone, PartialEq, Eq)]`
class NormalizedString {
 public:
  // `NormalizedString::default()`
  NormalizedString() = default;

  // `impl From<&str> for NormalizedString` / `impl From<String>`
  static NormalizedString From(absl::string_view s);

  // `NormalizedString::new(original, normalized, alignments, original_shift)`
  // (Rust: `#[cfg(test)] pub(crate)`; public here because other packages'
  // tests use it.)
  static NormalizedString New(std::string original, std::string normalized,
                              std::vector<std::pair<size_t, size_t>> alignments,
                              size_t original_shift);
  NormalizedString(std::string original, std::string normalized,
                   std::vector<std::pair<size_t, size_t>> alignments,
                   size_t original_shift)
      : original_(std::move(original)),
        normalized_(std::move(normalized)),
        alignments_(std::move(alignments)),
        original_shift_(original_shift) {}

  // Return the normalized string
  absl::string_view Get() const { return normalized_; }

  // Return the original string
  absl::string_view GetOriginal() const { return original_; }

  // Return the original offsets
  Offsets OffsetsOriginal() const {
    return {original_shift_, original_shift_ + LenOriginal()};
  }

  // Convert the given offsets range from one referential to the other one:
  // `Original => Normalized` or `Normalized => Original`
  //
  // Returns `None` when targeting something that is outside range
  std::optional<std::pair<size_t, size_t>> ConvertOffsets(
      const Range& range) const;

  // Return a range of the normalized string
  std::optional<absl::string_view> GetRange(const Range& range) const;

  // Return a range of the original string
  std::optional<absl::string_view> GetRangeOriginal(const Range& range) const;

  // Return a slice of the current NormalizedString
  // If the range is not on char boundaries, return None
  std::optional<NormalizedString> Slice(const Range& range) const;

  // Applies transformations to the current normalized version of the string,
  // while updating the alignments.
  // This method expect an Iterator yielding each char of the new normalized
  // string with a `change` isize equals to:
  //   - `1` if this is a new char
  //   - `-N` if the char is right before N removed chars
  //   - `0` if the char is replacing the existing one
  //
  // Since it is possible that the normalized string doesn't include some of
  // the characters at the beginning of the original one, we need an
  // `initial_offset` which represents the number of removed chars at the very
  // beginning.
  //
  // PORT NOTE: the Rust version panics when the range is not on char
  // boundaries (or out of range); this version leaves the string unchanged.
  void TransformRange(const Range& range,
                      const std::vector<std::pair<char32_t, int64_t>>& dest,
                      size_t initial_offset);

  // Applies transformations to the current normalized version of the string,
  // while updating the alignments.
  // This method expect an Iterator yielding each char of the new normalized
  // string with a `change` isize equals to:
  //   - `1` if this is a new char
  //   - `-N` if the char is right before N removed chars
  //   - `0` if the char is replacing the existing one
  //
  // Since it is possible that the normalized string doesn't include some of
  // the characters at the beginning of the original one, we need an
  // `initial_offset` which represents the number of removed chars at the very
  // beginning.
  void Transform(const std::vector<std::pair<char32_t, int64_t>>& dest,
                 size_t initial_offset);

  // Applies NFD normalization
  NormalizedString& Nfd();

  // Applies NFKD normalization
  NormalizedString& Nfkd();

  // Applies NFC normalization
  NormalizedString& Nfc();

  // Applies NFKC normalization
  NormalizedString& Nfkc();

  // Applies filtering over our characters
  NormalizedString& Filter(absl::FunctionRef<bool(char32_t)> keep);

  // Prepend the given string to ourself
  NormalizedString& Prepend(absl::string_view s);

  // Append the given string to ourself
  NormalizedString& Append(absl::string_view s);

  // Map our characters
  NormalizedString& Map(absl::FunctionRef<char32_t(char32_t)> map);

  // Calls the given function for each characters
  const NormalizedString& ForEach(
      absl::FunctionRef<void(char32_t)> foreach) const;

  // Lowercase
  NormalizedString& Lowercase();

  // Uppercase
  NormalizedString& Uppercase();

  // Replace anything that matches the pattern with the given content.
  absl::Status Replace(const Pattern& pattern, absl::string_view content);

  // Clear the normalized part of the string
  size_t Clear();

  // Split the current string in many subparts. Specify what to do with the
  // delimiter.
  //
  // ## Splitting Behavior for the delimiter
  //
  // The behavior can be one of the followings:
  // When splitting on `'-'` for example, with input `the-final--countdown`:
  //  - Removed => `[ "the", "", "final", "", "", "countdown" ]`
  //  - Isolated => `[ "the", "-", "final", "-", "-", "countdown" ]`
  //  - MergedWithPrevious => `[ "the-", "final-", "-", "countdown" ]`
  //  - MergedWithNext => `[ "the", "-final", "-", "-countdown" ]`
  absl::StatusOr<std::vector<NormalizedString>> Split(
      const Pattern& pattern, SplitDelimiterBehavior behavior) const;

  // Remove any leading space(s) of the normalized string
  NormalizedString& Lstrip();

  // Remove any trailing space(s) of the normalized string
  NormalizedString& Rstrip();

  // Remove any leading and trailing space(s) of the normalized string
  NormalizedString& Strip();

  // Returns the length of the normalized string (counting chars not bytes)
  // PORT NOTE: despite the Rust doc comment this is the length in *bytes*,
  // exactly like `String::len()`.
  size_t Len() const { return normalized_.size(); }

  // Returns the length of the original string (counting chars not bytes)
  // PORT NOTE: in bytes, see `Len()`.
  size_t LenOriginal() const { return original_.size(); }

  // Whether empty
  bool IsEmpty() const { return normalized_.empty(); }

  // Recalculate original alignments
  //
  // (Rust: `pub(crate)`; public here because other packages' tests use it.)
  // PORT NOTE: the Rust version panics on overlapping ranges; this version does
  // not check for them.
  std::vector<std::pair<size_t, size_t>> AlignmentsOriginal() const;

  // Mapping from normalized string to original one: (start, end) for each
  // byte of the normalized string (the `alignments` field).
  const std::vector<std::pair<size_t, size_t>>& Alignments() const {
    return alignments_;
  }

  // If this NormalizedString is a slice of a bigger one, we keep the track
  // of the missing part, so that we can still give offsets from this original
  // string (the `original_shift` field).
  size_t OriginalShift() const { return original_shift_; }

  friend bool operator==(const NormalizedString& a, const NormalizedString& b) {
    return a.original_ == b.original_ && a.normalized_ == b.normalized_ &&
           a.alignments_ == b.alignments_ &&
           a.original_shift_ == b.original_shift_;
  }
  friend bool operator!=(const NormalizedString& a, const NormalizedString& b) {
    return !(a == b);
  }

 private:
  // Validate the given range, to make sure it is on char boundaries
  std::optional<Range> ValidateRange(const Range& range) const;

  NormalizedString& Lrstrip(bool left, bool right);

  // The original version of the string, before any modification
  std::string original_;
  // The normalized version of the string, after all modifications
  std::string normalized_;
  // Mapping from normalized string to original one: (start, end) for each
  // byte of the normalized string
  std::vector<std::pair<size_t, size_t>> alignments_;
  // If this NormalizedString is a slice of a bigger one, we keep the track
  // of the missing part, so that we can still give offsets from this original
  // string.
  size_t original_shift_ = 0;
};

// Returns a range of the given string slice, by indexing chars instead of
// bytes
//
// `pub fn get_range_of<T: RangeBounds<usize>>(s: &str, range: T) ->
// Option<&str>`; `end == std::nullopt` means `start..`.
std::optional<absl::string_view> GetRangeOf(absl::string_view s, size_t start,
                                            std::optional<size_t> end);

// Convert the given range from bytes to char
std::optional<std::pair<size_t, size_t>> BytesToChar(
    absl::string_view s, std::pair<size_t, size_t> range);

// Convert the given range from char to bytes
std::optional<std::pair<size_t, size_t>> CharToBytes(
    absl::string_view s, std::pair<size_t, size_t> range);

}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_NORMALIZER_H_
