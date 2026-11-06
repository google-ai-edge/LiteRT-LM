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
// C++ port of `src/grapheme.rs` of the `unicode-segmentation` crate v1.13.3
// (https://github.com/unicode-rs/unicode-segmentation, MIT OR Apache-2.0,
// Copyright 2012-2015 The Rust Project Developers): extended (and legacy)
// grapheme cluster segmentation as used by the HuggingFace `tokenizers` crate
// v0.23.2 (`UnicodeSegmentation::graphemes(true)`). Kept line-by-line close to
// the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.
//
// Unicode version of the tables: 17.0.0 (see graphemes_tables.h).

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UNICODE_GRAPHEMES_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UNICODE_GRAPHEMES_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/unicode/graphemes_tables.h"

namespace tokenizers::unicode {

// ---- Convenience entry points ---------------------------------------------

// `UnicodeSegmentation::graphemes(s, is_extended)` drained into a vector: the
// grapheme clusters of the (valid UTF-8) string `s`, as sub-views of `s`.
// If `is_extended` is true, the iterator is over the
// [extended grapheme
// clusters](http://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries);
// otherwise, the iterator is over the
// [legacy grapheme
// clusters](http://www.unicode.org/reports/tr29/#Table_Combining_Char_Sequences_and_Grapheme_Clusters).
// [UAX#29](http://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
// recommends extended grapheme cluster boundaries for general processing.
std::vector<absl::string_view> Graphemes(absl::string_view s, bool is_extended);

// `UnicodeSegmentation::grapheme_indices(s, is_extended)` drained into a
// vector: the grapheme clusters of `s` and their byte offsets.
std::vector<std::pair<size_t, absl::string_view>> GraphemeIndices(
    absl::string_view s, bool is_extended);

// ---- `GraphemeCursor` -------------------------------------------------------

// An error return indicating that not enough content was available in the
// provided chunk to satisfy the query, and that more content must be provided.
struct GraphemeIncomplete {
  enum class Kind {
    // More pre-context is needed. The caller should call `ProvideContext`
    // with a chunk ending at the offset given, then retry the query. This
    // will only be returned if the `chunk_start` parameter is nonzero.
    kPreContext,

    // When requesting `PrevBoundary`, the cursor is moving past the beginning
    // of the current chunk, so the chunk before that is requested. This will
    // only be returned if the `chunk_start` parameter is nonzero.
    kPrevChunk,

    // When requesting `NextBoundary`, the cursor is moving past the end of the
    // current chunk, so the chunk after that is requested. This will only be
    // returned if the chunk ends before the `len` parameter provided on
    // creation of the cursor.
    kNextChunk,  // requesting chunk following the one given

    // An error returned when the chunk given does not contain the cursor
    // position.
    kInvalidOffset,
  };

  static GraphemeIncomplete PreContext(size_t offset) {
    return GraphemeIncomplete{Kind::kPreContext, offset};
  }
  static GraphemeIncomplete PrevChunk() {
    return GraphemeIncomplete{Kind::kPrevChunk, 0};
  }
  static GraphemeIncomplete NextChunk() {
    return GraphemeIncomplete{Kind::kNextChunk, 0};
  }
  static GraphemeIncomplete InvalidOffset() {
    return GraphemeIncomplete{Kind::kInvalidOffset, 0};
  }

  friend bool operator==(const GraphemeIncomplete& a,
                         const GraphemeIncomplete& b) {
    return a.kind == b.kind && a.pre_context_offset == b.pre_context_offset;
  }
  friend bool operator!=(const GraphemeIncomplete& a,
                         const GraphemeIncomplete& b) {
    return !(a == b);
  }
  friend std::ostream& operator<<(std::ostream& os,
                                  const GraphemeIncomplete& e);

  Kind kind;
  // The payload of `GraphemeIncomplete::PreContext(usize)`; 0 otherwise.
  size_t pre_context_offset;
};

// `Result<T, GraphemeIncomplete>`.
// PORT NOTE: `GraphemeIncomplete` is a control-flow signal of the chunked API
// rather than an error, so it is kept as a value type instead of being mapped
// to `absl::Status`.
template <typename T>
class GraphemeResult {
 public:
  GraphemeResult(T value) : value_(std::move(value)) {}  // NOLINT: Ok(value)
  GraphemeResult(GraphemeIncomplete error)               // NOLINT: Err(error)
      : value_(std::move(error)) {}

  bool ok() const { return std::holds_alternative<T>(value_); }
  // Precondition: `ok()`.
  const T& value() const { return std::get<T>(value_); }
  // Precondition: `!ok()`.
  const GraphemeIncomplete& error() const {
    return std::get<GraphemeIncomplete>(value_);
  }

  friend bool operator==(const GraphemeResult& a, const GraphemeResult& b) {
    return a.value_ == b.value_;
  }
  friend bool operator!=(const GraphemeResult& a, const GraphemeResult& b) {
    return !(a == b);
  }
  friend std::ostream& operator<<(std::ostream& os, const GraphemeResult& r) {
    if (r.ok()) {
      return os << "Ok(" << PrintValue(r.value()) << ")";
    }
    return os << "Err(" << r.error() << ")";
  }

 private:
  static std::string PrintValue(bool v) { return v ? "true" : "false"; }
  static std::string PrintValue(const std::optional<size_t>& v) {
    return v.has_value() ? "Some(" + std::to_string(*v) + ")" : "None";
  }

  std::variant<T, GraphemeIncomplete> value_;
};

// maybe unify with PairResult?
// An enum describing information about a potential boundary.
enum class GraphemeState {
  // No information is known.
  kUnknown,
  // It is known to not be a boundary.
  kNotBreak,
  // It is known to be a boundary.
  kBreak,
  // The codepoint after it has Indic_Conjunct_Break=Consonant,
  // so there is a break before so a boundary if it is preceded by another
  // InCB=Consonant follwoed by a sequence consisting of one or more InCB=Linker
  // and zero or more InCB = Extend (in any order).
  kInCbConsonant,
  // The codepoint after is a Regional Indicator Symbol, so a boundary iff
  // it is preceded by an even number of RIS codepoints. (GB12, GB13)
  kRegional,
  // The codepoint after is Extended_Pictographic,
  // so whether it's a boundary depends on pre-context according to GB11.
  kEmoji,
};

// Cursor-based segmenter for grapheme clusters.
//
// This allows working with ropes and other datastructures where the string is
// not contiguous or fully known at initialization time.
class GraphemeCursor {
 public:
  // Create a new cursor. The string and initial offset are given at creation
  // time, but the contents of the string are not. The `is_extended` parameter
  // controls whether extended grapheme clusters are selected.
  //
  // The `offset` parameter must be on a codepoint boundary.
  GraphemeCursor(size_t offset, size_t len, bool is_extended);

  // Set the cursor to a new location in the same string.
  void SetCursor(size_t offset);

  // The current offset of the cursor. Equal to the last value provided to
  // the constructor or `SetCursor()`, or returned from `NextBoundary()` or
  // `PrevBoundary()`.
  size_t CurCursor() const { return offset_; }

  // Provide additional pre-context when it is needed to decide a boundary.
  // The end of the chunk must coincide with the value given in the
  // `GraphemeIncomplete::PreContext` request.
  void ProvideContext(absl::string_view chunk, size_t chunk_start);

  // Determine whether the current cursor location is a grapheme cluster
  // boundary. Only a part of the string need be supplied. If `chunk_start` is
  // nonzero or the length of `chunk` is not equal to `len` on creation, then
  // this method may return `GraphemeIncomplete::PreContext`. The caller should
  // then call `ProvideContext` with the requested chunk, then retry calling
  // this method.
  //
  // For partial chunks, if the cursor is not at the beginning or end of the
  // string, the chunk should contain at least the codepoint following the
  // cursor. If the string is nonempty, the chunk must be nonempty.
  //
  // All calls should have consistent chunk contents (ie, if a chunk provides
  // content for a given slice, all further chunks covering that slice must
  // have the same content for it).
  GraphemeResult<bool> IsBoundary(absl::string_view chunk, size_t chunk_start);

  // Find the next boundary after the current cursor position. Only a part of
  // the string need be supplied. If the chunk is incomplete, then this
  // method might return `GraphemeIncomplete::PreContext` or
  // `GraphemeIncomplete::NextChunk`. In the former case, the caller should
  // call `ProvideContext` with the requested chunk, then retry. In the
  // latter case, the caller should provide the chunk following the one
  // given, then retry.
  //
  // See `IsBoundary` for expectations on the provided chunk.
  GraphemeResult<std::optional<size_t>> NextBoundary(absl::string_view chunk,
                                                     size_t chunk_start);

  // Find the previous boundary after the current cursor position. Only a part
  // of the string need be supplied. If the chunk is incomplete, then this
  // method might return `GraphemeIncomplete::PreContext` or
  // `GraphemeIncomplete::PrevChunk`. In the former case, the caller should
  // call `ProvideContext` with the requested chunk, then retry. In the
  // latter case, the caller should provide the chunk preceding the one
  // given, then retry.
  //
  // See `IsBoundary` for expectations on the provided chunk.
  GraphemeResult<std::optional<size_t>> PrevBoundary(absl::string_view chunk,
                                                     size_t chunk_start);

 private:
  GraphemeCat GraphemeCategory(char32_t ch);
  void Decide(bool is_break);
  GraphemeResult<bool> Decision(bool is_break);
  GraphemeResult<bool> IsBoundaryResult() const;
  void HandleIncbConsonant(absl::string_view chunk, size_t chunk_start);
  void HandleRegional(absl::string_view chunk, size_t chunk_start);
  void HandleEmoji(absl::string_view chunk, size_t chunk_start, bool seen_zwj);

  // Current cursor position.
  size_t offset_;
  // Total length of the string.
  size_t len_;
  // A config flag indicating whether this cursor computes legacy or extended
  // grapheme cluster boundaries (enables GB9a and GB9b if set).
  bool is_extended_;
  // Information about the potential boundary at `offset`
  GraphemeState state_;
  // Whether the ZWJ char has been seen already an only a
  // "\p{Extended_Pictographic} Extend*" part of GB11 has to be checked
  // (the payload of `GraphemeState::Emoji { seen_zwj }`).
  bool emoji_seen_zwj_ = false;
  // Category of codepoint immediately preceding cursor, if known.
  std::optional<GraphemeCat> cat_before_;
  // Category of codepoint immediately after cursor, if known.
  std::optional<GraphemeCat> cat_after_;
  // If set, at least one more codepoint immediately preceding this offset
  // is needed to resolve whether there's a boundary at `offset`.
  std::optional<size_t> pre_context_offset_;
  // The number of `InCB=Linker` codepoints preceding `offset`
  // (potentially intermingled with `InCB=Extend`).
  std::optional<size_t> incb_linker_count_;
  // The number of RIS codepoints preceding `offset`. If `pre_context_offset`
  // is set, then counts the number of RIS between that and `offset`, otherwise
  // is an accurate count relative to the string.
  std::optional<size_t> ris_count_;
  // Set if a call to `PrevBoundary` or `NextBoundary` was suspended due
  // to needing more input.
  bool resuming_;
  // Cached grapheme category and associated scalar value range.
  GraphemeCatRange grapheme_cat_cache_;
};

// ---- `Graphemes` iterator ---------------------------------------------------

// External iterator for a string's
// [grapheme
// clusters](http://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries).
//
// This struct is created by `GraphemesIterator::New` (the `graphemes` method on
// the `UnicodeSegmentation` trait).
// PORT NOTE: named `GraphemesIterator` because `Graphemes()` is the drained
// convenience function above.
class GraphemesIterator {
 public:
  // `new_graphemes(s, is_extended)`.
  GraphemesIterator(absl::string_view s, bool is_extended);

  // View the underlying data (the part yet to be iterated) as a slice of the
  // original string.
  absl::string_view AsStr() const;

  // `Iterator::size_hint`.
  std::pair<size_t, std::optional<size_t>> SizeHint() const;

  // `Iterator::next`.
  std::optional<absl::string_view> Next();

  // `DoubleEndedIterator::next_back`.
  std::optional<absl::string_view> NextBack();

 private:
  absl::string_view string_;
  GraphemeCursor cursor_;
  GraphemeCursor cursor_back_;
};

// External iterator for grapheme clusters and byte offsets.
//
// This struct is created by `GraphemeIndicesIterator::New` (the
// `grapheme_indices` method on the `UnicodeSegmentation` trait).
class GraphemeIndicesIterator {
 public:
  // `new_grapheme_indices(s, is_extended)`.
  GraphemeIndicesIterator(absl::string_view s, bool is_extended);

  // View the underlying data (the part yet to be iterated) as a slice of the
  // original string.
  absl::string_view AsStr() const { return iter_.AsStr(); }

  // `Iterator::size_hint`.
  std::pair<size_t, std::optional<size_t>> SizeHint() const {
    return iter_.SizeHint();
  }

  // `Iterator::next`.
  std::optional<std::pair<size_t, absl::string_view>> Next();

  // `DoubleEndedIterator::next_back`.
  std::optional<std::pair<size_t, absl::string_view>> NextBack();

 private:
  const char* start_offset_;
  GraphemesIterator iter_;
};

}  // namespace tokenizers::unicode

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UNICODE_GRAPHEMES_H_
