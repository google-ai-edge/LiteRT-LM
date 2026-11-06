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
// Copyright 2012-2015 The Rust Project Developers). Kept line-by-line close to
// the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/unicode/graphemes.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <ostream>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/unicode/graphemes_tables.h"
#include "support/tokenizer/tokenizers/unicode/utf8_chars.h"

namespace tokenizers::unicode {

using internal::DecodeUtf8;
using internal::DecodeUtf8Backward;
using internal::Utf8Len;

std::ostream& operator<<(std::ostream& os, const GraphemeIncomplete& e) {
  switch (e.kind) {
    case GraphemeIncomplete::Kind::kPreContext:
      return os << "PreContext(" << e.pre_context_offset << ")";
    case GraphemeIncomplete::Kind::kPrevChunk:
      return os << "PrevChunk";
    case GraphemeIncomplete::Kind::kNextChunk:
      return os << "NextChunk";
    case GraphemeIncomplete::Kind::kInvalidOffset:
      return os << "InvalidOffset";
  }
  return os;
}

// ---- Graphemes / GraphemeIndices -------------------------------------------

GraphemesIterator::GraphemesIterator(absl::string_view s, bool is_extended)
    : string_(s),
      cursor_(0, s.size(), is_extended),
      cursor_back_(s.size(), s.size(), is_extended) {}

absl::string_view GraphemesIterator::AsStr() const {
  return string_.substr(cursor_.CurCursor(),
                        cursor_back_.CurCursor() - cursor_.CurCursor());
}

std::pair<size_t, std::optional<size_t>> GraphemesIterator::SizeHint() const {
  const size_t slen = cursor_back_.CurCursor() - cursor_.CurCursor();
  return std::make_pair(std::min<size_t>(slen, 1), slen);
}

std::optional<absl::string_view> GraphemesIterator::Next() {
  const size_t start = cursor_.CurCursor();
  if (start == cursor_back_.CurCursor()) {
    return std::nullopt;
  }
  // `self.cursor.next_boundary(self.string, 0).unwrap().unwrap()`: the whole
  // string is given as the chunk, so the result is never incomplete.
  const GraphemeResult<std::optional<size_t>> result =
      cursor_.NextBoundary(string_, 0);
  if (!result.ok() || !result.value().has_value()) {
    return std::nullopt;
  }
  const size_t next = *result.value();
  return string_.substr(start, next - start);
}

std::optional<absl::string_view> GraphemesIterator::NextBack() {
  const size_t end = cursor_back_.CurCursor();
  if (end == cursor_.CurCursor()) {
    return std::nullopt;
  }
  // `self.cursor_back.prev_boundary(self.string, 0).unwrap().unwrap()`.
  const GraphemeResult<std::optional<size_t>> result =
      cursor_back_.PrevBoundary(string_, 0);
  if (!result.ok() || !result.value().has_value()) {
    return std::nullopt;
  }
  const size_t prev = *result.value();
  return string_.substr(prev, end - prev);
}

GraphemeIndicesIterator::GraphemeIndicesIterator(absl::string_view s,
                                                 bool is_extended)
    : start_offset_(s.data()), iter_(s, is_extended) {}

std::optional<std::pair<size_t, absl::string_view>>
GraphemeIndicesIterator::Next() {
  std::optional<absl::string_view> s = iter_.Next();
  if (!s.has_value()) return std::nullopt;
  return std::make_pair(static_cast<size_t>(s->data() - start_offset_), *s);
}

std::optional<std::pair<size_t, absl::string_view>>
GraphemeIndicesIterator::NextBack() {
  std::optional<absl::string_view> s = iter_.NextBack();
  if (!s.has_value()) return std::nullopt;
  return std::make_pair(static_cast<size_t>(s->data() - start_offset_), *s);
}

std::vector<absl::string_view> Graphemes(absl::string_view s,
                                         bool is_extended) {
  std::vector<absl::string_view> out;
  GraphemesIterator iter(s, is_extended);
  while (std::optional<absl::string_view> g = iter.Next()) {
    out.push_back(*g);
  }
  return out;
}

std::vector<std::pair<size_t, absl::string_view>> GraphemeIndices(
    absl::string_view s, bool is_extended) {
  std::vector<std::pair<size_t, absl::string_view>> out;
  GraphemeIndicesIterator iter(s, is_extended);
  while (std::optional<std::pair<size_t, absl::string_view>> g = iter.Next()) {
    out.push_back(*g);
  }
  return out;
}

// ---- GraphemeCursor ---------------------------------------------------------

namespace {

// An enum describing the result from lookup of a pair of categories.
enum class PairResult {
  // definitely not a break
  kNotBreak,
  // definitely a break
  kBreak,
  // a break iff not in extended mode
  kExtended,
  // a break unless in extended mode and preceded by
  // a sequence of 0 or more InCB=Extend and one or more
  // InCB = Linker (in any order),
  // preceded by another InCB=Consonant
  kInCbConsonant,
  // a break if preceded by an even number of RIS
  kRegional,
  // a break if preceded by emoji base and (Extend)*
  kEmoji,
};

inline PairResult CheckPair(GraphemeCat before, GraphemeCat after) {
  using GC = GraphemeCat;
  if (before == GC::kCR && after == GC::kLF) {
    return PairResult::kNotBreak;  // GB3
  }
  if (before == GC::kControl || before == GC::kCR || before == GC::kLF) {
    return PairResult::kBreak;  // GB4
  }
  if (after == GC::kControl || after == GC::kCR || after == GC::kLF) {
    return PairResult::kBreak;  // GB5
  }
  if (before == GC::kL && (after == GC::kL || after == GC::kV ||
                           after == GC::kLV || after == GC::kLVT)) {
    return PairResult::kNotBreak;  // GB6
  }
  if ((before == GC::kLV || before == GC::kV) &&
      (after == GC::kV || after == GC::kT)) {
    return PairResult::kNotBreak;  // GB7
  }
  if ((before == GC::kLVT || before == GC::kT) && after == GC::kT) {
    return PairResult::kNotBreak;  // GB8
  }
  if (after == GC::kExtend || after == GC::kZWJ) {
    return PairResult::kNotBreak;  // GB9
  }
  if (after == GC::kSpacingMark) {
    return PairResult::kExtended;  // GB9a
  }
  if (before == GC::kPrepend) {
    return PairResult::kExtended;  // GB9b
  }
  if (after == GC::kInCBConsonant) {
    return PairResult::kInCbConsonant;  // GB9c
  }
  if (before == GC::kZWJ && after == GC::kExtendedPictographic) {
    return PairResult::kEmoji;  // GB11
  }
  if (before == GC::kRegionalIndicator && after == GC::kRegionalIndicator) {
    return PairResult::kRegional;  // GB12, GB13
  }
  return PairResult::kBreak;  // GB999
}

// `chunk.chars().next().unwrap()`.
// PORT NOTE: `unwrap()` panics on an empty chunk in Rust (a caller error);
// the port yields U+0000 (a Control) instead of reading out of bounds.
inline char32_t FirstChar(absl::string_view chunk) {
  if (chunk.empty()) return U'\0';
  size_t pos = 0;
  return DecodeUtf8(chunk, &pos);
}

// `chunk.chars().next_back().unwrap()` (same PORT NOTE as `FirstChar`).
inline char32_t LastChar(absl::string_view chunk) {
  if (chunk.empty()) return U'\0';
  size_t end = chunk.size();
  return DecodeUtf8Backward(chunk, &end);
}

// `chunk.chars().rev()`.
class RevChars {
 public:
  explicit RevChars(absl::string_view chunk)
      : chunk_(chunk), end_(chunk.size()) {}

  std::optional<char32_t> Next() {
    if (end_ == 0) return std::nullopt;
    return DecodeUtf8Backward(chunk_, &end_);
  }

 private:
  absl::string_view chunk_;
  size_t end_;
};

// `usize::saturating_add`.
inline size_t SaturatingAdd(size_t a, size_t b) {
  const size_t sum = a + b;
  return sum < a ? static_cast<size_t>(-1) : sum;
}

// `usize::saturating_sub`.
inline size_t SaturatingSub(size_t a, size_t b) { return a > b ? a - b : 0; }

}  // namespace

GraphemeCursor::GraphemeCursor(size_t offset, size_t len, bool is_extended)
    : offset_(offset),
      len_(len),
      is_extended_(is_extended),
      state_(offset == 0 || offset == len ? GraphemeState::kBreak
                                          : GraphemeState::kUnknown),
      cat_before_(std::nullopt),
      cat_after_(std::nullopt),
      pre_context_offset_(std::nullopt),
      incb_linker_count_(std::nullopt),
      ris_count_(std::nullopt),
      resuming_(false),
      grapheme_cat_cache_{0, 0, GraphemeCat::kControl} {}

GraphemeCat GraphemeCursor::GraphemeCategory(char32_t ch) {
  if (ch <= 0x7e) {
    // Special-case optimization for ascii, except U+007F.  This
    // improves performance even for many primarily non-ascii texts,
    // due to use of punctuation and white space characters from the
    // ascii range.
    if (ch >= 0x20) {
      return GraphemeCat::kAny;
    } else if (ch == U'\n') {
      return GraphemeCat::kLF;
    } else if (ch == U'\r') {
      return GraphemeCat::kCR;
    } else {
      return GraphemeCat::kControl;
    }
  } else {
    // If this char isn't within the cached range, update the cache to the
    // range that includes it.
    if (static_cast<uint32_t>(ch) < grapheme_cat_cache_.lower ||
        static_cast<uint32_t>(ch) > grapheme_cat_cache_.upper) {
      grapheme_cat_cache_ = ::tokenizers::unicode::GraphemeCategory(ch);
    }
    return grapheme_cat_cache_.cat;
  }
}

// Not sure I'm gonna keep this, the advantage over new() seems thin.

void GraphemeCursor::SetCursor(size_t offset) {
  if (offset != offset_) {
    offset_ = offset;
    state_ = (offset == 0 || offset == len_) ? GraphemeState::kBreak
                                             : GraphemeState::kUnknown;
    // reset state derived from text around cursor
    cat_before_ = std::nullopt;
    cat_after_ = std::nullopt;
    incb_linker_count_ = std::nullopt;
    ris_count_ = std::nullopt;
  }
}

void GraphemeCursor::ProvideContext(absl::string_view chunk,
                                    size_t chunk_start) {
  // `assert!(chunk_start.saturating_add(chunk.len()) ==
  //          self.pre_context_offset.unwrap());`
  // PORT NOTE: a violated precondition panics in Rust; the port ignores the
  // call instead (library code must not abort).
  if (!pre_context_offset_.has_value() ||
      SaturatingAdd(chunk_start, chunk.size()) != *pre_context_offset_) {
    return;
  }
  pre_context_offset_ = std::nullopt;
  if (is_extended_ && chunk_start + chunk.size() == offset_) {
    const char32_t ch = LastChar(chunk);
    if (GraphemeCategory(ch) == GraphemeCat::kPrepend) {
      Decide(false);  // GB9b
      return;
    }
  }
  switch (state_) {
    case GraphemeState::kInCbConsonant:
      HandleIncbConsonant(chunk, chunk_start);
      break;
    case GraphemeState::kRegional:
      HandleRegional(chunk, chunk_start);
      break;
    case GraphemeState::kEmoji:
      HandleEmoji(chunk, chunk_start, emoji_seen_zwj_);
      break;
    default:
      if (!cat_before_.has_value() && offset_ == chunk.size() + chunk_start) {
        const char32_t ch = LastChar(chunk);
        cat_before_ = GraphemeCategory(ch);
      }
      break;
  }
}

void GraphemeCursor::Decide(bool is_break) {
  state_ = is_break ? GraphemeState::kBreak : GraphemeState::kNotBreak;
}

GraphemeResult<bool> GraphemeCursor::Decision(bool is_break) {
  Decide(is_break);
  return is_break;
}

GraphemeResult<bool> GraphemeCursor::IsBoundaryResult() const {
  if (state_ == GraphemeState::kBreak) {
    return true;
  } else if (state_ == GraphemeState::kNotBreak) {
    return false;
  } else if (pre_context_offset_.has_value()) {
    return GraphemeIncomplete::PreContext(*pre_context_offset_);
  } else {
    // `unreachable!("inconsistent state")`
    // PORT NOTE: cannot happen; reported as an invalid offset instead of
    // aborting.
    return GraphemeIncomplete::InvalidOffset();
  }
}

// For handling rule GB9c:
//
// There's an `InCB=Consonant` after this, and we need to look back
// to verify whether there should be a break.
//
// Seek backward to find an `InCB=Linker` preceded by an `InCB=Consonsnt`
// (potentially separated by some number of `InCB=Linker` or `InCB=Extend`).
// If we find the consonant in question, then there's no break; if we find a
// consonant with no linker, or a non-linker non-extend non-consonant, or the
// start of text, there's a break; otherwise we need more context
void GraphemeCursor::HandleIncbConsonant(absl::string_view chunk,
                                         size_t chunk_start) {
  // GB9c only applies to extended grapheme clusters
  if (!is_extended_) {
    Decide(true);
    return;
  }

  size_t incb_linker_count = incb_linker_count_.value_or(0);

  RevChars iter(chunk);
  while (const std::optional<char32_t> ch = iter.Next()) {
    if (IsIncbLinker(*ch)) {
      // We found an InCB linker
      incb_linker_count += 1;
      incb_linker_count_ = incb_linker_count;
    } else if (DerivedPropertyInCBExtend(*ch)) {
      // We ignore InCB extends, continue
    } else {
      // Prev character is neither linker nor extend, break suppressed iff it's
      // InCB=Consonant
      const bool result =
          !(incb_linker_count_.value_or(0) > 0 &&
            GraphemeCategory(*ch) == GraphemeCat::kInCBConsonant);
      Decide(result);
      return;
    }
  }

  if (chunk_start == 0) {
    // Start of text and we still haven't found a consonant, so break
    Decide(true);
  } else {
    // We need more context
    pre_context_offset_ = chunk_start;
    state_ = GraphemeState::kInCbConsonant;
  }
}

void GraphemeCursor::HandleRegional(absl::string_view chunk,
                                    size_t chunk_start) {
  size_t ris_count = ris_count_.value_or(0);
  RevChars iter(chunk);
  while (const std::optional<char32_t> ch = iter.Next()) {
    if (GraphemeCategory(*ch) != GraphemeCat::kRegionalIndicator) {
      ris_count_ = ris_count;
      Decide(ris_count % 2 == 0);
      return;
    }
    ris_count += 1;
  }
  ris_count_ = ris_count;
  if (chunk_start == 0) {
    Decide(ris_count % 2 == 0);
  } else {
    pre_context_offset_ = chunk_start;
    state_ = GraphemeState::kRegional;
  }
}

void GraphemeCursor::HandleEmoji(absl::string_view chunk, size_t chunk_start,
                                 bool seen_zwj) {
  // \p{Extended_Pictographic} Extend* ZWJ × \p{Extended_Pictographic}
  RevChars iter(chunk);
  if (!seen_zwj) {
    if (const std::optional<char32_t> ch = iter.Next()) {
      if (GraphemeCategory(*ch) != GraphemeCat::kZWJ) {
        Decide(true);
        return;
      } else {
        seen_zwj = true;
      }
    }
  }
  while (const std::optional<char32_t> ch = iter.Next()) {
    switch (GraphemeCategory(*ch)) {
      case GraphemeCat::kExtend:
        break;
      case GraphemeCat::kExtendedPictographic:
        Decide(false);
        return;
      default:
        Decide(true);
        return;
    }
  }
  if (chunk_start == 0) {
    Decide(true);
  } else {
    pre_context_offset_ = chunk_start;
    state_ = GraphemeState::kEmoji;
    emoji_seen_zwj_ = seen_zwj;
  }
}

GraphemeResult<bool> GraphemeCursor::IsBoundary(absl::string_view chunk,
                                                size_t chunk_start) {
  if (state_ == GraphemeState::kBreak) {
    return true;
  }
  if (state_ == GraphemeState::kNotBreak) {
    return false;
  }
  if ((offset_ < chunk_start ||
       offset_ >= SaturatingAdd(chunk_start, chunk.size())) &&
      (offset_ > SaturatingAdd(chunk_start, chunk.size()) ||
       !cat_after_.has_value())) {
    return GraphemeIncomplete::InvalidOffset();
  }
  if (pre_context_offset_.has_value()) {
    return GraphemeIncomplete::PreContext(*pre_context_offset_);
  }
  const size_t offset_in_chunk = SaturatingSub(offset_, chunk_start);
  if (!cat_after_.has_value()) {
    const char32_t ch = FirstChar(chunk.substr(offset_in_chunk));
    cat_after_ = GraphemeCategory(ch);
  }
  if (offset_ == chunk_start) {
    bool need_pre_context = true;
    switch (*cat_after_) {
      case GraphemeCat::kInCBConsonant:
        state_ = GraphemeState::kInCbConsonant;
        break;
      case GraphemeCat::kRegionalIndicator:
        state_ = GraphemeState::kRegional;
        break;
      case GraphemeCat::kExtendedPictographic:
        state_ = GraphemeState::kEmoji;
        emoji_seen_zwj_ = false;
        break;
      default:
        need_pre_context = !cat_before_.has_value();
        break;
    }
    if (need_pre_context) {
      pre_context_offset_ = chunk_start;
      return GraphemeIncomplete::PreContext(chunk_start);
    }
  }
  if (!cat_before_.has_value()) {
    const char32_t ch = LastChar(chunk.substr(0, offset_in_chunk));
    cat_before_ = GraphemeCategory(ch);
  }
  switch (CheckPair(*cat_before_, *cat_after_)) {
    case PairResult::kNotBreak:
      return Decision(false);
    case PairResult::kBreak:
      return Decision(true);
    case PairResult::kExtended: {
      const bool is_extended = is_extended_;
      return Decision(!is_extended);
    }
    case PairResult::kInCbConsonant: {
      HandleIncbConsonant(chunk.substr(0, offset_in_chunk), chunk_start);
      return IsBoundaryResult();
    }
    case PairResult::kRegional: {
      if (ris_count_.has_value()) {
        return Decision((*ris_count_ % 2) == 0);
      }
      HandleRegional(chunk.substr(0, offset_in_chunk), chunk_start);
      return IsBoundaryResult();
    }
    case PairResult::kEmoji: {
      HandleEmoji(chunk.substr(0, offset_in_chunk), chunk_start, false);
      return IsBoundaryResult();
    }
  }
  return IsBoundaryResult();
}

GraphemeResult<std::optional<size_t>> GraphemeCursor::NextBoundary(
    absl::string_view chunk, size_t chunk_start) {
  if (offset_ == len_) {
    return std::optional<size_t>(std::nullopt);
  }
  // `let mut iter = chunk[self.offset.saturating_sub(chunk_start)..].chars();`
  size_t iter_pos = SaturatingSub(offset_, chunk_start);
  if (iter_pos >= chunk.size()) {
    return GraphemeIncomplete::NextChunk();
  }
  char32_t ch = DecodeUtf8(chunk, &iter_pos);
  while (true) {
    if (resuming_) {
      if (!cat_after_.has_value()) {
        cat_after_ = GraphemeCategory(ch);
      }
    } else {
      offset_ = SaturatingAdd(offset_, Utf8Len(ch));
      state_ = GraphemeState::kUnknown;
      cat_before_ = cat_after_;  // `self.cat_after.take()`
      cat_after_ = std::nullopt;
      if (!cat_before_.has_value()) {
        cat_before_ = GraphemeCategory(ch);
      }
      if (IsIncbLinker(ch)) {
        incb_linker_count_ =
            incb_linker_count_.has_value() ? *incb_linker_count_ + 1 : 1;
      } else if (!DerivedPropertyInCBExtend(ch)) {
        incb_linker_count_ = 0;
      }
      if (*cat_before_ == GraphemeCat::kRegionalIndicator) {
        if (ris_count_.has_value()) ris_count_ = *ris_count_ + 1;
      } else {
        ris_count_ = 0;
      }
      if (iter_pos < chunk.size()) {
        ch = DecodeUtf8(chunk, &iter_pos);
        cat_after_ = GraphemeCategory(ch);
      } else if (offset_ == len_) {
        Decide(true);
      } else {
        resuming_ = true;
        return GraphemeIncomplete::NextChunk();
      }
    }
    resuming_ = true;
    const GraphemeResult<bool> is_boundary = IsBoundary(chunk, chunk_start);
    if (!is_boundary.ok()) {
      return is_boundary.error();  // `?`
    }
    if (is_boundary.value()) {
      resuming_ = false;
      return std::optional<size_t>(offset_);
    }
    resuming_ = false;
  }
}

GraphemeResult<std::optional<size_t>> GraphemeCursor::PrevBoundary(
    absl::string_view chunk, size_t chunk_start) {
  if (offset_ == 0) {
    return std::optional<size_t>(std::nullopt);
  }
  if (offset_ == chunk_start) {
    return GraphemeIncomplete::PrevChunk();
  }
  // `let mut iter = chunk[..self.offset.saturating_sub(chunk_start)]
  //      .chars().rev();`
  RevChars iter(chunk.substr(0, SaturatingSub(offset_, chunk_start)));
  // `let mut ch = iter.next().unwrap();`
  std::optional<char32_t> first = iter.Next();
  if (!first.has_value()) {
    // PORT NOTE: `unwrap()` on an empty chunk panics in Rust (the chunk must
    // contain the character before the cursor); the port reports an invalid
    // offset instead.
    return GraphemeIncomplete::InvalidOffset();
  }
  char32_t ch = *first;
  while (true) {
    if (offset_ == chunk_start) {
      resuming_ = true;
      return GraphemeIncomplete::PrevChunk();
    }
    if (resuming_) {
      cat_before_ = GraphemeCategory(ch);
    } else {
      offset_ -= Utf8Len(ch);
      cat_after_ = cat_before_;  // `self.cat_before.take()`
      cat_before_ = std::nullopt;
      state_ = GraphemeState::kUnknown;
      if (incb_linker_count_.has_value()) {
        const size_t incb_linker_count = *incb_linker_count_;
        if (incb_linker_count > 0 && IsIncbLinker(ch)) {
          incb_linker_count_ = incb_linker_count - 1;
        } else if (DerivedPropertyInCBExtend(ch)) {
          incb_linker_count_ = incb_linker_count;
        } else {
          incb_linker_count_ = std::nullopt;
        }
      }
      if (ris_count_.has_value()) {
        const size_t ris_count = *ris_count_;
        if (ris_count > 0) {
          ris_count_ = ris_count - 1;
        } else {
          ris_count_ = std::nullopt;
        }
      }
      if (const std::optional<char32_t> prev_ch = iter.Next()) {
        ch = *prev_ch;
        cat_before_ = GraphemeCategory(ch);
      } else if (offset_ == 0) {
        Decide(true);
      } else {
        resuming_ = true;
        cat_after_ = GraphemeCategory(ch);
        return GraphemeIncomplete::PrevChunk();
      }
    }
    resuming_ = true;
    const GraphemeResult<bool> is_boundary = IsBoundary(chunk, chunk_start);
    if (!is_boundary.ok()) {
      return is_boundary.error();  // `?`
    }
    if (is_boundary.value()) {
      resuming_ = false;
      return std::optional<size_t>(offset_);
    }
    resuming_ = false;
  }
}

}  // namespace tokenizers::unicode
