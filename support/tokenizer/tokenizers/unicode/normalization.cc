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
// C++ port of the `unicode_normalization_alignments` crate (the files
// `google/{lib,decompose,recompose}.rs` of the HuggingFace `tokenizers` crate
// v0.23.2, https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team) and of the parts of the
// `unicode-normalization` crate v0.1.25 it builds on (`src/{lookups,
// perfect_hash,normalize}.rs`, https://github.com/unicode-rs/unicode-
// normalization, MIT OR Apache-2.0, Copyright 2012-2019 The Rust Project
// Developers). Kept line-by-line close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/unicode/normalization.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/unicode/normalization_tables.h"
#include "support/tokenizer/tokenizers/unicode/utf8_chars.h"

namespace tokenizers::unicode {
namespace {

// ===========================================================================
// unicode-normalization/src/perfect_hash.rs
// Support for lookups based on minimal perfect hashing.
// ===========================================================================

// This function is based on multiplication being fast and is "good enough".
// Also it can share some work between the unsalted and salted versions.
inline size_t MyHash(uint32_t key, uint32_t salt, size_t n) {
  // `wrapping_add` / `wrapping_mul`: unsigned arithmetic wraps in C++ too.
  uint32_t y = (key + salt) * 2654435769u;
  y = y ^ (key * 0x31415926u);
  return static_cast<size_t>(
      (static_cast<uint64_t>(y) * static_cast<uint64_t>(n)) >> 32);
}

// Do a lookup using minimal perfect hashing.
//
// The table is stored as a sequence of "salt" values, then a sequence of
// values that contain packed key/value pairs. The strategy is to hash twice.
// The first hash retrieves a salt value that makes the second hash unique.
// The hash function doesn't have to be very good, just good enough that the
// resulting map is unique.
template <typename KV, typename V, typename FK, typename FV>
inline V MphLookup(uint32_t x, absl::Span<const uint16_t> salt,
                   absl::Span<const KV> kv, FK fk, FV fv, V default_value) {
  const uint32_t s = salt[MyHash(x, 0, salt.size())];
  const KV key_val = kv[MyHash(x, s, salt.size())];
  if (x == fk(key_val)) {
    return fv(key_val);
  } else {
    return default_value;
  }
}

// ===========================================================================
// unicode-normalization/src/lookups.rs
// Lookups of unicode properties using minimal perfect hashing.
// ===========================================================================

// Extract the key in a 24 bit key and 8 bit value packed in a u32.
inline uint32_t U8LookupFk(uint32_t kv) { return kv >> 8; }

// Extract the value in a 24 bit key and 8 bit value packed in a u32.
inline uint8_t U8LookupFv(uint32_t kv) {
  return static_cast<uint8_t>(kv & 0xff);
}

// Extract the key for a boolean lookup.
inline uint32_t BoolLookupFk(uint32_t kv) { return kv; }

// Extract the value for a boolean lookup.
inline bool BoolLookupFv(uint32_t /*kv*/) { return true; }

// Extract the key in a pair.
inline uint32_t PairLookupFk(const CompositionEntry& kv) { return kv.key; }
inline uint32_t PairLookupFk(const DecompositionEntry& kv) { return kv.key; }

// Extract the value in a pair, returning an option.
inline std::optional<char32_t> PairLookupFvOpt(const CompositionEntry& kv) {
  return kv.value;
}
inline std::optional<std::pair<uint16_t, uint16_t>> PairLookupFvOpt(
    const DecompositionEntry& kv) {
  return std::make_pair(kv.start, kv.len);
}

// `tables::composition_table_astral` (a `match` over the astral compositions).
std::optional<char32_t> CompositionTableAstral(char32_t c1, char32_t c2) {
  for (const AstralCompositionEntry& entry : kCompositionTableAstral) {
    if (entry.c1 == c1 && entry.c2 == c2) return entry.composed;
  }
  return std::nullopt;
}

std::optional<char32_t> CompositionTable(char32_t c1, char32_t c2) {
  if (c1 < 0x10000 && c2 < 0x10000) {
    return MphLookup<CompositionEntry, std::optional<char32_t>>(
        (static_cast<uint32_t>(c1) << 16) | static_cast<uint32_t>(c2),
        kCompositionTableSalt, kCompositionTableKv,
        [](const CompositionEntry& kv) { return PairLookupFk(kv); },
        [](const CompositionEntry& kv) { return PairLookupFvOpt(kv); },
        std::nullopt);
  } else {
    return CompositionTableAstral(c1, c2);
  }
}

std::optional<absl::Span<const char32_t>> CanonicalFullyDecomposed(char32_t c) {
  std::optional<std::pair<uint16_t, uint16_t>> start_len =
      MphLookup<DecompositionEntry,
                std::optional<std::pair<uint16_t, uint16_t>>>(
          static_cast<uint32_t>(c), kCanonicalDecomposedSalt,
          kCanonicalDecomposedKv,
          [](const DecompositionEntry& kv) { return PairLookupFk(kv); },
          [](const DecompositionEntry& kv) { return PairLookupFvOpt(kv); },
          std::nullopt);
  if (!start_len.has_value()) return std::nullopt;
  return kCanonicalDecomposedChars.subspan(start_len->first, start_len->second);
}

std::optional<absl::Span<const char32_t>> CompatibilityFullyDecomposed(
    char32_t c) {
  std::optional<std::pair<uint16_t, uint16_t>> start_len =
      MphLookup<DecompositionEntry,
                std::optional<std::pair<uint16_t, uint16_t>>>(
          static_cast<uint32_t>(c), kCompatibilityDecomposedSalt,
          kCompatibilityDecomposedKv,
          [](const DecompositionEntry& kv) { return PairLookupFk(kv); },
          [](const DecompositionEntry& kv) { return PairLookupFvOpt(kv); },
          std::nullopt);
  if (!start_len.has_value()) return std::nullopt;
  return kCompatibilityDecomposedChars.subspan(start_len->first,
                                               start_len->second);
}

// ===========================================================================
// unicode-normalization/src/normalize.rs
// Functions for computing canonical and compatible decompositions for Unicode
// characters.
// ===========================================================================

// Constants from Unicode 9.0.0 Section 3.12 Conjoining Jamo Behavior
// http://www.unicode.org/versions/Unicode9.0.0/ch03.pdf#M9.32468.Heading.310.Combining.Jamo.Behavior
constexpr uint32_t kSBase = 0xAC00;
constexpr uint32_t kLBase = 0x1100;
constexpr uint32_t kVBase = 0x1161;
constexpr uint32_t kTBase = 0x11A7;
constexpr uint32_t kLCount = 19;
constexpr uint32_t kVCount = 21;
constexpr uint32_t kTCount = 28;
constexpr uint32_t kNCount = kVCount * kTCount;
constexpr uint32_t kSCount = kLCount * kNCount;

constexpr uint32_t kSLast = kSBase + kSCount - 1;
constexpr uint32_t kLLast = kLBase + kLCount - 1;
constexpr uint32_t kVLast = kVBase + kVCount - 1;
constexpr uint32_t kTLast = kTBase + kTCount - 1;

// Composition only occurs for `TPart`s in `U+11A8 ..= U+11C2`,
// i.e. `kTBase + 1 ..= kTLast`.
constexpr uint32_t kTFirst = kTBase + 1;

// Safety-usable invariant: This ensures that c is a valid Hangul Syllable
// character (U+AC00..U+D7AF)
bool IsHangulSyllable(char32_t c) {
  // Safety: This checks the range 0xAC00 (kSBase) to 0xD7A4 (kSBase + kSCount),
  // upholding the safety-usable invariant
  return static_cast<uint32_t>(c) >= kSBase &&
         static_cast<uint32_t>(c) < (kSBase + kSCount);
}

// Decompose a precomposed Hangul syllable
// Safety: `s` MUST be a valid Hangul Syllable character, between U+AC00..U+D7AF
void DecomposeHangul(char32_t s, absl::FunctionRef<void(char32_t)> emit_char) {
  // This will be at most 0x2baf, the size of the Hangul Syllables block
  const uint32_t s_index = static_cast<uint32_t>(s) - kSBase;
  // This will be at most 0x2baf / (21 * 28), 19
  const uint32_t l_index = s_index / kNCount;
  // Safety: kLBase (0x1100) plus at most 19 is still going to be in range for
  // a valid Unicode code point in the BMP (< 0xD800)
  emit_char(static_cast<char32_t>(kLBase + l_index));

  // Safety: This will be at most (kNCount - 1) / kTCount = (V*T - 1) / T,
  // which gives us an upper bound of kVCount = 21
  const uint32_t v_index = (s_index % kNCount) / kTCount;
  // Safety: kVBase (0x1161) plus at most 21 is still going to be in range for
  // a valid Unicode code point in the BMP (< 0xD800)
  emit_char(static_cast<char32_t>(kVBase + v_index));

  // Safety: This will be at most kTCount - 1 (27)
  const uint32_t t_index = s_index % kTCount;
  if (t_index > 0) {
    // Safety: kTBase (0x11A7) plus at most 27 is still going to be in range
    // for a valid Unicode code point in the BMP (< 0xD800)
    emit_char(static_cast<char32_t>(kTBase + t_index));
  }
}

// Compose a pair of Hangul Jamo
std::optional<char32_t> ComposeHangul(char32_t a_char, char32_t b_char) {
  const uint32_t a = static_cast<uint32_t>(a_char);
  const uint32_t b = static_cast<uint32_t>(b_char);
  // Compose a leading consonant and a vowel together into an LV_Syllable
  if (kLBase <= a && a <= kLLast && kVBase <= b && b <= kVLast) {
    // Safety: based on the above bounds, l_index will be less than or equal to
    // kLCount (19) and v_index will be <= kVCount (21)
    const uint32_t l_index = a - kLBase;
    const uint32_t v_index = b - kVBase;
    // Safety: This will be <= 19 * (20 * 21) + (21 * 20), which is 8400.
    const uint32_t lv_index = l_index * kNCount + v_index * kTCount;
    // Safety: This is between 0xAC00 and 0xCCD0, which are in range for Hangul
    // Syllables (U+AC00..U+D7AF) and also in range for BMP unicode
    const uint32_t s = kSBase + lv_index;
    // Safety: We've verified this is in-range
    return static_cast<char32_t>(s);
  }
  // Compose an LV_Syllable and a trailing consonant into an LVT_Syllable
  if (kSBase <= a && a <= kSLast && kTFirst <= b && b <= kTLast &&
      (a - kSBase) % kTCount == 0) {
    // Safety: a is between 0xAC00 and (0xAC00 + 19 * 21 * 28). b - kTBase is
    // between 0 and 19. Adding a number 0 to 19 to a number that is at largest
    // 0xD7A4 will not go out of bounds to 0xD800 (where the surrogates start),
    // so this is safe.
    return static_cast<char32_t>(a + (b - kTBase));
  }
  return std::nullopt;
}

template <typename D>
void Decompose(char32_t c, D decompose_char,
               absl::FunctionRef<void(char32_t)> emit_char) {
  // 7-bit ASCII never decomposes
  if (c <= 0x7f) {
    emit_char(c);
    return;
  }

  // Perform decomposition for Hangul
  if (IsHangulSyllable(c)) {
    // Safety: Hangul Syllables invariant checked by is_hangul_syllable above
    DecomposeHangul(c, emit_char);
    return;
  }

  if (std::optional<absl::Span<const char32_t>> decomposed = decompose_char(c);
      decomposed.has_value()) {
    for (const char32_t d : *decomposed) {
      emit_char(d);
    }
    return;
  }

  // Finally bottom out.
  emit_char(c);
}

// ===========================================================================
// unicode_normalization_alignments: decompose.rs
// ===========================================================================

enum class DecompositionType {
  kCanonical,
  kCompatible,
};

// External iterator for a string decomposition's characters.
class Decompositions {
 public:
  static Decompositions NewCanonical(absl::string_view s) {
    return Decompositions(DecompositionType::kCanonical, s);
  }

  static Decompositions NewCompatible(absl::string_view s) {
    return Decompositions(DecompositionType::kCompatible, s);
  }

  // `Iterator::next`.
  std::optional<std::pair<char32_t, int64_t>> Next() {
    while (ready_end_ == 0) {
      const std::optional<char32_t> ch = iter_.Next();
      if (ch.has_value()) {
        bool first = true;
        const auto emit = [this, &first](char32_t d) {
          PushBack(d, first);
          first = false;
        };
        if (kind_ == DecompositionType::kCanonical) {
          DecomposeCanonical(*ch, emit);
        } else {
          DecomposeCompatible(*ch, emit);
        }
      } else {
        if (buffer_.empty()) {
          return std::nullopt;
        } else {
          SortPending();

          // This implementation means that we can call `next`
          // on an exhausted iterator; the last outer `next` call
          // will result in an inner `next` call. To make this
          // safe, we use `fuse`.
          break;
        }
      }
    }

    // We can assume here that, if `self.ready.end` is greater than zero,
    // it's also greater than `self.ready.start`. That's because we only
    // increment `self.ready.start` inside `increment_next_ready`, and
    // whenever it reaches equality with `self.ready.end`, we reset both
    // to zero, maintaining the invariant that:
    //      self.ready.start < self.ready.end || self.ready.end ==
    //      self.ready.start == 0
    //
    // This less-than-obviously-safe implementation is chosen for performance,
    // minimizing the number & complexity of branches in `next` in the common
    // case of buffering then unbuffering a single character with each call.
    const Entry entry = buffer_[ready_start_];
    IncrementNextReady();
    return std::make_pair(entry.ch, entry.size);
  }

  // A capacity hint for draining the iterator.
  size_t RemainingBytes() const { return iter_.RemainingBytes(); }

 private:
  // An element of `buffer`: `(u8, char, isize)`.
  struct Entry {
    uint8_t ccc;
    char32_t ch;
    int64_t size;
  };

  Decompositions(DecompositionType kind, absl::string_view s)
      : kind_(kind), iter_(s) {}

  void PushBack(char32_t ch, bool first) {
    const uint8_t cls = CanonicalCombiningClass(ch);

    if (cls == 0) {
      SortPending();
    }

    buffer_.push_back(Entry{cls, ch, first ? 0 : 1});
  }

  void SortPending() {
    // NB: `sort_by_key` is stable, so it will preserve the original text's
    // order within a combining class.
    std::stable_sort(
        buffer_.begin() + ready_end_, buffer_.end(),
        [](const Entry& a, const Entry& b) { return a.ccc < b.ccc; });
    ready_end_ = buffer_.size();
  }

  void ResetBuffer() {
    // Equivalent to `self.buffer.drain(0..self.ready.end)` (if SmallVec
    // supported this API)
    const size_t pending = buffer_.size() - ready_end_;
    for (size_t i = 0; i < pending; ++i) {
      buffer_[i] = buffer_[i + ready_end_];
    }
    buffer_.resize(pending);
    ready_start_ = 0;
    ready_end_ = 0;
  }

  void IncrementNextReady() {
    const size_t next = ready_start_ + 1;
    if (next == ready_end_) {
      ResetBuffer();
    } else {
      ready_start_ = next;
    }
  }

  DecompositionType kind_;
  internal::Utf8Chars iter_;

  // This buffer stores pairs of (canonical combining class, character),
  // pushed onto the end in text order.
  //
  // It's divided into up to three sections:
  // 1) A prefix that is free space;
  // 2) "Ready" characters which are sorted and ready to emit on demand;
  // 3) A "pending" block which stills needs more characters for us to be able
  //    to sort in canonical order and is not safe to emit.
  absl::InlinedVector<Entry, 4> buffer_;
  // `ready: Range<usize>`.
  size_t ready_start_ = 0;
  size_t ready_end_ = 0;
};

// ===========================================================================
// unicode_normalization_alignments: recompose.rs
// ===========================================================================

enum class RecompositionState {
  kComposing,
  kPurging,   // Purging(usize): the index is `state_next_`
  kFinished,  // Finished(usize): the index is `state_next_`
};

// External iterator for a string recomposition's characters.
class Recompositions {
 public:
  static Recompositions NewCanonical(absl::string_view s) {
    return Recompositions(Decompositions::NewCanonical(s));
  }

  static Recompositions NewCompatible(absl::string_view s) {
    return Recompositions(Decompositions::NewCompatible(s));
  }

  // `Iterator::next`.
  std::optional<std::pair<char32_t, int64_t>> Next() {
    while (true) {
      switch (state_) {
        case RecompositionState::kComposing: {
          while (const std::optional<std::pair<char32_t, int64_t>> item =
                     iter_.Next()) {
            const char32_t ch = item->first;
            const int64_t change = item->second;
            const uint8_t ch_class = CanonicalCombiningClass(ch);
            if (!composee_.has_value()) {
              if (ch_class != 0) {
                return std::make_pair(ch, change);
              }
              composee_ = std::make_pair(ch, change);
              continue;
            }
            const std::pair<char32_t, int64_t> k = *composee_;
            if (!last_ccc_.has_value()) {
              if (const std::optional<char32_t> r = Compose(k.first, ch);
                  r.has_value()) {
                composee_ = std::make_pair(*r, k.second + change - 1);
                continue;
              } else {
                if (ch_class == 0) {
                  composee_ = std::make_pair(ch, change);
                  return k;
                }
                buffer_.push_back(std::make_pair(ch, change));
                last_ccc_ = ch_class;
              }
            } else {
              const uint8_t l_class = *last_ccc_;
              if (l_class >= ch_class) {
                // `ch` is blocked from `composee`
                if (ch_class == 0) {
                  composee_ = std::make_pair(ch, change);
                  last_ccc_ = std::nullopt;
                  state_ = RecompositionState::kPurging;
                  state_next_ = 0;
                  return k;
                }
                buffer_.push_back(std::make_pair(ch, change));
                last_ccc_ = ch_class;
                continue;
              }
              if (const std::optional<char32_t> r = Compose(k.first, ch);
                  r.has_value()) {
                composee_ = std::make_pair(*r, k.second + change - 1);
                continue;
              } else {
                buffer_.push_back(std::make_pair(ch, change));
                last_ccc_ = ch_class;
              }
            }
          }
          state_ = RecompositionState::kFinished;
          state_next_ = 0;
          if (composee_.has_value()) {
            return TakeComposee();
          }
          break;
        }
        case RecompositionState::kPurging: {
          const size_t next = state_next_;
          if (next >= buffer_.size()) {
            buffer_.clear();
            state_ = RecompositionState::kComposing;
          } else {
            const std::pair<char32_t, int64_t> s = buffer_[next];
            state_next_ = next + 1;
            return s;
          }
          break;
        }
        case RecompositionState::kFinished: {
          const size_t next = state_next_;
          if (next >= buffer_.size()) {
            buffer_.clear();
            return TakeComposee();
          } else {
            const std::pair<char32_t, int64_t> s = buffer_[next];
            state_next_ = next + 1;
            return s;
          }
        }
      }
    }
  }

  // A capacity hint for draining the iterator.
  size_t RemainingBytes() const { return iter_.RemainingBytes(); }

 private:
  explicit Recompositions(Decompositions iter)
      : iter_(std::move(iter)), state_(RecompositionState::kComposing) {}

  // `self.composee.take()`.
  std::optional<std::pair<char32_t, int64_t>> TakeComposee() {
    std::optional<std::pair<char32_t, int64_t>> taken = composee_;
    composee_ = std::nullopt;
    return taken;
  }

  Decompositions iter_;
  RecompositionState state_;
  size_t state_next_ = 0;
  absl::InlinedVector<std::pair<char32_t, int64_t>, 4> buffer_;
  std::optional<std::pair<char32_t, int64_t>> composee_;
  std::optional<uint8_t> last_ccc_;
};

template <typename Iter>
std::vector<std::pair<char32_t, int64_t>> Collect(Iter iter) {
  std::vector<std::pair<char32_t, int64_t>> out;
  out.reserve(iter.RemainingBytes());
  while (const std::optional<std::pair<char32_t, int64_t>> item = iter.Next()) {
    out.push_back(*item);
  }
  return out;
}

}  // namespace

// ---- lookups.rs / normalize.rs public API --------------------------------

uint8_t CanonicalCombiningClass(char32_t c) {
  return MphLookup<uint32_t, uint8_t>(
      static_cast<uint32_t>(c), kCanonicalCombiningClassSalt,
      kCanonicalCombiningClassKv, U8LookupFk, U8LookupFv, 0);
}

bool IsCombiningMark(char32_t c) {
  return MphLookup<uint32_t, bool>(static_cast<uint32_t>(c), kCombiningMarkSalt,
                                   kCombiningMarkKv, BoolLookupFk, BoolLookupFv,
                                   false);
}

void DecomposeCanonical(char32_t c,
                        absl::FunctionRef<void(char32_t)> emit_char) {
  Decompose(c, CanonicalFullyDecomposed, emit_char);
}

void DecomposeCompatible(char32_t c,
                         absl::FunctionRef<void(char32_t)> emit_char) {
  const auto decompose_char =
      [](char32_t c) -> std::optional<absl::Span<const char32_t>> {
    std::optional<absl::Span<const char32_t>> d =
        CompatibilityFullyDecomposed(c);
    if (d.has_value()) return d;
    return CanonicalFullyDecomposed(c);
  };
  Decompose(c, decompose_char, emit_char);
}

std::optional<char32_t> Compose(char32_t a, char32_t b) {
  std::optional<char32_t> hangul = ComposeHangul(a, b);
  if (hangul.has_value()) return hangul;
  return CompositionTable(a, b);
}

// ---- unicode_normalization_alignments: lib.rs ------------------------------

std::vector<std::pair<char32_t, int64_t>> Nfd(absl::string_view s) {
  return Collect(Decompositions::NewCanonical(s));
}

std::vector<std::pair<char32_t, int64_t>> Nfkd(absl::string_view s) {
  return Collect(Decompositions::NewCompatible(s));
}

std::vector<std::pair<char32_t, int64_t>> Nfc(absl::string_view s) {
  return Collect(Recompositions::NewCanonical(s));
}

std::vector<std::pair<char32_t, int64_t>> Nfkc(absl::string_view s) {
  return Collect(Recompositions::NewCompatible(s));
}

}  // namespace tokenizers::unicode
