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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_CJK_BLOB_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_CJK_BLOB_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::omni::tts::kokoro {

// Readers for the Japanese and Chinese lexicon blobs, which are read in place.
//
// Rather than unpacking an archive to a cache directory, these blobs are mapped
// and used where they lie: the Japanese dictionary is over a hundred megabytes,
// and can afford neither the copy nor the disk. Nothing below allocates a copy
// of the data or assumes the buffer is aligned; every multi-byte field is read
// byte by byte.
//
// A blob's section table.
//
// Layout, all little-endian:
//
//     char   magic[8]          // "LTTSCJK1"
//     uint32 format_version    // 1
//     uint32 section_count
//     Entry  entries[section_count]
//     uint8  body[]            // each section 8-byte aligned
//
//     Entry := char name[8]; uint32 offset; uint32 size
//
// `offset` is absolute from the start of the blob, so a reader needs nothing
// but the header to hand a section to its consumer.
//
// Usage:
//   ASSERT_OK_AND_ASSIGN(CjkBlob blob, CjkBlob::Create(raw_bytes));
//   ASSERT_OK_AND_ASSIGN(absl::string_view keys, blob.Section("w-key"));
class CjkBlob {
 public:
  // Parses and validates the section directory of a `"LTTSCJK1"` container
  // buffer. The returned `CjkBlob` and all `Section()` views borrow directly
  // from `blob`, which must outlive them.
  //
  // args
  // - blob: Raw byte slice of the `"LTTSCJK1"` binary container.
  //
  // returns
  // - Parsed `CjkBlob` on success, or `absl::InvalidArgumentError` if the magic
  //   header, format version, section directory, or section bounds are invalid.
  static absl::StatusOr<CjkBlob> Create(absl::string_view blob);

  // Reports whether a section with the given name exists in this blob.
  //
  // args
  // - name: Section identifier (up to 8 ASCII bytes, e.g., `"w-idx"`).
  //
  // returns
  // - `true` if `name` is present in the section directory, `false` otherwise.
  bool Contains(absl::string_view name) const;

  // Looks up a section payload by name without copying.
  //
  // args
  // - name: Section identifier (up to 8 ASCII bytes, e.g., `"w-key"`).
  //
  // returns
  // - Zero-copy `absl::string_view` spanning the section payload, or
  //   `absl::NotFoundError` if `name` is absent.
  absl::StatusOr<absl::string_view> Section(absl::string_view name) const;

 private:
  explicit CjkBlob(
      absl::flat_hash_map<std::string, absl::string_view> sections);

  absl::flat_hash_map<std::string, absl::string_view> sections_;
};

// A sorted, NUL-separated key blob addressed by a `uint32` offset array.
//
// This backs both the Chinese word list and its phrase and character tables.
// Sorting by UTF-8 bytes makes exact lookup a binary search, and -- because a
// key's prefixes sort immediately before it -- makes "is this fragment a prefix
// of any entry" one as well. That second query is what jieba's DAG construction
// and pypinyin's maximum matching are built on, and it is why the blob stores
// only real entries instead of jieba's zero-frequency prefix padding.
//
// Usage:
//   ASSERT_OK_AND_ASSIGN(SortedStringIndex index,
//                        SortedStringIndex::Create(idx_bytes, key_bytes));
//   int pos = index.Find("一点");
//   bool has_prefix = index.HasPrefix("一");
class SortedStringIndex {
 public:
  // Parses and validates a little-endian `uint32_t` offset index of `size + 1`
  // entries over a NUL-separated `keys` payload. Both `index` and `keys` point
  // into the caller's buffer, which must outlive the returned object.
  //
  // args
  // - index: Byte slice of `size + 1` little-endian `uint32_t` offsets starting
  //   at `0` and monotonically non-decreasing within `keys.size()`.
  // - keys: Concatenated NUL-terminated UTF-8 key strings sorted in ascending
  //   lexicographical byte order.
  //
  // returns
  // - Validated `SortedStringIndex` on success, or `absl::InvalidArgumentError`
  //   if `index` is not a positive multiple of 4 bytes, does not start at `0`,
  //   is out of order, or references offsets past `keys.size()`.
  static absl::StatusOr<SortedStringIndex> Create(absl::string_view index,
                                                  absl::string_view keys);

  // Returns the number of keys stored in the index.
  //
  // returns
  // - Non-negative entry count (`index.size() / 4 - 1`).
  int Size() const;

  // Returns the key at `position` without its trailing NUL byte.
  //
  // args
  // - position: Zero-based entry index; must be in `[0, Size())`.
  //
  // returns
  // - Zero-copy `absl::string_view` of the key at `position`.
  absl::string_view Key(int position) const;

  // Searches for an exact match of `key` using binary search.
  //
  // args
  // - key: UTF-8 key string to look up.
  //
  // returns
  // - Zero-based position in `[0, Size())` if `key` is present, or `-1` if
  //   absent.
  int Find(absl::string_view key) const;

  // Reports whether at least one key in the index begins with `prefix`. An
  // empty `prefix` matches any non-empty index.
  //
  // args
  // - prefix: UTF-8 byte prefix to test.
  //
  // returns
  // - `true` if any key starts with `prefix`, `false` otherwise.
  bool HasPrefix(absl::string_view prefix) const;

 private:
  SortedStringIndex(absl::string_view index, int size, absl::string_view keys);

  // Returns the position of the first key that does not sort before `key`.
  int LowerBound(absl::string_view key) const;

  absl::string_view index_;
  int size_ = 0;
  absl::string_view keys_;
};

// A parallel `uint32` offset array over a NUL-separated value blob.
//
// Values are addressed by the position their key has in a `SortedStringIndex`,
// which keeps the two tables independent: the keys can be searched without
// touching the values, and the values can be any length without disturbing the
// search.
//
// Usage:
//   ASSERT_OK_AND_ASSIGN(StringValueTable values,
//                        StringValueTable::Create(vidx_bytes, val_bytes));
//   absl::string_view val = values.Value(pos);
class StringValueTable {
 public:
  // Parses and validates a little-endian `uint32_t` offset index of `size + 1`
  // entries over a NUL-separated `values` payload. Both `index` and `values`
  // point into the caller's buffer, which must outlive the returned object.
  //
  // args
  // - index: Byte slice of `size + 1` little-endian `uint32_t` offsets starting
  //   at `0` and monotonically non-decreasing within `values.size()`.
  // - values: Concatenated NUL-terminated UTF-8 value strings parallel to a
  //   `SortedStringIndex`.
  //
  // returns
  // - Validated `StringValueTable` on success, or `absl::InvalidArgumentError`
  //   if `index` is malformed or references offsets past `values.size()`.
  static absl::StatusOr<StringValueTable> Create(absl::string_view index,
                                                 absl::string_view values);

  // Returns the number of values stored in the table.
  //
  // returns
  // - Non-negative entry count (`index.size() / 4 - 1`).
  int Size() const;

  // Returns the value at `position` without its trailing NUL byte.
  //
  // args
  // - position: Zero-based entry index (typically returned by
  //   `SortedStringIndex::Find()`); must be in `[0, Size())`.
  //
  // returns
  // - Zero-copy `absl::string_view` of the value at `position`.
  absl::string_view Value(int position) const;

 private:
  StringValueTable(absl::string_view index, int size, absl::string_view values);

  absl::string_view index_;
  int size_ = 0;
  absl::string_view values_;
};

// MeCab's bundled Darts 0.32 double array, read in place.
//
// Each unit is `int32 base; uint32 check`. Traversal carries the parent's
// *base value* rather than its index: starting from `base[0]`, the child for
// byte `c` sits at `base + c + 1` and is valid when its `check` equals the
// parent's base. A node's value hangs off the slot at `base` itself, encoded as
// `-base - 1` and marked by a negative base.
//
// Reusing unidic-lite's own double array, rather than rebuilding one, is what
// makes Japanese segmentation byte-identical to MeCab by construction.
class MecabDoubleArray {
 public:
  struct Match {
    // MeCab packs two numbers into one value: the number of tokens sharing
    // this surface form in the low eight bits, and the index of the first of
    // them in the rest.
    int token_count;
    int token_index;
    // How many bytes of the query this key consumed.
    int length;
  };

  // Wraps a double array of 8-byte units without copying.
  //
  // args
  // - units: Raw `int32 base; uint32 check` units, which must outlive the
  //   result.
  //
  // returns
  // - The wrapped array, or `absl::InvalidArgumentError` if `units` is empty
  //   or not a whole number of 8-byte units.
  static absl::StatusOr<MecabDoubleArray> Create(absl::string_view units);

  // Finds every dictionary key that is a prefix of `key`.
  //
  // args
  // - key: UTF-8 bytes to match from their start.
  //
  // returns
  // - One `Match` per matching key, shortest first.
  std::vector<Match> CommonPrefixSearch(absl::string_view key) const;

 private:
  MecabDoubleArray(absl::string_view units, int unit_count);

  // Returns the `base` field of unit `index`, or 0 when out of range.
  int32_t Base(int64_t index) const;

  // Returns the `check` field of unit `index`, or a value that cannot match
  // when out of range.
  uint32_t Check(int64_t index) const;

  absl::string_view units_;
  int unit_count_ = 0;
};

// A row of a MeCab token table.
struct MecabToken {
  // The context ids the connection matrix is indexed by. MeCab calls them
  // `lcAttr` and `rcAttr`: `left_id` applies when the token is the right half
  // of a boundary, `right_id` when it is the left half.
  int left_id = 0;
  int right_id = 0;
  // MeCab's part-of-speech id.
  int pos_id = 0;
  // What the word costs on its own.
  int word_cost = 0;
  // The token's CSV feature record, pointing into the blob. unidic-lite ships
  // 26 columns; the pruned `ja-lexicon` keeps the eight the front end reads.
  absl::string_view feature;
};

// A MeCab token table: fixed 16-byte rows over a blob of NUL-terminated
// feature records.
//
// The rows are copied byte for byte out of unidic-lite, which is what makes
// segmentation identical to MeCab's by construction. The packer rewrites the
// feature blob, and the offset in each row along with it, but nothing the
// search reads.
class MecabTokenTable {
 public:
  // Wraps a token table and the feature blob its rows point into.
  //
  // args
  // - tokens: Concatenated 16-byte token rows.
  // - features: NUL-separated CSV feature records the rows point into.
  //   Both must outlive the result.
  //
  // returns
  // - The wrapped table, or `absl::InvalidArgumentError` if `tokens` is not a
  //   whole number of 16-byte rows.
  static absl::StatusOr<MecabTokenTable> Create(absl::string_view tokens,
                                                absl::string_view features);

  // Returns the number of rows in the table.
  //
  // returns
  // - Non-negative row count.
  int Size() const;

  // Decodes one row of the table.
  //
  // args
  // - index: Zero-based row index; must be in `[0, Size())`.
  //
  // returns
  // - The decoded row, with `feature` empty if its offset is out of range.
  MecabToken Token(int index) const;

 private:
  MecabTokenTable(absl::string_view tokens, int size,
                  absl::string_view features);

  absl::string_view tokens_;
  int size_ = 0;
  absl::string_view features_;
};

// MeCab's connection matrix: what every ordered pair of context ids costs.
//
// This is the largest section of the Japanese blob by a wide margin -- 58% of
// it once the features are pruned -- and the search reads at most one entry per
// candidate pair, so it stays exactly as MeCab wrote it.
class MecabConnectionMatrix {
 public:
  // Wraps MeCab's `matrix.bin` without copying.
  //
  // args
  // - matrix: `matrix.bin` bytes including its size header, which must
  //   outlive the result.
  //
  // returns
  // - The wrapped matrix, or `absl::InvalidArgumentError` if `matrix` is
  //   shorter than its header claims.
  static absl::StatusOr<MecabConnectionMatrix> Create(absl::string_view matrix);

  // Returns how many distinct left context ids the matrix covers. MeCab's
  // `matrix.bin` header calls this `rsize`.
  //
  // returns
  // - Non-negative left context id count.
  int LeftContextCount() const;

  // Returns how many distinct right context ids the matrix covers. MeCab's
  // `matrix.bin` header calls this `lsize`.
  //
  // returns
  // - Non-negative right context id count.
  int RightContextCount() const;

  // Looks up the cost of one node boundary.
  //
  // args
  // - right_id: Right context id of the left node.
  // - left_id: Left context id of the right node.
  //
  // returns
  // - The boundary cost, or 0 if either id is outside the matrix, so a
  //   truncated dictionary degrades instead of reading garbage.
  int Cost(int right_id, int left_id) const;

 private:
  MecabConnectionMatrix(absl::string_view costs, int left_context_count,
                        int right_context_count);

  absl::string_view costs_;
  int left_context_count_ = 0;
  int right_context_count_ = 0;
};

// MeCab's `char.bin`: which categories each code point belongs to, and what the
// tokenizer should do where the dictionary runs out.
class MecabCharProperty {
 public:
  // What the table records about one code point.
  struct CharInfo {
    // The categories the code point belongs to, one bit each. Two characters
    // group together exactly when these masks intersect.
    uint32_t types = 0;
    // The category that governs: its `unk.dic` entries are the ones guessed
    // with, and the three fields below are read off it rather than off any
    // other category in `types`.
    int default_type = 0;
    // The longest unknown word to guess, in characters.
    int length = 0;
    // Whether to guess even where the dictionary already matched.
    bool invoke = false;
    // Whether to also guess the whole run of characters sharing a category.
    bool group = false;
  };

  // A decoded character: what the table records for it and its byte length.
  struct DecodedChar {
    CharInfo info;
    int length = 1;
  };

  // Wraps MeCab's `char.bin` without copying.
  //
  // args
  // - data: `char.bin` bytes, which must outlive the result.
  //
  // returns
  // - The wrapped table, or `absl::InvalidArgumentError` if `data` declares
  //   no categories or more than it holds.
  static absl::StatusOr<MecabCharProperty> Create(absl::string_view data);

  // Returns the number of character categories.
  //
  // returns
  // - Positive category count.
  int CategoryCount() const;

  // Returns the name of a category, which is the key `unk.dic` files its
  // entries under.
  //
  // args
  // - index: Zero-based category index; must be in `[0, CategoryCount())`.
  //
  // returns
  // - The category name without NUL padding.
  absl::string_view CategoryName(int index) const;

  // Looks up what the table records for one code point.
  //
  // The table covers the basic multilingual plane only. MeCab reads off the
  // end of its own array for anything above; this returns the record for
  // U+0000, the default category, which is what that was meant to say.
  //
  // args
  // - code_point: Unicode code point to look up.
  //
  // returns
  // - The record for `code_point`.
  CharInfo Info(char32_t code_point) const;

  // Decodes the UTF-8 character at a byte offset and looks it up.
  //
  // Malformed input decodes to one byte of U+0000, as it does in MeCab, so a
  // bad byte costs a byte of progress rather than stalling the tokenizer.
  //
  // args
  // - text: UTF-8 text.
  // - offset: Byte offset of the character to decode.
  //
  // returns
  // - The character's record and byte length.
  DecodedChar InfoAt(absl::string_view text, int offset) const;

 private:
  MecabCharProperty(absl::string_view names, int category_count,
                    absl::string_view table, int code_point_count);

  absl::string_view names_;
  int category_count_ = 0;
  absl::string_view table_;
  int code_point_count_ = 0;
};

}  // namespace litert::omni::tts::kokoro

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_CJK_BLOB_H_
