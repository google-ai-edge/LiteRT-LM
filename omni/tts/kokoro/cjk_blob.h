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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_CJK_BLOB_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_CJK_BLOB_H_

#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::omni::tts::kokoro {

// Readers for the Japanese and Chinese lexicon blobs, which are read in place.
//
// Unlike `espeak-ng-data`, which rides in the .litertlm as an uncompressed zip
// and is extracted to a cache directory, these blobs are mapped and used where
// they lie: the Japanese dictionary is over a hundred megabytes, and can afford
// neither the copy nor the disk. Nothing below allocates a copy of the data or
// assumes the buffer is aligned; every multi-byte field is read byte by byte.
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

}  // namespace litert::omni::tts::kokoro

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_CJK_BLOB_H_
