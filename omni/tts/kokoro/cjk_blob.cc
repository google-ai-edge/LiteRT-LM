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

#include "omni/tts/kokoro/cjk_blob.h"

#include <cstdint>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::omni::tts::kokoro {
namespace {

constexpr absl::string_view kMagic = "LTTSCJK1";
constexpr uint32_t kFormatVersion = 1;
constexpr int kHeaderSize = 16;
constexpr int kEntrySize = 16;

// Reads a little-endian 32-bit word. The blobs are mapped wherever the model
// container happens to put them, so nothing may assume alignment.
uint32_t LoadUint32(absl::string_view data, int64_t offset) {
  const unsigned char* bytes =
      reinterpret_cast<const unsigned char*>(data.data()) + offset;
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

// Parses an offset array of `size + 1` words, checking that it is strictly
// ascending and stays inside `body`. Returns the entry count.
absl::StatusOr<int> ValidateOffsetIndex(absl::string_view index,
                                        absl::string_view body,
                                        absl::string_view what) {
  if (index.size() % 4 != 0 || index.size() < 4) {
    return absl::InvalidArgumentError(absl::StrCat(
        what, " offset index is ", index.size(),
        " bytes, which is not a positive whole number of 32-bit words"));
  }
  const int size = static_cast<int>(index.size() / 4) - 1;
  uint32_t previous = LoadUint32(index, 0);
  if (previous != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(what, " offset index starts at ", previous, ", not 0"));
  }
  for (int i = 1; i <= size; ++i) {
    const uint32_t offset = LoadUint32(index, i * 4);
    if (offset <= previous || offset > body.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat(what, " offset ", i, " is ", offset,
                       ", which is out of order or"
                       " past the end of a ",
                       body.size(), "-byte payload"));
    }
    previous = offset;
  }
  return size;
}

// Returns the NUL-terminated record at [start, end) of `body`, without its
// terminator.
absl::string_view RecordAt(absl::string_view body, uint32_t start,
                           uint32_t end) {
  if (end <= start) return "";
  // `end` points just past the terminator that the packer always writes.
  return body.substr(start, end - start - 1);
}

}  // namespace

CjkBlob::CjkBlob(absl::flat_hash_map<std::string, absl::string_view> sections)
    : sections_(std::move(sections)) {}

absl::StatusOr<CjkBlob> CjkBlob::Create(absl::string_view blob) {
  if (blob.size() < kHeaderSize) {
    return absl::InvalidArgumentError(absl::StrCat(
        "CJK lexicon blob is ", blob.size(), " bytes, too short for a header"));
  }
  if (blob.substr(0, kMagic.size()) != kMagic) {
    return absl::InvalidArgumentError(
        "CJK lexicon blob does not start with the expected magic");
  }
  const uint32_t format_version = LoadUint32(blob, 8);
  if (format_version != kFormatVersion) {
    return absl::InvalidArgumentError(
        absl::StrCat("CJK lexicon blob has format version ", format_version,
                     ", but this build understands ", kFormatVersion));
  }
  const uint32_t section_count = LoadUint32(blob, 12);
  const int64_t table_end =
      static_cast<int64_t>(kHeaderSize) + section_count * kEntrySize;
  if (table_end > static_cast<int64_t>(blob.size())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "CJK lexicon blob declares ", section_count,
        " sections, whose table does not fit in ", blob.size(), " bytes"));
  }

  absl::flat_hash_map<std::string, absl::string_view> sections;
  sections.reserve(section_count);
  for (uint32_t i = 0; i < section_count; ++i) {
    const int64_t entry = kHeaderSize + i * kEntrySize;
    absl::string_view name = blob.substr(entry, 8);
    const int64_t terminator = name.find('\0');
    if (terminator != absl::string_view::npos) {
      name = name.substr(0, terminator);
    }
    const uint32_t offset = LoadUint32(blob, entry + 8);
    const uint32_t size = LoadUint32(blob, entry + 12);
    if (static_cast<int64_t>(offset) + size >
        static_cast<int64_t>(blob.size())) {
      return absl::InvalidArgumentError(absl::StrCat(
          "CJK lexicon section \"", name, "\" spans [", offset, ", ",
          offset + size, "), past the end of a ", blob.size(), "-byte blob"));
    }
    if (!sections.try_emplace(name, blob.substr(offset, size)).second) {
      return absl::InvalidArgumentError(absl::StrCat(
          "CJK lexicon blob has more than one section named \"", name, "\""));
    }
  }
  return CjkBlob(std::move(sections));
}

bool CjkBlob::Contains(absl::string_view name) const {
  return sections_.contains(name);
}

absl::StatusOr<absl::string_view> CjkBlob::Section(
    absl::string_view name) const {
  const auto it = sections_.find(name);
  if (it == sections_.end()) {
    return absl::NotFoundError(
        absl::StrCat("CJK lexicon blob has no section named \"", name, "\""));
  }
  return it->second;
}

SortedStringIndex::SortedStringIndex(absl::string_view index, int size,
                                     absl::string_view keys)
    : index_(index), size_(size), keys_(keys) {}

absl::StatusOr<SortedStringIndex> SortedStringIndex::Create(
    absl::string_view index, absl::string_view keys) {
  ABSL_ASSIGN_OR_RETURN(const int size,
                        ValidateOffsetIndex(index, keys, "key"));
  return SortedStringIndex(index, size, keys);
}

int SortedStringIndex::Size() const { return size_; }

absl::string_view SortedStringIndex::Key(int position) const {
  return RecordAt(keys_, LoadUint32(index_, position * 4),
                  LoadUint32(index_, position * 4 + 4));
}

int SortedStringIndex::LowerBound(absl::string_view key) const {
  int low = 0;
  int high = size_;
  while (low < high) {
    const int middle = low + (high - low) / 2;
    if (Key(middle) < key) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

int SortedStringIndex::Find(absl::string_view key) const {
  const int position = LowerBound(key);
  if (position < size_ && Key(position) == key) {
    return position;
  }
  return -1;
}

bool SortedStringIndex::HasPrefix(absl::string_view prefix) const {
  const int position = LowerBound(prefix);
  return position < size_ && Key(position).substr(0, prefix.size()) == prefix;
}

StringValueTable::StringValueTable(absl::string_view index, int size,
                                   absl::string_view values)
    : index_(index), size_(size), values_(values) {}

absl::StatusOr<StringValueTable> StringValueTable::Create(
    absl::string_view index, absl::string_view values) {
  ABSL_ASSIGN_OR_RETURN(const int size,
                        ValidateOffsetIndex(index, values, "value"));
  return StringValueTable(index, size, values);
}

int StringValueTable::Size() const { return size_; }

absl::string_view StringValueTable::Value(int position) const {
  return RecordAt(values_, LoadUint32(index_, position * 4),
                  LoadUint32(index_, position * 4 + 4));
}

}  // namespace litert::omni::tts::kokoro
