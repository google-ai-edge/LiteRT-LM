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

#include "omni/text2speech/kokoro/cjk_blob.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

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
constexpr int kDoubleArrayUnitSize = 8;
constexpr int kTokenSize = 16;
constexpr int kCategoryNameSize = 32;

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

// Reads a little-endian 16-bit word, with the same alignment caveat.
uint16_t LoadUint16(absl::string_view data, int64_t offset) {
  const unsigned char* bytes =
      reinterpret_cast<const unsigned char*>(data.data()) + offset;
  return static_cast<uint16_t>(static_cast<uint16_t>(bytes[0]) |
                               (static_cast<uint16_t>(bytes[1]) << 8));
}

// A code point and the number of bytes it was decoded from.
struct DecodedCodePoint {
  char32_t code_point = 0;
  int length = 1;
};

// Decodes the UTF-8 character at byte `offset`.
//
// This follows MeCab's `utf8_to_ucs2` rather than being strict: anything
// malformed is one byte of U+0000. That matters because the tokenizer advances
// by whatever this reports, so refusing to decode would stall it, and U+0000
// lands in the default category, which is where unreadable bytes belong.
DecodedCodePoint DecodeMecabChar(absl::string_view text, int offset) {
  const int available = static_cast<int>(text.size()) - offset;
  if (available <= 0) return {};

  const unsigned char* bytes =
      reinterpret_cast<const unsigned char*>(text.data()) + offset;
  if (bytes[0] < 0x80) return {bytes[0], 1};
  if (available >= 2 && (bytes[0] & 0xE0) == 0xC0 &&
      (bytes[1] & 0xC0) == 0x80) {
    return {static_cast<char32_t>(((bytes[0] & 0x1F) << 6) | (bytes[1] & 0x3F)),
            2};
  }
  if (available >= 3 && (bytes[0] & 0xF0) == 0xE0 &&
      (bytes[1] & 0xC0) == 0x80 && (bytes[2] & 0xC0) == 0x80) {
    return {static_cast<char32_t>(((bytes[0] & 0x0F) << 12) |
                                  ((bytes[1] & 0x3F) << 6) | (bytes[2] & 0x3F)),
            3};
  }
  if (available >= 4 && (bytes[0] & 0xF8) == 0xF0 &&
      (bytes[1] & 0xC0) == 0x80 && (bytes[2] & 0xC0) == 0x80 &&
      (bytes[3] & 0xC0) == 0x80) {
    return {static_cast<char32_t>(((bytes[0] & 0x07) << 18) |
                                  ((bytes[1] & 0x3F) << 12) |
                                  ((bytes[2] & 0x3F) << 6) | (bytes[3] & 0x3F)),
            4};
  }
  return {};
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

MecabDoubleArray::MecabDoubleArray(absl::string_view units, int unit_count)
    : units_(units), unit_count_(unit_count) {}

absl::StatusOr<MecabDoubleArray> MecabDoubleArray::Create(
    absl::string_view units) {
  if (units.size() % kDoubleArrayUnitSize != 0 || units.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("double array is ", units.size(),
                     " bytes, which is not a positive"
                     " whole number of ",
                     kDoubleArrayUnitSize, "-byte units"));
  }
  return MecabDoubleArray(
      units, static_cast<int>(units.size() / kDoubleArrayUnitSize));
}

int32_t MecabDoubleArray::Base(int64_t index) const {
  if (index < 0 || index >= unit_count_) {
    return 0;
  }
  return static_cast<int32_t>(LoadUint32(units_, index * kDoubleArrayUnitSize));
}

uint32_t MecabDoubleArray::Check(int64_t index) const {
  if (index < 0 || index >= unit_count_) {
    // No traversal state can equal this, so an out-of-range slot never matches.
    return 0xFFFFFFFFu;
  }
  return LoadUint32(units_, index * kDoubleArrayUnitSize + 4);
}

std::vector<MecabDoubleArray::Match> MecabDoubleArray::CommonPrefixSearch(
    absl::string_view key) const {
  // Traversal carries the parent's base value, not its index; see the class
  // comment. An out-of-range slot reads back a `check` no base can equal, so
  // the bounds checks here cost one comparison and remove the need for any
  // other validation of the array.
  std::vector<Match> matches;
  int32_t base = Base(0);
  for (int i = 0; i < static_cast<int>(key.size()); ++i) {
    const auto byte = static_cast<unsigned char>(key[i]);
    const int64_t child = static_cast<int64_t>(base) + byte + 1;
    if (Check(child) != static_cast<uint32_t>(base)) break;
    base = Base(child);

    const int32_t terminal = Base(base);
    if (Check(base) == static_cast<uint32_t>(base) && terminal < 0) {
      const int value = -terminal - 1;
      matches.push_back({/*token_count=*/value & 0xFF,
                         /*token_index=*/value >> 8,
                         /*length=*/i + 1});
    }
  }
  return matches;
}

MecabTokenTable::MecabTokenTable(absl::string_view tokens, int size,
                                 absl::string_view features)
    : tokens_(tokens), size_(size), features_(features) {}

absl::StatusOr<MecabTokenTable> MecabTokenTable::Create(
    absl::string_view tokens, absl::string_view features) {
  if (tokens.size() % kTokenSize != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "MeCab token table is ", tokens.size(),
        " bytes, which is not a whole number of ", kTokenSize, " byte rows"));
  }
  return MecabTokenTable(tokens, static_cast<int>(tokens.size() / kTokenSize),
                         features);
}

int MecabTokenTable::Size() const { return size_; }

MecabToken MecabTokenTable::Token(int index) const {
  const int64_t offset = static_cast<int64_t>(index) * kTokenSize;
  MecabToken token;
  token.left_id = LoadUint16(tokens_, offset);
  token.right_id = LoadUint16(tokens_, offset + 2);
  token.pos_id = LoadUint16(tokens_, offset + 4);
  token.word_cost = static_cast<int16_t>(LoadUint16(tokens_, offset + 6));

  const uint32_t feature = LoadUint32(tokens_, offset + 8);
  if (feature < features_.size()) {
    const absl::string_view rest = features_.substr(feature);
    token.feature = rest.substr(0, rest.find('\0'));
  }
  return token;
}

MecabConnectionMatrix::MecabConnectionMatrix(absl::string_view costs,
                                             int left_context_count,
                                             int right_context_count)
    : costs_(costs),
      left_context_count_(left_context_count),
      right_context_count_(right_context_count) {}

absl::StatusOr<MecabConnectionMatrix> MecabConnectionMatrix::Create(
    absl::string_view matrix) {
  if (matrix.size() < 4) {
    return absl::InvalidArgumentError(
        absl::StrCat("MeCab connection matrix is ", matrix.size(),
                     " bytes, too short to hold its own size header"));
  }
  // MeCab writes `lsize` then `rsize`, and both names are the reverse of what
  // they count. `lsize` is the stride `Connector::cost` multiplies the *left*
  // context id by, so it is how many *right* context ids there are.
  const int right_count = LoadUint16(matrix, 0);
  const int left_count = LoadUint16(matrix, 2);
  const int64_t needed = 4 + 2 * static_cast<int64_t>(left_count) * right_count;
  if (static_cast<int64_t>(matrix.size()) < needed) {
    return absl::InvalidArgumentError(absl::StrCat(
        "MeCab connection matrix is ", matrix.size(), " bytes, but its ",
        right_count, " by ", left_count, " header needs ", needed));
  }
  return MecabConnectionMatrix(matrix.substr(4), left_count, right_count);
}

int MecabConnectionMatrix::LeftContextCount() const {
  return left_context_count_;
}

int MecabConnectionMatrix::RightContextCount() const {
  return right_context_count_;
}

int MecabConnectionMatrix::Cost(int right_id, int left_id) const {
  if (right_id < 0 || right_id >= right_context_count_ || left_id < 0 ||
      left_id >= left_context_count_) {
    return 0;
  }
  // `Connector::cost` reads `matrix[left->rcAttr + lsize * right->lcAttr]`, so
  // the left node's right context is the fast axis. Getting this backwards is
  // invisible in unidic-lite, where the matrix is square, but it segments the
  // language wrong.
  const int64_t entry =
      static_cast<int64_t>(left_id) * right_context_count_ + right_id;
  return static_cast<int16_t>(LoadUint16(costs_, entry * 2));
}

MecabCharProperty::MecabCharProperty(absl::string_view names,
                                     int category_count,
                                     absl::string_view table,
                                     int code_point_count)
    : names_(names),
      category_count_(category_count),
      table_(table),
      code_point_count_(code_point_count) {}

absl::StatusOr<MecabCharProperty> MecabCharProperty::Create(
    absl::string_view data) {
  if (data.size() < 4) {
    return absl::InvalidArgumentError(
        absl::StrCat("MeCab character property table is ", data.size(),
                     " bytes, too short to hold its category count"));
  }
  const uint32_t categories = LoadUint32(data, 0);
  const int64_t names_size =
      static_cast<int64_t>(categories) * kCategoryNameSize;
  if (categories == 0 ||
      4 + names_size + 4 > static_cast<int64_t>(data.size())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "MeCab character property table claims ", categories,
        " categories, which does not fit in ", data.size(), " bytes"));
  }
  const absl::string_view table = data.substr(4 + names_size);
  return MecabCharProperty(data.substr(4, names_size),
                           static_cast<int>(categories), table,
                           static_cast<int>(table.size() / 4));
}

int MecabCharProperty::CategoryCount() const { return category_count_; }

absl::string_view MecabCharProperty::CategoryName(int index) const {
  const absl::string_view name = names_.substr(
      static_cast<int64_t>(index) * kCategoryNameSize, kCategoryNameSize);
  return name.substr(0, name.find('\0'));
}

MecabCharProperty::CharInfo MecabCharProperty::Info(char32_t code_point) const {
  // MeCab's table stops at the end of the basic multilingual plane, and MeCab
  // reads past it rather than checking. The record for U+0000 is the default
  // category, which is what reading past it was meant to mean.
  const int64_t entry =
      static_cast<int64_t>(code_point) < code_point_count_ ? code_point : 0;
  const uint32_t packed = LoadUint32(table_, entry * 4);

  // MeCab declares these as one bitfield in the order `type`, `default_type`,
  // `length`, `group`, `invoke`, so `group` is bit 30 and `invoke` is bit 31 --
  // the reverse of the order the two are usually spoken about in.
  CharInfo info;
  info.types = packed & 0x3FFFF;
  info.default_type = static_cast<int>((packed >> 18) & 0xFF);
  info.length = static_cast<int>((packed >> 26) & 0xF);
  info.group = ((packed >> 30) & 1) != 0;
  info.invoke = ((packed >> 31) & 1) != 0;
  return info;
}

MecabCharProperty::DecodedChar MecabCharProperty::InfoAt(absl::string_view text,
                                                         int offset) const {
  const DecodedCodePoint decoded = DecodeMecabChar(text, offset);
  return {Info(decoded.code_point), decoded.length};
}

}  // namespace litert::omni::tts::kokoro
