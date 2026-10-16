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

#include "omni/text2speech/kokoro/traditional_chinese_converter.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl

namespace litert::omni::tts::kokoro {
namespace {

// Returns the byte length of the UTF-8 character starting at `text[pos]`,
// clamped to the remaining text. Invalid lead bytes count as one byte.
size_t Utf8CharLength(absl::string_view text, size_t pos) {
  const auto lead = static_cast<uint8_t>(text[pos]);
  size_t length = 1;
  if ((lead & 0xE0) == 0xC0) {
    length = 2;
  } else if ((lead & 0xF0) == 0xE0) {
    length = 3;
  } else if ((lead & 0xF8) == 0xF0) {
    length = 4;
  }
  return std::min(length, text.size() - pos);
}

// Returns the number of UTF-8 characters in `text`.
size_t Utf8CharCount(absl::string_view text) {
  size_t count = 0;
  for (size_t pos = 0; pos < text.size(); pos += Utf8CharLength(text, pos)) {
    ++count;
  }
  return count;
}

}  // namespace

absl::StatusOr<std::unique_ptr<TraditionalChineseConverter>>
TraditionalChineseConverter::Create(absl::string_view dictionary) {
  auto converter = absl::WrapUnique(new TraditionalChineseConverter());
  converter->dictionary_ = std::string(dictionary);

  int line_number = 0;
  for (absl::string_view line : absl::StrSplit(converter->dictionary_, '\n')) {
    ++line_number;
    absl::ConsumeSuffix(&line, "\r");
    if (line.empty() || line.front() == '#') continue;
    const size_t tab = line.find('\t');
    if (tab == absl::string_view::npos || tab == 0 || tab + 1 == line.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Traditional Chinese dictionary line ", line_number,
                       ": expected `traditional<TAB>simplified`"));
    }
    const absl::string_view key = line.substr(0, tab);
    absl::string_view value = line.substr(tab + 1);
    value = value.substr(0, value.find(' '));
    if (value.empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Traditional Chinese dictionary line ", line_number,
                       ": empty value"));
    }
    // The first entry for a key wins, as phrases are listed before characters.
    if (converter->entries_.emplace(key, value).second) {
      converter->max_key_chars_ =
          std::max(converter->max_key_chars_, Utf8CharCount(key));
    }
  }
  if (converter->entries_.empty()) {
    return absl::InvalidArgumentError(
        "Traditional Chinese dictionary has no entries");
  }
  return converter;
}

std::string TraditionalChineseConverter::Convert(absl::string_view text) const {
  std::string out;
  out.reserve(text.size());
  // Byte offsets of the next `max_key_chars_` character boundaries.
  std::vector<size_t> ends;
  ends.reserve(max_key_chars_);
  size_t pos = 0;
  while (pos < text.size()) {
    ends.clear();
    for (size_t end = pos; end < text.size() && ends.size() < max_key_chars_;) {
      end += Utf8CharLength(text, end);
      ends.push_back(end);
    }
    bool matched = false;
    for (auto it = ends.rbegin(); it != ends.rend(); ++it) {
      if (const auto entry = entries_.find(text.substr(pos, *it - pos));
          entry != entries_.end()) {
        out.append(entry->second);
        pos = *it;
        matched = true;
        break;
      }
    }
    if (!matched) {
      out.append(text.substr(pos, ends.front() - pos));
      pos = ends.front();
    }
  }
  return out;
}

size_t TraditionalChineseConverter::EntryCount() const {
  return entries_.size();
}

}  // namespace litert::omni::tts::kokoro
