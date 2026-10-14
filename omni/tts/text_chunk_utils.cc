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

#include "omni/tts/text_chunk_utils.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::omni::tts {

namespace {

// Words that are written with a trailing period but do not end a sentence.
// Kept short: the checks in IsSentenceFinalPeriod already cover dotted
// acronyms (EE.UU., e.g., a.m.), single-letter initials and periods followed
// by a lowercase word or a digit.
bool IsCommonAbbreviation(absl::string_view word) {
  static const absl::NoDestructor<absl::flat_hash_set<absl::string_view>>
      kAbbreviations({
          // English.
          "Mr",
          "Mrs",
          "Ms",
          "Dr",
          "Drs",
          "Prof",
          "St",
          "Jr",
          "Sr",
          "Mt",
          "vs",
          "etc",
          "Fig",
          "fig",
          "Vol",
          "vol",
          "No",
          "no",
          "Corp",
          "Inc",
          "Ltd",
          "Co",
          "ext",
          "approx",
          "Ave",
          "Blvd",
          "Dept",
          "Jan",
          "Feb",
          "Mar",
          "Apr",
          "Jun",
          "Jul",
          "Aug",
          "Sep",
          "Sept",
          "Oct",
          "Nov",
          "Dec",
          "Mon",
          "Tue",
          "Wed",
          "Thu",
          "Fri",
          "Sat",
          "Sun",
          // Spanish / Portuguese.
          "Sra",
          "Srta",
          "Dra",
          "Avda",
          "Av",
          "pág",
          "pag",
          "cap",
          "apdo",
          "Ing",
          "Lic",
          "oct",
          // Italian / French.
          "dott",
          "Dott",
          "sig",
          "Sig",
          "Mme",
          "Mlle",
      });
  return kAbbreviations->contains(word);
}

// Returns true if the ASCII period at `text[pos]` ends a sentence rather than
// belonging to a number (1.500, 12.5), an abbreviation (Dr., EE.UU., a.m.)
// or an initial. Splitting a chunk inside those would hand the text
// normalizer and the phonemizer half a token and insert a chunk pause there.
// A period at the very end of `text` (the next character may still be
// streaming in) counts as final after an ordinary word, which keeps streaming
// latency unchanged, but not after a number or an abbreviation.
bool IsSentenceFinalPeriod(absl::string_view text, size_t pos) {
  // The word before the period.
  size_t start = pos;
  while (start > 0 &&
         !absl::ascii_isspace(static_cast<unsigned char>(text[start - 1]))) {
    --start;
  }
  const absl::string_view word = text.substr(start, pos - start);
  if (absl::StrContains(word, '.')) return false;  // EE.UU.
  if (word.size() == 1 &&
      absl::ascii_isalpha(static_cast<unsigned char>(word[0]))) {
    return false;  // An initial: "J. Smith".
  }
  if (IsCommonAbbreviation(word)) return false;

  size_t next = pos + 1;
  // Closing quotes and brackets may sit between the period and the space.
  while (next < text.size() && absl::StrContains("\"')]", text[next])) {
    ++next;
  }
  if (next >= text.size()) {
    // End of the buffer: a digit before the period may be followed by more
    // digits ("$6." + "50").
    return !word.empty() &&
           !absl::ascii_isdigit(static_cast<unsigned char>(word.back()));
  }
  if (!absl::ascii_isspace(static_cast<unsigned char>(text[next]))) {
    return false;
  }
  // A sentence does not continue with a lowercase word or a number.
  size_t after = next;
  while (after < text.size() &&
         absl::ascii_isspace(static_cast<unsigned char>(text[after]))) {
    ++after;
  }
  return after >= text.size() ||
         !(absl::ascii_islower(static_cast<unsigned char>(text[after])) ||
           absl::ascii_isdigit(static_cast<unsigned char>(text[after])));
}

// Returns the position of the first occurrence of `delim` in `text` that
// counts as a chunk boundary, or npos.
size_t FindBoundary(absl::string_view text, absl::string_view delim) {
  size_t found_pos = text.find(delim);
  if (delim != ".") return found_pos;
  while (found_pos != absl::string_view::npos &&
         !IsSentenceFinalPeriod(text, found_pos)) {
    found_pos = text.find(delim, found_pos + 1);
  }
  return found_pos;
}

}  // namespace

DelimiterMatch FindFirstDelimiter(absl::string_view text,
                                  const TextChunkConfig& config) {
  DelimiterMatch result;
  for (const auto& delim : config.delimiters) {
    if (delim.empty()) continue;
    size_t found_pos = FindBoundary(text, delim);
    if (found_pos != absl::string_view::npos) {
      // Pick the delimiter that appears at the earliest position in `text`.
      // If two delimiters start at the exact same position, pick the longer
      // delimiter.
      if (result.pos == absl::string_view::npos || found_pos < result.pos ||
          (found_pos == result.pos && delim.length() > result.length)) {
        result.pos = found_pos;
        result.length = delim.length();
      }
    }
  }
  return result;
}

bool ShouldSchedule(absl::string_view buffer, bool is_finished,
                    const TextChunkConfig& config) {
  // If buffer has no non-whitespace characters, there is nothing to schedule.
  if (absl::StripAsciiWhitespace(buffer).empty()) {
    return false;
  }

  // 1. Trigger if buffer contains a matching delimiter.
  if (FindFirstDelimiter(buffer, config).found()) {
    return true;
  }
  // 2. Trigger if max buffer size is enabled and buffer length exceeds it.
  if (config.max_buffer_size > 0 && buffer.size() >= config.max_buffer_size) {
    return true;
  }

  // 3. Trigger if text stream/file is finished and buffer has trailing text.
  if (is_finished && !buffer.empty()) {
    return true;
  }

  return false;
}

std::optional<TextChunk> ExtractNextChunk(absl::string_view text,
                                          size_t start_index, bool is_finished,
                                          const TextChunkConfig& config) {
  size_t curr_index = start_index;
  while (curr_index < text.size()) {
    absl::string_view sub = text.substr(curr_index);
    if (sub.empty()) {
      break;
    }

    // Determine the delimiter search window. If max_buffer_size is configured
    // and buffer exceeds it, constrain the search window to max_buffer_size
    // so we don't ignore chunking when a delimiter appears far ahead.
    absl::string_view search_window = sub;
    if (config.max_buffer_size > 0 && sub.size() >= config.max_buffer_size) {
      search_window = sub.substr(0, config.max_buffer_size);
    }

    auto match = FindFirstDelimiter(search_window, config);
    if (match.found()) {
      absl::string_view chunk;
      size_t next_index = curr_index;
      if (config.include_delimiter) {
        size_t chunk_len = match.pos + match.length;
        chunk = sub.substr(0, chunk_len);
        next_index += chunk_len;
      } else {
        chunk = sub.substr(0, match.pos);
        next_index += match.pos + match.length;
      }
      // If chunk has no non-whitespace characters (e.g. isolated '\n' or
      // spaces), advance past it and continue searching without emitting an
      // empty chunk.
      if (!absl::StripAsciiWhitespace(chunk).empty()) {
        return TextChunk{.chunk = chunk, .next_start_index = next_index};
      }
      curr_index = next_index;
      continue;
    }

    // Buffer reaches max_buffer_size with no delimiter in the search window.
    if (config.max_buffer_size > 0 && sub.size() >= config.max_buffer_size) {
      // Search backwards for a whitespace boundary to avoid cutting words in
      // half.
      size_t split_pos = absl::string_view::npos;
      for (size_t i = config.max_buffer_size; i > (config.max_buffer_size / 2);
           --i) {
        if (absl::ascii_isspace(sub[i - 1])) {
          split_pos = i - 1;
          break;
        }
      }

      size_t cut_len = config.max_buffer_size;
      size_t advance_len = config.max_buffer_size;
      if (split_pos != absl::string_view::npos) {
        cut_len = split_pos;
        advance_len = split_pos + 1;  // Advance past the whitespace boundary.
      } else {
        // If no whitespace found, ensure we don't cut across a multi-byte UTF-8
        // continuation byte. Guard `cut_len < sub.size()` to prevent reading
        // past the end of `sub`.
        while (cut_len > 0 && cut_len < sub.size() &&
               (static_cast<unsigned char>(sub[cut_len]) & 0xC0) == 0x80) {
          cut_len--;
        }
        advance_len = std::max<size_t>(1, cut_len);
      }

      absl::string_view chunk = sub.substr(0, cut_len);
      size_t next_index = curr_index + advance_len;
      if (!absl::StripAsciiWhitespace(chunk).empty()) {
        return TextChunk{.chunk = chunk, .next_start_index = next_index};
      }
      curr_index = next_index;
      continue;
    }

    if (is_finished) {
      absl::string_view chunk = sub;
      size_t next_index = text.size();
      if (!absl::StripAsciiWhitespace(chunk).empty()) {
        return TextChunk{.chunk = chunk, .next_start_index = next_index};
      }
      return std::nullopt;
    }

    break;
  }
  return std::nullopt;
}

}  // namespace litert::omni::tts
