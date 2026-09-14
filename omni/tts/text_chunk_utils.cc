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

#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::omni::tts {

DelimiterMatch FindFirstDelimiter(absl::string_view text,
                                  const TextChunkConfig& config) {
  DelimiterMatch result;
  for (const auto& delim : config.delimiters) {
    if (delim.empty()) continue;
    size_t found_pos = text.find(delim);
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
