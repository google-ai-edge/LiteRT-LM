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

#include <cstddef>
#include <optional>
#include <string>

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
  while (curr_index < text.size() ||
         (is_finished && curr_index == text.size())) {
    absl::string_view sub = text.substr(curr_index);
    if (sub.empty() && !is_finished) {
      break;
    }

    auto match = FindFirstDelimiter(sub, config);
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
      if (!chunk.empty()) {
        return TextChunk{.chunk = chunk, .next_start_index = next_index};
      }
      // If chunk is empty (e.g. leading delimiter stripped), continue loop
      // to find the next chunk starting at next_index.
      curr_index = next_index;
      continue;
    }

    if (config.max_buffer_size > 0 && sub.size() >= config.max_buffer_size) {
      absl::string_view chunk = sub.substr(0, config.max_buffer_size);
      size_t next_index = curr_index + config.max_buffer_size;
      if (!chunk.empty()) {
        return TextChunk{.chunk = chunk, .next_start_index = next_index};
      }
      curr_index = next_index;
      continue;
    }

    if (is_finished) {
      absl::string_view chunk = sub;
      size_t next_index = text.size();
      if (!chunk.empty()) {
        return TextChunk{.chunk = chunk, .next_start_index = next_index};
      }
      return std::nullopt;
    }

    break;
  }
  return std::nullopt;
}

}  // namespace litert::omni::tts
