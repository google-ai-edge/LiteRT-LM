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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TEXT_CHUNK_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TEXT_CHUNK_UTILS_H_

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::omni::tts {

// Represents the result of searching for a delimiter match in text.
struct DelimiterMatch {
  // The character index position where the delimiter starts in the text,
  // or absl::string_view::npos if no delimiter was found.
  size_t pos = absl::string_view::npos;

  // The length in characters of the matched delimiter string.
  size_t length = 0;

  // Returns true if a valid delimiter match was found.
  bool found() const { return pos != absl::string_view::npos; }
};

// Pure configuration container specifying delimiter and buffer chunking
// options.
struct TextChunkConfig {
  // List of delimiter strings used to split text into synthesis chunks.
  // Default delimiters: sentence terminals (".", "?", "!"), clause break (";"),
  // and newline ("\n").
  std::vector<std::string> delimiters = {
      ".", "?", "!", ";", "\n", "，", "。", "！", "？",
  };

  // Whether to include the matching delimiter in the extracted output chunk.
  // If true, splitting "Hello. World" at "." yields "Hello.".
  // If false, it yields "Hello" and strips "." from the buffer.
  bool include_delimiter = true;

  // Optional maximum buffer size in bytes/characters.
  // If non-zero and buffer length reaches or exceeds this value, a chunk will
  // be triggered even if no delimiter is found. Set to 0 to disable.
  size_t max_buffer_size = 0;
};

// Represents an extracted text chunk string view and the position index
// for the next chunk extraction.
struct TextChunk {
  // View of the extracted text chunk backed by the input text.
  absl::string_view chunk;

  // Character offset index in the input text where the next chunk extraction
  // should start.
  size_t next_start_index = 0;
};

// Finds the earliest matching delimiter in `text` based on `config`.
//
// args
// - text: Input text string view to search.
// - config: Text chunk configuration specifying delimiters to match.
//
// returns
// - DelimiterMatch containing position and length of earliest match, or pos =
//   string_view::npos if no delimiter matched.
DelimiterMatch FindFirstDelimiter(absl::string_view text,
                                  const TextChunkConfig& config);

// Determines whether `buffer` contains text ready to be scheduled as an output
// chunk.
//
// args
// - buffer: The accumulated raw text buffer to evaluate.
// - is_finished: True if the input stream/file has reached end-of-file or
//   finished.
// - config: Text chunk configuration specifying delimiters and max buffer
//   limits.
//
// returns
// - True if buffer contains a matching delimiter, exceeds max_buffer_size,
//   or stream is finished with non-empty buffer.
bool ShouldSchedule(absl::string_view buffer, bool is_finished,
                    const TextChunkConfig& config);

// Extracts the next text chunk from `text` starting at `start_index`.
//
// args
// - text: Input text string view from which text is extracted.
// - start_index: Character offset in `text` to start searching from.
// - is_finished: True if the input stream/file has reached end-of-file or
//   finished.
// - config: Text chunk configuration controlling splitting and inclusion rules.
//
// returns
// - An optional TextChunk containing the extracted chunk string_view and
//   next_start_index, or std::nullopt if no chunk can be formed from
//   `start_index`.
std::optional<TextChunk> ExtractNextChunk(absl::string_view text,
                                          size_t start_index, bool is_finished,
                                          const TextChunkConfig& config);

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TEXT_CHUNK_UTILS_H_
