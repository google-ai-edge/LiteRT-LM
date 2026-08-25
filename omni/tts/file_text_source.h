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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_FILE_TEXT_SOURCE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_FILE_TEXT_SOURCE_H_

#include <cstddef>
#include <fstream>
#include <memory>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "omni/tts/text_chunk_utils.h"
#include "omni/tts/text_source.h"

namespace litert::omni::tts {

// TextSource implementation that reads input text from a file on disk.
//
// It reads data incrementally from `file_path` into an internal buffer as
// needed, splits the text into chunks according to `config`, and pushes the
// chunks into the output queue for speech synthesis.
class FileTextSource : public TextSource {
 public:
  // Factory method to create a FileTextSource instance.
  //
  // args
  // - file_path: Absolute or relative path to the text file on disk.
  // - config: Text chunk configuration specifying delimiters and buffer limits.
  //
  // returns
  // - StatusOr containing unique_ptr to FileTextSource, or NotFoundError
  //   if the file cannot be opened.
  static absl::StatusOr<std::unique_ptr<FileTextSource>> Create(
      std::string file_path, TextChunkConfig config = {});

  // Constructs a FileTextSource.
  //
  // args
  // - file_path: Path to the text file on disk.
  // - config: Text chunk configuration options for chunking.
  explicit FileTextSource(std::string file_path, TextChunkConfig config = {});

  ~FileTextSource() override = default;

  // Accessor for the active text chunk configuration.
  const TextChunkConfig& config() const { return config_; }

 protected:
  // Resets internal state and re-opens the file from the beginning.
  void ResetInternal() override;

  // Determines if ScheduleInternal should be called based on buffer state.
  // Side-effect free const method.
  bool NeedScheduleInternal() const override;

  // Schedules file reading and extracts text chunks into the stage output
  // deque.
  absl::Status ScheduleInternal() override;

 private:
  // Non-const helper method that reads blocks from file_stream_ into buffer_
  // until a delimiter is present in buffer_, max buffer limit is exceeded, or
  // EOF is reached.
  void ReadFromFileIfNeeded();

  // Path to the text file on disk.
  std::string file_path_;

  // Text chunk configuration for text chunking.
  TextChunkConfig config_;

  // File input stream for reading text data.
  std::ifstream file_stream_;

  // Internal buffer accumulating un-emitted text read from file.
  std::string buffer_;

  // Read offset index within buffer_.
  size_t buffer_start_index_ = 0;

  // True if end-of-file has been reached or stream open failed.
  bool is_finished_ = false;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_FILE_TEXT_SOURCE_H_
