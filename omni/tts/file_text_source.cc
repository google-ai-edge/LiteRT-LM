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

#include "omni/tts/file_text_source.h"

#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/tts/text_chunk_utils.h"

namespace litert::omni::tts {

absl::StatusOr<std::unique_ptr<FileTextSource>> FileTextSource::Create(
    std::string file_path, TextChunkConfig config) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    return absl::NotFoundError(
        absl::StrCat("Failed to open file: ", file_path));
  }
  file.close();
  return std::make_unique<FileTextSource>(std::move(file_path),
                                          std::move(config));
}

FileTextSource::FileTextSource(std::string file_path, TextChunkConfig config)
    : file_path_(std::move(file_path)), config_(std::move(config)) {
  file_stream_.open(file_path_);
  if (!file_stream_.is_open()) {
    is_finished_ = true;
  } else {
    ReadFromFileIfNeeded();
  }
}

void FileTextSource::ResetInternal() {
  buffer_.clear();
  buffer_start_index_ = 0;
  if (file_stream_.is_open()) {
    file_stream_.clear();
    file_stream_.seekg(0, std::ios::beg);
  } else {
    file_stream_.open(file_path_);
  }
  is_finished_ = !file_stream_.is_open();
  if (!is_finished_) {
    ReadFromFileIfNeeded();
  }
}

void FileTextSource::ReadFromFileIfNeeded() {
  if (is_finished_) return;
  char read_buf[1024];
  while (!is_finished_) {
    absl::string_view remaining =
        absl::string_view(buffer_).substr(buffer_start_index_);
    if (ShouldSchedule(remaining, /*is_finished=*/false, config_)) {
      break;
    }
    if (!file_stream_.is_open() || file_stream_.eof()) {
      is_finished_ = true;
      break;
    }
    file_stream_.read(read_buf, sizeof(read_buf));
    std::streamsize bytes_read = file_stream_.gcount();
    if (bytes_read > 0) {
      buffer_.append(read_buf, bytes_read);
    }
    if (file_stream_.eof() || file_stream_.fail()) {
      is_finished_ = true;
      break;
    }
  }
}

bool FileTextSource::NeedScheduleInternal() const {
  absl::string_view remaining =
      absl::string_view(buffer_).substr(buffer_start_index_);
  return ShouldSchedule(remaining, is_finished_, config_);
}

absl::Status FileTextSource::ScheduleInternal() {
  SetState(State::kRunning);
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
  auto chunk =
      ExtractNextChunk(buffer_, buffer_start_index_, is_finished_, config_);
  if (chunk.has_value()) {
    PushOutput(std::string(chunk->chunk));
    buffer_start_index_ = chunk->next_start_index;
    if (buffer_start_index_ >= buffer_.size()) {
      buffer_.clear();
      buffer_start_index_ = 0;
    }
    ReadFromFileIfNeeded();
  }
  return absl::OkStatus();
}

}  // namespace litert::omni::tts
