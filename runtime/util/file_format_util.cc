// Copyright 2025 The ODML Authors.
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

#include "runtime/util/file_format_util.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  //NOLINT

namespace litert::lm {

namespace {

constexpr size_t kMaxMagicSignatureLength = 8;

}  // namespace

absl::StatusOr<FileFormat> GetFileFormatFromFileContents(
    absl::string_view contents) {
  absl::string_view header = contents.substr(0, kMaxMagicSignatureLength);
  if (absl::StrContains(header, "TFL3")) {
    return FileFormat::TFLITE;
  } else if (absl::StrContains(header, "PK")) {
    return FileFormat::TASK;
  } else if (absl::StartsWith(header, "LITERTLM")) {
    return FileFormat::LITERT_LM;
  }
  return absl::InvalidArgumentError("Unsupported or unknown file format.");
}

absl::StatusOr<FileFormat> GetFileFormatFromPath(absl::string_view model_path) {
  if (absl::EndsWith(model_path, ".tflite")) {
    return FileFormat::TFLITE;
  } else if (absl::EndsWith(model_path, ".task")) {
    return FileFormat::TASK;
  } else if (absl::EndsWith(model_path, ".litertlm")) {
    return FileFormat::LITERT_LM;
  }
  return absl::InvalidArgumentError("Unsupported or unknown file format.");
}

absl::StatusOr<FileFormat> GetFileFormat(
    absl::string_view model_path, std::shared_ptr<ScopedFile> scoped_file) {
  // Trust the extension of the file path, if it matches a known format.
  auto format_from_path = GetFileFormatFromPath(model_path);
  if (format_from_path.ok()) {
    return *format_from_path;
  }

  // Otherwise, inspect the file contents to determine the file format.
  if (scoped_file) {
    // Map the first few bytes of the file.
    ASSIGN_OR_RETURN(size_t file_size,  // NOLINT
                     ScopedFile::GetSize(scoped_file->file()));
    const uint64_t bytes_to_map = std::min(file_size, kMaxMagicSignatureLength);
    ASSIGN_OR_RETURN(auto mapped_file,  // NOLINT
                     MemoryMappedFile::Create(scoped_file->file(), /*offset=*/0,
                                              /*size=*/bytes_to_map));

    absl::string_view header(reinterpret_cast<const char*>(mapped_file->data()),
                             mapped_file->length());

    return GetFileFormatFromFileContents(header);
  }

  return absl::InvalidArgumentError("Unsupported or unknown file format.");
}

}  // namespace litert::lm
