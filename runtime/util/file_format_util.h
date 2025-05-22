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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_FILE_FORMAT_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_FILE_FORMAT_UTIL_H_

#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

// Infer the file format from the contents of the file, or return an error if
// the file format cannot be inferred.
absl::StatusOr<FileFormat> GetFileFormatFromFileContents(
    absl::string_view contents);

// Infer the file format from the file path, or return an error if the file
// format cannot be inferred.
absl::StatusOr<FileFormat> GetFileFormatFromPath(absl::string_view model_path);

// Infer the file format from the file path or the file contents, or return an
// error if the file format cannot be inferred. This will prefer the model path
// if provided.
absl::StatusOr<FileFormat> GetFileFormat(
    absl::string_view model_path,
    std::shared_ptr<ScopedFile> scoped_file = nullptr);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_FILE_FORMAT_UTIL_H_
