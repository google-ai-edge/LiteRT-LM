/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_MEDIAPIPE_TASKS_CC_METADATA_UTILS_ZIP_UTILS_H_
#define THIRD_PARTY_MEDIAPIPE_TASKS_CC_METADATA_UTILS_ZIP_UTILS_H_

#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/util/external_file.pb.h"

namespace litert::lm {

// Extract files from the zip file.
// Input: Pointer and length of the zip file in memory.
// Outputs: A map with the filename as key and a pointer to the file contents
// as value. The file contents returned by this function are only guaranteed to
// stay valid while buffer_data is alive.
absl::Status ExtractFilesfromZipFile(
    const char* buffer_data, const size_t buffer_size,
    absl::flat_hash_map<std::string, absl::string_view>* files);

// Set the ExternalFile object by file_content in memory. By default,
// `is_copy=false` which means to set `file_pointer_meta` in ExternalFile which
// is the pointer points to location of a file in memory. Otherwise, if
// `is_copy=true`, copy the memory into `file_content` in ExternalFile.
void SetExternalFile(const absl::string_view& file_content,
                     proto::ExternalFile* model_file, bool is_copy = false);

}  // namespace litert::lm

#endif  // THIRD_PARTY_MEDIAPIPE_TASKS_CC_METADATA_UTILS_ZIP_UTILS_H_
