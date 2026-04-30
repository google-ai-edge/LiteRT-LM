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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_FILE_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_FILE_UTIL_H_

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

// Joins two file paths.
//
// TODO: b/419286976 - Support Windows. This currently assumes POSIX paths.
absl::StatusOr<std::string> JoinPath(absl::string_view path1,
                                     absl::string_view path2);

// Returns the basename of a file path.
//
// TODO: b/419286976 - Support Windows. This currently assumes POSIX paths.
absl::string_view Basename(absl::string_view path);

// Returns the dirname of a file path.
//
// TODO: b/419286976 - Support Windows. This currently assumes POSIX paths.
absl::string_view Dirname(absl::string_view path);

// Returns a unique identifier for the file based on its metadata (timestamp
// + file size).
absl::StatusOr<std::string> GetFileCacheIdentifier(absl::string_view path);

// Returns true if the file exists.
bool FileExists(absl::string_view path);

// Deletes all cache files for a given model and suffix.
// Any file matching the pattern (model_basename + * + suffix) or
// (model_basename + suffix + *) will be deleted.
absl::StatusOr<int> DeleteStaleCaches(absl::string_view cache_dir,
                                      absl::string_view model_basename,
                                      absl::string_view suffix);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_FILE_UTIL_H_
