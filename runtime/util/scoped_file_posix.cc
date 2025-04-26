// Copyright 2024 The ODML Authors.
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

#include <fcntl.h>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

// static
absl::StatusOr<ScopedFile> ScopedFile::Open(absl::string_view path) {
  int fd = open(path.data(), O_RDONLY);
  RET_CHECK_GE(fd, 0) << "open() failed: " << path;
  return ScopedFile(fd);
}

// static
absl::StatusOr<ScopedFile> ScopedFile::OpenWritable(absl::string_view path) {
  int fd = open(path.data(), O_RDWR);
  RET_CHECK_GE(fd, 0) << "open() failed: " << path;
  return ScopedFile(fd);
}

// static
void ScopedFile::CloseFile(int file) { close(file); }

}  // namespace litert::lm
