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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_FDINFO_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_FDINFO_H_

// Reader for the kernel's per-file-descriptor information,
// /proc/<pid>/fdinfo/<fd> (Linux and Android). Each file is a list of
// "<key>:\t<value>" lines; besides the generic ones (pos, flags, mnt_id, ino)
// a driver can add its own, which is how DMA-BUF buffers (size, exp_name,
// name) and DRM GPU clients (drm-driver, drm-client-id, drm-total-<region>,
// ...) report per-process memory. Internal to memory_profiling: the per-OS
// accelerator memory implementations are built on it.

#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

// The fields of one fdinfo file: key -> value, both with surrounding
// whitespace removed. Lines without a ':' are ignored; for a repeated key the
// last line wins.
using FdinfoFields = absl::flat_hash_map<std::string, std::string>;

// Parses the contents of one fdinfo file.
FdinfoFields ParseFdinfo(absl::string_view contents);

// Calls `fn(fd_name, fields)` for every file in `<proc_self_dir>/fdinfo`, in
// directory order. `fd_name` is the file name, i.e. the descriptor number.
// `proc_self_dir` is "/proc/self" in production; tests point it at a directory
// of fake fdinfo files. Does nothing if the directory cannot be read.
void ForEachFdinfo(
    absl::string_view proc_self_dir,
    absl::FunctionRef<void(absl::string_view fd_name, const FdinfoFields&)> fn);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_FDINFO_H_
