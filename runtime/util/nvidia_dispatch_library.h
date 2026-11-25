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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_NVIDIA_DISPATCH_LIBRARY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_NVIDIA_DISPATCH_LIBRARY_H_

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/file_util.h"

namespace litert::lm {

// The file names of the LiteRT dispatch library of NVIDIA's TensorRT-RTX
// backend, on Linux and on Windows.
inline constexpr absl::string_view kNvidiaDispatchLibraryFileNames[] = {
    "libLiteRtDispatch_Nvidia.so",
    "libLiteRtDispatch_Nvidia.dll",
};

// Whether `dispatch_lib_dir` holds NVIDIA's dispatch library. An executor on
// Backend::NPU that takes its dispatch library from that directory runs on
// NVIDIA's TensorRT-RTX backend. This only looks at the files, so it can be
// asked before LiteRT has loaded the library.
inline bool HasNvidiaDispatchLibrary(absl::string_view dispatch_lib_dir) {
  for (absl::string_view file_name : kNvidiaDispatchLibraryFileNames) {
    const absl::StatusOr<std::string> path =
        JoinPath(dispatch_lib_dir, file_name);
    if (path.ok() && FileExists(*path)) {
      return true;
    }
  }
  return false;
}

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_NVIDIA_DISPATCH_LIBRARY_H_
