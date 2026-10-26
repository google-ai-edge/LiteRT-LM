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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_HOST_MEMORY_LINUX_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_HOST_MEMORY_LINUX_H_

// Building block of the Linux / Android implementation of the platform
// contract in host_memory.h. Internal to memory_profiling: exposed only so
// that host_memory_linux_test.cc can exercise it on canned input.

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/host_memory.h"

namespace litert::lm {

// Builds a HostMemoryUsage from the contents of /proc/<pid>/status. Fields
// whose lines are missing are left unset; see host_memory_linux.cc for which
// lines feed which field.
HostMemoryUsage ParseProcStatus(absl::string_view status);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_HOST_MEMORY_LINUX_H_
