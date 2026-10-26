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

// Fallback implementation of the platform contract in host_memory.h for
// operating systems without a host_memory_<os>.cc. Nothing is measured.

#include "runtime/util/memory_profiling/host_memory.h"

namespace litert::lm {

HostMemoryUsage GetHostMemoryUsage() { return HostMemoryUsage(); }

}  // namespace litert::lm
