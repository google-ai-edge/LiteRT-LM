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

// Android implementation of the platform contract in
// accelerator_memory_monitor.h.
//
// What is measured: the dma-buf heap backed buffers the process holds open,
// read from /proc/self/fdinfo (accelerator_memory_dmabuf.h). In practice these
// are AHardwareBuffers, which is how LiteRT-LM's GPU and NPU backends allocate
// accelerator memory on Android. The breakdown is per exporter, i.e. per
// dma-buf heap ("system", "system-uncached", vendor heaps).
//
// What is not seen: memory the GPU driver allocates from its own pool (Mali
// kbase, Adreno KGSL) does not go through dma-buf and is invisible here, so the
// result is a lower bound on accelerator memory rather than a complete measure.

#include <optional>

#include "runtime/util/memory_profiling/accelerator_memory_dmabuf.h"
#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"

namespace litert::lm {

std::optional<AcceleratorMemoryUsage> GetProcessAcceleratorMemoryUsage() {
  return AcceleratorMemoryUsageFromDmaBuf(ReadDmaBufSummary("/proc/self"));
}

}  // namespace litert::lm
