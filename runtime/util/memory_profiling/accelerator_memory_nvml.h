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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_NVML_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_NVML_H_

// NVIDIA GPU memory of the current process on Linux, read through NVML (the
// NVIDIA Management Library, which ships with the driver and is what
// nvidia-smi uses). The second source of the Linux implementation of the
// platform contract in accelerator_memory.h, next to DRM fdinfo, which
// NVIDIA's driver does not report memory through. Internal to
// memory_profiling.
//
// NVML is loaded at run time with dlopen() and a handful of prototypes
// declared here, so that there is no build-time dependency on the CUDA
// toolkit and binaries run unchanged on machines without an NVIDIA driver.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace litert::lm {

// Framebuffer (VRAM) memory this process uses on one NVIDIA GPU.
struct NvmlDeviceMemory {
  // Device index as NVML enumerates it, 0-based.
  unsigned int index = 0;
  // Product name, e.g. "NVIDIA H100 80GB HBM3".
  std::string name;
  // Bytes of the GPU's framebuffer used by this process, whichever of its
  // compute (CUDA, OpenCL) and graphics (Vulkan, OpenGL) contexts hold them;
  // the figure nvidia-smi shows for the process.
  uint64_t used_bytes = 0;
  // Bytes of the GPU's framebuffer in use by all processes together, as
  // nvidia-smi's "Memory-Usage" column shows it. Diagnostic: what the GPU as
  // a whole is doing, independent of per-process attribution.
  uint64_t device_used_bytes = 0;
  // Number of processes the driver lists on this GPU that have no /proc entry
  // visible to this process, i.e. live in another PID namespace. Diagnostic:
  // inside a container this process's own use may be among them and missing
  // from `used_bytes`; see the limitation below.
  unsigned int foreign_process_count = 0;
};

// Reads this process's framebuffer use on every NVIDIA GPU in the machine.
// Returns std::nullopt when NVML is not available, i.e. no NVIDIA driver is
// installed or it cannot be initialized; otherwise one entry per GPU, with
// zero `used_bytes` for GPUs this process does not use. Loads and initializes
// NVML on first use and keeps it loaded.
//
// Known limitation: the driver records a process by its PID in the root PID
// namespace, so inside a container with its own PID namespace nothing matches
// getpid() and `used_bytes` is 0 even while the GPU is in use (the same reason
// nvidia-smi shows no processes inside Docker). `foreign_process_count` is
// then non-zero, and the first such reading logs a warning. Run with the
// host's PID namespace (e.g. `docker --pid=host`) to get per-process figures.
std::optional<std::vector<NvmlDeviceMemory>> ReadNvmlProcessMemory();

// One entry of an NVML per-process list, reduced to what is used here.
struct NvmlProcessEntry {
  unsigned int pid = 0;
  // Framebuffer bytes; NVML_VALUE_NOT_AVAILABLE already mapped to 0.
  uint64_t used_bytes = 0;
};

// Framebuffer bytes `pid` uses on a GPU, given the GPU's compute and graphics
// process lists. NVML reports a process's whole framebuffer use in every list
// the process appears in: one with both a compute and a graphics context
// (nvidia-smi type "C+G", e.g. OpenCL plus EGL) is in both lists with the
// same figure, which nvidia-smi shows once. So the entries for `pid` are not
// summed; the larger one stands. Exposed for testing.
uint64_t NvmlProcessBytes(unsigned int pid,
                          const std::vector<NvmlProcessEntry>& compute,
                          const std::vector<NvmlProcessEntry>& graphics);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_ACCELERATOR_MEMORY_NVML_H_
