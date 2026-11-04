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

#include "runtime/util/memory_profiling/accelerator_memory_nvml.h"

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl

namespace litert::lm {

namespace {

// The slice of the NVML C API used here, declared by hand so that no NVML
// header is needed. Names, signatures and struct layouts follow nvml.h of
// CUDA 11.x and later; the "_v2"/"_v3" suffixes pin the ABI versions these
// declarations match. NVML_VALUE_NOT_AVAILABLE and the struct layout of
// nvmlProcessInfo_t (pid, usedGpuMemory, gpuInstanceId, computeInstanceId)
// are part of that ABI. nvml.h spells the 64-bit fields `unsigned long long`;
// uint64_t has the same size and alignment on every Linux ABI, and the
// struct members are renamed to this code base's style, neither of which
// changes the layout.
using nvmlReturn_t = int;
constexpr nvmlReturn_t kNvmlSuccess = 0;
constexpr nvmlReturn_t kNvmlErrorInsufficientSize = 7;
constexpr uint64_t kNvmlValueNotAvailable = ~uint64_t{0};
constexpr unsigned int kNvmlDeviceNameBufferSize = 96;
using nvmlDevice_t = struct nvmlDevice_st*;
struct nvmlProcessInfo_t {
  unsigned int pid;
  uint64_t used_gpu_memory;
  unsigned int gpu_instance_id;
  unsigned int compute_instance_id;
};
struct nvmlMemory_t {
  uint64_t total;
  uint64_t free;
  uint64_t used;
};
using InitFn = nvmlReturn_t (*)();
using GetCountFn = nvmlReturn_t (*)(unsigned int*);
using GetHandleFn = nvmlReturn_t (*)(unsigned int, nvmlDevice_t*);
using GetNameFn = nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned int);
using GetMemoryInfoFn = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t*);
using GetProcessesFn = nvmlReturn_t (*)(nvmlDevice_t, unsigned int*,
                                        nvmlProcessInfo_t*);

// NVML, loaded once per process. Null function pointers mean "unavailable".
struct Nvml {
  GetCountFn get_count = nullptr;
  GetHandleFn get_handle = nullptr;
  GetNameFn get_name = nullptr;
  GetMemoryInfoFn get_memory_info = nullptr;
  GetProcessesFn get_compute_processes = nullptr;
  GetProcessesFn get_graphics_processes = nullptr;
  bool available() const { return get_count != nullptr; }
};

Nvml LoadNvml() {
  Nvml nvml;
  void* lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) lib = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) return nvml;  // No NVIDIA driver installed.

  const auto sym = [lib](const char* name) { return dlsym(lib, name); };
  const auto init = reinterpret_cast<InitFn>(sym("nvmlInit_v2"));
  const auto get_count =
      reinterpret_cast<GetCountFn>(sym("nvmlDeviceGetCount_v2"));
  const auto get_handle =
      reinterpret_cast<GetHandleFn>(sym("nvmlDeviceGetHandleByIndex_v2"));
  const auto get_name = reinterpret_cast<GetNameFn>(sym("nvmlDeviceGetName"));
  const auto get_memory_info =
      reinterpret_cast<GetMemoryInfoFn>(sym("nvmlDeviceGetMemoryInfo"));
  // _v3 (R470+) and _v2 share the four-field nvmlProcessInfo_t.
  auto get_compute = reinterpret_cast<GetProcessesFn>(
      sym("nvmlDeviceGetComputeRunningProcesses_v3"));
  if (get_compute == nullptr) {
    get_compute = reinterpret_cast<GetProcessesFn>(
        sym("nvmlDeviceGetComputeRunningProcesses_v2"));
  }
  auto get_graphics = reinterpret_cast<GetProcessesFn>(
      sym("nvmlDeviceGetGraphicsRunningProcesses_v3"));
  if (get_graphics == nullptr) {
    get_graphics = reinterpret_cast<GetProcessesFn>(
        sym("nvmlDeviceGetGraphicsRunningProcesses_v2"));
  }
  if (init == nullptr || get_count == nullptr || get_handle == nullptr ||
      get_name == nullptr || get_memory_info == nullptr ||
      get_compute == nullptr || get_graphics == nullptr) {
    ABSL_LOG(WARNING)
        << "libnvidia-ml.so is present but lacks the expected "
           "entry points; NVIDIA GPU memory will not be reported.";
    return nvml;
  }
  if (const nvmlReturn_t status = init(); status != kNvmlSuccess) {
    // Typically a driver / library version mismatch.
    ABSL_LOG(WARNING) << "nvmlInit_v2 failed with status " << status
                      << "; NVIDIA GPU memory will not be reported.";
    return nvml;
  }
  nvml.get_count = get_count;
  nvml.get_handle = get_handle;
  nvml.get_name = get_name;
  nvml.get_memory_info = get_memory_info;
  nvml.get_compute_processes = get_compute;
  nvml.get_graphics_processes = get_graphics;
  return nvml;
}

const Nvml& GetNvml() {
  static const Nvml* const nvml = new Nvml(LoadNvml());
  return *nvml;
}

// One of the two per-process lists (compute or graphics contexts) of
// `device`. Empty on error. NVML_VALUE_NOT_AVAILABLE (reported on some
// configurations) is mapped to 0, there being no better answer.
std::vector<NvmlProcessEntry> ReadProcessList(GetProcessesFn get_processes,
                                              nvmlDevice_t device) {
  std::vector<NvmlProcessEntry> entries;
  // Query the count first, then with some headroom in case processes start in
  // between.
  unsigned int count = 0;
  nvmlReturn_t status = get_processes(device, &count, nullptr);
  if (status != kNvmlSuccess && status != kNvmlErrorInsufficientSize) {
    return entries;
  }
  if (count == 0) return entries;
  std::vector<nvmlProcessInfo_t> infos(count + 8);
  count = infos.size();
  if (get_processes(device, &count, infos.data()) != kNvmlSuccess) {
    return entries;
  }
  for (unsigned int i = 0; i < count && i < infos.size(); ++i) {
    const nvmlProcessInfo_t& info = infos[i];
    entries.push_back(
        {.pid = info.pid,
         .used_bytes = info.used_gpu_memory == kNvmlValueNotAvailable
                           ? 0
                           : info.used_gpu_memory});
  }
  return entries;
}

// Number of distinct processes in the lists, other than `pid`, that have no
// /proc entry here. The driver names processes by their PID in the root PID
// namespace; one this namespace cannot see may be this very process under
// its outside PID.
unsigned int CountForeignProcesses(
    unsigned int pid, const std::vector<NvmlProcessEntry>& compute,
    const std::vector<NvmlProcessEntry>& graphics) {
  std::vector<unsigned int> foreign;
  for (const std::vector<NvmlProcessEntry>* list : {&compute, &graphics}) {
    for (const NvmlProcessEntry& entry : *list) {
      if (entry.pid == pid) continue;
      if (std::find(foreign.begin(), foreign.end(), entry.pid) !=
          foreign.end()) {
        continue;  // A "C+G" process is in both lists.
      }
      struct stat unused;
      if (stat(("/proc/" + std::to_string(entry.pid)).c_str(), &unused) != 0) {
        foreign.push_back(entry.pid);
      }
    }
  }
  return foreign.size();
}

// Warns once per process when its GPU use may be hidden by a PID namespace,
// which otherwise shows up only as a silent 0. Any thread may sample, so the
// once-only flag is atomic.
void WarnIfPossiblyHidden(const std::vector<NvmlDeviceMemory>& devices) {
  static std::atomic<bool> warned{false};
  if (warned.load(std::memory_order_relaxed)) return;
  for (const NvmlDeviceMemory& d : devices) {
    if (d.used_bytes == 0 && d.foreign_process_count > 0) {
      if (warned.exchange(true, std::memory_order_relaxed)) return;
      ABSL_LOG(WARNING)
          << "NVML lists " << d.foreign_process_count << " process(es) on GPU #"
          << d.index << " that are not visible in this PID namespace. If this "
          << "process runs in a container, its own GPU memory may be among "
          << "them and go unreported; run it in the host PID namespace "
          << "(e.g. `docker --pid=host`) to attribute it.";
      return;
    }
  }
}

}  // namespace

std::optional<std::vector<NvmlDeviceMemory>> ReadNvmlProcessMemory() {
  const Nvml& nvml = GetNvml();
  if (!nvml.available()) return std::nullopt;

  unsigned int device_count = 0;
  if (nvml.get_count(&device_count) != kNvmlSuccess) return std::nullopt;
  const unsigned int pid = getpid();
  std::vector<NvmlDeviceMemory> devices;
  devices.reserve(device_count);
  for (unsigned int i = 0; i < device_count; ++i) {
    nvmlDevice_t device = nullptr;
    if (nvml.get_handle(i, &device) != kNvmlSuccess) continue;
    NvmlDeviceMemory memory;
    memory.index = i;
    char name[kNvmlDeviceNameBufferSize] = {};
    if (nvml.get_name(device, name, sizeof(name)) == kNvmlSuccess) {
      memory.name = name;
    }
    nvmlMemory_t info = {};
    if (nvml.get_memory_info(device, &info) == kNvmlSuccess) {
      memory.device_used_bytes = info.used;
    }
    const std::vector<NvmlProcessEntry> compute =
        ReadProcessList(nvml.get_compute_processes, device);
    const std::vector<NvmlProcessEntry> graphics =
        ReadProcessList(nvml.get_graphics_processes, device);
    memory.used_bytes = NvmlProcessBytes(pid, compute, graphics);
    memory.foreign_process_count =
        CountForeignProcesses(pid, compute, graphics);
    devices.push_back(std::move(memory));
  }
  WarnIfPossiblyHidden(devices);
  return devices;
}

uint64_t NvmlProcessBytes(unsigned int pid,
                          const std::vector<NvmlProcessEntry>& compute,
                          const std::vector<NvmlProcessEntry>& graphics) {
  // Each list carries the process's whole framebuffer use (see the header),
  // so a process in both lists must not be counted twice. Within one list a
  // PID normally appears once; should it not, the larger entry stands too,
  // which errs on the side of under- rather than over-counting.
  uint64_t bytes = 0;
  for (const std::vector<NvmlProcessEntry>* list : {&compute, &graphics}) {
    for (const NvmlProcessEntry& entry : *list) {
      if (entry.pid == pid) bytes = std::max(bytes, entry.used_bytes);
    }
  }
  return bytes;
}

}  // namespace litert::lm
