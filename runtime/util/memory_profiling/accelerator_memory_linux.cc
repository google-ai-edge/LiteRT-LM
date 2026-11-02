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

// Linux implementation of the platform contract in
// accelerator_memory.h.
//
// What is measured, from two sources that do not overlap:
//
//   - DRM fdinfo, for every GPU driver but NVIDIA's. Every user-space GPU
//     stack on Linux (Mesa, AMD's and Intel's compute runtimes, ROCm) reaches
//     the GPU through a DRM kernel driver, and since kernel 5.19 the driver
//     reports per-client memory in the fdinfo of the open device node
//     (fdinfo.h), split into the GPU's memory regions ("vram", "gtt", "cpu"
//     for amdgpu; "system0", "local0" for i915; ...). This is the data
//     `gputop` and `nvtop` show for non-NVIDIA GPUs.
//   - NVML, for NVIDIA GPUs (accelerator_memory_nvml.h), whose driver does
//     not report memory through DRM fdinfo. It gives this process's
//     framebuffer use per GPU, reported here under the region "vram".
//
// The breakdown is per region summed over all GPUs; the per-client (DRM) and
// per-GPU (NVML) detail says which GPU holds what.
//
// Two figures per DRM region: `drm-resident-<region>` is what is in that
// memory right now, the GPU part of "memory resident in RAM";
// `drm-total-<region>` is everything allocated there, evicted buffers
// included, the GPU part of "memory attributed to the process by the OS"
// (Linux itself charges the process for none of it, so the whole figure is
// added to the host's). A driver that reports no residency is taken to have
// everything resident. A buffer shared between two clients counts for both.
// NVML reports one figure, framebuffer memory in use, which serves as both.
//
// None of this memory is in the host's figures (VmRSS, VmRSS + VmSwap; see
// host_memory_linux.cc). The DRM drivers map GPU buffers into the process
// as VM_PFNMAP mappings (TTM for amdgpu, radeon, nouveau and xe; i915; msm;
// the GEM shmem helpers behind panfrost, v3d, lima and others), as does
// NVIDIA's, and the kernel does not count such pages in a process's RSS. So
// the two *_counted_by_host figures are zero and no correction is needed: a
// GPU buffer appears under the GPU and nowhere else.
//
// What is not seen:
//   - Kernels before 5.19, and drivers that have not implemented the memory
//     keys (nouveau, older panfrost): their clients show up without regions,
//     and if nothing else can measure them, tracking is reported as
//     unsupported rather than as zero.
//   - System memory an NVIDIA GPU uses on the process's behalf; NVML only
//     reports framebuffer memory.
//   - NVIDIA memory of a process in a container with its own PID namespace:
//     the driver lists processes by root-namespace PID, which cannot match
//     getpid() there, so NVML attributes nothing to this process (see
//     accelerator_memory_nvml.h; a warning is logged when this is likely).

#include "runtime/util/memory_profiling/accelerator_memory_linux.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory.h"
#include "runtime/util/memory_profiling/accelerator_memory_nvml.h"
#include "runtime/util/memory_profiling/fdinfo.h"

namespace litert::lm {

namespace {

constexpr absl::string_view kTotalPrefix = "drm-total-";
constexpr absl::string_view kLegacyMemoryPrefix = "drm-memory-";
constexpr absl::string_view kResidentPrefix = "drm-resident-";

// Parses a DRM fdinfo size, "<n>" or "<n> <unit>" with unit KiB, MiB or GiB,
// to bytes.
std::optional<uint64_t> ParseDrmSize(absl::string_view value) {
  value = absl::StripAsciiWhitespace(value);
  uint64_t multiplier = 1;
  if (absl::ConsumeSuffix(&value, "KiB")) {
    multiplier = 1024;
  } else if (absl::ConsumeSuffix(&value, "MiB")) {
    multiplier = 1024 * 1024;
  } else if (absl::ConsumeSuffix(&value, "GiB")) {
    multiplier = 1024ull * 1024 * 1024;
  }
  uint64_t n = 0;
  if (!absl::SimpleAtoi(absl::StripAsciiWhitespace(value), &n)) {
    return std::nullopt;
  }
  return n * multiplier;
}

// Collects "<prefix><region>: <size>" keys into (region, bytes), ordered by
// region name.
std::vector<std::pair<std::string, uint64_t>> RegionsWithPrefix(
    const FdinfoFields& fields, absl::string_view prefix) {
  std::vector<std::pair<std::string, uint64_t>> regions;
  for (const auto& [key, value] : fields) {
    if (!absl::StartsWith(key, prefix)) continue;
    if (const std::optional<uint64_t> bytes = ParseDrmSize(value)) {
      regions.emplace_back(key.substr(prefix.size()), *bytes);
    }
  }
  std::sort(regions.begin(), regions.end());
  return regions;
}

std::string StringField(const FdinfoFields& fields, absl::string_view key) {
  const auto it = fields.find(key);
  return it == fields.end() ? std::string() : it->second;
}

// Orders the breakdown and the allocations largest first; ties broken by
// name so the output is deterministic.
void SortLargestFirst(AcceleratorMemoryUsage& usage) {
  const auto larger_first = [](const auto& a, const auto& b) {
    return std::tie(b.second, a.first) < std::tie(a.second, b.first);
  };
  std::sort(usage.breakdown.begin(), usage.breakdown.end(), larger_first);
  std::sort(usage.allocations.begin(), usage.allocations.end(), larger_first);
}

}  // namespace

std::optional<DrmClientMemory> ParseDrmFdinfo(const FdinfoFields& fields) {
  DrmClientMemory client;
  client.driver = StringField(fields, "drm-driver");
  if (client.driver.empty()) return std::nullopt;
  client.pdev = StringField(fields, "drm-pdev");
  client.client_id = StringField(fields, "drm-client-id");
  // Drivers that print both spellings print the same numbers under both, so
  // take one, preferring the current one.
  std::vector<std::pair<std::string, uint64_t>> totals =
      RegionsWithPrefix(fields, kTotalPrefix);
  if (totals.empty()) totals = RegionsWithPrefix(fields, kLegacyMemoryPrefix);
  const std::vector<std::pair<std::string, uint64_t>> residents =
      RegionsWithPrefix(fields, kResidentPrefix);
  client.regions.reserve(totals.size());
  for (auto& [name, total] : totals) {
    uint64_t resident = total;  // Residency not reported: assume all present.
    const auto it =
        std::find_if(residents.begin(), residents.end(),
                     [&name = name](const auto& r) { return r.first == name; });
    if (it != residents.end()) resident = it->second;
    client.regions.push_back({.name = std::move(name),
                              .total_bytes = total,
                              .resident_bytes = resident});
  }
  return client;
}

std::vector<DrmClientMemory> ReadDrmClients(absl::string_view proc_self_dir) {
  std::vector<DrmClientMemory> clients;
  absl::flat_hash_set<std::string> seen;
  ForEachFdinfo(proc_self_dir, [&](absl::string_view, const FdinfoFields& f) {
    std::optional<DrmClientMemory> client = ParseDrmFdinfo(f);
    if (!client.has_value()) return;
    // Client ids are per driver instance; qualify them so two GPUs with the
    // same driver cannot collide.
    const std::string key = absl::StrCat(client->driver, "\n", client->pdev,
                                         "\n", client->client_id);
    if (!seen.insert(key).second) return;  // Another descriptor, same client.
    clients.push_back(*std::move(client));
  });
  return clients;
}

std::optional<AcceleratorMemoryUsage> AcceleratorMemoryUsageFromDrmClients(
    const std::vector<DrmClientMemory>& clients) {
  AcceleratorMemoryUsage usage;
  if (clients.empty()) return usage;  // No GPU in use.

  absl::flat_hash_map<std::string, uint64_t> resident_by_region;
  bool any_memory = false;
  for (const DrmClientMemory& client : clients) {
    const std::string gpu = client.pdev.empty()
                                ? client.driver
                                : absl::StrCat(client.driver, " ", client.pdev);
    for (const DrmClientMemory::Region& region : client.regions) {
      any_memory = true;
      usage.total_bytes += region.resident_bytes;
      usage.attributed_bytes += region.total_bytes;
      resident_by_region[region.name] += region.resident_bytes;
      if (region.resident_bytes == 0) continue;
      usage.allocations.emplace_back(
          absl::StrCat(gpu, " client ", client.client_id, " (", region.name,
                       ")"),
          region.resident_bytes);
    }
  }
  // Clients exist but none reports memory: the kernel or driver cannot tell
  // us, which is different from "nothing allocated".
  if (!any_memory) return std::nullopt;
  // GPU buffers are PFN-mapped and so outside RSS, and Linux has no commit
  // charge: nothing here is in the host's figures (see the file comment).
  usage.resident_counted_by_host_bytes = 0;
  usage.attributed_counted_by_host_bytes = 0;

  usage.breakdown.assign(resident_by_region.begin(), resident_by_region.end());
  SortLargestFirst(usage);
  return usage;
}

std::optional<AcceleratorMemoryUsage> AcceleratorMemoryUsageFromLinuxSources(
    const std::vector<DrmClientMemory>& drm_clients,
    const std::optional<std::vector<NvmlDeviceMemory>>& nvml_devices) {
  std::optional<AcceleratorMemoryUsage> drm =
      AcceleratorMemoryUsageFromDrmClients(drm_clients);
  if (!nvml_devices.has_value()) return drm;

  // With NVML available, DRM clients without memory statistics are NVIDIA's
  // (nvidia-drm reports none), so NVML's figures stand in for them.
  AcceleratorMemoryUsage usage = drm.value_or(AcceleratorMemoryUsage());
  uint64_t nvidia_bytes = 0;
  for (const NvmlDeviceMemory& device : *nvml_devices) {
    if (device.used_bytes == 0) continue;
    nvidia_bytes += device.used_bytes;
    usage.allocations.emplace_back(
        absl::StrCat(device.name, " #", device.index, " (vram)"),
        device.used_bytes);
  }
  if (nvidia_bytes > 0) {
    // Framebuffer memory in use is both present and allocated; NVML draws no
    // finer distinction. None of it is in the host's figures (file comment).
    usage.total_bytes += nvidia_bytes;
    usage.attributed_bytes += nvidia_bytes;
    auto vram =
        std::find_if(usage.breakdown.begin(), usage.breakdown.end(),
                     [](const auto& region) { return region.first == "vram"; });
    if (vram == usage.breakdown.end()) {
      usage.breakdown.emplace_back("vram", nvidia_bytes);
    } else {
      vram->second += nvidia_bytes;
    }
    SortLargestFirst(usage);
  }
  return usage;
}

std::optional<AcceleratorMemoryUsage> GetProcessAcceleratorMemoryUsage() {
  return AcceleratorMemoryUsageFromLinuxSources(ReadDrmClients("/proc/self"),
                                                ReadNvmlProcessMemory());
}

}  // namespace litert::lm
