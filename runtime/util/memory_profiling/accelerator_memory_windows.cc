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

// Windows implementation of the platform contract in
// accelerator_memory.h.
//
// What is measured: the video memory the OS video memory manager attributes
// to this process, read through two Windows APIs. DXGI
// (IDXGIAdapter3::QueryVideoMemoryInfo) gives the process's current usage per
// GPU, split into the "local" and "non-local" segment groups: the video memory
// physically present, i.e. the GPU part of "memory resident in RAM". The
// kernel-mode statistics API (D3DKMTQueryStatistics) gives the process's
// committed bytes per memory segment: what the video memory manager charges
// to the process's commit charge, i.e. the GPU part of "memory attributed to
// the process by the OS"; see the background comment further down for why
// the two differ. Both come from the OS, not from the GPU vendor's driver, so
// they cover allocations made through D3D12, D3D11, Vulkan and OpenCL alike
// and match Task Manager's per-process GPU memory columns.
//
// Overlap with the host's resident figure (the working set): a GPU allocation
// that lives in a system-memory segment, which is all of an integrated GPU's
// memory and a discrete GPU's "shared" memory, is made of the process's own
// pages, and while it is in RAM it is in the working set. Measured on an
// Intel Arc 140V (integrated): a 256 MB Direct3D buffer the GPU filled raised
// the working set by 256 MB. Dedicated VRAM is not process memory and is not
// in the working set. The segment statistics say which kind each segment is,
// so the process's committed bytes in system-memory segments are reported as
// the part of the resident figure the working set already counts.
//
// What is not seen: nothing is missed on the GPU side, but the OS does not
// report which GPU LiteRT-LM's backend picked, so every GPU in the machine is
// queried and the unused ones contribute zero.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
// The headers below must come after windows.h. winternl.h provides NTSTATUS
// and wtypes.h PALETTEENTRY (which NOGDI keeps out of windows.h), both of
// which d3dkmthk.h (kernel-mode thunks, incl. D3DKMTQueryStatistics) needs.
#include <winternl.h>
#include <wtypes.h>
// d3dkmthk.h must come after the two headers above.
#include <d3dkmthk.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory.h"
#include "runtime/util/memory_profiling/accelerator_memory_windows.h"

namespace litert::lm {

namespace {

using Microsoft::WRL::ComPtr;

// Orders (label, bytes) pairs largest first; ties broken by label so the
// output is deterministic.
bool LargerFirst(const std::pair<std::string, uint64_t>& a,
                 const std::pair<std::string, uint64_t>& b) {
  return std::tie(b.second, a.first) < std::tie(a.second, b.first);
}

// One GPU, as seen through the two Windows APIs used here. Windows calls a GPU
// an "adapter": DXGI enumerates adapters, and the kernel-mode video memory
// manager identifies them by LUID. Software renderers (WARP, the "Microsoft
// Basic Render Driver") are adapters too, with no physical GPU behind them.
struct DxgiAdapter {
  ComPtr<IDXGIAdapter3> adapter;
  std::string description;
  // Identifies the GPU to the kernel-mode video memory manager.
  LUID luid = {};
  // Number of memory segments the video memory manager reports for the GPU,
  // or std::nullopt when it exposes no statistics for it (software renderers),
  // in which case the DXGI usage stands in.
  std::optional<ULONG> num_segments;
  // Per segment, whether it is system memory (an aperture into RAM, or RAM
  // the firmware set aside) rather than dedicated VRAM. Segments do not
  // change while the process runs.
  std::vector<bool> segment_is_system_memory;
};

std::string WideToUtf8(const wchar_t* wide) {
  const int size =
      WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
  if (size <= 1) return "";
  std::string utf8(size - 1, '\0');
  WideCharToMultiByte(CP_UTF8, 0, wide, -1, utf8.data(), size, nullptr,
                      nullptr);
  return utf8;
}

bool NtSucceeded(NTSTATUS status) { return status >= 0; }

// Number of memory segments of the GPU, or std::nullopt when the video memory
// manager has no statistics for it at all (software renderers).
std::optional<ULONG> GetNumSegments(const LUID& luid) {
  D3DKMT_QUERYSTATISTICS query = {};
  query.Type = D3DKMT_QUERYSTATISTICS_ADAPTER;
  query.AdapterLuid = luid;
  if (!NtSucceeded(D3DKMTQueryStatistics(&query))) return std::nullopt;
  return query.QueryResult.AdapterInformation.NbSegments;
}

// Whether segment `segment_id` of the GPU is system memory. An aperture
// segment addresses RAM by definition; SegmentProperties.SystemMemory (WDDM
// 2.9+, Windows 10 2004) also covers RAM the firmware reserved for the GPU at
// boot, which is how Intel integrated GPUs report their main segment. On
// older drivers the flag reads 0 and an unknown segment is taken to be VRAM,
// which falls back to counting its memory on both sides rather than hiding
// the host's memory behind it.
bool IsSystemMemorySegment(const LUID& luid, ULONG segment_id) {
  D3DKMT_QUERYSTATISTICS query = {};
  query.Type = D3DKMT_QUERYSTATISTICS_SEGMENT;
  query.AdapterLuid = luid;
  query.QuerySegment.SegmentId = segment_id;
  if (!NtSucceeded(D3DKMTQueryStatistics(&query))) return false;
  const D3DKMT_QUERYSTATISTICS_SEGMENT_INFORMATION& segment =
      query.QueryResult.SegmentInformation;
  return segment.Aperture != 0 || segment.SegmentProperties.SystemMemory != 0;
}

// Enumerates the GPUs that support per-process video memory accounting
// (IDXGIAdapter3, i.e. Windows 10+). Done once: GPUs do not change while the
// process runs and enumeration is far more expensive than a query.
//
// LiteRT-LM runs on a single GPU, but which one the backend picked is not
// known here, so every GPU is included, software renderers too. That is
// harmless: QueryVideoMemoryInfo reports only this process's usage, so a GPU
// the process does not use simply contributes zero, and the per-GPU detail in
// the diagnostic output shows which one was actually used.
const std::vector<DxgiAdapter>& GetDxgiAdapters() {
  // Intentionally leaked: COM objects must not be released during static
  // destruction, after DXGI may already have been unloaded.
  static const std::vector<DxgiAdapter>* const adapters = [] {
    auto* result = new std::vector<DxgiAdapter>();
    ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
      ABSL_LOG(WARNING) << "CreateDXGIFactory1 failed; accelerator memory "
                           "tracking is unavailable.";
      return result;
    }
    ComPtr<IDXGIAdapter1> adapter1;
    // Stop on any failure, not just DXGI_ERROR_NOT_FOUND, so an unexpected
    // error cannot turn this into an endless loop.
    for (UINT i = 0; SUCCEEDED(factory->EnumAdapters1(i, &adapter1)); ++i) {
      ComPtr<IDXGIAdapter3> adapter3;
      if (FAILED(adapter1.As(&adapter3))) continue;
      DxgiAdapter dxgi_adapter;
      dxgi_adapter.adapter = std::move(adapter3);
      DXGI_ADAPTER_DESC1 desc = {};
      if (SUCCEEDED(adapter1->GetDesc1(&desc))) {
        dxgi_adapter.description = WideToUtf8(desc.Description);
        dxgi_adapter.luid = desc.AdapterLuid;
        dxgi_adapter.num_segments = GetNumSegments(desc.AdapterLuid);
        for (ULONG segment_id = 0;
             segment_id < dxgi_adapter.num_segments.value_or(0); ++segment_id) {
          dxgi_adapter.segment_is_system_memory.push_back(
              IsSystemMemorySegment(desc.AdapterLuid, segment_id));
        }
      } else {
        dxgi_adapter.description = absl::StrCat("Adapter ", i);
      }
      result->push_back(std::move(dxgi_adapter));
    }
    return result;
  }();
  return *adapters;
}

// This process's video memory in the given segment group of `adapter`, or
// std::nullopt if the query fails.
std::optional<uint64_t> QueryVideoMemoryUsage(
    IDXGIAdapter3* adapter, DXGI_MEMORY_SEGMENT_GROUP segment_group) {
  DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
  if (FAILED(adapter->QueryVideoMemoryInfo(/*NodeIndex=*/0, segment_group,
                                           &info))) {
    return std::nullopt;
  }
  return info.CurrentUsage;
}

// Background, for readers unfamiliar with Windows memory accounting.
//
// The memory Windows attributes to a process
// (HostMemoryUsage::platform_attributed_bytes, host_memory_windows.cc) is its
// *commit charge*: every byte the OS has promised to keep for this process
// alone, in RAM or in the pagefile. It is a good footprint measure because it
// is exactly what the system gets back when the process exits, whether or not
// the memory is in RAM at that moment.
//
// GPU memory is managed by a Windows kernel component, the video memory
// manager, not by the GPU vendor's driver. The driver only describes the
// physical pools it has, called *segments*: dedicated VRAM on a graphics card,
// a block of RAM the firmware set aside for the GPU at boot, or plain system
// RAM that the GPU can address. Every GPU allocation lives in one segment, and
// the video memory manager keeps per-process, per-segment byte counts.
//
// The commit charge and the video memory overlap because the video memory
// manager promises that it can always page a GPU allocation out to system
// memory (to make room on the GPU, or when the application is in the
// background). To be able to keep that promise it charges the allocation's
// size to the owning *process's* commit when the allocation is created, the
// same way a malloc() would be, no matter which segment the allocation
// currently sits in. Measured on an Intel integrated GPU and on a discrete AMD
// GPU alike: a 256 MB buffer raises the process's commit by 256 MB whether it
// ends up in system RAM or in dedicated VRAM. So every GPU allocation of the
// process is in the commit charge as well as in the video memory figures.
//
// This function returns that part: everything the video memory manager has
// committed for this process on `dxgi_adapter`, summed over its segments. It
// is reported as AcceleratorMemoryUsage::attributed_bytes, all of which the
// commit charge already contains, so that a MemorySnapshot shows the commit
// charge as Task Manager does and files this part under the GPU.
//
// It also returns the share of it in system-memory segments. Those
// allocations are process pages, in the working set while in RAM (the file
// comment has the measurement), so that share is also the part of the
// resident video memory the host's resident figure already counts; a
// snapshot takes it out of the CPU side. Dedicated VRAM is not in the
// working set and is left out.
//
// The per-segment statistics API is used rather than DXGI because it reports
// committed bytes, which is what the commit charge is made of; DXGI reports
// the bytes currently resident, which can lag behind. DXGI is only the
// fallback for GPUs without statistics.
struct CommittedBytes {
  uint64_t total = 0;
  uint64_t system_memory = 0;
};
CommittedBytes QueryCommittedBytes(const DxgiAdapter& dxgi_adapter) {
  CommittedBytes bytes;
  if (!dxgi_adapter.num_segments.has_value()) {
    // No per-segment statistics for this GPU (software renderers, whose
    // memory is all system memory). Its resident usage is the best available
    // estimate; it is zero for a GPU the process never used.
    for (DXGI_MEMORY_SEGMENT_GROUP group :
         {DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
          DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL}) {
      bytes.total +=
          QueryVideoMemoryUsage(dxgi_adapter.adapter.Get(), group).value_or(0);
    }
    bytes.system_memory = bytes.total;
    return bytes;
  }
  // Ask the video memory manager how many bytes *this process* has committed
  // in each segment, and add them up. This is a commit figure like
  // PrivateUsage itself, so the two are directly comparable.
  for (ULONG segment_id = 0; segment_id < *dxgi_adapter.num_segments;
       ++segment_id) {
    D3DKMT_QUERYSTATISTICS query = {};
    query.Type = D3DKMT_QUERYSTATISTICS_PROCESS_SEGMENT;
    query.AdapterLuid = dxgi_adapter.luid;
    query.hProcess = GetCurrentProcess();
    query.QueryProcessSegment.SegmentId = segment_id;
    if (!NtSucceeded(D3DKMTQueryStatistics(&query))) continue;
    const uint64_t committed =
        query.QueryResult.ProcessSegmentInformation.BytesCommitted;
    bytes.total += committed;
    if (segment_id < dxgi_adapter.segment_is_system_memory.size() &&
        dxgi_adapter.segment_is_system_memory[segment_id]) {
      bytes.system_memory += committed;
    }
  }
  return bytes;
}

}  // namespace

AcceleratorMemoryUsage AcceleratorMemoryUsageFromAdapters(
    const std::vector<AdapterVideoMemory>& adapters) {
  AcceleratorMemoryUsage usage;
  uint64_t local_bytes = 0;
  uint64_t non_local_bytes = 0;
  uint64_t committed_bytes = 0;
  uint64_t in_working_set_bytes = 0;
  for (const AdapterVideoMemory& adapter : adapters) {
    local_bytes += adapter.local_bytes;
    non_local_bytes += adapter.non_local_bytes;
    committed_bytes += adapter.committed_bytes;
    // Committed system-memory allocations are in the working set while they
    // are resident; what is paged out is in neither figure, so cap at what
    // the GPU has resident.
    in_working_set_bytes +=
        std::min(adapter.system_memory_committed_bytes,
                 adapter.local_bytes + adapter.non_local_bytes);
    if (adapter.local_bytes > 0) {
      usage.allocations.emplace_back(
          absl::StrCat(adapter.description, " (local)"), adapter.local_bytes);
    }
    if (adapter.non_local_bytes > 0) {
      usage.allocations.emplace_back(
          absl::StrCat(adapter.description, " (non-local)"),
          adapter.non_local_bytes);
    }
  }
  usage.total_bytes = local_bytes + non_local_bytes;
  // Committed video memory is charged to the commit charge in full; the part
  // of it in system memory is in the working set as well (see
  // QueryCommittedBytes).
  usage.attributed_bytes = committed_bytes;
  usage.attributed_counted_by_host_bytes = committed_bytes;
  usage.resident_counted_by_host_bytes = in_working_set_bytes;
  // Always report both segments so the breakdown has a stable shape, even when
  // one of them is empty (e.g. no spill-over into non-local memory).
  usage.breakdown = {{"local", local_bytes}, {"non-local", non_local_bytes}};
  std::sort(usage.allocations.begin(), usage.allocations.end(), LargerFirst);
  return usage;
}

std::optional<AcceleratorMemoryUsage> GetProcessAcceleratorMemoryUsage() {
  std::vector<AdapterVideoMemory> adapters;
  for (const DxgiAdapter& dxgi_adapter : GetDxgiAdapters()) {
    const std::optional<uint64_t> local = QueryVideoMemoryUsage(
        dxgi_adapter.adapter.Get(), DXGI_MEMORY_SEGMENT_GROUP_LOCAL);
    const std::optional<uint64_t> non_local = QueryVideoMemoryUsage(
        dxgi_adapter.adapter.Get(), DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL);
    if (!local.has_value() && !non_local.has_value()) continue;
    const CommittedBytes committed = QueryCommittedBytes(dxgi_adapter);
    adapters.push_back({
        .description = dxgi_adapter.description,
        .local_bytes = local.value_or(0),
        .non_local_bytes = non_local.value_or(0),
        .committed_bytes = committed.total,
        .system_memory_committed_bytes = committed.system_memory,
    });
  }
  if (adapters.empty()) return std::nullopt;
  return AcceleratorMemoryUsageFromAdapters(adapters);
}

}  // namespace litert::lm
