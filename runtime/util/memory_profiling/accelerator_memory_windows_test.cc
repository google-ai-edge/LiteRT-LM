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

#include "runtime/util/memory_profiling/accelerator_memory_windows.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
// The headers below must come after windows.h; see
// accelerator_memory_windows.cc for why winternl.h and wtypes.h have to precede
// d3dkmthk.h.
#include <winternl.h>
#include <wtypes.h>
// d3dkmthk.h must come after the two headers above.
#include <d3d11.h>
#include <d3dkmthk.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"
#include "runtime/util/memory_profiling/host_memory.h"
#include "runtime/util/memory_profiling/memory_snapshot.h"

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;

constexpr uint64_t kMb = 1024 * 1024;

TEST(GetProcessAcceleratorMemoryUsageTest, Windows_IsSupported) {
  // Backed by DXGI. Every Windows 10+ machine exposes at least the Microsoft
  // Basic Render Driver (a software GPU), so tracking is always supported and
  // the breakdown always has the two segment groups.
  const std::optional<AcceleratorMemoryUsage> usage =
      GetProcessAcceleratorMemoryUsage();
  ASSERT_TRUE(usage.has_value());
  EXPECT_THAT(usage->breakdown, ElementsAre(Pair("local", testing::_),
                                            Pair("non-local", testing::_)));
  EXPECT_EQ(usage->total_bytes,
            usage->breakdown[0].second + usage->breakdown[1].second);
}

TEST(AcceleratorMemoryUsageFromAdaptersTest, NoAdaptersIsZero) {
  const AcceleratorMemoryUsage usage = AcceleratorMemoryUsageFromAdapters({});
  EXPECT_EQ(usage.total_bytes, 0);
  // The two segment groups are always present so the breakdown has a stable
  // shape for log parsers and dashboards.
  EXPECT_THAT(usage.breakdown,
              ElementsAre(Pair("local", 0), Pair("non-local", 0)));
  EXPECT_THAT(usage.allocations, IsEmpty());
}

TEST(AcceleratorMemoryUsageFromAdaptersTest, SumsSegmentsAcrossAdapters) {
  const std::vector<AdapterVideoMemory> adapters = {
      // Discrete: VRAM plus 200 MB spilled into shared system memory.
      {.description = "NVIDIA GeForce RTX 4070",
       .local_bytes = 1000 * kMb,
       .non_local_bytes = 200 * kMb,
       .committed_bytes = 1300 * kMb,
       .system_memory_committed_bytes = 200 * kMb},
      // Integrated: all system memory, 30 MB of it currently paged out.
      {.description = "Intel(R) UHD Graphics",
       .local_bytes = 0,
       .non_local_bytes = 50 * kMb,
       .committed_bytes = 80 * kMb,
       .system_memory_committed_bytes = 80 * kMb},
      {.description = "Microsoft Basic Render Driver",
       .local_bytes = 0,
       .non_local_bytes = 0,
       .committed_bytes = 0,
       .system_memory_committed_bytes = 0},
  };

  const AcceleratorMemoryUsage usage =
      AcceleratorMemoryUsageFromAdapters(adapters);

  // Present: what is resident in the segments now.
  EXPECT_EQ(usage.total_bytes, 1250 * kMb);
  EXPECT_THAT(usage.breakdown, ElementsAre(Pair("local", 1000 * kMb),
                                           Pair("non-local", 250 * kMb)));
  // Charged: what the video memory manager has committed, which the commit
  // charge contains in full.
  EXPECT_EQ(usage.attributed_bytes, 1380 * kMb);
  EXPECT_EQ(usage.attributed_counted_by_host_bytes, 1380 * kMb);
  // In the working set: the system-memory allocations while resident (200 MB
  // shared on the discrete GPU, and of the integrated GPU's 80 MB committed
  // only the 50 MB that are in RAM); VRAM never.
  EXPECT_EQ(usage.resident_counted_by_host_bytes, 250 * kMb);
  // Per-GPU detail, largest first; zero entries (and therefore GPUs the
  // process does not use) are omitted.
  EXPECT_THAT(
      usage.allocations,
      ElementsAre(Pair("NVIDIA GeForce RTX 4070 (local)", 1000 * kMb),
                  Pair("NVIDIA GeForce RTX 4070 (non-local)", 200 * kMb),
                  Pair("Intel(R) UHD Graphics (non-local)", 50 * kMb)));
}

TEST(AcceleratorMemoryUsageFromAdaptersTest, BreakdownSumsToTotal) {
  const std::vector<AdapterVideoMemory> adapters = {
      {.description = "A", .local_bytes = 3 * kMb, .non_local_bytes = 1 * kMb},
      {.description = "B", .local_bytes = 5 * kMb, .non_local_bytes = 7 * kMb},
  };
  const AcceleratorMemoryUsage usage =
      AcceleratorMemoryUsageFromAdapters(adapters);
  uint64_t breakdown_total = 0;
  for (const auto& [_, bytes] : usage.breakdown) breakdown_total += bytes;
  EXPECT_EQ(breakdown_total, usage.total_bytes);
  EXPECT_EQ(usage.total_bytes, 16 * kMb);
}

// A Direct3D 11 device on the machine's default GPU (Windows: "adapter"), used
// to allocate GPU memory that the OS attributes to this process.
struct D3D11TestDevice {
  Microsoft::WRL::ComPtr<ID3D11Device> device;
  Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
};

// Creates the test device, or returns std::nullopt with a reason suitable for
// GTEST_SKIP() when the default GPU is a software renderer (WARP / "Microsoft
// Basic Render Driver", e.g. on headless VMs): the video memory manager does
// not do per-process accounting for software renderers, so there is nothing to
// check.
std::optional<D3D11TestDevice> CreateD3D11TestDevice(std::string* skip_reason) {
  D3D11TestDevice result;
  const HRESULT hr = D3D11CreateDevice(
      nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
      D3D11_SDK_VERSION, &result.device, nullptr, &result.context);
  if (FAILED(hr)) {
    *skip_reason =
        absl::StrCat("No Direct3D 11 hardware device available: ", hr);
    return std::nullopt;
  }
  Microsoft::WRL::ComPtr<IDXGIDevice> dxgi_device;
  Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
  DXGI_ADAPTER_DESC adapter_desc = {};
  if (FAILED(result.device.As(&dxgi_device)) ||
      FAILED(dxgi_device->GetAdapter(&adapter)) ||
      FAILED(adapter->GetDesc(&adapter_desc))) {
    *skip_reason = "Failed to identify the default adapter.";
    return std::nullopt;
  }
  // The Microsoft Basic Render Driver has vendor ID 0x1414 and device ID 0x8C.
  if (adapter_desc.VendorId == 0x1414 && adapter_desc.DeviceId == 0x8C) {
    *skip_reason =
        "Default adapter is the software rasterizer, which has no per-process "
        "video memory accounting.";
    return std::nullopt;
  }
  return result;
}

// Allocates a default-usage buffer of `bytes`, which lives in GPU-accessible
// memory managed by the video memory manager, and has the GPU write it so that
// a lazy driver has to commit the allocation. The buffer is filled on the GPU
// (a UAV clear) rather than uploaded from the CPU, so no CPU-side copy of the
// data exists to disturb the process's memory counters.
Microsoft::WRL::ComPtr<ID3D11Buffer> AllocateD3D11Buffer(
    const D3D11TestDevice& d3d, uint64_t bytes) {
  D3D11_BUFFER_DESC desc = {};
  desc.ByteWidth = static_cast<UINT>(bytes);
  desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
  Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;
  if (FAILED(d3d.device->CreateBuffer(&desc, nullptr, &buffer))) {
    return nullptr;
  }
  D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
  uav_desc.Format = DXGI_FORMAT_R32_UINT;
  uav_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
  uav_desc.Buffer.NumElements = static_cast<UINT>(bytes / sizeof(uint32_t));
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> uav;
  if (FAILED(d3d.device->CreateUnorderedAccessView(buffer.Get(), &uav_desc,
                                                   &uav))) {
    return nullptr;
  }
  const UINT zeros[4] = {0, 0, 0, 0};
  d3d.context->ClearUnorderedAccessViewUint(uav.Get(), zeros);
  D3D11_QUERY_DESC query_desc = {};
  query_desc.Query = D3D11_QUERY_EVENT;
  Microsoft::WRL::ComPtr<ID3D11Query> query;
  if (SUCCEEDED(d3d.device->CreateQuery(&query_desc, &query))) {
    d3d.context->End(query.Get());
    d3d.context->Flush();
    while (d3d.context->GetData(query.Get(), nullptr, 0, 0) == S_FALSE) {
      absl::SleepFor(absl::Milliseconds(1));
    }
  } else {
    d3d.context->Flush();
  }
  return buffer;
}

// Logs how the OS accounts for video memory at this moment: the process's
// commit charge next to, for every GPU in the machine, its DXGI memory and
// every memory segment's properties and this process's committed bytes in it.
// Diagnostic only; it is what tells which segments end up in PrivateUsage on a
// given GPU, i.e. what AcceleratorMemoryUsage::attributed_bytes must count.
void LogVideoMemoryAccounting(absl::string_view when) {
  const std::optional<AcceleratorMemoryUsage> usage =
      GetProcessAcceleratorMemoryUsage();
  ABSL_LOG(INFO) << when << ": PrivateUsage="
                 << GetHostMemoryUsage().platform_attributed_bytes.value_or(0)
                 << " WorkingSetSize="
                 << GetHostMemoryUsage().resident_bytes.value_or(0)
                 << " video-memory-resident="
                 << (usage.has_value() ? usage->total_bytes : 0)
                 << " video-memory-committed="
                 << (usage.has_value() ? usage->attributed_bytes : 0)
                 << " video-memory-in-working-set="
                 << (usage.has_value() ? usage->resident_counted_by_host_bytes
                                       : 0);
  Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
  if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return;
  Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter1;
  for (UINT i = 0; SUCCEEDED(factory->EnumAdapters1(i, &adapter1)); ++i) {
    DXGI_ADAPTER_DESC1 desc = {};
    if (FAILED(adapter1->GetDesc1(&desc))) continue;
    std::string line = absl::StrCat(
        when, ": adapter ", i, " vendor=0x", absl::Hex(desc.VendorId),
        " dedicated_video=", desc.DedicatedVideoMemory,
        " dedicated_system=", desc.DedicatedSystemMemory,
        " shared_system=", desc.SharedSystemMemory);
    Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter3;
    if (SUCCEEDED(adapter1.As(&adapter3))) {
      for (const auto& [name, group] :
           {std::pair{"local", DXGI_MEMORY_SEGMENT_GROUP_LOCAL},
            std::pair{"non-local", DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL}}) {
        DXGI_QUERY_VIDEO_MEMORY_INFO info = {};
        if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, group, &info))) {
          absl::StrAppend(&line, " ", name, "={budget=", info.Budget,
                          " usage=", info.CurrentUsage, "}");
        }
      }
    }
    ABSL_LOG(INFO) << line;

    D3DKMT_QUERYSTATISTICS query = {};
    query.Type = D3DKMT_QUERYSTATISTICS_ADAPTER;
    query.AdapterLuid = desc.AdapterLuid;
    const NTSTATUS status = D3DKMTQueryStatistics(&query);
    if (status < 0) {
      ABSL_LOG(INFO) << when << ":   no adapter statistics, status=0x"
                     << absl::Hex(static_cast<uint32_t>(status));
      continue;
    }
    const ULONG num_segments = query.QueryResult.AdapterInformation.NbSegments;
    for (ULONG segment_id = 0; segment_id < num_segments; ++segment_id) {
      query = {};
      query.Type = D3DKMT_QUERYSTATISTICS_SEGMENT;
      query.AdapterLuid = desc.AdapterLuid;
      query.QuerySegment.SegmentId = segment_id;
      if (D3DKMTQueryStatistics(&query) < 0) continue;
      const D3DKMT_QUERYSTATISTICS_SEGMENT_INFORMATION& segment =
          query.QueryResult.SegmentInformation;
      std::string segment_line = absl::StrCat(
          when, ":   segment ", segment_id, " aperture=", segment.Aperture,
          " system_memory=",
          static_cast<uint64_t>(segment.SegmentProperties.SystemMemory),
          " firmware_reserved=",
          static_cast<uint64_t>(
              segment.SegmentProperties.PopulatedByReservedDDRByFirmware),
          " commit_limit=", segment.CommitLimit,
          " bytes_committed=", segment.BytesCommitted,
          " bytes_resident=", segment.BytesResident);
      query = {};
      query.Type = D3DKMT_QUERYSTATISTICS_PROCESS_SEGMENT;
      query.AdapterLuid = desc.AdapterLuid;
      query.hProcess = GetCurrentProcess();
      query.QueryProcessSegment.SegmentId = segment_id;
      if (D3DKMTQueryStatistics(&query) >= 0) {
        absl::StrAppend(
            &segment_line, " process_bytes_committed=",
            query.QueryResult.ProcessSegmentInformation.BytesCommitted);
      }
      ABSL_LOG(INFO) << segment_line;
    }
  }
}

// Diagnostic: logs the video memory accounting before and after a 256 MB GPU
// allocation and never fails.
//
// Why it exists: the GPU part of "memory attributed to the process by the OS"
// rests on an empirical fact about Windows, namely that the video memory
// manager charges every GPU allocation to the owning process's commit (and
// hence to PrivateUsage) regardless of the memory segment it lives in. That
// was established by running this test, not read from documentation: an Intel
// integrated GPU puts the buffer in an aperture segment flagged as
// firmware-reserved and reported to DXGI as "local", a discrete AMD GPU puts
// it in dedicated VRAM, and PrivateUsage grows by the buffer size on both. The
// only way to know what a given GPU and driver do is to allocate something and
// watch which segment's process_bytes_committed grows and whether PrivateUsage
// grows with it. This test prints exactly that, so a single run on a machine
// answers the question without a debugger.
//
// When to use it: run it (with --alsologtostderr) whenever
// Windows_SnapshotFilesChargedD3DAllocationsUnderTheGpu fails on a machine,
// when bringing up a GPU vendor or driver generation that has not been checked
// before (NVIDIA, Qualcomm, AMD integrated, a new Intel architecture), or when
// the CPU and GPU parts of the attributed figure look wrong. Compare the
// "before"/"after" lines: the sum of process_bytes_committed over all segments
// should grow by about as much as PrivateUsage does. If PrivateUsage grows by
// less, that GPU has segments the OS does not charge for, and
// AcceleratorMemoryUsage::attributed_counted_by_host_bytes would need to tell
// them apart by their flags.
TEST(AcceleratorMemoryUsageTest, Windows_LogsVideoMemoryAccounting) {
  constexpr uint64_t kAllocationBytes = 256 * kMb;
  std::string skip_reason;
  const std::optional<D3D11TestDevice> d3d =
      CreateD3D11TestDevice(&skip_reason);
  if (!d3d.has_value()) GTEST_SKIP() << skip_reason;

  LogVideoMemoryAccounting("before");
  const Microsoft::WRL::ComPtr<ID3D11Buffer> buffer =
      AllocateD3D11Buffer(*d3d, kAllocationBytes);
  ASSERT_NE(buffer, nullptr);
  LogVideoMemoryAccounting("after");
}

// Allocates GPU memory through Direct3D 11 on the default GPU and checks that
// the OS attributes it to this process.
TEST(GetProcessAcceleratorMemoryUsageTest, Windows_SeesD3DAllocations) {
  constexpr uint64_t kAllocationBytes = 256 * kMb;

  std::string skip_reason;
  const std::optional<D3D11TestDevice> d3d =
      CreateD3D11TestDevice(&skip_reason);
  if (!d3d.has_value()) GTEST_SKIP() << skip_reason;

  const std::optional<AcceleratorMemoryUsage> before =
      GetProcessAcceleratorMemoryUsage();
  ASSERT_TRUE(before.has_value());

  const Microsoft::WRL::ComPtr<ID3D11Buffer> buffer =
      AllocateD3D11Buffer(*d3d, kAllocationBytes);
  ASSERT_NE(buffer, nullptr);

  const std::optional<AcceleratorMemoryUsage> after =
      GetProcessAcceleratorMemoryUsage();
  ASSERT_TRUE(after.has_value());
  // Drivers may round or add bookkeeping, so only require most of the
  // allocation to show up.
  EXPECT_GE(after->total_bytes, before->total_bytes + kAllocationBytes / 2)
      << "before=" << before->total_bytes << " after=" << after->total_bytes
      << " breakdown=" << FormatAcceleratorMemoryBreakdown(*after);
  // The per-GPU detail names the GPU that holds the allocation.
  EXPECT_THAT(after->allocations, testing::Not(IsEmpty()));
}

// Checks that GPU memory the OS charges to the process's commit charge is
// recognised as such, so that a snapshot reports the commit charge as Task
// Manager shows it, with the allocation on its GPU side rather than its CPU
// side, and reports the allocation once in the memory resident in RAM.
//
// On every GPU measured so far (Intel integrated, AMD discrete) the OS charges
// the whole allocation, whether it lands in system memory or in dedicated
// VRAM, so the first branch is the expected one. The second branch guards the
// opposite case, should a platform turn out not to charge for some memory:
// then the GPU side of the attributed figure must not claim it either. The
// deltas are logged so the behaviour can be read off on a given machine.
//
// The resident check is the one that tells system memory from VRAM: on an
// integrated GPU (Intel Arc 140V) the allocation lands in the working set as
// well as in the video memory, and the snapshot must take it out of the CPU
// side; in dedicated VRAM it is in the video memory only. Either way it is
// present once.
TEST(AcceleratorMemoryUsageTest,
     Windows_SnapshotFilesChargedD3DAllocationsUnderTheGpu) {
  constexpr uint64_t kAllocationBytes = 256 * kMb;
  constexpr uint64_t kTolerance = kAllocationBytes / 2;

  std::string skip_reason;
  const std::optional<D3D11TestDevice> d3d =
      CreateD3D11TestDevice(&skip_reason);
  if (!d3d.has_value()) GTEST_SKIP() << skip_reason;

  const HostMemoryUsage raw_before = GetHostMemoryUsage();
  const MemorySnapshot before = TakeMemorySnapshot();
  ASSERT_TRUE(raw_before.platform_attributed_bytes.has_value());
  ASSERT_TRUE(before.attributed_by_os.has_value());
  ASSERT_TRUE(before.resident_in_ram.has_value());

  const Microsoft::WRL::ComPtr<ID3D11Buffer> buffer =
      AllocateD3D11Buffer(*d3d, kAllocationBytes);
  ASSERT_NE(buffer, nullptr);

  const HostMemoryUsage raw_after = GetHostMemoryUsage();
  const MemorySnapshot after = TakeMemorySnapshot();
  ASSERT_TRUE(raw_after.platform_attributed_bytes.has_value());
  ASSERT_TRUE(after.attributed_by_os.has_value());
  ASSERT_TRUE(after.resident_in_ram.has_value());

  const auto delta = [](uint64_t a, uint64_t b) {
    return static_cast<int64_t>(b) - static_cast<int64_t>(a);
  };
  // PrivateUsage as the OS reports it, GPU memory included.
  const int64_t raw_delta = delta(*raw_before.platform_attributed_bytes,
                                  *raw_after.platform_attributed_bytes);
  // The working set as the OS reports it, which on an integrated GPU grows
  // by the allocation too.
  const int64_t working_set_delta = delta(raw_before.resident_bytes.value_or(0),
                                          raw_after.resident_bytes.value_or(0));
  const int64_t attributed_delta = delta(before.attributed_by_os->total_bytes,
                                         after.attributed_by_os->total_bytes);
  const int64_t attributed_gpu_delta =
      delta(before.attributed_by_os->accelerator_bytes,
            after.attributed_by_os->accelerator_bytes);
  const int64_t attributed_cpu_delta = delta(before.attributed_by_os->cpu_bytes,
                                             after.attributed_by_os->cpu_bytes);
  const int64_t resident_delta = delta(before.resident_in_ram->total_bytes,
                                       after.resident_in_ram->total_bytes);
  const int64_t resident_cpu_delta = delta(before.resident_in_ram->cpu_bytes,
                                           after.resident_in_ram->cpu_bytes);
  ABSL_LOG(INFO) << "Allocating " << kAllocationBytes
                 << " bytes of GPU memory changed PrivateUsage by " << raw_delta
                 << " and the working set by " << working_set_delta
                 << "; the attributed figure by " << attributed_delta
                 << " (CPU " << attributed_cpu_delta << ", GPU "
                 << attributed_gpu_delta << ") and the resident figure by "
                 << resident_delta << " (CPU " << resident_cpu_delta
                 << "); snapshot is now " << after;

  // The attributed total is PrivateUsage, whatever happened.
  EXPECT_NEAR(static_cast<double>(attributed_delta),
              static_cast<double>(raw_delta), static_cast<double>(kMb));
  // The memory is present once, on the GPU side: whether or not the working
  // set grew by it, the CPU side of the resident figure did not.
  EXPECT_GE(resident_delta, static_cast<int64_t>(kTolerance));
  EXPECT_LT(resident_delta,
            static_cast<int64_t>(kAllocationBytes + kTolerance));
  EXPECT_LT(resident_cpu_delta, static_cast<int64_t>(kTolerance));
  if (raw_delta >= static_cast<int64_t>(kTolerance)) {
    // The allocation was charged to the process: it lands on the GPU side of
    // the attributed figure and leaves the CPU side alone.
    EXPECT_GE(attributed_gpu_delta, static_cast<int64_t>(kTolerance));
    EXPECT_LT(attributed_cpu_delta, static_cast<int64_t>(kTolerance));
  } else {
    // The allocation was not charged to the process: the GPU side of the
    // attributed figure must not claim it either.
    EXPECT_LT(attributed_gpu_delta, static_cast<int64_t>(kTolerance));
  }
}

}  // namespace
}  // namespace litert::lm
