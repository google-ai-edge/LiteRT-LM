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

// Live test of the NVIDIA path on a machine with an NVIDIA GPU: allocates
// framebuffer memory through the CUDA driver API and checks that NVML, and
// through it GetProcessAcceleratorMemoryUsage(), attributes it to this
// process. The CUDA driver API is loaded with dlopen() so that the test has
// no build-time dependency on the CUDA toolkit; any API that allocates
// framebuffer memory (OpenCL, Vulkan) would show up the same way.
//
// Under Forge the test runs in a PID namespace, where NVML cannot attribute
// memory to this process (see accelerator_memory_nvml.h): there it verifies
// that NVML loads, that the declared ABI matches the driver and that the
// device-level figure moves, and skips the per-process checks. Run it on a
// GPU workstation for the full check.

#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory.h"
#include "runtime/util/memory_profiling/accelerator_memory_nvml.h"

namespace litert::lm {
namespace {

using ::testing::Contains;
using ::testing::Pair;

constexpr uint64_t kMb = 1024 * 1024;
constexpr uint64_t kAllocationBytes = 256 * kMb;
constexpr uint64_t kTolerance = kAllocationBytes / 2;

// The slice of the CUDA driver API used to allocate framebuffer memory.
// cuda.h spells CUdeviceptr `unsigned long long`; uint64_t has the same size
// and alignment on every Linux ABI.
using CUresult = int;
using CUdevice = int;
using CUcontext = struct CUctx_st*;
using CUdeviceptr = uint64_t;
constexpr CUresult kCudaSuccess = 0;

// A CUDA context on device 0 holding one framebuffer allocation.
class CudaAllocation {
 public:
  // Returns nullptr, with the reason in `skip_reason`, if the CUDA driver is
  // not usable on this machine.
  static std::unique_ptr<CudaAllocation> Create(uint64_t bytes,
                                                std::string* skip_reason) {
    void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) {
      *skip_reason = "libcuda.so.1 not found: no NVIDIA driver.";
      return nullptr;
    }
    auto self = std::unique_ptr<CudaAllocation>(new CudaAllocation);
    const auto init =
        reinterpret_cast<CUresult (*)(unsigned int)>(dlsym(lib, "cuInit"));
    const auto device_get = reinterpret_cast<CUresult (*)(CUdevice*, int)>(
        dlsym(lib, "cuDeviceGet"));
    const auto ctx_create =
        reinterpret_cast<CUresult (*)(CUcontext*, unsigned int, CUdevice)>(
            dlsym(lib, "cuCtxCreate_v2"));
    const auto mem_alloc = reinterpret_cast<CUresult (*)(CUdeviceptr*, size_t)>(
        dlsym(lib, "cuMemAlloc_v2"));
    self->mem_free_ =
        reinterpret_cast<CUresult (*)(CUdeviceptr)>(dlsym(lib, "cuMemFree_v2"));
    self->ctx_destroy_ = reinterpret_cast<CUresult (*)(CUcontext)>(
        dlsym(lib, "cuCtxDestroy_v2"));
    if (init == nullptr || device_get == nullptr || ctx_create == nullptr ||
        mem_alloc == nullptr || self->mem_free_ == nullptr ||
        self->ctx_destroy_ == nullptr) {
      *skip_reason = "libcuda.so.1 lacks the expected entry points.";
      return nullptr;
    }
    CUdevice device = 0;
    if (CUresult r = init(0); r != kCudaSuccess) {
      *skip_reason = absl::StrCat("cuInit failed: ", r);
      return nullptr;
    }
    if (CUresult r = device_get(&device, 0); r != kCudaSuccess) {
      *skip_reason = absl::StrCat("cuDeviceGet failed: ", r);
      return nullptr;
    }
    if (CUresult r = ctx_create(&self->ctx_, 0, device); r != kCudaSuccess) {
      *skip_reason = absl::StrCat("cuCtxCreate failed: ", r);
      return nullptr;
    }
    if (CUresult r = mem_alloc(&self->ptr_, bytes); r != kCudaSuccess) {
      *skip_reason = absl::StrCat("cuMemAlloc failed: ", r);
      return nullptr;
    }
    return self;
  }

  ~CudaAllocation() {
    if (ptr_ != 0) mem_free_(ptr_);
    if (ctx_ != nullptr) ctx_destroy_(ctx_);
  }

 private:
  CudaAllocation() = default;
  CUresult (*mem_free_)(CUdeviceptr) = nullptr;
  CUresult (*ctx_destroy_)(CUcontext) = nullptr;
  CUcontext ctx_ = nullptr;
  CUdeviceptr ptr_ = 0;
};

// True when the driver lists GPU processes this PID namespace cannot see. In
// a container (Forge test sandboxes included) this process is one of them,
// under a PID that cannot match getpid(); see accelerator_memory_nvml.h.
bool HasForeignGpuProcesses(const std::vector<NvmlDeviceMemory>& devices) {
  for (const NvmlDeviceMemory& d : devices) {
    if (d.foreign_process_count > 0) return true;
  }
  return false;
}

uint64_t SumUsed(const std::vector<NvmlDeviceMemory>& devices) {
  uint64_t bytes = 0;
  for (const NvmlDeviceMemory& d : devices) bytes += d.used_bytes;
  return bytes;
}

uint64_t SumDeviceUsed(const std::vector<NvmlDeviceMemory>& devices) {
  uint64_t bytes = 0;
  for (const NvmlDeviceMemory& d : devices) bytes += d.device_used_bytes;
  return bytes;
}

TEST(NvidiaAcceleratorMemoryTest, Nvidia_NvmlSeesCudaAllocation) {
  const std::optional<std::vector<NvmlDeviceMemory>> before =
      ReadNvmlProcessMemory();
  ASSERT_TRUE(before.has_value()) << "NVML not available on a GPU machine.";
  ASSERT_FALSE(before->empty()) << "NVML reports no NVIDIA GPU.";

  std::string skip_reason;
  const std::unique_ptr<CudaAllocation> allocation =
      CudaAllocation::Create(kAllocationBytes, &skip_reason);
  if (allocation == nullptr) GTEST_SKIP() << skip_reason;

  const std::optional<std::vector<NvmlDeviceMemory>> after =
      ReadNvmlProcessMemory();
  ASSERT_TRUE(after.has_value());
  for (const NvmlDeviceMemory& d : *after) {
    ABSL_LOG(INFO) << "GPU #" << d.index << " " << d.name << ": this process "
                   << d.used_bytes << " bytes, all processes "
                   << d.device_used_bytes << ", foreign processes "
                   << d.foreign_process_count;
  }

  // The GPU as a whole saw the allocation (plus the context's own overhead):
  // this checks that NVML loaded, initialized and that the declared ABI
  // matches the driver.
  EXPECT_GE(SumDeviceUsed(*after), SumDeviceUsed(*before) + kTolerance);

  // And it is attributed to this process. In a PID namespace the driver's
  // PIDs cannot match ours (documented limitation), so only insist when the
  // driver lists no process we cannot see.
  const int64_t process_delta = static_cast<int64_t>(SumUsed(*after)) -
                                static_cast<int64_t>(SumUsed(*before));
  if (process_delta < static_cast<int64_t>(kTolerance) &&
      HasForeignGpuProcesses(*after)) {
    GTEST_SKIP() << "Running in a PID namespace: NVML cannot attribute "
                    "memory to this process (it reports root-namespace PIDs).";
  }
  EXPECT_GE(process_delta, static_cast<int64_t>(kTolerance));
}

TEST(NvidiaAcceleratorMemoryTest, Nvidia_PlatformContractReportsVram) {
  std::string skip_reason;
  const std::unique_ptr<CudaAllocation> allocation =
      CudaAllocation::Create(kAllocationBytes, &skip_reason);
  if (allocation == nullptr) GTEST_SKIP() << skip_reason;

  const std::optional<AcceleratorMemoryUsage> usage =
      GetProcessAcceleratorMemoryUsage();
  ASSERT_TRUE(usage.has_value());
  if (usage->total_bytes < kTolerance) {
    const std::optional<std::vector<NvmlDeviceMemory>> devices =
        ReadNvmlProcessMemory();
    if (devices.has_value() && HasForeignGpuProcesses(*devices)) {
      GTEST_SKIP() << "Running in a PID namespace; see the test above.";
    }
  }
  EXPECT_GE(usage->total_bytes, kTolerance)
      << "breakdown=" << FormatAcceleratorMemoryBreakdown(*usage);
  // One figure serves as present and as allocated (another GPU's DRM client
  // may add evicted buffers to the latter), and none of it is in the host's
  // figures.
  EXPECT_GE(usage->attributed_bytes, usage->total_bytes);
  EXPECT_EQ(usage->resident_counted_by_host_bytes, 0);
  EXPECT_EQ(usage->attributed_counted_by_host_bytes, 0);
  EXPECT_THAT(usage->breakdown,
              Contains(Pair("vram", testing::Ge(kTolerance))));
  // The per-GPU detail names the NVIDIA GPU.
  bool names_nvidia = false;
  for (const auto& [label, bytes] : usage->allocations) {
    names_nvidia |=
        absl::StrContains(label, "NVIDIA") && absl::EndsWith(label, "(vram)");
  }
  EXPECT_TRUE(names_nvidia);
}

}  // namespace
}  // namespace litert::lm
