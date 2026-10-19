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

// Measures, on a Mac with a Metal GPU, how Metal allocations show up in the
// two host figures of host_memory.h (resident_size and phys_footprint) and in
// Metal's own per-process figure (MTLDevice.currentAllocatedSize). Its output
// is the data the Apple implementation of the platform contract in
// accelerator_memory.h will be built on; the test itself only asserts that
// Metal accounts for what it allocates. Run it explicitly:
//
//   bazel test --config=darwin_arm64 --test_output=all \
//     //runtime/util/memory_profiling:accelerator_memory_apple_metal_test
//
// Three kinds of buffer are measured, each 256 MB:
//   - StorageModeShared, allocated by Metal and written by the CPU: the
//     ordinary case for activations and KV cache.
//   - StorageModePrivate, filled by the GPU: GPU-only memory.
//   - StorageModeShared over an mmap()ed, read-only file, through
//     newBufferWithBytesNoCopy: how ML Drift's Metal backend
//     (gpu_backend_metal.mm) hands LiteRT-LM's weights, mapped from the model
//     file with PROT_READ|PROT_WRITE, MAP_PRIVATE and never written, to the
//     GPU. Clean file-backed pages are not in phys_footprint, so this case
//     decides whether the weights appear in the accelerator figure, the host
//     figure, both or neither.

#import <Metal/Metal.h>

#include <fcntl.h>
#include <libproc.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/host_memory.h"

namespace litert::lm {
namespace {

constexpr uint64_t kMb = 1024 * 1024;
constexpr uint64_t kBufferBytes = 256 * kMb;

// One reading of the figures of interest: the two of host_memory.h, Metal's
// own, and the parts of the kernel's ledger that explain them. `internal` /
// `external` split the resident size into anonymous and file-backed pages;
// `graphics` is the part of phys_footprint the kernel attributes to the GPU
// (task_vm_info.ledger_tag_graphics_footprint, incl. compressed), `neural`
// the part attributed to the Neural Engine.
struct Reading {
  uint64_t resident = 0;
  uint64_t footprint = 0;
  uint64_t metal_allocated = 0;
  uint64_t internal = 0;
  uint64_t external = 0;
  // Pages from device / shared-memory pagers: neither anonymous nor
  // file-backed (task_vm_info.device).
  uint64_t device = 0;
  uint64_t graphics = 0;
  // Graphics-tagged memory the kernel keeps out of phys_footprint
  // (task_vm_info.ledger_tag_graphics_nofootprint).
  uint64_t graphics_nofootprint = 0;
  uint64_t neural = 0;
};

Reading Read(id<MTLDevice> device) {
  Reading reading;
  const HostMemoryUsage host = GetHostMemoryUsage();
  reading.resident = host.resident_bytes.value_or(0);
  reading.footprint = host.platform_attributed_bytes.value_or(0);
  reading.metal_allocated = static_cast<uint64_t>(device.currentAllocatedSize);
  task_vm_info_data_t info = {};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) ==
      KERN_SUCCESS) {
    reading.internal = info.internal;
    reading.external = info.external;
    reading.device = info.device;
    reading.graphics =
        info.ledger_tag_graphics_footprint + info.ledger_tag_graphics_footprint_compressed;
    reading.graphics_nofootprint =
        info.ledger_tag_graphics_nofootprint + info.ledger_tag_graphics_nofootprint_compressed;
    reading.neural = info.ledger_tag_neural_footprint + info.ledger_tag_neural_footprint_compressed;
  }
  return reading;
}

std::string SignedMb(uint64_t before, uint64_t after) {
  const double mb =
      (static_cast<double>(after) - static_cast<double>(before)) / static_cast<double>(kMb);
  return absl::StrFormat("%+8.2f", mb);
}

// Logs how each figure moved from `before` to `after`, in MB.
void LogDelta(const std::string& step, const Reading& before, const Reading& after) {
  ABSL_LOG(INFO) << absl::StrFormat(
      "%-42s resident %s (internal %s, external %s, device %s)  footprint %s "
      "(graphics %s, neural %s; graphics outside footprint %s)  metal %s",
      step, SignedMb(before.resident, after.resident), SignedMb(before.internal, after.internal),
      SignedMb(before.external, after.external), SignedMb(before.device, after.device),
      SignedMb(before.footprint, after.footprint), SignedMb(before.graphics, after.graphics),
      SignedMb(before.neural, after.neural),
      SignedMb(before.graphics_nofootprint, after.graphics_nofootprint),
      SignedMb(before.metal_allocated, after.metal_allocated));
}

// --- The task's VM map --------------------------------------------------------
//
// Whether the overlap between the host figures and Metal's can be measured
// rather than assumed depends on what the kernel's VM map shows for a file
// mapping once Metal wraps it and the GPU uses it: a changed tag or share
// mode, a new region, pages moving between regions. These helpers log that.

// One region of the map, from mach_vm_region_recurse().
struct Region {
  mach_vm_address_t address = 0;
  mach_vm_size_t size = 0;
  vm_region_submap_info_data_64_t info = {};
};

// Every region of the task, submaps expanded.
std::vector<Region> ReadRegions() {
  std::vector<Region> regions;
  mach_vm_address_t address = 0;
  natural_t depth = 0;
  while (true) {
    Region region;
    mach_msg_type_number_t count = VM_REGION_SUBMAP_INFO_COUNT_64;
    if (mach_vm_region_recurse(mach_task_self(), &address, &region.size, &depth,
                               reinterpret_cast<vm_region_recurse_info_t>(&region.info),
                               &count) != KERN_SUCCESS) {
      break;
    }
    if (region.info.is_submap) {
      ++depth;
      continue;
    }
    region.address = address;
    regions.push_back(region);
    address += region.size;
  }
  return regions;
}

const char* ShareModeName(unsigned char share_mode) {
  switch (share_mode) {
    case SM_COW:
      return "cow";
    case SM_PRIVATE:
      return "private";
    case SM_EMPTY:
      return "empty";
    case SM_SHARED:
      return "shared";
    case SM_TRUESHARED:
      return "trueshared";
    case SM_PRIVATE_ALIASED:
      return "private-aliased";
    case SM_SHARED_ALIASED:
      return "shared-aliased";
    case SM_LARGE_PAGE:
      return "large-page";
    default:
      return "?";
  }
}

// Logs the regions overlapping [start, start + length): the mapping under
// study, and whatever the driver may have put on top of it. Also asks the
// page-level query (mach_vm_page_range_query) how many of the range's pages it
// considers present, dirty, and referenced: whether that tracks the task's own
// resident count (the pmap) or the VM object's pages decides whether it can
// measure the overlap between the two.
void LogRegionsOverlapping(const std::string& step, const void* start, uint64_t length) {
  const auto begin = reinterpret_cast<mach_vm_address_t>(start);
  const mach_vm_address_t end = begin + length;
  for (const Region& r : ReadRegions()) {
    if (r.address + r.size <= begin || r.address >= end) continue;
    char path[PROC_PIDPATHINFO_MAXSIZE] = {};
    proc_regionfilename(getpid(), r.address, path, sizeof(path));
    ABSL_LOG(INFO) << absl::StrFormat(
        "%-42s region %#llx+%.1f MB prot=%d/%d share=%s tag=%u resident=%.1f MB "
        "dirty=%.1f MB external_pager=%d%s%s",
        step, r.address, r.size / static_cast<double>(kMb), r.info.protection,
        r.info.max_protection, ShareModeName(r.info.share_mode), r.info.user_tag,
        r.info.pages_resident * static_cast<double>(vm_page_size) / kMb,
        r.info.pages_dirtied * static_cast<double>(vm_page_size) / kMb, r.info.external_pager,
        path[0] != '\0' ? " file=" : "", path);
  }
  const uint64_t pages = length / vm_page_size;
  std::vector<int> dispositions(pages, 0);
  mach_vm_size_t count = pages;
  if (mach_vm_page_range_query(mach_task_self(), begin, length,
                               reinterpret_cast<mach_vm_address_t>(dispositions.data()),
                               &count) == KERN_SUCCESS) {
    uint64_t present = 0, dirty = 0, referenced = 0, external = 0;
    for (mach_vm_size_t i = 0; i < count; ++i) {
      present += (dispositions[i] & VM_PAGE_QUERY_PAGE_PRESENT) != 0;
      dirty += (dispositions[i] & VM_PAGE_QUERY_PAGE_DIRTY) != 0;
      referenced += (dispositions[i] & VM_PAGE_QUERY_PAGE_REF) != 0;
      external += (dispositions[i] & VM_PAGE_QUERY_PAGE_EXTERNAL) != 0;
    }
    ABSL_LOG(INFO) << absl::StrFormat(
        "%-42s page query: present %.1f MB, dirty %.1f MB, referenced %.1f MB, "
        "external %.1f MB",
        step, present * static_cast<double>(vm_page_size) / kMb,
        dirty * static_cast<double>(vm_page_size) / kMb,
        referenced * static_cast<double>(vm_page_size) / kMb,
        external * static_cast<double>(vm_page_size) / kMb);
  }
}

// Resident bytes per VM tag over the whole task: a new mapping the driver
// creates elsewhere (over the same pages, say) shows up as growth under its
// tag.
std::map<unsigned int, uint64_t> ResidentBytesByTag() {
  std::map<unsigned int, uint64_t> by_tag;
  for (const Region& r : ReadRegions()) {
    by_tag[r.info.user_tag] += static_cast<uint64_t>(r.info.pages_resident) * vm_page_size;
  }
  return by_tag;
}

// Logs the tags whose resident total moved by a megabyte or more.
void LogTagDelta(const std::string& step, const std::map<unsigned int, uint64_t>& before,
                 const std::map<unsigned int, uint64_t>& after) {
  std::string moved;
  for (const auto& [tag, bytes] : after) {
    const auto it = before.find(tag);
    const int64_t delta =
        static_cast<int64_t>(bytes) - static_cast<int64_t>(it == before.end() ? 0 : it->second);
    if (delta >= static_cast<int64_t>(kMb) || delta <= -static_cast<int64_t>(kMb)) {
      absl::StrAppendFormat(&moved, " tag %u %+.1f MB;", tag, delta / static_cast<double>(kMb));
    }
  }
  ABSL_LOG(INFO) << absl::StrFormat("%-42s resident by VM tag:%s", step,
                                    moved.empty() ? " no change" : moved.c_str());
}

// Makes the GPU write every byte of `buffer`.
void GpuFill(id<MTLCommandQueue> queue, id<MTLBuffer> buffer) {
  id<MTLCommandBuffer> commands = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
  [blit fillBuffer:buffer range:NSMakeRange(0, buffer.length) value:1];
  [blit endEncoding];
  [commands commit];
  [commands waitUntilCompleted];
}

// Makes the GPU read every byte of `source` by copying it into `scratch` (a
// private buffer of the same size), so that pages the GPU has not touched yet
// get faulted in on its behalf.
void GpuRead(id<MTLCommandQueue> queue, id<MTLBuffer> source, id<MTLBuffer> scratch) {
  id<MTLCommandBuffer> commands = [queue commandBuffer];
  id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
  [blit copyFromBuffer:source
           sourceOffset:0
               toBuffer:scratch
      destinationOffset:0
                   size:source.length];
  [blit endEncoding];
  [commands commit];
  [commands waitUntilCompleted];
}

// A 256 MB file of non-zero bytes, mapped with the given mmap() protection
// and flags. LiteRT-LM maps a model (memory_mapped_file_posix.cc) with
// PROT_READ|PROT_WRITE, MAP_PRIVATE and never writes to it, so its pages stay
// clean and file-backed.
class MappedFile {
 public:
  static MappedFile Create(int prot, int flags) {
    MappedFile file;
    const char* tmpdir = std::getenv("TEST_TMPDIR");
    file.path_ = absl::StrCat(tmpdir != nullptr ? tmpdir : "/tmp", "/metal_weights.bin");
    const int fd = open(file.path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return file;
    std::vector<uint8_t> chunk(kMb, 0x5a);
    for (uint64_t written = 0; written < kBufferBytes; written += kMb) {
      if (write(fd, chunk.data(), chunk.size()) != static_cast<ssize_t>(kMb)) {
        close(fd);
        return file;
      }
    }
    void* data = mmap(nullptr, kBufferBytes, prot, flags, fd, 0);
    close(fd);
    if (data != MAP_FAILED) file.data_ = data;
    return file;
  }

  ~MappedFile() {
    if (data_ != nullptr) munmap(data_, kBufferBytes);
    if (!path_.empty()) unlink(path_.c_str());
  }

  void* data() const { return data_; }

  // Reads every page so that it is resident, as a model's weights are after
  // the first run.
  uint64_t TouchAllPages() const {
    uint64_t sum = 0;
    const auto* bytes = static_cast<const volatile uint8_t*>(data_);
    for (uint64_t offset = 0; offset < kBufferBytes; offset += 4096) {
      sum += bytes[offset];
    }
    return sum;
  }

 private:
  std::string path_;
  void* data_ = nullptr;
};

class AppleMetalMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = MTLCreateSystemDefaultDevice();
    if (device_ == nil) GTEST_SKIP() << "No Metal device on this machine.";
    queue_ = [device_ newCommandQueue];
    ABSL_LOG(INFO) << "Device: " << device_.name.UTF8String
                   << ", unified memory: " << (device_.hasUnifiedMemory ? "yes" : "no")
                   << ", recommended max working set: "
                   << device_.recommendedMaxWorkingSetSize / kMb << " MB";
    const Reading start = Read(device_);
    ABSL_LOG(INFO) << absl::StrFormat(
        "Start: resident %.2f MB (internal %.2f, external %.2f), footprint "
        "%.2f MB (graphics %.2f, neural %.2f), metal %.2f MB",
        start.resident / static_cast<double>(kMb), start.internal / static_cast<double>(kMb),
        start.external / static_cast<double>(kMb), start.footprint / static_cast<double>(kMb),
        start.graphics / static_cast<double>(kMb), start.neural / static_cast<double>(kMb),
        start.metal_allocated / static_cast<double>(kMb));
  }

  // Maps the file with `prot` / `flags`, has the CPU read it, wraps it for the
  // GPU without a copy, has the GPU read it, then the CPU again, logging the
  // figures after each step.
  void MeasureMappedWeights(int prot, int flags, const std::string& label) {
    const MappedFile file = MappedFile::Create(prot, flags);
    ASSERT_NE(file.data(), nullptr) << "Could not create or map the file.";

    const Reading before = Read(device_);
    const uint64_t checksum = file.TouchAllPages();
    const Reading touched = Read(device_);
    LogDelta(label + ": after CPU read of every page", before, touched);
    ASSERT_GT(checksum, 0u);
    LogRegionsOverlapping(label + ": after CPU read", file.data(), kBufferBytes);
    std::map<unsigned int, uint64_t> tags = ResidentBytesByTag();

    // Expected from the host side alone: resident up by ~256 MB, footprint
    // flat, since the pages are clean and file-backed.
    Reading gpu_read;
    @autoreleasepool {
      // The GPU reads the weights by copying them into `scratch`. Back
      // scratch with a GPU fill first (which, as PrivateBufferFilledByGpu
      // shows, is what charges a private buffer to phys_footprint), so that
      // the copy's only new effect is the read of the weights.
      id<MTLBuffer> scratch = [device_ newBufferWithLength:kBufferBytes
                                                   options:MTLResourceStorageModePrivate];
      ASSERT_NE(scratch, nil);
      GpuFill(queue_, scratch);
      const Reading with_scratch = Read(device_);
      LogDelta(label + ": (scratch private buffer, GPU-filled)", touched, with_scratch);
      {
        const std::map<unsigned int, uint64_t> now = ResidentBytesByTag();
        LogTagDelta(label + ": (scratch private buffer)", tags, now);
        tags = now;
      }

      id<MTLBuffer> weights = [device_ newBufferWithBytesNoCopy:file.data()
                                                         length:kBufferBytes
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
      if (weights == nil) {
        ABSL_LOG(INFO) << label << ": newBufferWithBytesNoCopy refused the mapping.";
        return;
      }
      const Reading wrapped = Read(device_);
      LogDelta(label + ": after newBufferWithBytesNoCopy", with_scratch, wrapped);
      LogRegionsOverlapping(label + ": after wrap", file.data(), kBufferBytes);
      {
        const std::map<unsigned int, uint64_t> now = ResidentBytesByTag();
        LogTagDelta(label + ": after wrap", tags, now);
        tags = now;
      }

      GpuRead(queue_, weights, scratch);
      gpu_read = Read(device_);
      LogDelta(label + ": after GPU read of every page", wrapped, gpu_read);
      LogRegionsOverlapping(label + ": after GPU read", file.data(), kBufferBytes);
      {
        const std::map<unsigned int, uint64_t> now = ResidentBytesByTag();
        LogTagDelta(label + ": after GPU read", tags, now);
        tags = now;
      }

      // Whether pages the GPU has read are still readable from the page
      // cache for free, or were moved / evicted and fault back in.
      ASSERT_GT(file.TouchAllPages(), 0u);
      const Reading reread = Read(device_);
      LogDelta(label + ": after CPU re-read of every page", gpu_read, reread);
      LogRegionsOverlapping(label + ": after CPU re-read", file.data(), kBufferBytes);
      {
        const std::map<unsigned int, uint64_t> now = ResidentBytesByTag();
        LogTagDelta(label + ": after CPU re-read", tags, now);
        tags = now;
      }
      gpu_read = reread;

      weights = nil;
      scratch = nil;
    }
    LogDelta(label + ": after releasing both buffers", gpu_read, Read(device_));
    LogRegionsOverlapping(label + ": after release", file.data(), kBufferBytes);
    LogTagDelta(label + ": after release", tags, ResidentBytesByTag());
  }

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
};

TEST_F(AppleMetalMemoryTest, SharedBufferWrittenByCpu) {
  Reading written;
  @autoreleasepool {
    const Reading before = Read(device_);
    id<MTLBuffer> buffer = [device_ newBufferWithLength:kBufferBytes
                                                options:MTLResourceStorageModeShared];
    ASSERT_NE(buffer, nil);
    const Reading allocated = Read(device_);
    LogDelta("shared: allocated, untouched", before, allocated);

    std::memset(buffer.contents, 1, kBufferBytes);
    written = Read(device_);
    LogDelta("shared: after CPU write", allocated, written);
    // What a Metal-owned mapping looks like in the VM map (its tag is the
    // handle for telling Metal's memory from the process's own).
    LogRegionsOverlapping("shared: Metal's own mapping", buffer.contents, kBufferBytes);

    // The one invariant this test insists on: Metal accounts for what it
    // allocates. Everything else is measurement.
    EXPECT_GE(written.metal_allocated, before.metal_allocated + kBufferBytes / 2);

    buffer = nil;
  }
  LogDelta("shared: after release", written, Read(device_));
}

TEST_F(AppleMetalMemoryTest, PrivateBufferFilledByGpu) {
  @autoreleasepool {
    const Reading before = Read(device_);
    id<MTLBuffer> buffer = [device_ newBufferWithLength:kBufferBytes
                                                options:MTLResourceStorageModePrivate];
    ASSERT_NE(buffer, nil);
    const Reading allocated = Read(device_);
    LogDelta("private: allocated, untouched", before, allocated);

    GpuFill(queue_, buffer);
    const Reading filled = Read(device_);
    LogDelta("private: after GPU fill", allocated, filled);

    EXPECT_GE(filled.metal_allocated, before.metal_allocated + kBufferBytes / 2);
  }
}

// A shared buffer only the GPU writes: the pattern of a KV cache or of
// activations. Whether its pages are in the task's resident count without the
// CPU ever touching them, and what the region query says about them in that
// case, decides whether the region query can stand in for the task's own
// resident count when attributing resident memory to the GPU.
TEST_F(AppleMetalMemoryTest, SharedBufferFilledByGpu) {
  Reading filled;
  @autoreleasepool {
    const Reading before = Read(device_);
    const std::map<unsigned int, uint64_t> tags = ResidentBytesByTag();
    id<MTLBuffer> buffer = [device_ newBufferWithLength:kBufferBytes
                                                options:MTLResourceStorageModeShared];
    ASSERT_NE(buffer, nil);
    const Reading allocated = Read(device_);
    LogDelta("shared/gpu: allocated, untouched", before, allocated);
    LogRegionsOverlapping("shared/gpu: untouched", buffer.contents, kBufferBytes);

    GpuFill(queue_, buffer);
    filled = Read(device_);
    LogDelta("shared/gpu: after GPU fill", allocated, filled);
    LogRegionsOverlapping("shared/gpu: after GPU fill", buffer.contents, kBufferBytes);
    LogTagDelta("shared/gpu: after GPU fill", tags, ResidentBytesByTag());

    // And once the CPU reads what the GPU wrote.
    uint64_t sum = 0;
    const auto* bytes = static_cast<const volatile uint8_t*>(buffer.contents);
    for (uint64_t offset = 0; offset < kBufferBytes; offset += 4096) sum += bytes[offset];
    ASSERT_GT(sum, 0u);
    const Reading read = Read(device_);
    LogDelta("shared/gpu: after CPU read", filled, read);
    LogRegionsOverlapping("shared/gpu: after CPU read", buffer.contents, kBufferBytes);
    LogTagDelta("shared/gpu: after CPU read", tags, ResidentBytesByTag());

    EXPECT_GE(filled.metal_allocated, before.metal_allocated + kBufferBytes / 2);
    filled = read;
    buffer = nil;
  }
  LogDelta("shared/gpu: after release", filled, Read(device_));
}

// The mapping LiteRT-LM uses for a model: writable and private, never written.
TEST_F(AppleMetalMemoryTest, MappedWeightsWrappedWithoutCopy) {
  MeasureMappedWeights(PROT_READ | PROT_WRITE, MAP_PRIVATE, "mmap rw/private");
}

// The same with a read-only shared mapping (what llama.cpp uses), to tell
// whether it is the writable private mapping that makes the kernel turn the
// pages into anonymous memory once the GPU has used them.
TEST_F(AppleMetalMemoryTest, ReadOnlyMappedWeightsWrappedWithoutCopy) {
  MeasureMappedWeights(PROT_READ, MAP_SHARED, "mmap ro/shared");
}

// And read-only but still private: whether it is the write permission or the
// private mapping that matters. A loader that must map some of a file
// copy-on-write (to patch it) can still map the rest like this.
TEST_F(AppleMetalMemoryTest, ReadOnlyPrivateMappedWeightsWrappedWithoutCopy) {
  MeasureMappedWeights(PROT_READ, MAP_PRIVATE, "mmap ro/private");
}

}  // namespace
}  // namespace litert::lm
