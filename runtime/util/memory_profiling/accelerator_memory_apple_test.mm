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

// Tests of the Apple implementation: the rule that combines the kernel's
// ledger with Metal's figure, on fixed inputs, and live checks on a Mac with
// a Metal GPU that a Metal buffer and weights mapped from a file end up where
// the design says (see the top of accelerator_memory_apple.mm) in
// GetProcessAcceleratorMemoryUsage() and in a MemorySnapshot. For the raw
// measurements the design rests on, see accelerator_memory_apple_metal_test.

#import <Metal/Metal.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory.h"
#include "runtime/util/memory_profiling/accelerator_memory_apple.h"
#include "runtime/util/memory_profiling/host_memory.h"
#include "runtime/util/memory_profiling/memory_snapshot.h"

namespace litert::lm {
namespace {

using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Pair;

constexpr uint64_t kMb = 1024 * 1024;
constexpr uint64_t kBufferBytes = 256 * kMb;
// Half the allocation: generous against driver overhead and the odd page.
constexpr uint64_t kTolerance = kBufferBytes / 2;
// How far the host figure may drift from unrelated activity during a test.
constexpr uint64_t kHostJitter = 32 * kMb;

// --- The rule, on fixed inputs -----------------------------------------------

TEST(AcceleratorMemoryUsageFromAppleSourcesTest, NothingUsedIsZero) {
  const AcceleratorMemoryUsage usage = AcceleratorMemoryUsageFromAppleSources(
      AppleAcceleratorLedger(), {{.name = "Apple M3", .allocated_bytes = 0}});
  EXPECT_EQ(usage.total_bytes, 0);
  EXPECT_EQ(usage.attributed_bytes, 0);
  EXPECT_THAT(usage.breakdown, IsEmpty());
  EXPECT_THAT(usage.allocations, ElementsAre(Pair("Apple M3 (Metal)", 0)));
}

TEST(AcceleratorMemoryUsageFromAppleSourcesTest, ChargedMemoryWithOverheadIsTheLedger) {
  // A 256 MB private buffer as the M5 Pro accounts for it: the kernel charges
  // 300 MB, Metal counts 256 MB. The kernel's figure stands, and all of it is
  // already in phys_footprint.
  const AcceleratorMemoryUsage usage = AcceleratorMemoryUsageFromAppleSources(
      {.graphics_bytes = 300 * kMb}, {{.name = "Apple M5 Pro", .allocated_bytes = 256 * kMb}});
  EXPECT_EQ(usage.total_bytes, 300 * kMb);
  EXPECT_EQ(usage.attributed_bytes, 300 * kMb);
  EXPECT_EQ(usage.attributed_counted_by_host_bytes, 300 * kMb);
  EXPECT_EQ(usage.resident_counted_by_host_bytes, 0);
  EXPECT_THAT(usage.breakdown, ElementsAre(Pair("graphics", 300 * kMb)));
}

TEST(AcceleratorMemoryUsageFromAppleSourcesTest, MappedWeightsArePresentButNotAttributed) {
  // The same buffer plus 4 GB of weights wrapped without a copy: Metal counts
  // them, the kernel does not. They are in RAM, so present; charged to
  // nobody, so not attributed.
  const AcceleratorMemoryUsage usage = AcceleratorMemoryUsageFromAppleSources(
      {.graphics_bytes = 300 * kMb},
      {{.name = "Apple M5 Pro", .allocated_bytes = (256 + 4096) * kMb}});
  EXPECT_EQ(usage.total_bytes, (256 + 4096) * kMb);
  EXPECT_EQ(usage.attributed_bytes, 300 * kMb);
  EXPECT_EQ(usage.attributed_counted_by_host_bytes, 300 * kMb);
  EXPECT_THAT(usage.breakdown,
              ElementsAre(Pair("graphics", 300 * kMb), Pair("mapped weights", (4096 - 44) * kMb)));
}

TEST(AcceleratorMemoryUsageFromAppleSourcesTest, CompressedPagesAreAttributedButNotPresent) {
  // 150 MB of graphics memory of which the kernel has compressed 50 MB:
  // charged in full, 100 MB in RAM, nothing left over for mapped weights.
  const AcceleratorMemoryUsage usage = AcceleratorMemoryUsageFromAppleSources(
      {.graphics_bytes = 100 * kMb, .graphics_compressed_bytes = 50 * kMb},
      {{.name = "Apple M5 Pro", .allocated_bytes = 150 * kMb}});
  EXPECT_EQ(usage.total_bytes, 100 * kMb);
  EXPECT_EQ(usage.attributed_bytes, 150 * kMb);
  EXPECT_EQ(usage.attributed_counted_by_host_bytes, 150 * kMb);
  EXPECT_THAT(usage.breakdown, ElementsAre(Pair("graphics", 100 * kMb)));
}

TEST(AcceleratorMemoryUsageFromAppleSourcesTest, NeuralEngineIsItsOwnRegion) {
  const AcceleratorMemoryUsage usage = AcceleratorMemoryUsageFromAppleSources(
      {.graphics_bytes = 10 * kMb, .neural_bytes = 20 * kMb}, {});
  EXPECT_EQ(usage.total_bytes, 30 * kMb);
  EXPECT_EQ(usage.attributed_bytes, 30 * kMb);
  EXPECT_THAT(usage.breakdown, ElementsAre(Pair("graphics", 10 * kMb), Pair("neural", 20 * kMb)));
  EXPECT_THAT(usage.allocations, IsEmpty());
}

// --- Live, on a Metal GPU ----------------------------------------------------

// A 256 MB file of non-zero bytes mapped read-only and shared, so that the
// GPU reads the page cache in place and the pages stay clean, file-backed and
// outside phys_footprint throughout (accelerator_memory_apple_metal_test).
class MappedFile {
 public:
  static MappedFile Create() {
    MappedFile file;
    const char* tmpdir = std::getenv("TEST_TMPDIR");
    file.path_ = absl::StrCat(tmpdir != nullptr ? tmpdir : "/tmp", "/apple_test_weights.bin");
    const int fd = open(file.path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return file;
    std::vector<uint8_t> chunk(kMb, 0x5a);
    for (uint64_t written = 0; written < kBufferBytes; written += kMb) {
      if (write(fd, chunk.data(), chunk.size()) != static_cast<ssize_t>(kMb)) {
        close(fd);
        return file;
      }
    }
    void* data = mmap(nullptr, kBufferBytes, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (data != MAP_FAILED) file.data_ = data;
    return file;
  }
  ~MappedFile() {
    if (data_ != nullptr) munmap(data_, kBufferBytes);
    if (!path_.empty()) unlink(path_.c_str());
  }
  void* data() const { return data_; }

 private:
  std::string path_;
  void* data_ = nullptr;
};

class AppleAcceleratorMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = MTLCreateSystemDefaultDevice();
    if (device_ == nil) GTEST_SKIP() << "No Metal device on this machine.";
    queue_ = [device_ newCommandQueue];
  }

  // Makes the GPU write every byte of `buffer` (which backs a private
  // buffer's pages and charges it to the process).
  void GpuFill(id<MTLBuffer> buffer) {
    id<MTLCommandBuffer> commands = [queue_ commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
    [blit fillBuffer:buffer range:NSMakeRange(0, buffer.length) value:1];
    [blit endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
  }

  // Makes the GPU read every byte of `source` by copying it into `scratch`.
  void GpuRead(id<MTLBuffer> source, id<MTLBuffer> scratch) {
    id<MTLCommandBuffer> commands = [queue_ commandBuffer];
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

  static int64_t Delta(uint64_t before, uint64_t after) {
    return static_cast<int64_t>(after) - static_cast<int64_t>(before);
  }

  static void LogSnapshot(const char* title, const MemorySnapshot& snapshot) {
    ABSL_LOG(INFO) << title << ": " << snapshot;
  }

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
};

TEST_F(AppleAcceleratorMemoryTest, Apple_SourcesAreAvailable) {
  EXPECT_TRUE(ReadAppleAcceleratorLedger().has_value());
  const std::vector<MetalDeviceMemory> devices = ReadMetalAllocatedMemory();
  ASSERT_FALSE(devices.empty());
  bool names_this_device = false;
  for (const MetalDeviceMemory& d : devices) {
    names_this_device |= d.name == device_.name.UTF8String;
  }
  EXPECT_TRUE(names_this_device);
  EXPECT_TRUE(GetProcessAcceleratorMemoryUsage().has_value());
}

// A buffer the CPU has written is charged to phys_footprint: the attributed
// figure grows by it exactly as Activity Monitor would show, with the growth
// on its GPU side and its CPU side flat. Having been written by the CPU, the
// buffer is in the process's resident count as well as on the GPU, and the
// resident figure shows it on both sides (see the file comment of
// accelerator_memory_apple.mm for why that cannot be corrected).
TEST_F(AppleAcceleratorMemoryTest, Apple_ChargedBufferIsOnTheGpuSideOfTheAttributedFigure) {
  const MemorySnapshot before = TakeMemorySnapshot();
  const HostMemoryUsage host_before = GetHostMemoryUsage();
  LogSnapshot("before", before);
  ASSERT_TRUE(before.accelerator.has_value());
  ASSERT_TRUE(before.attributed_by_os.has_value());
  ASSERT_TRUE(before.resident_in_ram.has_value());
  ASSERT_TRUE(host_before.platform_attributed_bytes.has_value());

  @autoreleasepool {
    id<MTLBuffer> buffer = [device_ newBufferWithLength:kBufferBytes
                                                options:MTLResourceStorageModeShared];
    ASSERT_NE(buffer, nil);
    std::memset(buffer.contents, 1, kBufferBytes);

    const MemorySnapshot after = TakeMemorySnapshot();
    const HostMemoryUsage host_after = GetHostMemoryUsage();
    LogSnapshot("with a 256 MB shared buffer", after);
    ASSERT_TRUE(after.accelerator.has_value());
    ASSERT_TRUE(after.attributed_by_os.has_value());
    ASSERT_TRUE(after.resident_in_ram.has_value());
    ASSERT_TRUE(host_after.platform_attributed_bytes.has_value());

    // The GPU detail sees the buffer as graphics memory, present and charged.
    EXPECT_GE(Delta(before.accelerator->total_bytes, after.accelerator->total_bytes),
              static_cast<int64_t>(kTolerance));
    EXPECT_GE(Delta(before.accelerator->attributed_bytes, after.accelerator->attributed_bytes),
              static_cast<int64_t>(kTolerance));
    EXPECT_THAT(after.accelerator->breakdown, Contains(Pair("graphics", testing::Ge(kTolerance))));

    // The attributed figure is phys_footprint as the OS reports it ...
    const int64_t footprint_delta =
        Delta(*host_before.platform_attributed_bytes, *host_after.platform_attributed_bytes);
    EXPECT_GE(footprint_delta, static_cast<int64_t>(kTolerance));
    EXPECT_EQ(Delta(before.attributed_by_os->total_bytes, after.attributed_by_os->total_bytes),
              footprint_delta);
    // ... with the buffer on its accelerator side and its CPU side unmoved.
    EXPECT_GE(Delta(before.attributed_by_os->accelerator_bytes,
                    after.attributed_by_os->accelerator_bytes),
              static_cast<int64_t>(kTolerance));
    EXPECT_LE(
        std::abs(Delta(before.attributed_by_os->cpu_bytes, after.attributed_by_os->cpu_bytes)),
        static_cast<int64_t>(kHostJitter));

    // The resident figure has it on the accelerator side, and (documented) on
    // the CPU side too since the CPU wrote it.
    EXPECT_GE(
        Delta(before.resident_in_ram->accelerator_bytes, after.resident_in_ram->accelerator_bytes),
        static_cast<int64_t>(kTolerance));

    bool names_this_device = false;
    for (const auto& [label, bytes] : after.accelerator->allocations) {
      names_this_device |= absl::StrContains(label, device_.name.UTF8String);
    }
    EXPECT_TRUE(names_this_device);
  }
}

// Weights mapped from a file and handed to the GPU without a copy are not in
// phys_footprint, so the attributed figure does not move, but they do occupy
// RAM the model needs: they are present on the GPU side of the resident
// figure, as "mapped weights".
TEST_F(AppleAcceleratorMemoryTest, Apple_MappedWeightsAreResidentOnTheGpuNotAttributed) {
  const MappedFile file = MappedFile::Create();
  ASSERT_NE(file.data(), nullptr) << "Could not create or map the file.";

  @autoreleasepool {
    // The GPU reads the weights by copying them into `scratch`; back it first
    // so that it is a constant between the two snapshots.
    id<MTLBuffer> scratch = [device_ newBufferWithLength:kBufferBytes
                                                 options:MTLResourceStorageModePrivate];
    ASSERT_NE(scratch, nil);
    GpuFill(scratch);

    const MemorySnapshot before = TakeMemorySnapshot();
    LogSnapshot("before", before);
    ASSERT_TRUE(before.accelerator.has_value());
    ASSERT_TRUE(before.attributed_by_os.has_value());
    ASSERT_TRUE(before.resident_in_ram.has_value());

    id<MTLBuffer> weights = [device_ newBufferWithBytesNoCopy:file.data()
                                                       length:kBufferBytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
    ASSERT_NE(weights, nil) << "newBufferWithBytesNoCopy refused the mapping.";
    GpuRead(weights, scratch);

    const MemorySnapshot after = TakeMemorySnapshot();
    LogSnapshot("with 256 MB of mapped weights", after);
    ASSERT_TRUE(after.accelerator.has_value());
    ASSERT_TRUE(after.attributed_by_os.has_value());
    ASSERT_TRUE(after.resident_in_ram.has_value());

    EXPECT_GE(Delta(before.accelerator->total_bytes, after.accelerator->total_bytes),
              static_cast<int64_t>(kTolerance));
    EXPECT_THAT(after.accelerator->breakdown,
                Contains(Pair("mapped weights", testing::Ge(kTolerance))));
    EXPECT_LE(
        std::abs(Delta(before.accelerator->attributed_bytes, after.accelerator->attributed_bytes)),
        static_cast<int64_t>(kHostJitter));
    // Attributed: unmoved, as Activity Monitor. Resident: the weights, on the
    // accelerator side.
    EXPECT_LE(
        std::abs(Delta(before.attributed_by_os->total_bytes, after.attributed_by_os->total_bytes)),
        static_cast<int64_t>(kHostJitter));
    EXPECT_GE(
        Delta(before.resident_in_ram->accelerator_bytes, after.resident_in_ram->accelerator_bytes),
        static_cast<int64_t>(kTolerance));
    EXPECT_GE(Delta(before.resident_in_ram->total_bytes, after.resident_in_ram->total_bytes),
              static_cast<int64_t>(kTolerance));
  }
}

}  // namespace
}  // namespace litert::lm
