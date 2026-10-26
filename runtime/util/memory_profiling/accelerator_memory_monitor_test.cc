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

#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::IsEmpty;
using ::testing::Pair;

constexpr uint64_t kMb = 1024 * 1024;

// Replays a fixed sequence of samples. Once the sequence is exhausted, the
// last sample is repeated and `exhausted` is notified so tests can stop the
// monitor deterministically. SleepFor() is a no-op so tests run fast.
class ScriptedSampler : public AcceleratorMemoryMonitor::Sampler {
 public:
  ScriptedSampler(std::vector<std::optional<AcceleratorMemoryUsage>> samples,
                  absl::Notification* exhausted)
      : samples_(std::move(samples)), exhausted_(exhausted) {}

  bool IsSupported() override { return true; }

  std::optional<AcceleratorMemoryUsage> Sample() override {
    absl::MutexLock lock(mutex_);
    if (next_ < samples_.size()) {
      return samples_[next_++];
    }
    if (!exhausted_->HasBeenNotified()) {
      exhausted_->Notify();
    }
    return samples_.empty() ? std::nullopt : samples_.back();
  }

  void SleepFor(absl::Duration duration) override {}

 private:
  absl::Mutex mutex_;
  std::vector<std::optional<AcceleratorMemoryUsage>> samples_;
  size_t next_ ABSL_GUARDED_BY(mutex_) = 0;
  absl::Notification* exhausted_;
};

class UnsupportedSampler : public AcceleratorMemoryMonitor::Sampler {
 public:
  bool IsSupported() override { return false; }
  std::optional<AcceleratorMemoryUsage> Sample() override {
    ADD_FAILURE() << "Sample() must not be called when unsupported.";
    return std::nullopt;
  }
};

AcceleratorMemoryUsage MakeUsage(
    uint64_t total_bytes,
    std::vector<std::pair<std::string, uint64_t>> breakdown = {}) {
  AcceleratorMemoryUsage usage;
  usage.total_bytes = total_bytes;
  usage.breakdown = std::move(breakdown);
  return usage;
}

// Whether tracking is supported depends on the platform and is checked by the
// per-platform tests; what every platform must satisfy is the contract on the
// shape of the result.
TEST(GetProcessAcceleratorMemoryUsageTest, BreakdownSumsToTotalWhenSupported) {
  const std::optional<AcceleratorMemoryUsage> usage =
      GetProcessAcceleratorMemoryUsage();
  if (!usage.has_value()) {
    GTEST_SKIP() << "Accelerator memory tracking is unsupported here.";
  }
  if (usage->breakdown.empty()) {
    GTEST_SKIP() << "This platform offers no breakdown.";
  }
  uint64_t breakdown_total = 0;
  for (const auto& [_, bytes] : usage->breakdown) breakdown_total += bytes;
  EXPECT_EQ(breakdown_total, usage->total_bytes);
}

TEST(AcceleratorMemoryBytesToMbTest, ConvertsMebibytes) {
  EXPECT_THAT(AcceleratorMemoryBytesToMb(0), FloatEq(0.0f));
  EXPECT_THAT(AcceleratorMemoryBytesToMb(kMb), FloatEq(1.0f));
  EXPECT_THAT(AcceleratorMemoryBytesToMb(kMb + kMb / 2), FloatEq(1.5f));
}

TEST(FormatAcceleratorMemoryBreakdownTest, EmptyBreakdownFormatsAsEmptyString) {
  EXPECT_THAT(FormatAcceleratorMemoryBreakdown(MakeUsage(5 * kMb)), IsEmpty());
}

TEST(FormatAcceleratorMemoryBreakdownTest, FormatsEntriesInOrder) {
  const AcceleratorMemoryUsage usage =
      MakeUsage(3 * kMb, {{"local", 2 * kMb}, {"non-local", kMb}});
  EXPECT_EQ(FormatAcceleratorMemoryBreakdown(usage),
            "local=2.00 MB, non-local=1.00 MB");
}

TEST(FormatAcceleratorMemoryAllocationsTest, NoAllocationsFormatsAsNoLines) {
  EXPECT_THAT(FormatAcceleratorMemoryAllocations(MakeUsage(5 * kMb)),
              IsEmpty());
}

TEST(FormatAcceleratorMemoryAllocationsTest, ListsAllocationsOfAtLeastOneMb) {
  AcceleratorMemoryUsage usage = MakeUsage(13 * kMb);
  usage.allocations = {
      {"weights (Inode 12345)", 10 * kMb},
      {"kv_cache (Inode 67890)", 2 * kMb},
      {"small (Inode 99999)", 512 * 1024},  // Below 1 MiB: omitted.
      {"exactly_one_mb (Inode 1)", kMb},
  };

  EXPECT_THAT(FormatAcceleratorMemoryAllocations(usage),
              ElementsAre("  -    10.00 MB  weights (Inode 12345)",
                          "  -     2.00 MB  kv_cache (Inode 67890)",
                          "  -     1.00 MB  exactly_one_mb (Inode 1)"));
}

TEST(AcceleratorMemoryMonitorTest, UnsupportedSamplerIsNoOp) {
  AcceleratorMemoryMonitor monitor(absl::Milliseconds(1),
                                   std::make_unique<UnsupportedSampler>());
  EXPECT_FALSE(monitor.IsSupported());
  monitor.Start();
  monitor.Stop();
  EXPECT_FALSE(monitor.GetPeakUsage().has_value());
}

TEST(AcceleratorMemoryMonitorTest, NullSamplerIsUnsupported) {
  AcceleratorMemoryMonitor monitor(absl::Milliseconds(1),
                                   /*sampler=*/nullptr);
  EXPECT_FALSE(monitor.IsSupported());
  monitor.Start();
  monitor.Stop();
  EXPECT_FALSE(monitor.GetPeakUsage().has_value());
}

TEST(AcceleratorMemoryMonitorTest, DefaultSamplerMatchesPlatformSupport) {
  AcceleratorMemoryMonitor monitor(absl::Milliseconds(1));
  EXPECT_EQ(monitor.IsSupported(),
            GetProcessAcceleratorMemoryUsage().has_value());
}

TEST(AcceleratorMemoryMonitorTest, NoPeakBeforeStart) {
  absl::Notification exhausted;
  AcceleratorMemoryMonitor monitor(
      absl::Milliseconds(1),
      std::make_unique<ScriptedSampler>(
          std::vector<std::optional<AcceleratorMemoryUsage>>{MakeUsage(kMb)},
          &exhausted));
  EXPECT_TRUE(monitor.IsSupported());
  EXPECT_FALSE(monitor.GetPeakUsage().has_value());
  // Stopping a monitor that was never started must be harmless.
  monitor.Stop();
  EXPECT_FALSE(monitor.GetPeakUsage().has_value());
}

TEST(AcceleratorMemoryMonitorTest, TracksPeakTotalAndItsBreakdown) {
  absl::Notification exhausted;
  AcceleratorMemoryMonitor monitor(
      absl::Milliseconds(1),
      std::make_unique<ScriptedSampler>(
          std::vector<std::optional<AcceleratorMemoryUsage>>{
              MakeUsage(1 * kMb, {{"local", 1 * kMb}, {"non-local", 0}}),
              MakeUsage(7 * kMb, {{"local", 5 * kMb}, {"non-local", 2 * kMb}}),
              MakeUsage(3 * kMb, {{"local", 3 * kMb}, {"non-local", 0}}),
          },
          &exhausted));

  monitor.Start();
  ASSERT_TRUE(exhausted.WaitForNotificationWithTimeout(absl::Seconds(5)));
  monitor.Stop();

  const std::optional<AcceleratorMemoryUsage> peak = monitor.GetPeakUsage();
  ASSERT_TRUE(peak.has_value());
  EXPECT_EQ(peak->total_bytes, 7 * kMb);
  // The breakdown is the one captured alongside the peak total, not a
  // per-category maximum, so the categories always sum to the total.
  EXPECT_THAT(peak->breakdown,
              ElementsAre(Pair("local", 5 * kMb), Pair("non-local", 2 * kMb)));
}

TEST(AcceleratorMemoryMonitorTest, IgnoresFailedSamples) {
  absl::Notification exhausted;
  AcceleratorMemoryMonitor monitor(
      absl::Milliseconds(1),
      std::make_unique<ScriptedSampler>(
          std::vector<std::optional<AcceleratorMemoryUsage>>{
              std::nullopt,
              MakeUsage(4 * kMb),
              std::nullopt,
          },
          &exhausted));

  monitor.Start();
  ASSERT_TRUE(exhausted.WaitForNotificationWithTimeout(absl::Seconds(5)));
  monitor.Stop();

  const std::optional<AcceleratorMemoryUsage> peak = monitor.GetPeakUsage();
  ASSERT_TRUE(peak.has_value());
  EXPECT_EQ(peak->total_bytes, 4 * kMb);
  EXPECT_THAT(peak->breakdown, IsEmpty());
}

TEST(AcceleratorMemoryMonitorTest, PeakSurvivesStopAndRestart) {
  absl::Notification exhausted;
  AcceleratorMemoryMonitor monitor(
      absl::Milliseconds(1),
      std::make_unique<ScriptedSampler>(
          std::vector<std::optional<AcceleratorMemoryUsage>>{
              MakeUsage(9 * kMb),
              MakeUsage(2 * kMb),
          },
          &exhausted));

  monitor.Start();
  // Repeated Start() calls are no-ops.
  monitor.Start();
  ASSERT_TRUE(exhausted.WaitForNotificationWithTimeout(absl::Seconds(5)));
  monitor.Stop();
  // Repeated Stop() calls are no-ops.
  monitor.Stop();
  ASSERT_TRUE(monitor.GetPeakUsage().has_value());
  EXPECT_EQ(monitor.GetPeakUsage()->total_bytes, 9 * kMb);

  // Restarting keeps sampling (the sampler now repeats its last, smaller
  // sample) without lowering the recorded peak.
  monitor.Start();
  monitor.Stop();
  ASSERT_TRUE(monitor.GetPeakUsage().has_value());
  EXPECT_EQ(monitor.GetPeakUsage()->total_bytes, 9 * kMb);
}

TEST(AcceleratorMemoryMonitorTest, DestructorStopsRunningMonitor) {
  absl::Notification exhausted;
  auto monitor = std::make_unique<AcceleratorMemoryMonitor>(
      absl::Milliseconds(1),
      std::make_unique<ScriptedSampler>(
          std::vector<std::optional<AcceleratorMemoryUsage>>{MakeUsage(kMb)},
          &exhausted));
  monitor->Start();
  ASSERT_TRUE(exhausted.WaitForNotificationWithTimeout(absl::Seconds(5)));
  // Must join the sampling thread rather than crash or leak it.
  monitor.reset();
}

}  // namespace
}  // namespace litert::lm
