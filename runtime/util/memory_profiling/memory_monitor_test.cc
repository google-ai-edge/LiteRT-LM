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

#include "runtime/util/memory_profiling/memory_monitor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"
#include "runtime/util/memory_profiling/memory_snapshot.h"

namespace litert::lm {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Pair;

constexpr uint64_t kMb = 1024 * 1024;

// Replays a fixed sequence of samples. Once the sequence is exhausted, the
// last sample is repeated and `exhausted` is notified so tests can stop the
// monitor deterministically. The wait between samples is a no-op so tests run
// fast.
class ScriptedSampler : public MemoryMonitor::Sampler {
 public:
  ScriptedSampler(std::vector<MemorySnapshot> samples,
                  absl::Notification* exhausted)
      : samples_(std::move(samples)), exhausted_(exhausted) {}

  MemorySnapshot Sample() override {
    absl::MutexLock lock(mutex_);
    if (next_ < samples_.size()) {
      return samples_[next_++];
    }
    if (!exhausted_->HasBeenNotified()) {
      exhausted_->Notify();
    }
    return samples_.back();
  }

  void WaitBetweenSamples(absl::Duration interval,
                          const absl::Notification& stop) override {}

 private:
  absl::Mutex mutex_;
  std::vector<MemorySnapshot> samples_;
  size_t next_ ABSL_GUARDED_BY(mutex_) = 0;
  absl::Notification* exhausted_;
};

// A snapshot with the given memory resident in RAM and attributed by the OS
// (nullopt for unavailable), each as CPU + accelerator, and accelerator
// detail of `accelerator_mb` (nullopt for unsupported), all in MB.
MemorySnapshot Snapshot(std::optional<MemorySplit> resident_in_ram,
                        std::optional<MemorySplit> attributed_by_os,
                        std::optional<uint64_t> accelerator_mb) {
  MemorySnapshot snapshot;
  snapshot.resident_in_ram = resident_in_ram;
  snapshot.attributed_by_os = attributed_by_os;
  if (accelerator_mb.has_value()) {
    AcceleratorMemoryUsage accelerator;
    accelerator.total_bytes = *accelerator_mb * kMb;
    accelerator.breakdown = {{"local", *accelerator_mb * kMb},
                             {"non-local", 0}};
    accelerator.allocations = {{"Some GPU (local)", *accelerator_mb * kMb}};
    snapshot.accelerator = std::move(accelerator);
  }
  return snapshot;
}

// A split of `cpu_mb` + `accelerator_mb`.
MemorySplit Split(uint64_t cpu_mb, uint64_t accelerator_mb) {
  return {.total_bytes = (cpu_mb + accelerator_mb) * kMb,
          .cpu_bytes = cpu_mb * kMb,
          .accelerator_bytes = accelerator_mb * kMb};
}

MATCHER_P2(SplitsAs, cpu_mb, accelerator_mb, "") {
  if (!arg.has_value()) {
    *result_listener << "is nullopt";
    return false;
  }
  *result_listener << "is " << FormatMemorySplit(*arg);
  return arg->total_bytes == (cpu_mb + accelerator_mb) * kMb &&
         arg->cpu_bytes == cpu_mb * kMb &&
         arg->accelerator_bytes == accelerator_mb * kMb;
}

// Runs `samples` through a monitor and returns the peaks.
MemoryPeaks PeaksOf(std::vector<MemorySnapshot> samples) {
  absl::Notification exhausted;
  MemoryMonitor monitor(
      absl::Milliseconds(1),
      std::make_unique<ScriptedSampler>(std::move(samples), &exhausted));
  monitor.Start();
  EXPECT_TRUE(exhausted.WaitForNotificationWithTimeout(absl::Seconds(5)));
  monitor.Stop();
  return monitor.GetPeaks();
}

TEST(MemoryMonitorTest, NoPeaksBeforeStart) {
  MemoryMonitor monitor(absl::Milliseconds(1));
  const MemoryPeaks peaks = monitor.GetPeaks();
  EXPECT_EQ(peaks.num_samples, 0);
  EXPECT_FALSE(peaks.resident_in_ram.has_value());
  EXPECT_FALSE(peaks.attributed_by_os.has_value());
  EXPECT_FALSE(peaks.accelerator.has_value());
  EXPECT_FALSE(monitor.GetLatest().has_value());
  EXPECT_EQ(peaks.ToString(), "(no samples)");
  // Stopping a monitor that was never started must be harmless.
  monitor.Stop();
}

TEST(MemoryMonitorTest, TracksEachFiguresPeakAsAWholeSample) {
  // Memory resident in RAM peaks in the second sample, memory attributed by
  // the OS in the third, the accelerator detail in the second.
  const MemoryPeaks peaks = PeaksOf({
      Snapshot(Split(/*cpu_mb=*/90, /*accelerator_mb=*/10), Split(100, 10), 10),
      Snapshot(Split(40, 100), Split(50, 100), 100),
      Snapshot(Split(95, 0), Split(160, 0), 0),
  });

  // Each peak is the whole split of its sample, so CPU + accelerator add up
  // to it, rather than a per-part maximum.
  EXPECT_THAT(peaks.resident_in_ram, SplitsAs(40, 100));
  EXPECT_THAT(peaks.attributed_by_os, SplitsAs(160, 0));
  ASSERT_TRUE(peaks.accelerator.has_value());
  EXPECT_EQ(peaks.accelerator->total_bytes, 100 * kMb);
  // Likewise the breakdown is the one captured alongside the peak, so the
  // categories always sum to the total.
  EXPECT_THAT(peaks.accelerator->breakdown,
              ElementsAre(Pair("local", 100 * kMb), Pair("non-local", 0)));
  // The scripted sampler repeats the last sample until the monitor stops, so
  // at least the three distinct ones were seen.
  EXPECT_GE(peaks.num_samples, 3);
}

TEST(MemoryMonitorTest, IgnoresUnavailableFigures) {
  const MemoryPeaks peaks = PeaksOf({
      Snapshot(std::nullopt, std::nullopt, std::nullopt),
      Snapshot(std::nullopt, Split(40, 0), std::nullopt),
      Snapshot(Split(30, 0), std::nullopt, std::nullopt),
  });
  EXPECT_THAT(peaks.resident_in_ram, SplitsAs(30, 0));
  EXPECT_THAT(peaks.attributed_by_os, SplitsAs(40, 0));
  EXPECT_FALSE(peaks.accelerator.has_value());
}

TEST(MemoryMonitorTest, PeaksSurviveStopAndRestart) {
  absl::Notification exhausted;
  MemoryMonitor monitor(absl::Milliseconds(1),
                        std::make_unique<ScriptedSampler>(
                            std::vector<MemorySnapshot>{
                                Snapshot(Split(80, 9), Split(90, 9), 9),
                                Snapshot(Split(10, 2), Split(20, 2), 2),
                            },
                            &exhausted));

  monitor.Start();
  // Repeated Start() calls are no-ops.
  monitor.Start();
  ASSERT_TRUE(exhausted.WaitForNotificationWithTimeout(absl::Seconds(5)));
  monitor.Stop();
  // Repeated Stop() calls are no-ops.
  monitor.Stop();
  EXPECT_THAT(monitor.GetPeaks().attributed_by_os, SplitsAs(90, 9));
  ASSERT_TRUE(monitor.GetLatest().has_value());
  EXPECT_THAT(monitor.GetLatest()->attributed_by_os, SplitsAs(20, 2));

  // Restarting keeps sampling (the sampler now repeats its last, smaller
  // sample) without lowering the recorded peaks.
  const int samples_before = monitor.GetPeaks().num_samples;
  monitor.Start();
  monitor.Stop();
  EXPECT_THAT(monitor.GetPeaks().attributed_by_os, SplitsAs(90, 9));
  EXPECT_GT(monitor.GetPeaks().num_samples, samples_before);
}

TEST(MemoryMonitorTest, DestructorStopsRunningMonitor) {
  absl::Notification exhausted;
  auto monitor = std::make_unique<MemoryMonitor>(
      absl::Milliseconds(1),
      std::make_unique<ScriptedSampler>(
          std::vector<MemorySnapshot>{Snapshot(Split(1, 0), Split(1, 0), 1)},
          &exhausted));
  monitor->Start();
  ASSERT_TRUE(exhausted.WaitForNotificationWithTimeout(absl::Seconds(5)));
  // Must join the sampling thread rather than crash or leak it.
  monitor.reset();
}

TEST(MemoryMonitorTest, DefaultSamplerTakesRealSnapshots) {
  MemoryMonitor monitor(absl::Milliseconds(1));
  monitor.Start();
  monitor.Stop();
  const MemoryPeaks peaks = monitor.GetPeaks();
  EXPECT_GE(peaks.num_samples, 1);
  // Every supported host reports at least its resident size.
  EXPECT_TRUE(peaks.resident_in_ram.has_value());
}

// Counts its samples and lets a test wait for a given number of them. Uses
// the real wait between samples, so that the test sees how Stop() interacts
// with it.
class CountingSampler : public MemoryMonitor::Sampler {
 public:
  MemorySnapshot Sample() override {
    absl::MutexLock lock(mutex_);
    ++count_;
    return MemorySnapshot();
  }

  int count() const {
    absl::MutexLock lock(mutex_);
    return count_;
  }

  bool WaitForCount(int count, absl::Duration timeout) {
    absl::MutexLock lock(mutex_);
    target_ = count;
    return mutex_.AwaitWithTimeout(
        absl::Condition(this, &CountingSampler::TargetReached), timeout);
  }

 private:
  bool TargetReached() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return count_ >= target_;
  }

  mutable absl::Mutex mutex_;
  int count_ ABSL_GUARDED_BY(mutex_) = 0;
  int target_ ABSL_GUARDED_BY(mutex_) = 0;
};

TEST(MemoryMonitorTest, StopDoesNotWaitOutTheIntervalAndTakesALastSample) {
  auto sampler = std::make_unique<CountingSampler>();
  CountingSampler* counting = sampler.get();
  // An interval far longer than the test: Stop() must cut the wait short.
  MemoryMonitor monitor(absl::Hours(1), std::move(sampler));
  monitor.Start();
  ASSERT_TRUE(counting->WaitForCount(1, absl::Seconds(5)));
  const absl::Time before_stop = absl::Now();
  monitor.Stop();
  EXPECT_LT(absl::Now() - before_stop, absl::Seconds(5));
  // The sample taken at Start(), and exactly one more after Stop(): whether
  // Stop() lands while that first sample is still in flight or during the
  // wait, the thread takes one sample that began after Stop() and no other.
  EXPECT_EQ(counting->count(), 2);
  EXPECT_EQ(monitor.GetPeaks().num_samples, 2);
}

TEST(MemoryPeaksToStringTest, SummarisesOnOneLine) {
  const MemoryPeaks peaks =
      PeaksOf({Snapshot(Split(90, 10), Split(100, 10), 10),
               Snapshot(Split(40, 100), Split(50, 100), 100)});
  const std::string text = peaks.ToString();
  EXPECT_THAT(text, HasSubstr("peak resident in RAM=140.00 MB (CPU 40.00 MB + "
                              "accelerator 100.00 MB), "));
  EXPECT_THAT(text, HasSubstr("peak attributed by OS=150.00 MB (CPU 50.00 MB + "
                              "accelerator 100.00 MB), "));
  EXPECT_THAT(text, HasSubstr("peak accelerator detail=100.00 MB (local=100.00 "
                              "MB, non-local=0.00 MB), "));
  EXPECT_THAT(text, HasSubstr(" samples"));
  // The same text comes out of streams and absl string building.
  std::ostringstream stream;
  stream << "after run " << peaks;
  EXPECT_EQ(stream.str(), absl::StrCat("after run ", text));
  EXPECT_EQ(absl::StrCat(peaks), text);
}

TEST(FormatMemoryPeaksTest, FormatsOnePeakPerLine) {
  const MemoryPeaks peaks =
      PeaksOf({Snapshot(Split(90, 10), Split(100, 10), 10),
               Snapshot(Split(40, 100), Split(50, 100), 100)});
  const std::string text = FormatMemoryPeaks(peaks, "Benchmark");
  EXPECT_THAT(text, HasSubstr("Benchmark ("));
  EXPECT_THAT(text, HasSubstr(" samples):\n"));
  EXPECT_THAT(text, HasSubstr("\n  Peak memory resident in RAM: 140.00 MB (CPU "
                              "40.00 MB + accelerator 100.00 MB)\n"));
  EXPECT_THAT(text,
              HasSubstr("\n  Peak memory attributed to the process by the OS: "
                        "150.00 MB (CPU 50.00 MB + accelerator 100.00 MB)\n"));
  EXPECT_THAT(text, HasSubstr("\n  Peak accelerator memory detail: 100.00 MB "
                              "(local=100.00 MB, non-local=0.00 MB)\n"));
  EXPECT_THAT(text, HasSubstr("\n    -   100.00 MB  Some GPU (local)"));
}

TEST(FormatMemoryPeaksTest, SaysWhenThereAreNoSamples) {
  EXPECT_EQ(FormatMemoryPeaks(MemoryPeaks(), "Benchmark"),
            "Benchmark: (no samples)");
}

}  // namespace
}  // namespace litert::lm
