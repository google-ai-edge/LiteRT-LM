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

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory.h"
#include "runtime/util/memory_profiling/memory_snapshot.h"

namespace litert::lm {

namespace {

std::string Mb(uint64_t bytes) {
  return absl::StrFormat("%.2f MB", BytesToMb(bytes));
}

// "<total> (<breakdown>)", or just "<total>" without a breakdown.
std::string AcceleratorSummary(const AcceleratorMemoryUsage& accelerator) {
  std::string summary = Mb(accelerator.total_bytes);
  if (!accelerator.breakdown.empty()) {
    absl::StrAppend(&summary, " (",
                    FormatAcceleratorMemoryBreakdown(accelerator), ")");
  }
  return summary;
}

// Keeps `candidate` as the peak if it is the first or the larger sample.
void UpdatePeak(const std::optional<MemorySplit>& candidate,
                std::optional<MemorySplit>& peak) {
  if (candidate.has_value() &&
      (!peak.has_value() || candidate->total_bytes > peak->total_bytes)) {
    peak = candidate;
  }
}

// Folds `sample` into `peaks`.
void UpdatePeaks(const MemorySnapshot& sample, MemoryPeaks& peaks) {
  ++peaks.num_samples;
  UpdatePeak(sample.resident_in_ram, peaks.resident_in_ram);
  UpdatePeak(sample.attributed_by_os, peaks.attributed_by_os);
  if (sample.accelerator.has_value() &&
      (!peaks.accelerator.has_value() ||
       sample.accelerator->total_bytes > peaks.accelerator->total_bytes)) {
    peaks.accelerator = sample.accelerator;
  }
}

}  // namespace

std::string MemoryPeaks::ToString() const {
  if (num_samples == 0) return "(no samples)";
  std::vector<std::string> parts;
  if (resident_in_ram.has_value()) {
    parts.push_back(absl::StrCat("peak resident in RAM=",
                                 FormatMemorySplit(*resident_in_ram)));
  }
  if (attributed_by_os.has_value()) {
    parts.push_back(absl::StrCat("peak attributed by OS=",
                                 FormatMemorySplit(*attributed_by_os)));
  }
  parts.push_back(absl::StrCat("peak accelerator detail=",
                               accelerator.has_value()
                                   ? AcceleratorSummary(*accelerator)
                                   : std::string("unsupported")));
  parts.push_back(
      absl::StrCat(num_samples, num_samples == 1 ? " sample" : " samples"));
  return absl::StrJoin(parts, ", ");
}

std::ostream& operator<<(std::ostream& os, const MemoryPeaks& peaks) {
  return os << peaks.ToString();
}

std::string FormatMemoryPeaks(const MemoryPeaks& peaks,
                              absl::string_view title) {
  if (peaks.num_samples == 0) return absl::StrCat(title, ": (no samples)");
  std::vector<std::string> lines;
  lines.push_back(
      absl::StrCat(title, " (", peaks.num_samples,
                   peaks.num_samples == 1 ? " sample):" : " samples):"));
  if (peaks.resident_in_ram.has_value()) {
    lines.push_back(absl::StrCat("  Peak memory resident in RAM: ",
                                 FormatMemorySplit(*peaks.resident_in_ram)));
  }
  if (peaks.attributed_by_os.has_value()) {
    lines.push_back(
        absl::StrCat("  Peak memory attributed to the process by the OS: ",
                     FormatMemorySplit(*peaks.attributed_by_os)));
  }
  if (peaks.accelerator.has_value()) {
    lines.push_back(absl::StrCat("  Peak accelerator memory detail: ",
                                 AcceleratorSummary(*peaks.accelerator)));
    for (const std::string& line :
         FormatAcceleratorMemoryAllocations(*peaks.accelerator)) {
      lines.push_back(absl::StrCat("  ", line));
    }
  } else {
    lines.push_back("  Peak accelerator memory detail: unsupported");
  }
  return absl::StrJoin(lines, "\n");
}

void MemoryMonitor::Sampler::WaitBetweenSamples(
    absl::Duration interval, const absl::Notification& stop) {
  stop.WaitForNotificationWithTimeout(interval);
}

MemoryMonitor::MemoryMonitor(absl::Duration sampling_interval,
                             std::unique_ptr<Sampler> sampler)
    : sampler_(sampler != nullptr ? std::move(sampler)
                                  : std::make_unique<Sampler>()),
      sampling_interval_(sampling_interval) {}

MemoryMonitor::~MemoryMonitor() { StopInternal(); }

void MemoryMonitor::Start() {
  if (sampling_thread_ != nullptr) {
    ABSL_LOG(INFO) << "Memory monitoring has already started.";
    return;
  }
  stop_signal_ = std::make_unique<absl::Notification>();
  sampling_thread_ = std::make_unique<std::thread>([this]() {
    // The first sample is taken immediately, so that short runs still record
    // at least one data point, and one more after Stop() has been requested,
    // so that the state at the end of the run is recorded. Whether Stop() was
    // requested is read before each sample, so a sample that was already in
    // flight when Stop() came does not count as that last one. The wait in
    // between ends early when Stop() is called.
    while (true) {
      const bool stopping = stop_signal_->HasBeenNotified();
      MemorySnapshot sample = sampler_->Sample();
      {
        absl::MutexLock lock(mutex_);
        UpdatePeaks(sample, peaks_);
        latest_ = std::move(sample);
      }
      if (stopping) break;
      sampler_->WaitBetweenSamples(sampling_interval_, *stop_signal_);
    }
  });
}

void MemoryMonitor::Stop() {
  if (sampling_thread_ == nullptr) {
    ABSL_LOG(INFO) << "Memory monitoring has not started yet or has already "
                      "stopped.";
    return;
  }
  StopInternal();
}

void MemoryMonitor::StopInternal() {
  if (sampling_thread_ == nullptr) return;
  stop_signal_->Notify();
  sampling_thread_->join();
  sampling_thread_.reset();
  stop_signal_.reset();
}

MemoryPeaks MemoryMonitor::GetPeaks() const {
  absl::MutexLock lock(mutex_);
  return peaks_;
}

std::optional<MemorySnapshot> MemoryMonitor::GetLatest() const {
  absl::MutexLock lock(mutex_);
  return latest_;
}

void MemoryMonitor::LogPeaks(absl::string_view title) const {
  for (absl::string_view line :
       absl::StrSplit(FormatMemoryPeaks(GetPeaks(), title), '\n')) {
    ABSL_LOG(INFO) << line;
  }
}

}  // namespace litert::lm
