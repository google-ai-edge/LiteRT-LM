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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl

namespace litert::lm {

namespace {

// Allocations smaller than this are omitted from the diagnostic listing.
constexpr uint64_t kMinLoggedAllocationBytes = 1024 * 1024;

}  // namespace

float AcceleratorMemoryBytesToMb(uint64_t bytes) {
  return static_cast<float>(static_cast<double>(bytes) / (1024.0 * 1024.0));
}

std::string FormatAcceleratorMemoryBreakdown(
    const AcceleratorMemoryUsage& usage) {
  return absl::StrJoin(
      usage.breakdown, ", ",
      [](std::string* out, const std::pair<std::string, uint64_t>& entry) {
        absl::StrAppendFormat(out, "%s=%.2f MB", entry.first,
                              AcceleratorMemoryBytesToMb(entry.second));
      });
}

std::vector<std::string> FormatAcceleratorMemoryAllocations(
    const AcceleratorMemoryUsage& usage) {
  std::vector<std::string> lines;
  for (const auto& [label, bytes] : usage.allocations) {
    if (bytes < kMinLoggedAllocationBytes) continue;
    // The size comes before the label so that generic "<name>: <value> <unit>"
    // log parsers do not mistake these lines for metrics.
    lines.push_back(absl::StrFormat("  - %8.2f MB  %s",
                                    AcceleratorMemoryBytesToMb(bytes), label));
  }
  return lines;
}

void AcceleratorMemoryMonitor::Sampler::SleepFor(absl::Duration duration) {
  absl::SleepFor(duration);
}

AcceleratorMemoryMonitor::AcceleratorMemoryMonitor(
    absl::Duration sampling_interval, std::unique_ptr<Sampler> sampler)
    : sampler_(std::move(sampler)), sampling_interval_(sampling_interval) {
  is_supported_ = sampler_ != nullptr && sampler_->IsSupported();
  if (!is_supported_) {
    // A monitor is typically created per benchmark iteration; only log once.
    ABSL_LOG_FIRST_N(INFO, 1)
        << "Accelerator memory tracking is not supported on this platform.";
  }
}

AcceleratorMemoryMonitor::~AcceleratorMemoryMonitor() { StopInternal(); }

void AcceleratorMemoryMonitor::Start() {
  if (!is_supported_) return;
  if (sampling_thread_ != nullptr) {
    ABSL_LOG(INFO) << "Accelerator memory monitoring has already started.";
    return;
  }

  stop_signal_ = std::make_unique<absl::Notification>();
  sampling_thread_ = std::make_unique<std::thread>([this]() {
    // The first sample is taken immediately so that short runs still record
    // at least one data point.
    while (true) {
      std::optional<AcceleratorMemoryUsage> sample = sampler_->Sample();
      if (sample.has_value()) {
        absl::MutexLock lock(mutex_);
        if (!peak_.has_value() || sample->total_bytes > peak_->total_bytes) {
          peak_ = std::move(sample);
        }
      }
      if (stop_signal_->HasBeenNotified()) break;
      sampler_->SleepFor(sampling_interval_);
    }
  });
}

void AcceleratorMemoryMonitor::Stop() {
  if (!is_supported_) return;
  if (sampling_thread_ == nullptr) {
    ABSL_LOG(INFO) << "Accelerator memory monitoring has not started yet or "
                      "has already stopped.";
    return;
  }
  StopInternal();
}

void AcceleratorMemoryMonitor::StopInternal() {
  if (sampling_thread_ == nullptr) return;
  stop_signal_->Notify();
  sampling_thread_->join();
  sampling_thread_.reset();
  stop_signal_.reset();
}

std::optional<AcceleratorMemoryUsage> AcceleratorMemoryMonitor::GetPeakUsage()
    const {
  if (!is_supported_) return std::nullopt;
  absl::MutexLock lock(mutex_);
  return peak_;
}

}  // namespace litert::lm
