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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl

#if defined(__ANDROID__)
#include <dirent.h>
#include <sys/stat.h>

#include <fstream>
#include <unordered_map>

#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#endif  // defined(__ANDROID__)

namespace litert::lm {

namespace {

// Allocations smaller than this are omitted from the diagnostic listing.
constexpr uint64_t kMinLoggedAllocationBytes = 1024 * 1024;

#if defined(__ANDROID__)
// Reads the DMA-BUF buffers held by this process from /proc/self/fdinfo.
// Buffers that are mapped through several file descriptors are counted once,
// keyed by inode.
DmaBufSummary GetProcessDmaBufUsage() {
  DmaBufSummary summary;
  DIR* dir = opendir("/proc/self/fdinfo");
  if (dir == nullptr) {
    return summary;
  }

  struct dirent* de = nullptr;
  struct DmaBufItem {
    uint64_t size = 0;
    std::string name;
    std::string exp_name;
  };
  std::unordered_map<uint64_t, DmaBufItem> buffers;

  while ((de = readdir(dir)) != nullptr) {
    if (de->d_name[0] == '.') {
      continue;
    }
    std::string fd_path = absl::StrCat("/proc/self/fdinfo/", de->d_name);
    std::ifstream file(fd_path);
    if (!file.is_open()) {
      continue;
    }

    uint64_t cur_ino = 0;
    uint64_t cur_size = 0;
    std::string cur_name;
    std::string cur_exp;
    std::string line;

    while (std::getline(file, line)) {
      if (absl::StartsWith(line, "size:\t") ||
          absl::StartsWith(line, "size: ")) {
        (void)absl::SimpleAtoi(line.substr(line.find_first_of(" \t") + 1),
                               &cur_size);
      } else if (absl::StartsWith(line, "ino:\t") ||
                 absl::StartsWith(line, "ino: ")) {
        (void)absl::SimpleAtoi(line.substr(line.find_first_of(" \t") + 1),
                               &cur_ino);
      } else if (absl::StartsWith(line, "exp_name:\t") ||
                 absl::StartsWith(line, "exp_name: ")) {
        cur_exp = std::string(absl::StripTrailingAsciiWhitespace(
            line.substr(line.find_first_of(" \t") + 1)));
      } else if (absl::StartsWith(line, "name:\t") ||
                 absl::StartsWith(line, "name: ")) {
        cur_name = std::string(absl::StripTrailingAsciiWhitespace(
            line.substr(line.find_first_of(" \t") + 1)));
      }
    }

    if (cur_size != 0 && !cur_exp.empty()) {
      if (cur_ino == 0) {
        struct stat st;
        if (stat(absl::StrCat("/proc/self/fd/", de->d_name).c_str(), &st) ==
            0) {
          cur_ino = st.st_ino;
        } else {
          (void)absl::SimpleAtoi(de->d_name, &cur_ino);
        }
      }
      buffers[cur_ino] = DmaBufItem{
          .size = cur_size,
          .name = cur_name.empty() ? cur_exp : cur_name,
          .exp_name = cur_exp,
      };
    }
  }
  closedir(dir);

  summary.entries.reserve(buffers.size());
  for (const auto& [ino, item] : buffers) {
    summary.total_bytes += item.size;
    summary.entries.push_back({
        .inode = ino,
        .size_bytes = item.size,
        .name = item.name,
        .exp_name = item.exp_name,
    });
  }

  std::sort(
      summary.entries.begin(), summary.entries.end(),
      [](const auto& a, const auto& b) { return a.size_bytes > b.size_bytes; });
  return summary;
}
#endif  // defined(__ANDROID__)

}  // namespace

AcceleratorMemoryUsage AcceleratorMemoryUsageFromDmaBuf(
    const DmaBufSummary& dmabuf) {
  AcceleratorMemoryUsage usage;
  usage.total_bytes = dmabuf.total_bytes;

  // Break the total down by exporter, i.e. the dma-buf heap the buffer was
  // allocated from (e.g. "system-uncached", "system"), and keep every buffer
  // for the diagnostic listing. `dmabuf.entries` is already ordered largest
  // first.
  absl::flat_hash_map<std::string, uint64_t> bytes_by_exporter;
  usage.allocations.reserve(dmabuf.entries.size());
  for (const DmaBufSummary::Entry& entry : dmabuf.entries) {
    bytes_by_exporter[entry.exp_name] += entry.size_bytes;
    usage.allocations.emplace_back(
        absl::StrFormat("%s (Inode %u)", entry.name, entry.inode),
        entry.size_bytes);
  }
  usage.breakdown.assign(bytes_by_exporter.begin(), bytes_by_exporter.end());
  // Largest first; ties broken by name so the output is deterministic.
  std::sort(usage.breakdown.begin(), usage.breakdown.end(),
            [](const auto& a, const auto& b) {
              return std::tie(b.second, a.first) < std::tie(a.second, b.first);
            });
  return usage;
}

#if defined(__ANDROID__)
// On Android, report the dma-buf heap backed buffers held by the process (in
// practice AHardwareBuffers). Memory the GPU driver allocates from its own
// pool (Mali kbase, Adreno KGSL) is not visible here, so this is a lower
// bound on accelerator memory rather than a complete measure.
std::optional<AcceleratorMemoryUsage> GetProcessAcceleratorMemoryUsage() {
  return AcceleratorMemoryUsageFromDmaBuf(GetProcessDmaBufUsage());
}
#else
// Remaining platform implementations are added incrementally; until then
// accelerator memory tracking is reported as unsupported.
std::optional<AcceleratorMemoryUsage> GetProcessAcceleratorMemoryUsage() {
  return std::nullopt;
}
#endif  // defined(__ANDROID__)

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
