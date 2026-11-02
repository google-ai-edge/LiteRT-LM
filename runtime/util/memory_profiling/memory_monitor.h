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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_MEMORY_MONITOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_MEMORY_MONITOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <thread>  // NOLINT(build/c++11)

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/util/memory_profiling/accelerator_memory_monitor.h"
#include "runtime/util/memory_profiling/memory_snapshot.h"

namespace litert::lm {

// The peaks a MemoryMonitor has seen since it was started.
//
// Each figure peaks at its own moment (memory resident in RAM typically while
// the model loads, memory attributed by the OS while it runs), so the two
// peaks below are independent maxima. Each is kept as the whole split of the
// sample it peaked in, so that its CPU and accelerator parts add up to it.
struct MemoryPeaks {
  // The sample with the most memory resident in RAM, and the one with the
  // most memory attributed to the process by the OS (MemorySnapshot,
  // memory_snapshot.h). nullopt if no sample had the figure.
  std::optional<MemorySplit> resident_in_ram;
  std::optional<MemorySplit> attributed_by_os;
  // The accelerator detail of the sample with the most accelerator memory
  // present, kept whole so its breakdown still sums to its total. nullopt
  // where accelerator memory tracking is unsupported.
  std::optional<AcceleratorMemoryUsage> accelerator;
  // How many samples the peaks are based on; 0 means "no peaks yet".
  int num_samples = 0;

  // Single-line summary, e.g.
  //   peak resident in RAM=1548.02 MB (CPU 352.63 MB + accelerator 1195.39
  //   MB), peak attributed by OS=783.41 MB (CPU 353.03 MB + accelerator
  //   430.38 MB), peak accelerator detail=1195.39 MB (graphics=430.38 MB,
  //   mapped weights=765.02 MB), 44 samples
  // or "(no samples)". Also what `<<` and absl::StrCat() print. See
  // FormatMemoryPeaks() for the detailed, multi-line form.
  std::string ToString() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const MemoryPeaks& peaks) {
    sink.Append(peaks.ToString());
  }
};

std::ostream& operator<<(std::ostream& os, const MemoryPeaks& peaks);

// Detailed, multi-line form: `title` and the sample count on the first line,
// then
//   Peak memory resident in RAM: <total> MB (CPU <cpu> MB + accelerator ...)
//   Peak memory attributed to the process by the OS: <total> MB (CPU ... )
//   Peak accelerator memory detail: <total> MB (<breakdown>)
//     - <bytes> MB  <allocation>
// for the available figures. Note that log parsers which look for
// "<name>: <value> <unit>" lines will pick these up as metrics.
std::string FormatMemoryPeaks(const MemoryPeaks& peaks,
                              absl::string_view title = "Peak memory usage");

// Tracks the peak memory usage of the process by taking a MemorySnapshot from
// a background thread at a fixed interval.
//
// Typical use around a benchmark run:
//
//   MemoryMonitor monitor;
//   monitor.Start();
//   ... run ...
//   monitor.Stop();
//   std::cerr << "run " << monitor.GetPeaks() << "\n";
class MemoryMonitor {
 public:
  // Produces the samples and paces them. Can be overridden for testing.
  class Sampler {
   public:
    virtual ~Sampler() = default;
    virtual MemorySnapshot Sample() { return TakeMemorySnapshot(); }
    // Waits between two samples: for `interval`, or until `stop` is notified,
    // whichever comes first, so that Stop() does not have to wait out the
    // interval.
    virtual void WaitBetweenSamples(absl::Duration interval,
                                    const absl::Notification& stop);
  };

  // A null `sampler` means the default one, which calls TakeMemorySnapshot().
  explicit MemoryMonitor(
      absl::Duration sampling_interval = absl::Milliseconds(50),
      std::unique_ptr<Sampler> sampler = nullptr);
  // Stops the monitor if it is still running.
  ~MemoryMonitor();

  MemoryMonitor(const MemoryMonitor&) = delete;
  MemoryMonitor& operator=(const MemoryMonitor&) = delete;

  // Starts sampling; the first sample is taken immediately. Calling Start()
  // while already running is a no-op.
  void Start() ABSL_LOCKS_EXCLUDED(mutex_);
  // Stops sampling and joins the sampling thread. The thread finishes any
  // sample in flight and then takes one more, so that the state at the end of
  // the run is recorded; Stop() therefore returns after at most two samples'
  // worth of time, not after the rest of the sampling interval. Calling Stop()
  // when not running is a no-op. The peaks remain available after Stop(), and
  // Start() continues to accumulate into them.
  void Stop() ABSL_LOCKS_EXCLUDED(mutex_);

  // The peaks so far. Safe to call while sampling. Before the first sample,
  // `num_samples` is 0 and every field is nullopt.
  MemoryPeaks GetPeaks() const ABSL_LOCKS_EXCLUDED(mutex_);
  // The most recent sample, or nullopt before the first one.
  std::optional<MemorySnapshot> GetLatest() const ABSL_LOCKS_EXCLUDED(mutex_);

  // Logs FormatMemoryPeaks(GetPeaks(), title) at INFO, one log line per text
  // line.
  void LogPeaks(absl::string_view title = "Peak memory usage") const;

 private:
  void StopInternal();

  mutable absl::Mutex mutex_;
  std::unique_ptr<Sampler> sampler_;
  absl::Duration sampling_interval_;
  std::unique_ptr<absl::Notification> stop_signal_;
  std::unique_ptr<std::thread> sampling_thread_;
  MemoryPeaks peaks_ ABSL_GUARDED_BY(mutex_);
  std::optional<MemorySnapshot> latest_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_MEMORY_PROFILING_MEMORY_MONITOR_H_
