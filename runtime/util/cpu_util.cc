// Copyright 2025 The ODML Authors.
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

#include "runtime/util/cpu_util.h"

#include <set>

#if defined(_WIN32) || defined(__APPLE__)
#else
#include <unistd.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"  // from @com_google_absl
#endif

namespace litert::lm {

std::set<int> GetBigCoreIds() {
#if defined(_WIN32) || defined(__APPLE__)
  return std::set<int>();
#else
  static const absl::NoDestructor<std::set<int>> bigCoreIds([]() {
    int num_cores = sysconf(_SC_NPROCESSORS_CONF);
    if (num_cores <= 0) {
      return std::set<int>();
    }

    std::vector<std::pair<int, uint64_t>> core_freqs;
    uint64_t max_freq = 0;

    for (int i = 0; i < num_cores; ++i) {
      std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(i) +
                         "/cpufreq/cpuinfo_max_freq";
      std::ifstream file(path);
      if (file.is_open()) {
        uint64_t freq = 0;
        if (file >> freq) {
          core_freqs.push_back({i, freq});
          if (freq > max_freq) {
            max_freq = freq;
          }
        }
      }
    }

    std::set<int> big_cores;
    if (!core_freqs.empty()) {
      uint64_t min_freq = core_freqs[0].second;
      for (const auto& p : core_freqs) {
        if (p.second < min_freq) {
          min_freq = p.second;
        }
      }

      // If all cores have the same maximum frequency, return empty to apply
      // no pin.
      if (max_freq == min_freq) {
        return std::set<int>();
      }

      for (const auto& p : core_freqs) {
        if (p.second > min_freq) {
          big_cores.insert(p.first);
        }
      }
    }
    return big_cores;
  }());

  return *bigCoreIds;
#endif
}

}  // namespace litert::lm
