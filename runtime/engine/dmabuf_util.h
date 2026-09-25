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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_DMABUF_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_DMABUF_UTIL_H_

#include <cstdint>
#include <string>
#include <vector>

namespace litert::lm {

// Summary of DMA-BUF memory allocations obtained from /proc/self/fdinfo.
struct DmaBufSummary {
  uint64_t total_bytes = 0;
  struct Entry {
    uint64_t inode = 0;
    uint64_t size_bytes = 0;
    std::string name;
    std::string exp_name;
  };
  std::vector<Entry> entries;
};

// Returns DMA-BUF memory usage for the current process on Android.
// On non-Android platforms, returns an empty DmaBufSummary.
DmaBufSummary GetProcessDmaBufUsage();

// Logs DMA-BUF memory usage summary and detailed buffer entries to
// ABSL_LOG(INFO) if dmabuf.total_bytes > 0. If report_out is non-null, appends
// the formatted lines to report_out.
void LogDmaBufUsage(const DmaBufSummary& dmabuf,
                    std::string* report_out = nullptr);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_DMABUF_UTIL_H_
