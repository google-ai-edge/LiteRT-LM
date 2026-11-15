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

#include "runtime/engine/cpu_affinity_utils.h"

#if defined(__ANDROID__)
#include <sched.h>
#include <sys/system_properties.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/util/status_macros.h"
#endif  // defined(__ANDROID__)

namespace litert::lm {

#if defined(__ANDROID__)

namespace {

enum class PixelSoc {
  kUnknown,
  kTensorG3,
  kTensorG4,
  kTensorG5,
  kTensorG6,
};

// Defines the CPU cores to use for a given Pixel SoC.
struct TensorCoreAffinity {
  PixelSoc soc;            // The Pixel SoC identifier.
  std::vector<int> cores;  // The CPU cores to use for affinity.
};

// Cores are mid and big cores to optimize performance.
const TensorCoreAffinity kTensorAffinities[] = {
    {PixelSoc::kTensorG3, {4, 5, 6, 7, 8}},
    {PixelSoc::kTensorG4, {4, 5, 6, 7}},
    {PixelSoc::kTensorG5, {2, 3, 4, 5, 6, 7}},
    {PixelSoc::kTensorG6, {2, 3, 4, 5, 6, 7}},
};

// Queries Android system properties to identify the current Pixel SoC.
// The result is cached to avoid repeated property lookups.
PixelSoc GetCurrentPixelSoc() {
  static const PixelSoc soc = []() {
    char manufacturer[PROP_VALUE_MAX] = {0};
    char soc_model[PROP_VALUE_MAX] = {0};
    __system_property_get("ro.soc.manufacturer", manufacturer);
    __system_property_get("ro.soc.model", soc_model);

    if (absl::string_view(manufacturer) != "Google") {
      return PixelSoc::kUnknown;
    }

    absl::string_view soc_str(soc_model);
    if (soc_str == "Tensor G3") return PixelSoc::kTensorG3;
    if (soc_str == "Tensor G4") return PixelSoc::kTensorG4;
    if (soc_str == "Tensor G5") return PixelSoc::kTensorG5;
    if (soc_str == "Tensor G6") return PixelSoc::kTensorG6;
    return PixelSoc::kUnknown;
  }();
  return soc;
}

}  // namespace

bool IsPixelTensorDevice() {
  return GetCurrentPixelSoc() != PixelSoc::kUnknown;
}

std::vector<int> GetPixelPerformanceCores() {
  PixelSoc soc = GetCurrentPixelSoc();
  for (const auto& affinity : kTensorAffinities) {
    if (soc == affinity.soc) {
      return affinity.cores;
    }
  }
  return {};
}

absl::Status SetCpuAffinity(const std::vector<int>& cpu_affinity_cores) {
  if (cpu_affinity_cores.empty()) {
    ABSL_LOG(WARNING) << "CPU affinity cores are empty, skipping CPU affinity "
                         "setting.";
    return absl::OkStatus();
  }

  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int cpu : cpu_affinity_cores) {
    CPU_SET(cpu, &mask);
  }
  if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
    return absl::InternalError(
        absl::StrCat("Failed to set CPU affinity: ", strerror(errno)));
  }

  ABSL_VLOG(1) << "Successfully set CPU affinity.";
  return absl::OkStatus();
}

#else

bool IsPixelTensorDevice() { return false; }

std::vector<int> GetPixelPerformanceCores() { return {}; }

absl::Status SetCpuAffinity(const std::vector<int>& cpu_affinity_cores) {
  return absl::OkStatus();
}
#endif  // defined(__ANDROID__)

}  // namespace litert::lm
