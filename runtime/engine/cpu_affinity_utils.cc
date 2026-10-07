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
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/status_macros.h"
#endif  // defined(__ANDROID__)

namespace litert::lm {

#if defined(__ANDROID__)

namespace {

enum class SupportedSoc {
  kUnknown,
  kTensorG3,
  kTensorG4,
  kTensorG5,
  kTensorG6,
  kSnapdragon8Elite,
};

// Defines the CPU cores to use for a given SoC.
struct SocCoreAffinity {
  SupportedSoc soc;        // The SoC identifier.
  std::vector<int> cores;  // The CPU cores to use for affinity.
};

// Cores are mid and big cores to optimize performance.
const SocCoreAffinity kSocAffinities[] = {
    {SupportedSoc::kTensorG3, {4, 5, 6, 7, 8}},
    {SupportedSoc::kTensorG4, {4, 5, 6, 7}},
    {SupportedSoc::kTensorG5, {2, 3, 4, 5, 6, 7}},
    {SupportedSoc::kTensorG6, {2, 3, 4, 5, 6, 7}},
    // Qualcomm Snapdragon 8 Elite (SM8850 / SM8750):
    // 6 Performance cores (0-5) + 2 Prime cores (6-7).
    // Cores {4, 5, 6, 7} lock execution to the 2 Prime cores (4.74 GHz)
    // and 2 highest Performance cores (3.63 GHz).
    {SupportedSoc::kSnapdragon8Elite, {4, 5, 6, 7}},
};

// Queries Android system properties to identify the current SoC.
// The result is cached to avoid repeated property lookups.
SupportedSoc GetCurrentSupportedSoc() {
  static const SupportedSoc soc = []() {
    char manufacturer[PROP_VALUE_MAX] = {0};
    char soc_model[PROP_VALUE_MAX] = {0};
    __system_property_get("ro.soc.manufacturer", manufacturer);
    __system_property_get("ro.soc.model", soc_model);

    absl::string_view mfg_str(manufacturer);
    absl::string_view soc_str(soc_model);

    if (mfg_str == "Google") {
      if (soc_str == "Tensor G3") return SupportedSoc::kTensorG3;
      if (soc_str == "Tensor G4") return SupportedSoc::kTensorG4;
      if (soc_str == "Tensor G5") return SupportedSoc::kTensorG5;
      if (soc_str == "Tensor G6") return SupportedSoc::kTensorG6;
    }

    if (mfg_str == "QTI" || mfg_str == "Qualcomm" || mfg_str.empty()) {
      if (absl::StartsWith(soc_str, "SM8850") ||
          absl::StartsWith(soc_str, "SM8750")) {
        return SupportedSoc::kSnapdragon8Elite;
      }
    }

    // Direct SoC model fallback regardless of manufacturer string.
    if (absl::StartsWith(soc_str, "SM8850") ||
        absl::StartsWith(soc_str, "SM8750")) {
      return SupportedSoc::kSnapdragon8Elite;
    }

    return SupportedSoc::kUnknown;
  }();
  return soc;
}

}  // namespace

bool HasPerformanceCores() {
  return GetCurrentSupportedSoc() != SupportedSoc::kUnknown;
}

std::vector<int> GetPerformanceCores() {
  SupportedSoc soc = GetCurrentSupportedSoc();
  for (const auto& affinity : kSocAffinities) {
    if (soc == affinity.soc) {
      return affinity.cores;
    }
  }
  return {};
}

bool IsPixelTensorDevice() {
  return HasPerformanceCores();
}

std::vector<int> GetPixelPerformanceCores() {
  return GetPerformanceCores();
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

bool HasPerformanceCores() { return false; }

std::vector<int> GetPerformanceCores() { return {}; }

bool IsPixelTensorDevice() { return false; }

std::vector<int> GetPixelPerformanceCores() { return {}; }

absl::Status SetCpuAffinity(const std::vector<int>& cpu_affinity_cores) {
  return absl::OkStatus();
}
#endif  // defined(__ANDROID__)

}  // namespace litert::lm
