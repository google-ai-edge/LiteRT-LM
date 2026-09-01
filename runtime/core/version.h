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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_VERSION_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_VERSION_H_

#include <algorithm>
#include <cstddef>
#include <vector>

#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

// LITERT_LM_VERSION_STRING is passed as a preprocessor macro define
// (-DLITERT_LM_VERSION_STRING="\"...\"") by Bazel from version.bzl.
// In non-Bazel / CMake builds where the macro is not set, default to "0.0.0".
#ifndef LITERT_LM_VERSION_STRING
#define LITERT_LM_VERSION_STRING "0.0.0"
#endif

// The current version of the LiteRT-LM runtime as a string view (e.g.
// "0.17.0").
inline constexpr absl::string_view LITERT_LM_VERSION =
    LITERT_LM_VERSION_STRING;

// Parses a SemVer version string (e.g., "0.17.0", "1.2.3-alpha", "v2.0") into
// an array of integer version components [major, minor, patch, ...].
// Non-digit suffixes (such as "-alpha", "-dev") on individual components are
// stripped before parsing.
inline std::vector<int> ParseVersion(absl::string_view version_str) {
  std::vector<int> parts;
  for (absl::string_view part : absl::StrSplit(version_str, '.')) {
    size_t non_digit = part.find_first_not_of("0123456789");
    absl::string_view digits = (non_digit == absl::string_view::npos)
                                  ? part
                                  : part.substr(0, non_digit);
    int val = 0;
    if (absl::SimpleAtoi(digits, &val)) {
      parts.push_back(val);
    }
  }
  return parts;
}

// Compares two semantic version strings (e.g., "0.17.0" vs "0.18.0").
// Missing trailing components are treated as 0 (e.g. "0.17" is equivalent to
// "0.17.0").
//
// Returns:
//   - A negative value (-1) if v1 < v2 (v1 is older than v2).
//   - Zero (0) if v1 == v2 (v1 and v2 represent the same version).
//   - A positive value (1) if v1 > v2 (v1 is newer than v2).
inline int CompareVersions(absl::string_view v1, absl::string_view v2) {
  auto p1 = ParseVersion(v1);
  auto p2 = ParseVersion(v2);
  size_t max_size = std::max(p1.size(), p2.size());
  p1.resize(max_size, 0);
  p2.resize(max_size, 0);
  if (p1 < p2) return -1;
  if (p1 > p2) return 1;
  return 0;
}

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_VERSION_H_
