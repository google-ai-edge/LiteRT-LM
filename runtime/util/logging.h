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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LOGGING_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LOGGING_H_

#include <cstdint>
#include <iostream>
#include <optional>
#include <variant>
#include <vector>

#include "absl/strings/str_join.h"  // from @com_google_absl

namespace litert::lm {

enum class LogSeverity {
  kVerbose = 0,
  kDebug = 1,
  kInfo = 2,
  kWarning = 3,
  kError = 4,
  kFatal = 5,
  kSilent = 1000,
};

// Sets the minimum log severity. Note that for LiteRT, this value is applied
// during the creation of the Environment and do not change for an already
// existing Environment.
void SetMinLogSeverity(LogSeverity severity);

std::optional<LogSeverity> GetMinLogSeverity();

// Converts the given LogSeverity to the underlying int8_t value representing
// LiteRtLogSeverity.
int8_t ToLiteRtLogSeverityInt8(LogSeverity severity);

// Helper function to print a vector of elements.
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& data) {
  os << "vector of " << data.size() << " elements: ["
     << absl::StrJoin(data, ", ") << "]";
  return os;
}

// Helper function to print a std::optional of data.
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::optional<T>& data) {
  if (data.has_value()) {
    return os << *data;
  }
  return os << "Not set";
}

// Helper function to print a std::variant of data.
template <typename... T>
std::ostream& operator<<(std::ostream& os, const std::variant<T...>& data) {
  std::visit([&os](const auto& arg) { os << arg; }, data);
  return os;
}

}  // namespace litert::lm


#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LOGGING_H_
