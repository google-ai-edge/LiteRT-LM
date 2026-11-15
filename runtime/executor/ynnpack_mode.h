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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_YNNPACK_MODE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_YNNPACK_MODE_H_

#include <ostream>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

// Delegation mode for YNNPACK.
class YnnpackMode {
 public:
  enum class Value {
    kOff = 0,
    kOn = 1,
    kAuto = 2,
  };

  static constexpr Value kOff = Value::kOff;
  static constexpr Value kOn = Value::kOn;
  static constexpr Value kAuto = Value::kAuto;

  constexpr YnnpackMode() : value_(Value::kOff) {}
  constexpr YnnpackMode(Value v) : value_(v) {}  // NOLINT
  constexpr YnnpackMode(bool b)  // NOLINT
      : value_(b ? Value::kOn : Value::kOff) {}

  constexpr Value value() const { return value_; }
  constexpr bool is_on() const { return value_ == Value::kOn; }
  constexpr bool is_off() const { return value_ == Value::kOff; }
  constexpr bool is_auto() const { return value_ == Value::kAuto; }

  constexpr explicit operator bool() const { return is_on(); }

  constexpr bool operator==(const YnnpackMode& other) const = default;
  constexpr bool operator==(Value v) const { return value_ == v; }
  constexpr bool operator==(bool b) const {
    return value_ == (b ? Value::kOn : Value::kOff);
  }

 private:
  Value value_;
};

inline absl::string_view YnnpackModeName(YnnpackMode mode) {
  switch (mode.value()) {
    case YnnpackMode::Value::kOff:
      return "false";
    case YnnpackMode::Value::kOn:
      return "true";
    case YnnpackMode::Value::kAuto:
      return "auto";
  }
  return "false";
}

template <typename Sink>
void AbslStringify(Sink& sink, YnnpackMode mode) {
  sink.Append(YnnpackModeName(mode));
}

inline std::ostream& operator<<(std::ostream& os, YnnpackMode mode) {
  switch (mode.value()) {
    case YnnpackMode::Value::kOff:
      return os << "0";
    case YnnpackMode::Value::kOn:
      return os << "1";
    case YnnpackMode::Value::kAuto:
      return os << "auto";
  }
  return os << "0";
}

inline bool AbslParseFlag(absl::string_view text, YnnpackMode* mode,
                          std::string* error) {
  if (absl::EqualsIgnoreCase(text, "true") || text == "1" ||
      absl::EqualsIgnoreCase(text, "on")) {
    *mode = YnnpackMode::kOn;
    return true;
  }
  if (absl::EqualsIgnoreCase(text, "false") || text == "0" ||
      absl::EqualsIgnoreCase(text, "off")) {
    *mode = YnnpackMode::kOff;
    return true;
  }
  if (absl::EqualsIgnoreCase(text, "auto")) {
    *mode = YnnpackMode::kAuto;
    return true;
  }
  *error = absl::StrCat("Invalid --enable_ynnpack value: '", text,
                        "'. Expected true, false, or auto.");
  return false;
}

inline std::string AbslUnparseFlag(YnnpackMode mode) {
  switch (mode.value()) {
    case YnnpackMode::Value::kOff:
      return "false";
    case YnnpackMode::Value::kOn:
      return "true";
    case YnnpackMode::Value::kAuto:
      return "auto";
  }
  return "false";
}

inline absl::StatusOr<YnnpackMode> ParseYnnpackMode(absl::string_view text) {
  YnnpackMode mode;
  std::string error;
  if (AbslParseFlag(text, &mode, &error)) {
    return mode;
  }
  return absl::InvalidArgumentError(error);
}

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_YNNPACK_MODE_H_
