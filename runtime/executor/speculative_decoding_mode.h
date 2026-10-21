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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_SPECULATIVE_DECODING_MODE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_SPECULATIVE_DECODING_MODE_H_

#include <optional>
#include <ostream>

#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

// How speculative decoding (MTP) is used during decode.
enum class SpeculativeDecodingMode {
  // Never speculate.
  kDisabled = 0,
  // Speculate on every decode step (loads the drafter lazily if needed).
  kAlways = 1,
  // Speculate only while it is measured to be faster than plain decoding
  // (loads the drafter lazily if needed).
  kAdaptive = 2,
};

inline absl::string_view SpeculativeDecodingModeName(
    SpeculativeDecodingMode mode) {
  switch (mode) {
    case SpeculativeDecodingMode::kDisabled:
      return "DISABLED";
    case SpeculativeDecodingMode::kAlways:
      return "ALWAYS";
    case SpeculativeDecodingMode::kAdaptive:
      return "ADAPTIVE";
  }
  return "UNKNOWN";
}

template <typename Sink>
void AbslStringify(Sink& sink, SpeculativeDecodingMode mode) {
  sink.Append(SpeculativeDecodingModeName(mode));
}

inline std::ostream& operator<<(std::ostream& os,
                                SpeculativeDecodingMode mode) {
  return os << SpeculativeDecodingModeName(mode);
}

// Per-session speculative decoding (MTP) options.
struct SpeculativeDecodingConfig {
  // How speculative decoding is used. std::nullopt inherits the engine's
  // configuration: if the MTP drafter was loaded when the engine was created,
  // backends that support adaptive gating behave like kAdaptive and the others
  // speculate on every step; otherwise, no speculation.
  std::optional<SpeculativeDecodingMode> mode;

  bool operator==(const SpeculativeDecodingConfig&) const = default;
};

template <typename Sink>
void AbslStringify(Sink& sink, const SpeculativeDecodingConfig& config) {
  sink.Append("SpeculativeDecodingConfig{mode: ");
  sink.Append(config.mode.has_value()
                  ? SpeculativeDecodingModeName(*config.mode)
                  : "Not set");
  sink.Append("}");
}

inline std::ostream& operator<<(std::ostream& os,
                                const SpeculativeDecodingConfig& config) {
  return os << absl::StrCat(config);
}

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_SPECULATIVE_DECODING_MODE_H_
