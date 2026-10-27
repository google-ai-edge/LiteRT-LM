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
  kOff = 0,
  // Speculate on every decode step (loads the drafter lazily if needed).
  kOn = 1,
  // Speculate only while it is measured to be faster than plain decoding
  // (loads the drafter lazily if needed). Backends that do not support
  // adaptive gating speculate on every step, like kOn.
  kAdaptive = 2,
};

inline absl::string_view SpeculativeDecodingModeName(
    SpeculativeDecodingMode mode) {
  switch (mode) {
    case SpeculativeDecodingMode::kOff:
      return "OFF";
    case SpeculativeDecodingMode::kOn:
      return "ON";
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

// Speculative decoding (MTP) options.
//
// The engine holds the default configuration (`EngineSettings`), and each
// session can override it (`SessionConfig`). Every field is optional: a field
// left unset in the session inherits the engine's value, and a field left
// unset in the engine uses the runtime default.
struct SpeculativeDecodingConfig {
  // How speculative decoding is used. Runtime default: kOff.
  std::optional<SpeculativeDecodingMode> mode;

  // Lenience factor c in [0, 1] of the speculative decoding acceptance test.
  // - std::nullopt or 0 (default): exact; emitted tokens follow the session's
  //   sampling distribution p.
  // - c > 0: a drafted token is also accepted with probability c when the
  //   verifier disagrees with it, so emitted tokens follow
  //   (1 - c) * p + c * delta(draft). This trades output quality for speed;
  //   c = 1 accepts every drafted token.
  // Only applies to speculative decode steps; the adaptive gating policy never
  // changes it. The NPU executor does not support lenience: it logs a warning
  // and stays exact.
  std::optional<float> lenience;

  bool operator==(const SpeculativeDecodingConfig&) const = default;
};

template <typename Sink>
void AbslStringify(Sink& sink, const SpeculativeDecodingConfig& config) {
  sink.Append("SpeculativeDecodingConfig{mode: ");
  sink.Append(config.mode.has_value()
                  ? SpeculativeDecodingModeName(*config.mode)
                  : "Not set");
  sink.Append(", lenience: ");
  sink.Append(config.lenience.has_value() ? absl::StrCat(*config.lenience)
                                          : "Not set");
  sink.Append("}");
}

inline std::ostream& operator<<(std::ostream& os,
                                const SpeculativeDecodingConfig& config) {
  return os << absl::StrCat(config);
}

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_SPECULATIVE_DECODING_MODE_H_
