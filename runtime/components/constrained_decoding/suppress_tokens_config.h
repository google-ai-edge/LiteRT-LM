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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_SUPPRESS_TOKENS_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_SUPPRESS_TOKENS_CONFIG_H_

#include <utility>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl

namespace litert::lm {

// A configuration storing the tokens to suppress.
//
// Parameters are immutable once constructed to prevent unexpected divergence
// between the active penalty settings and a running generative sequence.
class SuppressTokensConfig {
 public:
  explicit SuppressTokensConfig(absl::flat_hash_set<int> suppress_tokens)
      : suppress_tokens_(std::move(suppress_tokens)) {}

  // Returns a default config with no tokens to suppress.
  static SuppressTokensConfig Default() { return SuppressTokensConfig({}); }

  const absl::flat_hash_set<int>& suppress_tokens() const {
    return suppress_tokens_;
  }

  // Returns whether the suppress tokens config is enabled. If the suppress
  // tokens set is empty, the config is disabled.
  bool enabled() const { return !suppress_tokens_.empty(); }

 private:
  absl::flat_hash_set<int> suppress_tokens_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_SUPPRESS_TOKENS_CONFIG_H_
