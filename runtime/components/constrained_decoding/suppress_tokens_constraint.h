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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_SUPPRESS_TOKENS_CONSTRAINT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_SUPPRESS_TOKENS_CONSTRAINT_H_

#include <memory>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/logit_mask.h"
#include "runtime/components/constrained_decoding/suppress_tokens_config.h"

namespace litert::lm {

// A constraint that forbids specified tokens from being generated.
class SuppressTokensConstraint : public Constraint {
 public:
  class SuppressTokensState : public Constraint::State {
   public:
    SuppressTokensState() = default;
  };

  SuppressTokensConstraint(int vocab_size, SuppressTokensConfig config);
  SuppressTokensConstraint(int vocab_size,
                           absl::flat_hash_set<int> suppress_tokens);

  std::unique_ptr<State> Start() const override;

  bool IsEnded(const State& state) const override;

  int GetVocabularySize() const override { return vocab_size_; }

  absl::StatusOr<std::unique_ptr<State>> ComputeNext(const State& state,
                                                     int token) const override;

  absl::StatusOr<std::unique_ptr<LogitMask>> ComputeMask(
      const State& state) const override;

  const SuppressTokensConfig& config() const { return config_; }

 private:
  const int vocab_size_;
  const SuppressTokensConfig config_;
  std::unique_ptr<BitmapLogitMask> mask_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_SUPPRESS_TOKENS_CONSTRAINT_H_
