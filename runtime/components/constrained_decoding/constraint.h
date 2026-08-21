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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_CONSTRAINT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_CONSTRAINT_H_

#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/logit_mask.h"

namespace litert::lm {

// The constraint can be used to compute the next state and the allowed tokens
// given the current state and the token to be consumed. The constraint should
// be maintained by the executor during decoding.

// A constraint is always created by the ConstraintProvider.
class Constraint {
 public:
  // The state of the constraint.
  class State {
   public:
    virtual ~State() = default;
  };

  virtual ~Constraint() = default;

  // Gets the start state of the constraint.
  virtual std::unique_ptr<State> Start() const = 0;

  // Returns true if the constraint is at the end state.
  virtual bool IsEnded(const State& state) const = 0;

  // Gets the vocabulary size of the constraint.
  virtual int GetVocabularySize() const = 0;

  // Computes the next state given the current state and the latest decoded
  // token.
  virtual absl::StatusOr<std::unique_ptr<State>> ComputeNext(
      const State& state, int token) const = 0;

  // Computes the logit mask given the current state.
  // By default, wraps the result of ComputeBitmap into a BitmapLogitMask.
  virtual absl::StatusOr<std::unique_ptr<LogitMask>> ComputeMask(
      const State& state) const;

  // Computes the allowed tokens bitmap given the current state.
  // Deprecated: Use ComputeMask instead.
  [[deprecated("Use ComputeMask instead.")]]
  virtual absl::StatusOr<std::unique_ptr<Bitmap>> ComputeBitmap(
      const State& state) const;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_CONSTRAINT_H_
