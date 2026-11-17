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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_RANDOM_STATE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_RANDOM_STATE_H_

#include <cstdint>
#include <random>

namespace litert::lm {

// All random state that a session consumes while generating. The session owns
// it and lends it to every call that can draw, so executors hold no random
// state of their own.
//
// It is a value type: a copy continues exactly where the original was, and
// from then on the two advance independently.
struct RandomState {
  // Returns the state that a session with sampler seed `seed` starts from.
  static RandomState FromSeed(int32_t seed);

  // Draws the uniforms that samplers turn into tokens.
  std::default_random_engine sampling;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_RANDOM_STATE_H_
