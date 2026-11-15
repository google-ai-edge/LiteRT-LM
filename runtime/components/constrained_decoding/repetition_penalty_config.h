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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_REPETITION_PENALTY_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_REPETITION_PENALTY_CONFIG_H_

#include <algorithm>

namespace litert::lm {

// A configuration storing the penalty parameters for the
// `RepetitionPenaltyConstraint`. `RepetitionPenaltyConstraint` only observes
// the sequence generated in Decode of the current decode loop.
//
// Clamps the given parameters to their valid lower bounds during construction.
// Parameters are immutable once constructed to prevent unexpected divergence
// between the active penalty settings and a running generative sequence's
// state.
//
// When multiple penalties are active, the order of application is:
// 1. Multiplicative penalty (Repetition Penalty)
// 2. Subtractive penalties (Presence and Frequency Penalties)
class RepetitionPenaltyConfig {
 public:
  // Constructs a config with the given bounds, clamping where necessary.
  //
  // @param repetition_penalty A multiplicative penalty for any token already
  // generated (e.g., 1.0 = no penalty, 1.2 = moderate penalty). Positive logits
  // are divided by this penalty, and negative logits are multiplied
  // (HuggingFace style). The value is clamped to [1.0, inf).
  // @param presence_penalty A scalar subtracted globally from a logit if a
  // token has appeared at least once within the currently generated sequence.
  // Positive values discourage repetition, while negative values reward
  // repeating tokens (OpenAI style).
  // @param frequency_penalty A scalar subtracted from a token's logit, scaled
  // linearly by the *number of times* that token has previously appeared
  // within the currently generated sequence. Positive values discourage
  // repetition, while negative values reward repeating tokens
  // (OpenAI style).
  // @param window_size The maximum number of recent tokens to consider for
  // penalization. Tokens older than this are forgotten. A value of 0 means
  // track all infinite history. The value is clamped to [0, inf).
  RepetitionPenaltyConfig(float repetition_penalty, float presence_penalty,
                          float frequency_penalty, int window_size)
      : repetition_penalty_(std::max(1.0f, repetition_penalty)),
        presence_penalty_(presence_penalty),
        frequency_penalty_(frequency_penalty),
        window_size_(std::max(0, window_size)) {}

  // Returns a default config with all penalties disabled.
  static RepetitionPenaltyConfig Default() {
    return RepetitionPenaltyConfig(1.0f, 0.0f, 0.0f, 0);
  }

  float repetition_penalty() const { return repetition_penalty_; }
  float presence_penalty() const { return presence_penalty_; }
  float frequency_penalty() const { return frequency_penalty_; }
  int window_size() const { return window_size_; }

  // Returns whether the penalty config is enabled. If all penalties are
  // disabled, the config is disabled.
  bool enabled() const {
    return repetition_penalty_ > 1.0f || presence_penalty_ != 0.0f ||
           frequency_penalty_ != 0.0f;
  }

 private:
  float repetition_penalty_;
  float presence_penalty_;
  float frequency_penalty_;
  int window_size_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_REPETITION_PENALTY_CONFIG_H_
