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

// Functions for operating on 1D (like audio) signals represented as vectors
// or other compatible container types.

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_SIGNAL_VECTOR_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_SIGNAL_VECTOR_UTIL_H_

#include <cmath>

#include "absl/base/nullability.h"  // from @com_google_absl

namespace litert::lm {

// Computes the coefficient used by the smoothers, where scale specifies the
// standard deviation in units of samples of the approximately-Gaussian impulse
// response.
//
// A description of the filter strategy and coefficient formula is at
// http://en.wikipedia.org/wiki/Scale_space_implementation#Recursive_filters
// The 2/t is unlike the 1/t in dicklyon's matlab smooth1d.m, because
// this coefficient is for a 4-pass version compared to that 2-pass
// version.  With four passes, the corresponding continuous-time
// impulse response has continuous first and second derivatives,
// unlike the 2-pass or double- exponential smoothing filter, whose
// impulse repsonse has a cusp (a discontinuity of first derivative)
// at the time origin.  The more smooth shape of the 4-pass smoother
// makes it more "Gaussian-like".
inline float SmootherCoefficientFromScale(float scale) {
  if (scale <= 0.01) return 1.0;  // Negligible smoothing requested.
  const float t = scale * scale;  // Kernel variance, TP Lindeberg's t notation.
  const float coefficient =
      std::sqrt(powf(1.0 + 2.0 / t, 2) - 1.0) - 2.0 / t;
  return coefficient;
}

// Except for ForwardSmoothVector, which is a "causal" smoothing filter, the
// smoothing functions all require Reversible Containers.  They work on real
// and complex value_types.

template <typename ContainerType, typename SampleType>
void ForwardSmoothVector(float coefficient, SampleType* absl_nonnull state,
                         ContainerType* absl_nonnull signal) {
  SampleType local_state = *state;
  for (auto it = signal->begin(); it != signal->end(); ++it) {
    local_state += coefficient * (*it - local_state);
    *it = local_state;
  }
  *state = local_state;
}

template <typename ContainerType, typename SampleType>
void BackwardSmoothVector(float coefficient, SampleType* absl_nonnull state,
                          ContainerType* absl_nonnull signal) {
  SampleType local_state = *state;
  for (auto it = signal->rbegin(); it != signal->rend(); ++it) {  // Reversed.
    local_state += coefficient * (*it - local_state);
    *it = local_state;
  }
  *state = local_state;
}

// A Gaussian-like smoother, made by cascading four one-pole smoothers, two
// in forward direction and two backward, for net zero phase.
template <typename ContainerType>
void SmoothVector(float coefficient, ContainerType* absl_nonnull signal) {
  // Two passes, each a forward and a backward one-pole smoother.
  auto state = *(signal->begin());
  for (int count = 0; count < 2; ++count) {
    state *= (1.0f - coefficient);  // A compromise starting edge state.
    ForwardSmoothVector(coefficient, &state, signal);
    state *= (1.0f - coefficient);  // A compromise ending edge state.
    BackwardSmoothVector(coefficient, &state, signal);
  }
}

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_PREPROCESSOR_SIGNAL_VECTOR_UTIL_H_
