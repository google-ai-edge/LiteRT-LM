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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLING_UNIFORMS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLING_UNIFORMS_H_

#include <random>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// Helpers for the uniform random values that a sampling call consumes: one
// value in [0, 1) per sampled token, used to pick the token by inverse
// transform sampling.

// Fills `uniforms` with float(uniform_real_distribution<double>(0, 1)(rng)),
// clamped below 1.
void DrawUniforms(std::default_random_engine& rng, absl::Span<float> uniforms);

// Fills `uniforms` for a sampling call with `sampler_params`: draws them from
// `rng`, or sets them to 0 without consuming `rng` for greedy sampling
// (k <= 1).
void DrawSamplingUniforms(const proto::SamplerParameters& sampler_params,
                          std::default_random_engine& rng,
                          absl::Span<float> uniforms);

// Returns `InvalidArgumentError` unless `uniforms` has `expected_count` values,
// each in [0, 1).
absl::Status ValidateUniforms(absl::Span<const float> uniforms,
                              int expected_count);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLING_UNIFORMS_H_
