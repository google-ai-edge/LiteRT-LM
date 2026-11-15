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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLER_PARAMS_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLER_PARAMS_UTIL_H_

#include <optional>

#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// Rewrites `GREEDY` and `TOP_K` `SamplerParameters` into equivalent `TOP_P`
// parameters (`GREEDY` -> `k = 1, p = 0, temperature = 1`; `TOP_K` -> `p = 1`
// with `k` and `temperature` unchanged), leaving other fields and types intact.
proto::SamplerParameters NormalizeSamplerParams(
    proto::SamplerParameters sampler_params);

// Returns an UnimplementedError unless `type` is `TYPE_UNSPECIFIED` or `TOP_P`,
// or an InvalidArgumentError if `k` is not positive, `p` is outside [0, 1], or
// `temperature` is negative or not finite. The samplers call it on normalized
// parameters (see `NormalizeSamplerParams`). They sample `TYPE_UNSPECIFIED`
// like `TOP_P`, as they did before checking the type.
absl::Status ValidateTopPSamplingParams(proto::SamplerParameters::Type type,
                                        int k, float p, float temperature);

// Returns `TOP_P` parameters with `k = 1` (i.e. greedy sampling), `p = 0`,
// `temperature = 1`, and `seed = 0`.
proto::SamplerParameters GreedySamplerParams();

// Returns whether `a` and `b` select the same sampler configuration after
// normalizing `GREEDY` and `TOP_K` into their equivalent `TOP_P` parameters.
// Unset parameters and parameters with type `TYPE_UNSPECIFIED` both select the
// default sampler, so they compare equal to each other whatever their other
// fields hold. Otherwise all normalized fields are compared; an unset `seed`
// equals seed 0, which is how the samplers read it. `ignore_seed` skips the
// seed, which only picks the random stream, not the sampled distribution or the
// sampler's cost.
//
// Note: the fields are compared one by one; update this function when a field
// is added to `SamplerParameters`.
bool SamplerParamsEquivalent(const std::optional<proto::SamplerParameters>& a,
                             const std::optional<proto::SamplerParameters>& b,
                             bool ignore_seed = false);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLER_PARAMS_UTIL_H_
