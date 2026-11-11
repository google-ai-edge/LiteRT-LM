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

#include "runtime/components/sampler_params_util.h"

#include <cmath>
#include <optional>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

proto::SamplerParameters NormalizeSamplerParams(
    proto::SamplerParameters sampler_params) {
  switch (sampler_params.type()) {
    case proto::SamplerParameters::GREEDY:
      sampler_params.set_type(proto::SamplerParameters::TOP_P);
      sampler_params.set_k(1);
      sampler_params.set_p(0.0f);
      sampler_params.set_temperature(1.0f);
      break;
    case proto::SamplerParameters::TOP_K:
      sampler_params.set_type(proto::SamplerParameters::TOP_P);
      sampler_params.set_p(1.0f);
      break;
    default:
      break;
  }
  return sampler_params;
}

absl::Status ValidateTopPSamplingParams(proto::SamplerParameters::Type type,
                                        int k, float p, float temperature) {
  switch (type) {
    case proto::SamplerParameters::TYPE_UNSPECIFIED:
    case proto::SamplerParameters::TOP_P:
      break;
    default:
      return absl::UnimplementedError(
          absl::StrCat("Sampler type: ", type, " not implemented yet."));
  }
  if (k <= 0) {
    return absl::InvalidArgumentError("k must be positive.");
  }
  if (!(p >= 0.0f && p <= 1.0f)) {
    return absl::InvalidArgumentError("p must be in [0, 1].");
  }
  if (!(temperature >= 0.0f && std::isfinite(temperature))) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Temperature must be >= 0 and finite, but got ", temperature));
  }
  return absl::OkStatus();
}

proto::SamplerParameters GreedySamplerParams() {
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_k(1);
  sampler_params.set_p(0.0f);
  sampler_params.set_temperature(1.0f);
  sampler_params.set_seed(0);
  return sampler_params;
}

bool SamplerParamsEquivalent(const std::optional<proto::SamplerParameters>& a,
                             const std::optional<proto::SamplerParameters>& b,
                             bool ignore_seed) {
  const auto is_default = [](const std::optional<proto::SamplerParameters>& p) {
    return !p.has_value() ||
           p->type() == proto::SamplerParameters::TYPE_UNSPECIFIED;
  };
  const bool a_is_default = is_default(a);
  const bool b_is_default = is_default(b);
  if (a_is_default || b_is_default) {
    return a_is_default == b_is_default;
  }
  const proto::SamplerParameters norm_a = NormalizeSamplerParams(*a);
  const proto::SamplerParameters norm_b = NormalizeSamplerParams(*b);
  return norm_a.type() == norm_b.type() && norm_a.k() == norm_b.k() &&
         norm_a.p() == norm_b.p() &&
         norm_a.temperature() == norm_b.temperature() &&
         (ignore_seed || norm_a.seed() == norm_b.seed()) &&
         norm_a.backend() == norm_b.backend() &&
         norm_a.compute_exact_log_probs() == norm_b.compute_exact_log_probs();
}

}  // namespace litert::lm
