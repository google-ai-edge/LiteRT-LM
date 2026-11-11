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

#include "runtime/components/sampling_uniforms.h"

#include <algorithm>
#include <cstddef>
#include <random>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/proto/sampler_params.pb.h"
#include "support/util/status_macros.h"

namespace litert::lm {

void DrawUniforms(std::default_random_engine& rng, absl::Span<float> uniforms) {
  // The largest float below 1. Converting a double in [0, 1) to float may
  // round it up to 1.
  constexpr float kMaxUniform = 0x1.fffffep-1f;
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (float& uniform : uniforms) {
    uniform = std::min(static_cast<float>(distribution(rng)), kMaxUniform);
  }
}

void DrawSamplingUniforms(const proto::SamplerParameters& sampler_params,
                          std::default_random_engine& rng,
                          absl::Span<float> uniforms) {
  if (sampler_params.type() != proto::SamplerParameters::GREEDY &&
      sampler_params.k() > 1) {
    DrawUniforms(rng, uniforms);
  } else {
    std::fill(uniforms.begin(), uniforms.end(), 0.0f);
  }
}

absl::Status ValidateUniforms(absl::Span<const float> uniforms,
                              int expected_count) {
  RET_CHECK_GT(expected_count, 0);
  if (uniforms.size() != static_cast<size_t>(expected_count)) {
    return absl::InvalidArgumentError(absl::StrCat("Expected ", expected_count,
                                                   " uniform values, but got ",
                                                   uniforms.size(), "."));
  }
  for (const float uniform : uniforms) {
    // Written so that NaN fails the check too.
    if (!(uniform >= 0.0f && uniform < 1.0f)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Uniform values must be in [0, 1), but got ", uniform, "."));
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
