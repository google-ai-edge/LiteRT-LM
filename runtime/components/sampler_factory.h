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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLER_FACTORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLER_FACTORY_H_

#include <functional>
#include <memory>
#include <optional>
#include <random>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// Creates a Sampler instance based on the provided parameters.
//
// Args:
//   backend: The backend implementation of the sampler (CPU / GPU / ...).
//   batch_size: The batch size for the input logits.
//   sampler_params: The parameters for the sampler.
//   The following parameters are optional and only used for GPU backend.
//   env: The litert environment to use for the sampler.
//   sequence_size: The sequence size for the sampler.
//   vocab_size: The vocabulary size for the sampler.
//   activation_data_type: The activation data type for the sampler.
//
// Returns:
//   The created Sampler instance.
absl::StatusOr<std::unique_ptr<Sampler>> CreateSampler(
    Backend backend, int batch_size, proto::SamplerParameters sampler_params,
    std::optional<std::reference_wrapper<const litert::Environment>> env =
        std::nullopt,
    std::optional<int> sequence_size = std::nullopt,
    std::optional<int> vocab_size = std::nullopt,
    std::optional<ActivationDataType> activation_data_type = std::nullopt);

// Reconfigures `sampler` with `sampler_params` and `rand_gen`.
//
// GPU samplers fix their top-k capacity at creation. If `sampler_params.k()`
// exceeds `max_top_k` (the capacity of `sampler`), or `sampler` is null, the
// sampler is first recreated with `CreateSampler()` (with the remaining
// arguments) and `max_top_k` is updated. If recreation fails, `sampler` and
// `max_top_k` are left unchanged.
// Then `sampler->UpdateConfig()` is called with `batch_size` and `rand_gen`.
//
// `UpdateConfig()` ignores `sampler_params.seed()`: a non-null `rand_gen`
// replaces the sampler's generator, so that the sampler shares the caller's
// random stream, and a null `rand_gen` keeps the existing generator. The seed
// therefore only takes effect when the sampler is recreated; pass a generator
// to apply a new seed.
absl::Status UpdateOrRecreateSampler(
    std::unique_ptr<Sampler>& sampler, int& max_top_k, Backend backend,
    int batch_size, const proto::SamplerParameters& sampler_params,
    std::shared_ptr<std::default_random_engine> rand_gen,
    std::optional<std::reference_wrapper<const litert::Environment>> env =
        std::nullopt,
    std::optional<int> sequence_size = std::nullopt,
    std::optional<int> vocab_size = std::nullopt,
    std::optional<ActivationDataType> activation_data_type = std::nullopt);

// Returns TOP_P parameters with k = 1 (i.e. greedy sampling), temperature 1 and
// seed 0.
proto::SamplerParameters GreedySamplerParams();

// Returns whether `a` and `b` select the same sampler configuration. Unset
// parameters and parameters with type `TYPE_UNSPECIFIED` both select the
// default sampler, so they compare equal to each other whatever their other
// fields hold. Otherwise all fields are compared; an unset `seed` equals seed
// 0, which is how the samplers read it.
//
// Note: the fields are compared one by one; update this function when a field
// is added to `SamplerParameters`.
bool SamplerParamsEquivalent(const std::optional<proto::SamplerParameters>& a,
                             const std::optional<proto::SamplerParameters>& b);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLER_FACTORY_H_
