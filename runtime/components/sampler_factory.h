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

#include <memory>
#include <optional>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert {
class Environment;
}  // namespace litert

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
    litert::Environment* env = nullptr,
    std::optional<int> sequence_size = std::nullopt,
    std::optional<int> vocab_size = std::nullopt,
    std::optional<ActivationDataType> activation_data_type = std::nullopt);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLER_FACTORY_H_
