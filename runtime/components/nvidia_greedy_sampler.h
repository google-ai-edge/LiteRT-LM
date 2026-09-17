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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_NVIDIA_GREEDY_SAMPLER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_NVIDIA_GREEDY_SAMPLER_H_

#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// Metadata-only eligibility check. Malformed shapes on otherwise matching
// NVIDIA float buffers are errors; other backends/types/configs are
// unsupported.
absl::StatusOr<bool> SupportsNvidiaGreedySampler(
    Backend backend, const proto::SamplerParameters& sampler_params,
    TensorBufferType buffer_type, const RankedTensorType& logits_type);

// Optional MTP argmax extension. Returns nullptr to keep the existing CPU/GPU
// sampler when disabled, unsupported, or absent from the already-loaded NVIDIA
// dispatch library. Once matched, operational errors are returned, not hidden.
// Requires LITERT_NVIDIA_MTP_GPU_SAMPLING=1; never loads a new dispatch
// instance.
absl::StatusOr<std::unique_ptr<Sampler>> TryCreateNvidiaGreedySampler(
    const Environment& env, Backend backend,
    const proto::SamplerParameters& sampler_params,
    const TensorBuffer& logits_tensor);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_NVIDIA_GREEDY_SAMPLER_H_
