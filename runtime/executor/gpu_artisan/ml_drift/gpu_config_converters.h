// Copyright 2024 The ODML Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_GPU_CONFIG_CONVERTERS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_GPU_CONFIG_CONVERTERS_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"
#include "runtime/executor/gpu_artisan/proto/llm_params.pb.h"
#include "runtime/executor/gpu_artisan/proto/transformer_params.pb.h"

namespace odml::infra::gpu {

// Converts a `LlmParameters` proto to a `ml_drift::LlmConfig`.
absl::StatusOr<ml_drift::LlmConfig> CreateGpuConfigFromLlmParametersProto(
    const proto::LlmParameters& llm_params, bool use_submodel = false);

// Updates `ml_drift::LlmConfig` for audio gemini.
absl::Status UpdateGpuConfigForAudio(ml_drift::LlmConfig& llm_config,
                                     int max_seq_length,
                                     int max_audio_seq_length,
                                     bool is_streaming, int tokens_per_frame);

// Creates a `TransformerConfig` from `TransformerParameters`; prefer other
// calls when possible.
ml_drift::TransformerConfig CreateTransformerConfig(
    const proto::TransformerParameters& params);

}  // namespace odml::infra::gpu

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_GPU_CONFIG_CONVERTERS_H_
