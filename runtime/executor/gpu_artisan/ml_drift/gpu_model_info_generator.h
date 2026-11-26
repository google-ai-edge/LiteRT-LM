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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_GPU_MODEL_INFO_GENERATOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_GPU_MODEL_INFO_GENERATOR_H_

#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/proto/llm_inference_engine.pb.h"

namespace odml::infra::gpu {

// Creates a `CreateGpuModelInfo` from the provided boolean configuration flags.
ml_drift::CreateGpuModelInfo GenerateGpuModelInfo(
    bool allow_precision_loss, bool use_low_power, bool prefer_texture_weights,
    bool enable_host_mapped_pointer, bool enable_fast_tuning,
    bool enable_winograd_opt, bool disallow_8bit_convs);

// Creates a `CreateGpuModelInfo` from the provided `MlDriftOptions` and
// `ActivationDataType`.
ml_drift::CreateGpuModelInfo GenerateGpuModelInfo(
    const proto::SessionConfig::MlDriftOptions& mldrift_options,
    const proto::SessionConfig::ActivationDataType& activation_data_type);

}  // namespace odml::infra::gpu

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_GPU_MODEL_INFO_GENERATOR_H_
