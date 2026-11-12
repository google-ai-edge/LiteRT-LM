// Copyright 2024 The MediaPipe Authors.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_BYOM_BYOM_FACTORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_BYOM_BYOM_FACTORY_H_

#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"
#include "runtime/executor/gpu_artisan/proto/llm_params.pb.h"

namespace odml_byom {

// TODO: b/319523756 - Deprecate this after we switch to component-based logic.
// Factory method to construct a `LlmBuilder` based on `config.model_type`.
// Any non-BYOM model is assumed to be compatible with the default `LlmBuilder`.
absl::StatusOr<std::unique_ptr<ml_drift::TransformerBuilder>>
CreateTransformerBuilder(const ml_drift::TransformerConfig& config,
                         odml::infra::proto::LlmModelType model_type,
                         int kv_cache_batch_size,
                         const ml_drift::GpuInfo& gpu_info,
                         const ml_drift::CreateGpuModelInfo& create_info,
                         odml::infra::gpu::LlmTensorLoader* tensor_loader);

}  // namespace odml_byom

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_BYOM_BYOM_FACTORY_H_
