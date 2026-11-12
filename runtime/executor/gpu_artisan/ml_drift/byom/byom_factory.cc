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

#include "runtime/executor/gpu_artisan/ml_drift/byom/byom_factory.h"

#include <memory>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/byom/falcon.h"
#include "runtime/executor/gpu_artisan/ml_drift/byom/phi.h"
#include "runtime/executor/gpu_artisan/ml_drift/byom/stablelm.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"
#include "runtime/executor/gpu_artisan/proto/llm_params.pb.h"

namespace odml_byom {
namespace {
using ::odml::infra::proto::LlmModelType;
}

absl::StatusOr<std::unique_ptr<ml_drift::TransformerBuilder>>
CreateTransformerBuilder(const ml_drift::TransformerConfig& config,
                         odml::infra::proto::LlmModelType model_type,
                         int kv_cache_batch_size,
                         const ml_drift::GpuInfo& gpu_info,
                         const ml_drift::CreateGpuModelInfo& create_info,
                         odml::infra::gpu::LlmTensorLoader* tensor_loader) {
  if (config.number_of_kv_heads < 1 ||
      config.number_of_kv_heads > config.number_of_heads) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid number of kv heads. kv heads - ", config.number_of_kv_heads,
        " vs heads - ", config.number_of_heads));
  }
  switch (model_type) {
    case LlmModelType::LLM_MODEL_TYPE_FALCON_RW_1B:
      return std::make_unique<odml_byom::FalconTransformerBuilder>(
          config, kv_cache_batch_size, gpu_info, create_info, tensor_loader);
    case LlmModelType::LLM_MODEL_TYPE_STABLELM_4E1T_3B:
      return std::make_unique<odml_byom::StablelmTransformerBuilder>(
          config, kv_cache_batch_size, gpu_info, create_info, tensor_loader);
    case LlmModelType::LLM_MODEL_TYPE_PHI_2:
      return std::make_unique<odml_byom::PhiTransformerBuilder>(
          config, kv_cache_batch_size, gpu_info, create_info, tensor_loader);
    default:
      return std::make_unique<ml_drift::TransformerBuilder>(
          config, kv_cache_batch_size, gpu_info, create_info, tensor_loader);
  }
}

}  // namespace odml_byom
