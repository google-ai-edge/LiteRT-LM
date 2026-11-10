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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <queue>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/audio_encoder.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"
#include "runtime/executor/gpu_artisan/proto/llm_inference_engine.pb.h"
#include "runtime/executor/gpu_artisan/proto/llm_params.pb.h"
#include "runtime/executor/gpu_artisan/proto/transformer_params.pb.h"

namespace ml_drift {

// Holds data needed to save/restore the context of the LLM.
struct ContextData {
  ContextData();
  ~ContextData();

  // Move only.
  ContextData(ContextData&& other);
  ContextData& operator=(ContextData&& other);

  ContextData(const ContextData&) = delete;
  ContextData& operator=(const ContextData& other) = delete;

  std::optional<uint32_t> lora_id;
  std::vector<std::unique_ptr<GpuSpatialTensor>> key_tensors;
  std::vector<std::unique_ptr<GpuSpatialTensor>> value_tensors;
  std::vector<int> tokens;

  // Speculative decoding related data.

  // Tokens that are ready to be returned from the speculative runner. The
  // speculative runner can produce more tokens than requested, so tokens that
  // haven't been returned yet are queued up here.
  std::queue<int> ready_tokens;
  // The ContextData of the speculative decoding drafter.
  std::unique_ptr<ContextData> drafter_context;
};

GPUOperation CreateWriteParamsOp(const TensorDescriptor& dst, int params_count);

GPUOperation CreateCopyNthToSingleOp(const GpuInfo& gpu_info,
                                     const TensorDescriptor& src,
                                     const TensorDescriptor& dst);

// Copies tokens or positions to the destination tensors. No support for
// batching, expected to be used only in DecodeToLogits with embeddings
// as input.
// dst: destination tensor for tokens or positions.
GPUOperation CreateCopyTokensOrPositionsOp(const TensorDescriptor& dst);

GPUOperation CreateCopyExternalEmbeddingsOp(const TensorDescriptor& dst);

// Copies a slice of the source tensor into the destination tensor at the
// specified index. specifically: dst[0, 0, index:index + x, :] = src.
// src: source tensor of shape [1, 1, x, emb_dim].
// dst: destination tensor of shape [1, 1, media_cache_size, emb_dim].
GPUOperation CreateCopyMediaEmbeddingsOp(const TensorDescriptor& src,
                                         const TensorDescriptor& dst);

// Copies tokens and positions to the destination tensors. No support for
// batching, expected to be used only in DecodeToLogits with token as input.
// dst_tokens: destination tensor for tokens.
// dst_positions: destination tensor for positions.
GPUOperation CreateCopyTokensAndPositionsOp(
    const TensorDescriptor& dst_tokens, const TensorDescriptor& dst_positions);

// Copies masks to the destination tensors. No support for batching, expected
// to be used only in DecodeToLogits
// dst_local: destination tensor for local attention mask.
// dst_global: destination tensor for global attention mask.
// mask_width: width of the single token row in the mask, which is a max
//             sequence length.
GPUOperation CreateCopyMasksOp(const TensorDescriptor& dst_local,
                               const TensorDescriptor& dst_global,
                               int mask_width);

// Configuration of LLM-based model. It contains one or more transformers and
// other components.
struct LlmConfig {
  // TODO: b/319523756 - Delete this field once BYOM models are supported with
  // component-based logic.
  odml::infra::proto::LlmModelType model_type =
      odml::infra::proto::LLM_MODEL_TYPE_UNKNOWN;

  std::optional<TransformerConfig> text_decoder = std::nullopt;

  std::optional<AudioEncoderConfig> audio_encoder = std::nullopt;

  std::vector<int> suppressed_tokens;
};

inline std::ostream& operator<<(std::ostream& os, const LlmConfig& config) {
  os << " model_type: " << config.model_type;
  if (config.text_decoder.has_value()) {
    os << " text decoder: (" << config.text_decoder.value() << ")";
  }
  if (config.audio_encoder.has_value()) {
    os << " audio encoder: (" << config.audio_encoder.value() << ")";
  }
  return os;
}

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_LLM_H_
