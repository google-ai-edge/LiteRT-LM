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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_LLM_EXECUTOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_LLM_EXECUTOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/lora_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/kv_cache_interface.h"
#include "runtime/executor/litert/kv_cache.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_interface.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {

// Common core for all LlmExecutor implementations to share common logic and
// components.
class LlmExecutorCommonCore {
 public:
  absl::StatusOr<std::unique_ptr<KVCacheInterface>> CreateKVCache();

  absl::StatusOr<int> LoadLoRA(const ModelAssets& model_assets);

  absl::Status UnloadLoRA(int lora_id);

  absl::Status Cancel();

  absl::Status Prefill(absl::string_view prefill_signature,
                       absl::flat_hash_map<absl::string_view, TensorBuffer>&
                           prefill_input_buffers,
                       absl::Span<const int> ids, int next_position,
                       LitertKVCache& kv_cache, std::optional<int> lora_id);

  absl::StatusOr<absl::flat_hash_map<absl::string_view, TensorBuffer>>
  CreatePrefillInputBuffers(absl::string_view prefill_signature,
                            int sequence_length, int context_length);

  absl::Status BindTensorsAndRunPrefill(
      absl::string_view signature,
      absl::flat_hash_map<absl::string_view, TensorBuffer>& input_buffers,
      LitertKVCache& kv_cache, std::optional<int> lora_id);

  absl::StatusOr<TensorBuffer> Step(int token, int next_position,
                                    LitertKVCache& kv_cache,
                                    std::optional<int> lora_id);

  absl::StatusOr<TensorBuffer> BindTensorsAndRunStep(
      LitertKVCache& kv_cache, std::optional<int> lora_id);

  absl::StatusOr<int> InferVocabSize();

  LlmExecutorCommonCore(
      LlmExecutorSettings executor_settings, Environment& env,
      const Model& model, CompiledModel compiled_model,
      absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers,
      absl::flat_hash_map<absl::string_view, TensorBuffer>
          decode_output_buffers,
      ModelSignatures signatures,
      std::unique_ptr<EmbeddingLookupManager> embedding_lookup,
      std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup,
      bool use_fp16_precision)
      : executor_settings_(std::move(executor_settings)),
        env_(env),
        model_(model),
        compiled_model_(std::move(compiled_model)),
        decode_input_buffers_(std::move(decode_input_buffers)),
        decode_output_buffers_(std::move(decode_output_buffers)),
        signatures_(signatures),
        embedding_lookup_(std::move(embedding_lookup)),
        per_layer_embedding_lookup_(std::move(per_layer_embedding_lookup)),
        use_fp16_precision_(use_fp16_precision) {}

  LlmExecutorSettings executor_settings_;
  Environment& env_;
  const Model& model_;
  CompiledModel compiled_model_;

  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers_;
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers_;

  // The signatures of the model.
  ModelSignatures signatures_;

  // The embedding lookup for the optional embedder model.
  std::unique_ptr<EmbeddingLookupManager> embedding_lookup_;

  // The embedding lookup for the optional per layer embedder model.
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup_;

  // Whether to use FP16 precision for the calculation.
  bool use_fp16_precision_;
};

// LiteRT-LM executor with external sampler and static shapes. Supports CPU
// and GPU backends.
class LlmExecutorExternalSamplerStatic
    : public LlmExecutorExternalSamplerInterface {
 public:
  static absl::StatusOr<std::unique_ptr<LlmExecutorExternalSamplerStatic>>
  Create(LlmExecutorSettings executor_settings, Environment& env,
         ModelResources& resources);

  absl::StatusOr<std::unique_ptr<KVCacheInterface>> CreateKVCache() override;

  absl::StatusOr<int> LoadLoRA(const ModelAssets& model_assets) override;

  absl::Status UnloadLoRA(int lora_id) override;

  absl::Status Cancel() override;

  absl::Status Prefill(ExecutorInputs&& input_data, KVCacheInterface& kv_cache,
                       std::optional<int> lora_id) override;

  absl::StatusOr<TensorBuffer> Step(ExecutorInputs&& input_data,
                                    KVCacheInterface& kv_cache,
                                    std::optional<int> lora_id) override;

 private:
  friend class LlmExecutorInternalSamplerStatic;

  LlmExecutorExternalSamplerStatic(
      std::unique_ptr<LlmExecutorCommonCore> common_core,
      ActivationDataType logits_data_type,
      SortedPrefillSignatureMap prefill_signature_map)
      : common_core_(std::move(common_core)),
        logits_data_type_(logits_data_type),
        prefill_signature_map_(std::move(prefill_signature_map)) {}

  std::unique_ptr<LlmExecutorCommonCore> common_core_;
  // The logits data type of the model, used to determine the data type of the
  // logits tensor for gpu sampling.
  ActivationDataType logits_data_type_;

  SortedPrefillSignatureMap prefill_signature_map_;

  // Signature names are unique across all signatures in a model so it is safe
  // to refer to them by just their unique name.
  absl::flat_hash_map<
      std::string /*prefill_signature_name*/,
      absl::flat_hash_map<absl::string_view /*input_name*/, TensorBuffer>>
      prefill_input_buffers_;
};

// LiteRT-LM executor with external sampler and dynamic shapes. Supports CPU
// backend only.
class LlmExecutorExternalSamplerDynamic
    : public LlmExecutorExternalSamplerInterface {
 public:
  static absl::StatusOr<std::unique_ptr<LlmExecutorExternalSamplerDynamic>>
  Create(LlmExecutorSettings executor_settings, Environment& env,
         ModelResources& resources);

  absl::StatusOr<std::unique_ptr<KVCacheInterface>> CreateKVCache() override;

  absl::StatusOr<int> LoadLoRA(const ModelAssets& model_assets) override;

  absl::Status UnloadLoRA(int lora_id) override;

  absl::Status Cancel() override;

  absl::Status Prefill(ExecutorInputs&& input_data, KVCacheInterface& kv_cache,
                       std::optional<int> lora_id) override;

  absl::StatusOr<TensorBuffer> Step(ExecutorInputs&& input_data,
                                    KVCacheInterface& kv_cache,
                                    std::optional<int> lora_id) override;

 private:
  LlmExecutorExternalSamplerDynamic(
      std::unique_ptr<LlmExecutorCommonCore> common_core,
      int prefill_chunk_size, uint32_t kv_increament_size)
      : common_core_(std::move(common_core)),
        prefill_chunk_size_(prefill_chunk_size),
        kv_increament_size_(kv_increament_size) {}

  std::unique_ptr<LlmExecutorCommonCore> common_core_;
  int prefill_chunk_size_;
  uint32_t kv_increament_size_;
};

// LiteRT-LM executor with internal sampler and static shapes. Recommended to
// use for backends that benefit from reduced data and control plane movement
// (e.g. GPU).
class LlmExecutorInternalSamplerStatic
    : public LlmExecutorInternalSamplerInterface {
 public:
  static absl::StatusOr<std::unique_ptr<LlmExecutorInternalSamplerStatic>>
  Create(LlmExecutorSettings executor_settings, Environment& env,
         ModelResources& resources);

  absl::StatusOr<std::unique_ptr<KVCacheInterface>> CreateKVCache() override;

  absl::StatusOr<int> LoadLoRA(const ModelAssets& model_assets) override;

  absl::Status UnloadLoRA(int lora_id) override;

  absl::Status Cancel() override;

  absl::Status Prefill(ExecutorInputs&& input_data, KVCacheInterface& kv_cache,
                       std::optional<int> lora_id) override;

  absl::StatusOr<std::vector<std::vector<int>>> SampleTokens(
      int num_steps, ExecutorInputs&& input_data, KVCacheInterface& kv_cache,
      std::optional<int> lora_id) override;

 private:
  LlmExecutorInternalSamplerStatic(
      std::unique_ptr<LlmExecutorExternalSamplerStatic>
          external_sampler_static_executor,
      std::unique_ptr<Sampler> sampler)
      : external_sampler_static_executor_(
            std::move(external_sampler_static_executor)),
        sampler_(std::move(sampler)) {}

  std::unique_ptr<LlmExecutorExternalSamplerStatic>
      external_sampler_static_executor_;

  std::unique_ptr<Sampler> sampler_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_LLM_EXECUTOR_H_
