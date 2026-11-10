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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TRANSFORMER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TRANSFORMER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/model.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/task/weights_layout.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/runtime_params.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/embedding_updater.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/placeholder_tensor_loader.h"
#include "runtime/executor/gpu_artisan/proto/llm_params.pb.h"
#include "runtime/executor/gpu_artisan/proto/transformer_params.pb.h"

namespace ml_drift {

// Uses the 8 MSB of the uint32 to isolate the different option layers.
inline constexpr uint32_t kLoRAAttentionOptionTag = 1u << 24;
inline constexpr uint32_t kLoRAFeedForwardOptionTag = 2u << 24;
inline constexpr uint32_t kTransformerLayerOptionTag = 4u << 24;
inline constexpr uint32_t kFinalLnOptionTag = 8u << 24;
inline constexpr uint32_t kSharedKvCacheLayersOptionTag = 16u << 24;

static constexpr char kGemmaBaseName[] = "transformer";
static constexpr char kGemmaMtpName[] = "eevee";

std::unique_ptr<GPUOperation> CreateFillTensorOp(const TensorDescriptor& dst);

class VisionBlockIndicesOp : public GPUOperation {
 public:
  VisionBlockIndicesOp() = default;
  int3 GetGridSize() const override { return int3(1, batch_size_, 1); }
  int batch_size_ = 1;
};

VisionBlockIndicesOp CreateVisionBlockIndicesOp(
    const TensorDescriptor& tokens_desc,
    const TensorDescriptor& block_indices_desc, int batch_size);

// Configuration of a single transformer.
struct TransformerConfig {
  // TODO: b/349164917 - Consider moving this to LlmConfig as it is model-level
  // parameter. Not possible to do right now as it is used as
  // num_output_candidates (check SessionConfig).
  int batch_size;
  int sequence_size;
  int model_dimension;
  int hidden_dimension;
  int head_dimension;
  int number_of_heads;
  int number_of_kv_heads;
  // If set, this overrides head_dimension for global attention layers. (V3+)
  std::optional<int> global_head_dimension = std::nullopt;
  // If set, this overrides hidden_dimension for kv_shared ffw layers. (V3+)
  std::optional<int> kv_shared_hidden_dimension = std::nullopt;
  // If true, this uses k=v for global attention layers. (V3+)
  bool global_k_eq_v = false;
  // If set, this overrides number_of_kv_heads for global attention layers (V3+)
  std::optional<int> global_number_of_kv_heads = std::nullopt;
  int vocabulary_size;
  int stack_size;
  // The number of stacks that are treated as "extra", which may have slightly
  // different loading behavior.
  int extra_stack_size = 0;
  std::vector<int> lora_ranks;
  int max_top_k = 1;
  int cache_size = 0;

  // insert rms_norm after feed forward layers
  bool post_norm_ff = true;

  // insert rms_norm after attentions
  bool post_norm_attention = true;

  // insert separate weighted norms for q and k at beginning of attentions
  bool qk_norm = false;
  // When using qk-norm with an architecture that may or may not use weights
  // (like V3), this parameter tells us which of those cases we have.
  bool qk_norm_has_scale = false;

  // Applies to V3+ RMSNormalization ops, except for audio, laurel, & PLE.
  float rms_norm_scale_shift = 0.0f;

  // The value to set the soft cap (Tanh) before calling the final project
  // layer and attention softmax. <=0 indicates there is no cap. Default to the
  // Gemini model setting.
  float attn_soft_cap_value = 50.0f;
  float final_proj_soft_cap_value = 30.0f;

  // Whether to skip or add absolute positional embeddings.
  bool skip_absolute_positional_embeddings = false;

  // Whether to use optional transformer layers.
  bool use_optional_transformer_layers = false;

  // Whether to use placeholder for layers.
  bool use_placeholder_for_layers = false;

  // The numbers of the selectable final ln.
  std::optional<int> selectable_final_ln = std::nullopt;

  // Attention scale type to be applied within the transformer.
  // Default to be inv sqrt scale type as most gemini and gemma models use.
  odml::infra::proto::TransformerParameters::AttentionScaleType
      attention_scale_type = odml::infra::proto::TransformerParameters::
          SCALE_TYPE_INV_SQRT_HEAD_DIM;
  float query_rescale_factor = 0.0f;

  odml::infra::proto::TransformerParameters::AttentionMaskType
      attention_mask_type = odml::infra::proto::TransformerParameters::CAUSAL;
  bool vision_bidirectional_attention = false;

  int num_local_layers_per_global = 0;
  std::optional<int> global_layer_offset = std::nullopt;
  std::optional<int> sliding_window_size = std::nullopt;
  std::optional<float> global_rope_wavelength = std::nullopt;
  std::optional<float> global_rope_scaling = std::nullopt;
  std::optional<float> global_rope_proportion = std::nullopt;
  std::optional<int> ngram_vocabulary_size = std::nullopt;
  std::optional<int> per_layer_embedding_dimension = std::nullopt;
  bool load_per_layer_embedding_dynamically = false;
  bool load_ngrammer_embedding_dynamically = false;
  bool pq_ple_with_layers = false;

  std::optional<
      odml::infra::proto::TransformerParameters::ResidualAdapterParameters>
      residual_adapter = std::nullopt;

  std::optional<int> num_draft_tokens = std::nullopt;

  // Typically only scale LoRA weights by 1/lora_rank for Gemini v >= 2.
  // However, there are exceptions such as Gemini Audio v1.5.
  bool scale_lora = false;

  // Vision modality.
  // With MobileNet vision encoder, vision_tokens_num denotes the number of
  // tokens per image, which is the same for all images. With ViT vision
  // encoder, vision_tokens_num denotes the maximum number of tokens per image.
  int vision_tokens_num = 0;
  int max_num_images = 0;

  // Audio modality.
  // Number of media embedding cache slots to reserve for audio embeddings
  // (including end-of-audio embeddings), after the slots reserved for images.
  // This is used when the audio encoder is not part of this model (e.g. a
  // LiteRT audio encoder), so the cache can't be sized from `audio_encoder`.
  int max_num_audio_tokens = 0;

  // Audio hard token encoding params.
  // Which index audio hard tokens start at.
  int audio_vocab_offset = 0;
  // Length of audio_input_embedding_table_extra variable.
  int audio_vocab_extra_dim = 0;
  // Dimension which we embed audio hard tokens to.
  int audio_input_embedding_dim = 0;
  // Whether to apply RMS normalization to the audio hard tokens.
  // Only implemented for v1 transformers.
  bool audio_dual_norm_soft_and_hard_tokens = true;

  // Enable LlmGpuRunner::DecodeToLogits call.
  bool enable_decode_logits = false;

  // The maximum number of indices to be removed from the cache with "clean
  // cache at idx".
  int max_indices_count_in_clean_cache_at = 32;

  // Laurel configs.
  // The embedding w/ shape [B, L, d_model] will first be projected to
  // [B, L, laurel_rank] and then projected to [B, L, d_model]. When it is
  // greater than 0, the laurel layer will be applied.
  int laurel_rank = 0;
  // The scale shift for the post laurel rms normalization. The default value
  // is 0.0f, which means no scale shift.
  float post_laurel_norm_scale_shift = 0.0f;

  // The start layer of the kv cache that is shared from the previous layer.
  std::optional<int> kv_cache_shared_start_layer = std::nullopt;

  // The scalar multiplier for the standard deviation used in statistical top-k
  // thresholding for activation sparsity.
  std::optional<float> activation_sparsity_stddev_multiplier = std::nullopt;

  // Altup configs.
  struct AltupConfig {
    // The number of modality routers.
    // This value needs to be initialized based on the real model config when
    // the transformer object is created.
    int altup_num_modalities = 4;
    // The number of AltUp towers (blocks), i.e., the expansion factor $K$.
    int altup_num_inputs = 8;
    // The activated block index.
    int altup_active_idx = 0;
    // Runtime optimization, whether to directly stream the active_x from the
    // prediction to the correction.
    bool direct_active_x_stream = false;

    friend std::ostream& operator<<(std::ostream& os,
                                    const AltupConfig& config) {
      return os << "altup_num_modalities: " << config.altup_num_modalities
                << " altup_num_inputs: " << config.altup_num_inputs
                << " altup_active_idx: " << config.altup_active_idx
                << " direct_active_x_stream: " << config.direct_active_x_stream;
    }
  };
  std::optional<AltupConfig> altup_config;

  // MoE configs.
  struct MoeConfig {
    int hidden_dimension;
    int num_experts;
    int num_active_experts;

    friend std::ostream& operator<<(std::ostream& os, const MoeConfig& config) {
      return os << "hidden_dimension: " << config.hidden_dimension
                << " num_experts: " << config.num_experts
                << " num_active_experts: " << config.num_active_experts;
    }
  };
  std::optional<MoeConfig> moe_config;

  // Centroid Embedder configs.
  struct CentroidEmbedderConfig {
    int num_centroids;
    int top_k;

    friend std::ostream& operator<<(std::ostream& os,
                                    const CentroidEmbedderConfig& config) {
      return os << "num_centroids: " << config.num_centroids
                << " top_k: " << config.top_k;
    }
  };
  std::optional<CentroidEmbedderConfig> centroid_embedder_config;

  // MTP configs
  std::optional<int> post_proj_size;
  bool combine_activations_with_input = false;

  // If we need to apply a special workaround when running in an f16 environment
  // for models (like Gemma3) which were trained using bfloat16, in order to
  // avoid overflows.
  bool gemma3_bfloat16_fix = false;

  int light_conv_kernel_size = 0;

  struct RelativePositionalEmbeddingConfig {
    int min_timescale;
    int max_timescale;
    int block_size;  // Deprecated. Use kConformerBlockSize in audio_encoder.cc.
    int past_horizon;
    int future_horizon;

    friend std::ostream& operator<<(
        std::ostream& os, const RelativePositionalEmbeddingConfig& config) {
      return os << "min_timescale: " << config.min_timescale
                << " max_timescale: " << config.max_timescale
                << " block_size: " << config.block_size
                << " past_horizon: " << config.past_horizon
                << " future_horizon: " << config.future_horizon;
    }
  };
  std::optional<RelativePositionalEmbeddingConfig>
      relative_positional_embedding_config = std::nullopt;

  // Whether to apply SRQ. If true, depending on the model setup, static
  // range quantization will be applied to the corresponding layers, e.g.,
  // attn_vec_einsum, kv_einsum, q_einsum, mlp/gating_einsum and mlp/linear.
  bool apply_srq = false;

  // If applying srq, whether to do so also for PLE.
  bool apply_srq_to_ple = false;

  // Whether to use int8 for KV cache. Only used when apply_srq is true.
  bool int8_kv_cache = true;

  // Forces int8 kv cache when apply_srq is false. Uses estimated SRQs.
  // TODO: Figure out how to best trigger this on demand.
  bool force_int8_kv_cache = false;

  // Sets the size of the local cache ringbuffer. This must be strictly greater
  // than the sliding window size. Larger values will retain more history, for
  // context editing. Smaller values will reduce memory. Without this feature,
  // large context windows can take up ~6x as much memory!
  // TODO: Figure out how to best trigger this on demand.
  // TODO: Ensure this plays nicely with context Clone/Clear/Save/Restore.
  std::optional<int> local_ring_cache_size = std::nullopt;

  // If set, automatically use ringbuffers for local KV caches with size
  // sliding_window + max_prefill_seq_len.
  bool use_autosized_ringbuffers = false;

  int layer_skip_start = 0;
  int num_layers_skipped = 0;
  int matform_factor = 0;

  // Some models have special scaling factors applied to final outputs of
  // specific layers. This is currently only applied to global layers, and
  // is called `skip_scale` elsewhere.
  bool scale_layer = false;

  // Whether to enable external embedding. If it's true, the model (from V3)
  // will take external embedding as input input instead tokens.
  bool enable_external_embeddings = false;
};

int SkipMatformIndex(int i, const TransformerConfig& config);

// Holds information on LoRA tensors that will allow loading from a separate
// tensor loader.
struct LoRATensorInfo {
  WeightsDescription weights_desc;
  OHWI shape;
  GpuModelBuilder::ValueId id;
  bool swap_dims;
};

inline std::ostream& operator<<(std::ostream& os,
                                const TransformerConfig& config) {
  os << "batch_size: " << config.batch_size
     << " sequence_size: " << config.sequence_size
     << " model_dimension: " << config.model_dimension
     << " hidden_dimension: " << config.hidden_dimension
     << " head_dimension: " << config.head_dimension
     << " number_of_heads: " << config.number_of_heads
     << " number_of_kv_heads: " << config.number_of_kv_heads
     << " global_head_dimension: "
     << (config.global_head_dimension.has_value()
             ? std::to_string(*config.global_head_dimension)
             : std::string("std::nullopt"))
     << " kv_shared_hidden_dimension: "
     << (config.kv_shared_hidden_dimension.has_value()
             ? std::to_string(*config.kv_shared_hidden_dimension)
             : std::string("std::nullopt"))
     << " global_k_eq_v: " << config.global_k_eq_v
     << " global_number_of_kv_heads: "
     << (config.global_number_of_kv_heads.has_value()
             ? std::to_string(*config.global_number_of_kv_heads)
             : std::string("std::nullopt"))
     << " vocabulary_size: " << config.vocabulary_size
     << " stack_size: " << config.stack_size
     << " extra_stack_size: " << config.extra_stack_size
     << " lora_rank: " << absl::StrJoin(config.lora_ranks, ", ")
     << " max_top_k: " << config.max_top_k
     << " cache_size: " << config.cache_size
     << " post_norm_ff: " << config.post_norm_ff
     << " post_norm_attention: " << config.post_norm_attention
     << " qk_norm: " << config.qk_norm
     << " qk_norm_has_scale: " << config.qk_norm_has_scale
     << " rms_norm_scale_shift: " << config.rms_norm_scale_shift
     << " attn_soft_cap_value: " << config.attn_soft_cap_value
     << " final_proj_soft_cap_value: " << config.final_proj_soft_cap_value
     << " skip_absolute_positional_embeddings: "
     << config.skip_absolute_positional_embeddings
     << " use_optional_transformer_layers: "
     << config.use_optional_transformer_layers
     << " use_placeholder_for_layers: " << config.use_placeholder_for_layers
     << " selectable_final_ln: "
     << (config.selectable_final_ln.has_value()
             ? std::to_string(*config.selectable_final_ln)
             : std::string("std::nullopt"))
     << " attention_scale_type: " << config.attention_scale_type
     << (config.query_rescale_factor > 0.0f
             ? absl::StrCat("query_rescale_factor: ",
                            config.query_rescale_factor)
             : "")
     << " attention_mask_type: " << config.attention_mask_type
     << " num_local_layers_per_global: " << config.num_local_layers_per_global
     << " global_layer_offset: "
     << (config.global_layer_offset.has_value()
             ? std::to_string(*config.global_layer_offset)
             : std::string("std::nullopt"))
     << " sliding_window_size: "
     << (config.sliding_window_size.has_value()
             ? std::to_string(*config.sliding_window_size)
             : std::string("std::nullopt"))
     << " global_rope_wavelength: "
     << (config.global_rope_wavelength.has_value()
             ? std::to_string(*config.global_rope_wavelength)
             : std::string("std::nullopt"))
     << " global_rope_scaling: "
     << (config.global_rope_scaling.has_value()
             ? std::to_string(*config.global_rope_scaling)
             : std::string("std::nullopt"))
     << " global_rope_proportion: "
     << (config.global_rope_proportion.has_value()
             ? std::to_string(*config.global_rope_proportion)
             : std::string("std::nullopt"))
     << " ngram_vocabulary_size: "
     << (config.ngram_vocabulary_size.has_value()
             ? std::to_string(*config.ngram_vocabulary_size)
             : std::string("std::nullopt"))
     << " load_ngrammer_embedding_dynamically: "
     << config.load_ngrammer_embedding_dynamically
     << " per_layer_embedding_dimension: "
     << (config.per_layer_embedding_dimension.has_value()
             ? std::to_string(*config.per_layer_embedding_dimension)
             : std::string("std::nullopt"))
     << " load_per_layer_embedding_dynamically: "
     << config.load_per_layer_embedding_dynamically
     << " pq_ple_with_layers: " << config.pq_ple_with_layers
     << " residual_adapter_interleave: "
     << (config.residual_adapter.has_value()
             ? std::to_string(static_cast<int>(
                   config.residual_adapter->where_to_interleave()))
             : std::string("std::nullopt"))
     << " residual_adapter_bottleneck_dimension: "
     << (config.residual_adapter.has_value()
             ? std::to_string(config.residual_adapter->bottleneck_dimension())
             : std::string("std::nullopt"))
     << " num_draft_tokens: "
     << (config.num_draft_tokens.has_value()
             ? std::to_string(*config.num_draft_tokens)
             : std::string("std::nullopt"))
     << " scale_lora: " << config.scale_lora
     << " vision_tokens_num: " << config.vision_tokens_num
     << " max_num_images: " << config.max_num_images
     << " max_num_audio_tokens: " << config.max_num_audio_tokens
     << " audio_vocab_offset: " << config.audio_vocab_offset
     << " audio_vocab_extra_dim: " << config.audio_vocab_extra_dim
     << " audio_input_embedding_dim: " << config.audio_input_embedding_dim
     << " audio_dual_norm_soft_and_hard_tokens: "
     << config.audio_dual_norm_soft_and_hard_tokens
     << " enable_decode_logits: " << config.enable_decode_logits
     << " max_indices_count_in_clean_cache_at: "
     << config.max_indices_count_in_clean_cache_at
     << " laurel_rank: " << config.laurel_rank
     << " post_laurel_norm_scale_shift: " << config.post_laurel_norm_scale_shift
     << " kv_cache_shared_start_layer: "
     << (config.kv_cache_shared_start_layer.has_value()
             ? std::to_string(*config.kv_cache_shared_start_layer)
             : std::string("std::nullopt"))
     << " activation_sparsity_stddev_multiplier: "
     << (config.activation_sparsity_stddev_multiplier.has_value()
             ? std::to_string(*config.activation_sparsity_stddev_multiplier)
             : std::string("std::nullopt"));

  if (config.altup_config.has_value()) {
    os << " altup_config: " << *config.altup_config;
  } else {
    os << " altup_config: std::nullopt";
  }

  if (config.moe_config.has_value()) {
    os << " moe_config: " << *config.moe_config;
  } else {
    os << " moe_config: std::nullopt";
  }

  if (config.centroid_embedder_config.has_value()) {
    os << " centroid_embedder_config: " << *config.centroid_embedder_config;
  } else {
    os << " centroid_embedder_config: std::nullopt";
  }

  os << " post_proj_size: "
     << (config.post_proj_size.has_value()
             ? std::to_string(*config.post_proj_size)
             : std::string("std::nullopt"))
     << " combine_activations_with_input: "
     << config.combine_activations_with_input
     << " bfloat16 fix: " << config.gemma3_bfloat16_fix
     << " light_conv_kernel_size: " << config.light_conv_kernel_size;

  if (config.relative_positional_embedding_config.has_value()) {
    os << " relative_positional_embedding_config: "
       << *config.relative_positional_embedding_config;
  } else {
    os << " relative_positional_embedding_config: std::nullopt";
  }

  os << " apply_srq: " << config.apply_srq
     << " apply_srq_to_ple: " << config.apply_srq_to_ple
     << " int8_kv_cache: " << config.int8_kv_cache
     << " force_int8_kv_cache: " << config.force_int8_kv_cache;

  if (config.local_ring_cache_size.has_value()) {
    os << " local_ring_cache_size: " << *config.local_ring_cache_size;
  } else {
    os << " local_ring_cache_size: std::nullopt";
  }

  os << " layer_skip_start: " << config.layer_skip_start
     << " num_layers_skipped: " << config.num_layers_skipped
     << " matform_factor: " << config.matform_factor
     << " scale_layer: " << config.scale_layer
     << " enable_external_embeddings: " << config.enable_external_embeddings;

  return os;
}

class CopySubLinearTensorOp : public ml_drift::GPUOperation {
 public:
  CopySubLinearTensorOp(const TensorDescriptor& src,
                        const TensorDescriptor& dst, int offset = 0);
  ml_drift::int3 GetGridSize() const override;

  // Move only
  CopySubLinearTensorOp(CopySubLinearTensorOp&& operation) = default;
  CopySubLinearTensorOp& operator=(CopySubLinearTensorOp&& operation) = default;
  CopySubLinearTensorOp(const CopySubLinearTensorOp&) = delete;
  CopySubLinearTensorOp& operator=(const CopySubLinearTensorOp&) = delete;
};

class TransformerBuilder {
 public:
  TransformerBuilder(const TransformerConfig& config, int kv_cache_batch_size,
                     const GpuInfo& gpu_info,
                     const CreateGpuModelInfo& create_info,
                     odml::infra::gpu::LlmTensorLoader* tensor_loader)
      : config_(config),
        gpu_info_(gpu_info),
        create_info_(create_info),
        kv_cache_batch_size_(kv_cache_batch_size),
        data_type_(DeduceDataTypeFromPrecision(create_info_.precision)),
        tensor_loader_(tensor_loader) {
    tensor_loader_->SetGpuInfo(gpu_info_);
    tensor_loader_->SetDataType(data_type_);
    gemma3_bfloat16_fix_ =
        config.gemma3_bfloat16_fix && data_type_ == DataType::kFloat16;
  }
  virtual ~TransformerBuilder() = default;

  void SetLocalRingCacheSize(int local_ring_cache_size) {
    config_.local_ring_cache_size = local_ring_cache_size;
  }

  absl::StatusOr<GpuModel> CreateStackedTransformerModel(
      int sequence_size, int batch_size,
      GpuModelBuilder::TensorHandle* input_handle,
      GpuModelBuilder::TensorHandle* media_handle,
      GpuModelBuilder::TensorHandle* params_i32_handle,
      std::vector<GpuModelBuilder::TensorHandle>* key_cached_tensors,
      std::vector<GpuModelBuilder::TensorHandle>* value_cached_tensors,
      GpuModelBuilder::TensorHandle* output_handle);

  // If segment_pos_tensor, global_mask_tensor or local_mask_tensor is not
  // nullptr, they will have to be generated and provided outside of the
  // transformer model. If they are null, they will be generated as part of
  // the transformer model.
  absl::StatusOr<GpuModel> CreateStackedTransformerModelV2(
      int sequence_size, int batch_size,
      GpuModelBuilder::TensorHandle* current_tokens,
      GpuModelBuilder::TensorHandle* media_handle,
      GpuModelBuilder::TensorHandle* tokens_ids_handle,
      GpuModelBuilder::TensorHandle* params_i32_handle,
      std::vector<GpuModelBuilder::TensorHandle>* key_cached_tensors,
      std::vector<GpuModelBuilder::TensorHandle>* value_cached_tensors,
      GpuModelBuilder::TensorHandle* output_handle,
      std::vector<odml::infra::gpu::EmbeddingWeights>* ple_weights,
      std::vector<odml::infra::gpu::EmbeddingWeights>* ngrammer_weights,
      std::vector<GpuModelBuilder::TensorHandle>* vision_ple_tensors,
      GpuModelBuilder::TensorHandle* segment_pos_tensor,
      GpuModelBuilder::TensorHandle* global_mask_tensor,
      GpuModelBuilder::TensorHandle* local_mask_tensor);

  absl::StatusOr<GpuModel> CreateStackedTransformerModelV3(
      int sequence_size, int batch_size,
      GpuModelBuilder::TensorHandle* current_tokens,
      GpuModelBuilder::TensorHandle* media_handle,
      GpuModelBuilder::TensorHandle* tokens_ids_handle,
      GpuModelBuilder::TensorHandle* params_i32_handle,
      std::vector<GpuModelBuilder::TensorHandle>* key_cached_tensors,
      std::vector<GpuModelBuilder::TensorHandle>* value_cached_tensors,
      GpuModelBuilder::TensorHandle* output_handle,
      std::vector<odml::infra::gpu::EmbeddingWeights>* ple_weights,
      std::vector<GpuModelBuilder::TensorHandle>* vision_ple_tensors,
      GpuModelBuilder::TensorHandle* segment_pos_tensor,
      GpuModelBuilder::TensorHandle* global_mask_tensor,
      GpuModelBuilder::TensorHandle* local_mask_tensor,
      GpuModelBuilder::TensorHandle* external_embeddings_handle,
      std::vector<GpuModelBuilder::TensorHandle>* external_ple_handles,
      GpuModelBuilder::TensorHandle* input_activations_handle,
      GpuModelBuilder::TensorHandle* output_activations_handle,
      const std::string& base_name);

  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeMatMulSoftmaxMatMul(
      GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& q,
      const GpuModelBuilder::TensorHandle& k,
      const GpuModelBuilder::TensorHandle* k_scaled_weights,
      const GpuModelBuilder::TensorHandle& v,
      const GpuModelBuilder::TensorHandle* v_scaled_weights,
      const GpuModelBuilder::TensorHandle& params_tensor,
      const std::function<GpuModelBuilder::TensorHandle(
          GpuModelBuilder::TensorHandle)>& qk_post_proc_graph,
      bool is_global);

  // Create a GpuModel that performs the final embedding to logits operation.
  absl::StatusOr<GpuModel> CreateFinalEmbeddingToLogitsModel(
      GpuModelBuilder::TensorHandle* input_handle,
      GpuModelBuilder::TensorHandle* logits_handle, int sequence_size);

  // Creates a GpuModel that performs only the main token embedding lookup.
  absl::StatusOr<GpuModel> CreateEmbeddingLookupModel(
      int sequence_size, int batch_size,
      GpuModelBuilder::TensorHandle* current_tokens,
      GpuModelBuilder::TensorHandle* output_embeddings,
      std::vector<odml::infra::gpu::EmbeddingWeights>* ple_weights,
      std::vector<GpuModelBuilder::TensorHandle>* output_ple_embeddings);

  // The PostProcessing model will be created such that it'll take in
  // sequence_size tokens in the sequence length direction of each output head,
  // as in the input_handle will be of contain the following dimensions:
  // batch_size, sequence_size, model_dimension
  absl::StatusOr<GpuModel> CreatePostProcessingModel(
      GpuModelBuilder::TensorHandle* input_handle,
      GpuModelBuilder::TensorHandle* params_i32_handle,
      GpuModelBuilder::TensorHandle* params_f32_handle,
      GpuModelBuilder::TensorHandle* tokens_ids_handle,
      GpuModelBuilder::TensorHandle* stacked_transformer_input_handle,
      GpuModelBuilder::TensorHandle* pre_softmax_handle,
      GpuModelBuilder::TensorHandle* topk_tokens_handle,
      GpuModelBuilder::TensorHandle* topk_probabilities_handle,
      GpuModelBuilder::TensorHandle* constraint_mask_handle, int sequence_size);

  absl::StatusOr<GpuModel> CreateSoftmaxModel(
      GpuModelBuilder::TensorHandle* input,
      GpuModelBuilder::TensorHandle* output);

  absl::StatusOr<GpuModel> CreateCleanCacheModel(
      std::vector<GpuModelBuilder::TensorHandle>* key_cached_tensors,
      std::vector<GpuModelBuilder::TensorHandle>* value_cached_tensors,
      GpuModelBuilder::TensorHandle* params_i32_handle,
      GpuModelBuilder::TensorHandle* indices_to_remove_handle = nullptr);

  // Shared tensors, can be used by few models.
  TensorDescriptor GetParamsTensorDescriptor() const;
  TensorDescriptor GetKCacheTensorDescriptor(bool is_global = false) const;
  static TensorDescriptor GetKCacheTensorDescriptor(
      const TransformerConfig& config, int kv_cache_batch_size, int cache_size,
      DataType data_type, bool is_global = false);
  TensorDescriptor GetVCacheTensorDescriptor(bool is_global = false) const;
  static TensorDescriptor GetVCacheTensorDescriptor(
      const TransformerConfig& config, int kv_cache_batch_size, int cache_size,
      DataType data_type, bool is_global = false);
  TensorDescriptor GetTokensTensorDescriptor() const;
  TensorDescriptor GetInputTensorDescriptor(int batch_size,
                                            int sequence_size) const;
  TensorDescriptor GetVisionPleTensorDescriptor() const;
  TensorDescriptor GetMediaEmbeddingsTensorDescriptor() const;

  // Use small local cache in attention calculation
  static bool UseSmallLocalCache(const TransformerConfig& config) {
    return config.sliding_window_size.has_value() &&
           config.num_local_layers_per_global > 0 &&
           config.cache_size > config.sliding_window_size.value();
  }
  static bool UseRingBuffer(const TransformerConfig& config) {
    return UseSmallLocalCache(config) &&
           config.local_ring_cache_size.has_value();
  }
  static int GetCacheSize(const TransformerConfig& config, bool is_global);

  const absl::flat_hash_map<ValueId, GpuSpatialTensor*>&
  GetExternalImmutableTensors() const {
    return external_immutable_tensors_;
  }
  const absl::flat_hash_map<ValueId, TensorDescriptor>&
  GetExternalMutableTensors() const {
    return external_mutable_tensors_;
  }
  const absl::flat_hash_map<std::string, GpuModelBuilder::TensorHandle>&
  GetDebugTensors() {
    return debug_tensors_;
  }

  absl::flat_hash_map<int, absl::flat_hash_map<std::string, LoRATensorInfo>>&
  lora_tensors() {
    return lora_tensors_;
  }

  absl::flat_hash_map<std::string, std::vector<odml::infra::gpu::LoadTensorFn>>&
  tensor_reload_fns() {
    return tensor_reload_fns_;
  }

  std::vector<std::string> GetLayerNames() const;
  std::vector<std::string> GetLayerNames(const std::string& prefix) const;
  std::string GetPerLayerEmbeddingName(int layer) const;
  std::string GetNgrammerEmbeddingName(int index) const;

  // Distinguishing features for now are absence of ngram_vocabulary_size, with
  // local-global attention patterns, use of per layer embedding,
  // kv cache sharing, and laurel presence.
  bool UseGemmaModelV3N() const {
    return config_.num_local_layers_per_global > 0 &&
           config_.sliding_window_size.has_value() &&
           *config_.sliding_window_size > 0 &&
           !config_.ngram_vocabulary_size.has_value() &&
           config_.per_layer_embedding_dimension &&
           *config_.per_layer_embedding_dimension > 0 &&
           config_.kv_cache_shared_start_layer.has_value() &&
           *config_.kv_cache_shared_start_layer > 0 && config_.laurel_rank > 0;
  }

  // Gemma V4 models have different head dimensions for the global and local
  // layers. Currently we use this feature alone to distinguish from other model
  // families. Since we have some tests for global_head_dimension usage, we
  // also check for sliding_window_size so that test code doesn't trigger V3+
  // paths necessarily.
  bool UseGemmaModelV4() const {
    return config_.global_head_dimension.has_value() &&
           config_.sliding_window_size.has_value() &&
           *config_.sliding_window_size > 0;
  }

  // Determines whether we use new V3+ code path. Shared by both V3N and V4.
  bool UseGemmaModelV3NPlus() const {
    return UseGemmaModelV3N() || UseGemmaModelV4();
  }

  // Gemini V2 models should use the StackedTransformerModelV2.
  bool UseGoogleInternalModelV2() const {
    return config_.num_local_layers_per_global > 0 &&
           config_.sliding_window_size.has_value() &&
           *config_.sliding_window_size > 0 &&
           config_.ngram_vocabulary_size.has_value() &&
           *config_.ngram_vocabulary_size > 0 &&
           config_.per_layer_embedding_dimension &&
           *config_.per_layer_embedding_dimension > 0;
  }

  void SetMediaCacheSize(int media_cache_size) {
    media_cache_size_ = media_cache_size;
  }

  void ChangePrecisionIfBfloat16Fix(CalculationsPrecision precision);
  GpuModelBuilder::TensorHandle CastIfBfloat16Fix(
      const GpuModelBuilder::TensorHandle& src, DataType data_type);

 protected:
  // For debugging only. Saves a specific internal tensor for later inspection,
  // using the supplied name as a lookup key.
  void AddDebugTensor(const std::string& name,
                      const GpuModelBuilder::TensorHandle& tensor);

  // Default LLM building scales our embeddings by sqrt(model_dimension).
  // This is overridden by BYOM models, which don't have such a re-scaling step.
  virtual bool ScaleEmbeddingsBySqrtModelDim() { return true; }

  struct LlmWeightsDesc {
    bool IsQuantized() const {
      return description.type == DataType::kUint8 ||
             description.type == DataType::kUint4 ||
             description.type == DataType::kUint2;
    }
    WeightsDescription description;
    OHWI shape;
    OHWI scale_zp_shape;
    GpuModelBuilder::TensorHandle weights;
    GpuModelBuilder::TensorHandle scale;
    GpuModelBuilder::TensorHandle zero_point;
  };
  void CreateCache(
      std::vector<GpuModelBuilder::TensorHandle>* key_cached_tensors,
      std::vector<GpuModelBuilder::TensorHandle>* value_cached_tensors);
  GpuModelBuilder::TensorHandle DequantizeInt8Cache(
      GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& int8_cache,
      const std::string& scale_name);
  virtual absl::StatusOr<GpuModelBuilder::TensorHandle> MakeFFLayer(
      const GpuModelBuilder::TensorHandle& src, const std::string& name);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeFFLayerV2(
      const GpuModelBuilder::TensorHandle& src, const std::string& name);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeFFLayerV3(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      bool use_statistical_top_k, bool is_global, bool is_kv_shared_layer);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeFFLayerMoe(
      const GpuModelBuilder::TensorHandle& src, const std::string& name);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeFFLayerExperts(
      const GpuModelBuilder::TensorHandle& src,
      const GpuModelBuilder::TensorHandle& src_no_norm,
      const std::string& name);
  absl::StatusOr<GpuModelBuilder::TensorHandle> CreateComputeRouterModalitiesV3(
      const GpuModelBuilder::TensorHandle& src, const std::string& name);
  absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
  CreateAltupPredictV3(
      const std::vector<GpuModelBuilder::TensorHandle>& input_handles,
      const std::string& name);
  absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
  CreateAltupCorrectV3(
      const std::vector<GpuModelBuilder::TensorHandle>& predictions,
      const GpuModelBuilder::TensorHandle& activated, const std::string& name);
  absl::StatusOr<GpuModelBuilder::TensorHandle>
  CreatePostProcessEmbeddingsWithAltupV3(
      const std::vector<GpuModelBuilder::TensorHandle>& input_embeds,
      const std::string& name);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeLaurelLayerV3(
      const GpuModelBuilder::TensorHandle& src,
      const GpuModelBuilder::TensorHandle& normed_src, const std::string& name);
  absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>> PreprocessAltupV3(
      const GpuModelBuilder::TensorHandle& src, const std::string& name);
  absl::StatusOr<GpuModelBuilder::TensorHandle> ApplyRotqatRotate(
      const GpuModelBuilder::TensorHandle& src, int hadamard_dim,
      const std::string& scale_name);
  absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
  MakeProjectedPleInputsV3(const GpuModelBuilder::TensorHandle& src,
                           const std::string& projection_name,
                           const std::string& norm_name);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakePleLayer(
      const GpuModelBuilder::TensorHandle& src,
      const GpuModelBuilder::TensorHandle& per_layer_embedding,
      const std::string& name);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakePleLayerV3(
      const GpuModelBuilder::TensorHandle& src,
      const GpuModelBuilder::TensorHandle& per_layer_embedding,
      const std::string& name);
  absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>> MakeGetQKV(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      bool is_global = false);
  absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>> MakeGetQ(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      bool is_global = false);

  virtual absl::StatusOr<GpuModelBuilder::TensorHandle> MakeSelfAttention(
      const GpuModelBuilder::TensorHandle& src,
      const GpuModelBuilder::TensorHandle& attention_mask,
      const GpuModelBuilder::TensorHandle& segment_pos,
      const GpuModelBuilder::TensorHandle& key_tensor,
      const GpuModelBuilder::TensorHandle& value_tensor,
      const GpuModelBuilder::TensorHandle& params_tensor, bool is_global,
      const std::string& name);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeSelfAttentionV2(
      const GpuModelBuilder::TensorHandle& src,
      const GpuModelBuilder::TensorHandle& attention_mask,
      const GpuModelBuilder::TensorHandle& segment_pos,
      const GpuModelBuilder::TensorHandle& key_tensor,
      const GpuModelBuilder::TensorHandle& value_tensor,
      const GpuModelBuilder::TensorHandle& params_tensor, bool is_global,
      const std::string& name);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeSelfAttentionV2Subgraph(
      const std::string& subgraph_id, const BHWC& src_sh,
      GpuModelBuilder::TensorHandle q, GpuModelBuilder::TensorHandle k,
      GpuModelBuilder::TensorHandle v, GpuModelBuilder::TensorHandle mask,
      GpuModelBuilder::TensorHandle fire_bias,
      GpuModelBuilder::TensorHandle segment_pos,
      GpuModelBuilder::TensorHandle key_tensor,
      GpuModelBuilder::TensorHandle value_tensor,
      GpuModelBuilder::TensorHandle params_tensor, bool is_global);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeSelfAttentionV3(
      const GpuModelBuilder::TensorHandle& src,
      const GpuModelBuilder::TensorHandle& attention_mask,
      const GpuModelBuilder::TensorHandle& segment_pos,
      const GpuModelBuilder::TensorHandle& key_tensor,
      const GpuModelBuilder::TensorHandle& value_tensor,
      const GpuModelBuilder::TensorHandle& params_tensor, bool is_global,
      const std::string& name, bool shared_kv_cache,
      const std::string& kv_layer_name);
  GpuModelBuilder::TensorHandle MakeRMSNormalizationWithBuilder(
      GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& src,
      const std::string& name, float scale_shift = 1.0f);
  absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
  MakeSelfAttentionV3SubgraphPreKvCacheWrite(
      const std::string& subgraph_id, GpuModelBuilder::TensorHandle q,
      GpuModelBuilder::TensorHandle k, GpuModelBuilder::TensorHandle v,
      GpuModelBuilder::TensorHandle segment_pos, bool is_global,
      bool shared_kv_cache, const std::string& qk_norm_prefix);
  absl::StatusOr<GpuModelBuilder::TensorHandle>
  MakeSelfAttentionV3SubgraphPostKvCacheWrite(
      const std::string& subgraph_id, const BHWC& src_sh,
      GpuModelBuilder::TensorHandle q, GpuModelBuilder::TensorHandle mask,
      GpuModelBuilder::TensorHandle key_tensor,
      GpuModelBuilder::TensorHandle value_tensor,
      GpuModelBuilder::TensorHandle params_tensor,
      const std::string& kv_layer_name, bool is_global);
  virtual absl::StatusOr<GpuModelBuilder::TensorHandle> MakeLayer(
      const GpuModelBuilder::TensorHandle& inputs,
      const GpuModelBuilder::TensorHandle& attention_mask,
      const GpuModelBuilder::TensorHandle& segment_pos,
      const GpuModelBuilder::TensorHandle& key_tensor,
      const GpuModelBuilder::TensorHandle& value_tensor,
      const GpuModelBuilder::TensorHandle& params_tensor, bool is_global,
      const std::string& name);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeLayerV2(
      const GpuModelBuilder::TensorHandle& inputs,
      const GpuModelBuilder::TensorHandle& attention_mask,
      const GpuModelBuilder::TensorHandle& segment_pos,
      const GpuModelBuilder::TensorHandle& key_tensor,
      const GpuModelBuilder::TensorHandle& value_tensor,
      const GpuModelBuilder::TensorHandle* per_layer_embedding,
      const GpuModelBuilder::TensorHandle& params_tensor, bool is_global,
      const std::string& name);
  absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>> MakeLayerV3(
      const std::vector<GpuModelBuilder::TensorHandle>& inputs,
      const GpuModelBuilder::TensorHandle& attention_mask,
      const GpuModelBuilder::TensorHandle& segment_pos,
      const GpuModelBuilder::TensorHandle& key_tensor,
      const GpuModelBuilder::TensorHandle& value_tensor,
      const GpuModelBuilder::TensorHandle* per_layer_embedding,
      const GpuModelBuilder::TensorHandle& params_tensor, bool is_global,
      const std::string& name, int layer_num, const std::string& kv_layer_name);
  GpuModelBuilder::TensorHandle MakeFillSegmentPos(
      const GpuModelBuilder::TensorHandle& params_i32);
  virtual absl::StatusOr<GpuModelBuilder::TensorHandle> MakeFillAttention(
      const GpuModelBuilder::TensorHandle& params_i32,
      const GpuModelBuilder::TensorHandle* current_tokens = nullptr);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeFillLocalAttention(
      const GpuModelBuilder::TensorHandle& params_i32,
      const GpuModelBuilder::TensorHandle* current_tokens = nullptr);
  absl::StatusOr<int> GetMaskWidth(
      odml::infra::proto::TransformerParameters::AttentionMaskType
          attn_mask_type);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeFillAttentionMask(
      const GpuModelBuilder::TensorHandle& params_i32,
      const GpuModelBuilder::TensorHandle* current_tokens, bool is_local);
  // `value` must be >= 0.
  GpuModelBuilder::TensorHandle MakeAddWithOverflow(
      const GpuModelBuilder::TensorHandle& src, int value);
  // Entries of `src1` must be >= 0.
  GpuModelBuilder::TensorHandle MakeAddWithOverflow2Input(
      const GpuModelBuilder::TensorHandle& src0,
      const GpuModelBuilder::TensorHandle& src1);
  // `value` must be >= 0.
  GpuModelBuilder::TensorHandle MakeMulWithOverflow(
      const GpuModelBuilder::TensorHandle& src, int value);
  virtual std::string GetEmbeddingLookupName() const;
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeEmbLookUp(
      const GpuModelBuilder::TensorHandle& src);
  virtual GpuModelBuilder::TensorHandle MakeNormalization(
      const GpuModelBuilder::TensorHandle& src, const std::string& name);

  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeLinearAutoPrecision(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      int out_channels);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeLinearAutoPrecision(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      int out_channels, const GpuModelBuilder::TensorHandle* lora_src,
      std::optional<odml::infra::gpu::LlmTensorLoader::MatformOptions>
          matform_options);

  // When apply_srq is false, it is the same as MakeLinearAutoPrecision.
  // When apply_srq is true, it will apply static range quantization.
  // Specifically, it will apply fake quant to the inputs and apply fake quant
  // to the outputs from MakeLinearAutoPrecision.
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeLinearAutoPrecisionWithSrq(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      int out_channels, bool apply_srq,
      std::optional<odml::infra::gpu::LlmTensorLoader::MatformOptions>
          matform_options = std::nullopt);

  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeConvAutoPrecision(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      Layout src_layout, int out_channels);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeConvAutoPrecision(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      Layout src_layout, int out_channels,
      const GpuModelBuilder::TensorHandle* lora_src,
      std::optional<odml::infra::gpu::LlmTensorLoader::MatformOptions>
          matform_options);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeAutoPrecisionOIConvWithIds(
      const GpuModelBuilder::TensorHandle& src,
      const GpuModelBuilder::TensorHandle& batched_ids, const std::string& name,
      int out_channels);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeConv(
      const GpuModelBuilder::TensorHandle& src,
      const GpuModelBuilder::Weights& weights,
      const GpuModelBuilder::TensorHandle* bias_th = nullptr);
  absl::StatusOr<GpuModelBuilder::TensorHandle> AddLoRA(
      const GpuModelBuilder::TensorHandle& lora_input,
      const GpuModelBuilder::TensorHandle& base, const std::string& name,
      Layout layout);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeInputLoRAInjection(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      int out_channels);
  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeLoRAInjection(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      Layout src_layout, int out_channels, int lora_rank);

  absl::StatusOr<GpuModelBuilder::Weights> GetWeightsTensor(
      const std::string& w_name, Layout layout, int in_channels,
      int out_channels, DataType float_type,
      std::optional<odml::infra::gpu::LlmTensorLoader::MatformOptions>
          matform_options,
      int weights_batch = 1);
  absl::StatusOr<GpuModelBuilder::Weights> GetWeightsFloatTensor(
      const std::string& w_name, Layout layout, int in_channels,
      int out_channels, DataType data_type, int weights_batch = 1);
  absl::StatusOr<GpuModelBuilder::Weights> GetWeightsI8Tensor(
      const std::string& w_name, Layout layout, int in_channels,
      int out_channels, int weights_batch = 1);
  absl::StatusOr<GpuModelBuilder::Weights> GetWeightsI4Tensor(
      const std::string& w_name, Layout layout, int in_channels,
      int out_channels,
      std::optional<odml::infra::gpu::LlmTensorLoader::MatformOptions>
          matform_options,
      int weights_batch = 1);
  absl::StatusOr<GpuModelBuilder::Weights> GetWeightsI2Tensor(
      const std::string& w_name, Layout layout, int in_channels,
      int out_channels, int weights_batch = 1);
  absl::StatusOr<GpuModelBuilder::Weights> GetLoraWeightsTensor(
      const std::string& w_name, Layout layout, int in_channels,
      int out_channels, DataType data_type, int lora_rank);

  // RMSNormalization with a scale shift added to the scale tensor loaded by
  // tensor loader.
  GpuModelBuilder::TensorHandle MakeRMSNormalization(
      const GpuModelBuilder::TensorHandle& src, const std::string& name,
      float scale_shift = 1.0f);

  // Loads per-layer embedding tables and performs embedding lookups on them.
  // `ple_tensors` and `ple_scale_tensors` will be populated with the loaded
  // embedding tables and their quantization scales. The returned tensor handles
  // are the result of the embedding lookups for the input `tokens`.
  absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
  LoadPerLayerEmbeddings(
      const GpuModelBuilder::TensorHandle& tokens,
      std::vector<odml::infra::gpu::EmbeddingWeights>* ple_weights);

  GpuModelBuilder::TensorHandle DecompressPqPleLayers(
      const GpuModelBuilder::TensorHandle& compressed_table,
      const BHWC& in_shape_one, const BHWC& in_shape_two,
      const BHWC& out_shape);

  absl::StatusOr<GpuModelBuilder::TensorHandle> GetCentroidEmbeddings(
      const GpuModelBuilder::TensorHandle& activations,
      const std::string& name);

  absl::StatusOr<LlmWeightsDesc> LoadWeights(const std::string& name,
                                             const OHWI& weights_shape,
                                             bool swap_dims);

  GpuModelBuilder::TensorHandle MakeEmbLookUpOp(
      const GpuModelBuilder::TensorHandle& src,
      const LlmWeightsDesc& embedding);

  absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
  GetNGrammerEmbeddings(const GpuModelBuilder::TensorHandle& current_tokens,
                        const GpuModelBuilder::TensorHandle& previous_tokens,
                        absl::Span<const int> primes);

  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeNGrammerLayer(
      const GpuModelBuilder::TensorHandle& embeddings,
      const std::vector<GpuModelBuilder::TensorHandle>& ngrammer_embs,
      const std::string& name);

  absl::StatusOr<std::vector<GpuModelBuilder::TensorHandle>>
  GetPerLayerEmbeddings(const GpuModelBuilder::TensorHandle& tokens);

  GpuModelBuilder::TensorHandle MakeRotaryEmbedding(
      GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& src,
      const GpuModelBuilder::TensorHandle& segment_pos, float min_timescale,
      float max_timescale, float proportion);

  // Q: Reshape -> Transpose -> Reshape
  // K: Reshape -> Transpose -> Reshape
  // V: Reshape -> Transpose -> Reshape
  void MakeTransformQKV(GpuModelBuilder& builder,
                        GpuModelBuilder::TensorHandle* q,
                        GpuModelBuilder::TensorHandle* k,
                        GpuModelBuilder::TensorHandle* v,
                        bool is_global = false);

  // for Q: Reshape->Transpose->Reshape -> Split->RoPE->Concat
  // for K: Reshape->Transpose->Reshape -> Split->RoPE->Concat
  // for V: Reshape->Transpose->Reshape
  void MakeTransformQKVApplyRoPE(GpuModelBuilder& builder,
                                 const GpuModelBuilder::TensorHandle& position,
                                 GpuModelBuilder::TensorHandle* q,
                                 GpuModelBuilder::TensorHandle* k,
                                 GpuModelBuilder::TensorHandle* v,
                                 float min_timescale, float max_timescale);

  // for Q: Reshape->Transpose->Reshape -> RMSNorm -> Split->RoPE->Concat
  // for K: Reshape->Transpose->Reshape -> RMSNorm -> Split->RoPE->Concat
  // for V: Reshape->Transpose->Reshape
  absl::Status MakeTransformQKVRmsNormApplyRoPE(
      GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& position,
      GpuModelBuilder::TensorHandle* q, GpuModelBuilder::TensorHandle* k,
      GpuModelBuilder::TensorHandle* v, bool is_global, float min_timescale,
      float max_timescale, const std::string& qk_norm_prefix = "",
      float scale_shift = 0.0f, float proportion = 1.0f);

  // Dispatches QKV transformation and RoPE/RMSNorm application across Q, K, and
  // V tensors based on model configuration and available fused kernels.
  absl::Status QKVRoPE(GpuModelBuilder& builder,
                       const GpuModelBuilder::TensorHandle& segment_pos,
                       GpuModelBuilder::TensorHandle* q,
                       GpuModelBuilder::TensorHandle* k = nullptr,
                       GpuModelBuilder::TensorHandle* v = nullptr,
                       bool is_global = false,
                       const std::string& qk_norm_prefix = "",
                       std::optional<float> scale_shift = std::nullopt,
                       std::optional<float> proportion = std::nullopt);

  GpuModelBuilder::TensorHandle ApplyFireBias(
      const GpuModelBuilder::TensorHandle& position, float log_scale_factor,
      float threshold, const BHWC& dst_shape);

  absl::StatusOr<GpuModelBuilder::TensorHandle> PerDimScale(
      GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& src,
      const std::string& name);

  GpuModelBuilder::TensorHandle GatedResidualConnection(
      const GpuModelBuilder::TensorHandle& x,
      const GpuModelBuilder::TensorHandle& y, const std::string& name);

  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeMatMulSoftmaxMatMul(
      GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& q,
      const GpuModelBuilder::TensorHandle& k,
      const GpuModelBuilder::TensorHandle& v,
      const GpuModelBuilder::TensorHandle& params_tensor,
      const std::function<GpuModelBuilder::TensorHandle(
          GpuModelBuilder::TensorHandle)>& qk_post_proc_graph,
      bool is_global);

  absl::StatusOr<GpuModelBuilder::TensorHandle> MakeResidualAdapter(
      const GpuModelBuilder::TensorHandle& feature, const std::string& name);

  absl::StatusOr<GpuModelBuilder::TensorHandle> InterleaveTextAndMedia(
      const GpuModelBuilder::TensorHandle& text,
      const GpuModelBuilder::TensorHandle& media);

  GpuModelBuilder::TensorHandle MakeMagnitude(
      const GpuModelBuilder::TensorHandle& src, Axis axis);

  // Load the scale tensor according to the scale_name and apply fake quant.
  GpuModelBuilder::TensorHandle MakeFakeQuant(
      const GpuModelBuilder::TensorHandle& src, const std::string& scale_name,
      DataType data_type = DataType::kInt8);

  GpuModelBuilder::TensorHandle MakeQuantInt8(
      const GpuModelBuilder::TensorHandle& src, const std::string& scale_name,
      const DataType& scale_type);
  GpuModelBuilder::TensorHandle MakeQuantInt8(
      const GpuModelBuilder::TensorHandle& src, float scale,
      const DataType& scale_type);

  float GetQuantScale(const std::string& scale_name, const DataType& src_type);

  GpuModelBuilder::TensorHandle CreateKVCacheQuantizedScaleTensor(
      GpuModelBuilder& builder, DataType type, int channel_count, float scale);

  absl::StatusOr<GpuModelBuilder::TensorHandle> GatherMedia(
      const GpuModelBuilder::TensorHandle& media,
      const GpuModelBuilder::TensorHandle& media_index);

  void CopyTokensToOutputBuffer(
      const GpuModelBuilder::TensorHandle& tokens,
      const GpuModelBuilder::TensorHandle& params_i32_handle,
      const GpuModelBuilder::TensorHandle& output);

  GpuModelBuilder::TensorHandle MakeRelu1p5(
      const GpuModelBuilder::TensorHandle& src);

  // Can handle automatically batch broadcasting for src k/v
  void AddValuesToCache(GpuModelBuilder& builder,
                        const GpuModelBuilder::TensorHandle& k,
                        const GpuModelBuilder::TensorHandle& v,
                        const GpuModelBuilder::TensorHandle& params_tensor,
                        const GpuModelBuilder::TensorHandle& cache_k,
                        const GpuModelBuilder::TensorHandle& cache_v,
                        bool is_global = false);

  void ExtractLocalCache(GpuModelBuilder& builder,
                         const GpuModelBuilder::TensorHandle& k,
                         const GpuModelBuilder::TensorHandle& v,
                         const GpuModelBuilder::TensorHandle& params_tensor,
                         const GpuModelBuilder::TensorHandle& k_local,
                         const GpuModelBuilder::TensorHandle& v_local);

  void ExtractLocalSlidingWindowMask(
      GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& src_mask,
      const GpuModelBuilder::TensorHandle& dst_mask,
      const GpuModelBuilder::TensorHandle& params_tensor);

  GpuModelBuilder::TensorHandle ExtractFirstBatch(
      GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& cache);

  absl::StatusOr<GpuModelBuilder::TensorHandle> GetLocalMask(
      GpuModelBuilder::TensorHandle* params_tensor,
      GpuModelBuilder::TensorHandle* external_local_mask,
      GpuModelBuilder::TensorHandle* current_tokens = nullptr);

  void WriteCurrentTokens(
      const GpuModelBuilder::TensorHandle& current_tokens,
      const GpuModelBuilder::TensorHandle& tokens_ids_handle,
      const GpuModelBuilder::TensorHandle& params_tensor);
  void ReadPreviousTokens(
      const GpuModelBuilder::TensorHandle& previous_tokens,
      const GpuModelBuilder::TensorHandle& tokens_ids_handle,
      const GpuModelBuilder::TensorHandle& params_tensor);

  int GetI32ParamsCount() const { return LlmRuntimeParams::kTotalParamsCount; }

  // params_i32 values: same as above
  // params_f32 values:
  // [0:out_batch_size*sequence_size-1] sampling params, there's one for each
  // sampled output token, that is batch_size * sequence_size.
  // For the ordering of the sampling params, or which item corresponds to which
  // batch and sequence index, see MakeSampling() for details.
  // out_batch_size: 1.0f / temperature
  int GetF32ParamsCount(int sequence_size) const {
    return kv_cache_batch_size_ * sequence_size + 1;
  }

  TensorDescriptor GetEmbeddingDesc(const BHWC& shape, int vocab_size,
                                    const std::string& base_name) const;
  absl::StatusOr<GpuModelBuilder::TensorHandle> DequantizeEmbedding(
      const BHWC& dst_shape, const GpuModelBuilder::TensorHandle& weights,
      const GpuModelBuilder::TensorHandle& scales,
      const GpuModelBuilder::TensorHandle* lookup = nullptr);

  GpuModelBuilder::TensorHandle AddExternalTensor(GpuSpatialTensor* tensor);
  GpuModelBuilder::TensorHandle AddExternalMutableTensor(
      const TensorDescriptor& desc);

  GpuModelBuilder::TensorHandle AddMutableTensor(
      TensorDescriptor tensor_descriptor);

  DataType GetKVCacheType() const {
    if (config_.apply_srq && config_.int8_kv_cache) {
      return DataType::kUint8;
    } else if (config_.force_int8_kv_cache) {
      return DataType::kUint8;
    } else {
      return data_type_;
    }
  }

  // In all cases actual size: sliding_window_size.value() + sequence_size - 1.
  // But for sequence_size != 1 we will use sliding_window_size.value() +
  // sequence_size for better alignment.
  int GetLocalCacheSize() const;

  static WeightsDescription GetKWeightsDescription(
      const TransformerConfig& config, int kv_cache_batch_size, int cache_size,
      DataType data_type, bool is_global = false);
  static OHWI GetKWeightsShape(const TransformerConfig& config,
                               int kv_cache_batch_size, int cache_size,
                               bool is_global = false);

  static WeightsDescription GetVWeightsDescription(
      const TransformerConfig& config, int kv_cache_batch_size, int cache_size,
      DataType data_type, bool is_global = false);
  static OHWI GetVWeightsShape(const TransformerConfig& config,
                               int kv_cache_batch_size, int cache_size,
                               bool is_global = false);

  GpuModelBuilder::TensorHandle CapInfinity(
      const GpuModelBuilder::TensorHandle& src);

  std::string GetConvId(const GpuModelBuilder::TensorHandle& src,
                        const OHWI& weights_shape,
                        const WeightsDescription& weights_desc);

  // If the converter stored `<scales_name>.f16_rescale_inv`, records it as the
  // output multiplier for convs using `scale_th` (applied in MakeConv).
  void RegisterF16RescaleMultiplier(
      const std::string& scales_name,
      const GpuModelBuilder::TensorHandle& scale_th);

  void CleanState();

  TransformerConfig config_;
  GpuInfo gpu_info_;
  CreateGpuModelInfo create_info_;
  absl::flat_hash_map<ValueId, GpuSpatialTensor*> external_immutable_tensors_;
  absl::flat_hash_map<ValueId, TensorDescriptor> external_mutable_tensors_;
  GpuModelBuilder model_builder_;
  // This is the number of output heads.
  int kv_cache_batch_size_;
  // multimodal cache size
  int media_cache_size_;
  DataType data_type_;
  bool gemma3_bfloat16_fix_ = false;

  odml::infra::gpu::LlmTensorLoader* tensor_loader_ = nullptr;
  absl::flat_hash_map<std::string, GpuModelBuilder::TensorHandle>
      debug_tensors_;

  // Maps a weights-scale tensor id to the multiplier that must be applied to
  // the conv output, for scales that were pre-multiplied by a power of two to
  // avoid F16 subnormals.
  absl::flat_hash_map<ValueId, float> weights_output_multipliers_;

  // Holds the LoRA tensors created while building the LLM. The first key is the
  // LoRA rank and the second key is the name of the tensor.
  absl::flat_hash_map<int, absl::flat_hash_map<std::string, LoRATensorInfo>>
      lora_tensors_;

  // Holds the layer tensors reload functions for different transformer layer.
  absl::flat_hash_map<std::string, std::vector<odml::infra::gpu::LoadTensorFn>>
      tensor_reload_fns_;
};

absl::StatusOr<std::vector<odml::infra::gpu::EmbeddingRaw>>
GetDynamicPleEmbeddings(const TransformerConfig& config,
                        odml::infra::gpu::LlmTensorLoader* tensor_loader);

absl::StatusOr<std::vector<odml::infra::gpu::EmbeddingRaw>>
GetDynamicNgrammerEmbeddings(const TransformerConfig& config,
                             odml::infra::gpu::LlmTensorLoader* tensor_loader);

// Creates an operation to copy values from one KV cache to another.
std::unique_ptr<GPUOperation> CreateCopyCacheOp(const TensorDescriptor& cache_k,
                                                const TensorDescriptor& cache_v,
                                                const TransformerConfig& config,
                                                int kv_cache_batch_size,
                                                bool is_global = false);

CreateGpuModelInfo GetCreateGpuModelInfo(
    const CreateGpuModelInfo& create_info,
    const absl::flat_hash_map<ValueId, GpuSpatialTensor*>&
        external_immutable_tensors,
    const absl::flat_hash_map<ValueId, TensorDescriptor>&
        external_mutable_tensors);

std::vector<int32_t> GetLlmRuntimeParamsBuffer(
    const LlmRuntimeParams& param_vals, const TransformerConfig& config);

GpuModelBuilder::TensorHandle MakeSelectMaskOp(
    GpuModelBuilder& builder, const GpuModelBuilder::TensorHandle& src,
    const GpuModelBuilder::TensorHandle& mask);

}  // namespace ml_drift

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_GPU_ARTISAN_ML_DRIFT_TRANSFORMER_H_
