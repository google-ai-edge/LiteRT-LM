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

#include "runtime/executor/gpu_artisan/ml_drift/gpu_config_converters.h"

#include <algorithm>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/audio_encoder.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"
#include "runtime/executor/gpu_artisan/proto/audio_encoder_params.pb.h"
#include "runtime/executor/gpu_artisan/proto/llm_params.pb.h"
#include "runtime/executor/gpu_artisan/proto/transformer_params.pb.h"
#include "runtime/util/status_macros.h"

namespace odml::infra::gpu {

namespace {

using ::ml_drift::AudioEncoderConfig;
using ::ml_drift::ConvConfig;
using ::ml_drift::HW;
using ::ml_drift::LlmConfig;
using ::ml_drift::MLPAdapterConfig;
using ::ml_drift::ProjectionConfig;
using ::ml_drift::SubSampleConvProjectionConfig;
using ::ml_drift::TransformerConfig;
using ::odml::infra::proto::AudioEncoderParameters;
using ::odml::infra::proto::LlmParameters;
using ::odml::infra::proto::TransformerParameters;
using SubSampleConvProjectionParameters = ::odml::infra::proto::
    AudioEncoderParameters::SubSampleConvProjectionParameters;
using MLPAdapterParameters =
    ::odml::infra::proto::AudioEncoderParameters::MLPAdapterParameters;
using ProjectionParameters =
    ::odml::infra::proto::AudioEncoderParameters::ProjectionParameters;

ml_drift::SubSamplePaddingType ConvertSubSamplePaddingType(
    AudioEncoderParameters::SubSamplePaddingType type) {
  switch (type) {
    case AudioEncoderParameters::CAUSAL:
      return ml_drift::SubSamplePaddingType::CAUSAL;
    case AudioEncoderParameters::REVERSE_CAUSAL:
      return ml_drift::SubSamplePaddingType::REVERSE_CAUSAL;
    case AudioEncoderParameters::SAME:
      return ml_drift::SubSamplePaddingType::SAME;
    default:
      return ml_drift::SubSamplePaddingType::UNSPECIFIED;
  }
}

SubSampleConvProjectionConfig CreateSubSampleConvProjectionConfig(
    const SubSampleConvProjectionParameters& params) {
  SubSampleConvProjectionConfig config;
  config.input_dims = params.input_dims();
  config.model_dims = params.model_dims();
  config.convs = std::vector<ConvConfig>(params.convs_size());
  for (int i = 0; i < params.convs_size(); ++i) {
    const auto& conv = params.convs(i);
    config.convs[i].channel_size = conv.channel_size();
    config.convs[i].stride = HW(conv.stride().h(), conv.stride().w());
    config.convs[i].kernel_size =
        HW(conv.kernel_size().h(), conv.kernel_size().w());
    config.convs[i].padding.h = ConvertSubSamplePaddingType(conv.padding().h());
    config.convs[i].padding.w = ConvertSubSamplePaddingType(conv.padding().w());
    config.convs[i].bias = conv.bias();
  }
  config.is_causal = params.is_causal();
  config.proj_bias = params.proj_bias();
  config.norm_bias = params.norm_bias();
  config.shrink_factor = params.shrink_factor();
  config.take_every_n = params.take_every_n();
  config.pad_for_streaming = 0;
  return config;
}

MLPAdapterConfig CreateMLPAdapterConfig(const MLPAdapterParameters& params) {
  MLPAdapterConfig config;
  config.num_layers = params.num_layers();
  config.output_dim = params.output_dim();
  config.initial_layer_norm = params.initial_layer_norm();
  return config;
}

ProjectionConfig CreateProjectionConfig(const ProjectionParameters& params) {
  ProjectionConfig config;
  config.output_dim = params.output_dim();
  return config;
}

AudioEncoderConfig CreateAudioEncoderConfig(
    const AudioEncoderParameters& params) {
  AudioEncoderConfig config;
  config.feature = CreateSubSampleConvProjectionConfig(
      params.sub_sample_conv_projection_parameters());
  config.transformer = CreateTransformerConfig(params.transformer_parameters());
  if (params.has_mlp_adapter_parameters()) {
    config.adapter = CreateMLPAdapterConfig(params.mlp_adapter_parameters());
  } else if (params.has_projection_parameters()) {
    config.projection = CreateProjectionConfig(params.projection_parameters());
  }
  if (params.has_usm_projection_parameters()) {
    config.usm_projection =
        CreateProjectionConfig(params.usm_projection_parameters());
  }
  // TODO: b/349164917 - Consider moving vocab_size to TransformerParameters.
  config.transformer.vocabulary_size = params.vocab_size();
  return config;
}

}  // namespace

TransformerConfig CreateTransformerConfig(const TransformerParameters& params) {
  TransformerConfig config;

  config.batch_size = params.batch_size();
  config.sequence_size = params.max_seq_length();
  config.model_dimension = params.embedding_dim();
  config.hidden_dimension = params.hidden_dimension();
  config.head_dimension = params.head_dimension();
  config.number_of_heads = params.num_heads();
  if (params.num_kv_heads() > 0) {
    config.number_of_kv_heads = params.num_kv_heads();
  } else {
    // If num_kv_heads is not set (or equal to 0), use num_heads as the number
    // of kv heads (i.e. Multi-head Attention).
    config.number_of_kv_heads = params.num_heads();
  }
  config.stack_size = params.num_stacks();
  config.extra_stack_size = params.num_extra_stacks();

  if (params.global_head_dimension() > 0) {
    config.global_head_dimension = params.global_head_dimension();
  }
  if (params.kv_shared_hidden_dimension() > 0) {
    config.kv_shared_hidden_dimension = params.kv_shared_hidden_dimension();
  }
  if (params.global_num_kv_heads() > 0) {
    config.global_number_of_kv_heads = params.global_num_kv_heads();
  }
  config.global_k_eq_v = params.global_k_eq_v();

  // Override default CAUSAL attention mask type only if it is specified in the
  // params.
  if (params.self_attention_parameters().attention_mask_type() !=
      TransformerParameters::UNSPECIFIED) {
    config.attention_mask_type =
        params.self_attention_parameters().attention_mask_type();
  }

  config.cache_size = params.max_seq_length();

  config.post_norm_ff = params.feed_forward_parameters().post_norm() !=
                        TransformerParameters::NO_NORM;

  config.post_norm_attention =
      params.post_norm() != TransformerParameters::NO_NORM;

  if (params.self_attention_parameters().soft_cap_value() >= 0.0f) {
    config.attn_soft_cap_value =
        params.self_attention_parameters().soft_cap_value();
  }
  if (params.final_project_parameters().soft_cap_value() >= 0.0f) {
    config.final_proj_soft_cap_value =
        params.final_project_parameters().soft_cap_value();
  }
  config.skip_absolute_positional_embeddings =
      params.skip_absolute_positional_embeddings();

  if (params.self_attention_parameters().has_attention_scale_type()) {
    config.attention_scale_type =
        params.self_attention_parameters().attention_scale_type();
    // query_rescale_factor seems to be set for entire transformer stack, but we
    // only use and apply it for self-attention layer when a scale type is used
    // which requires such a factor.
    if (config.attention_scale_type ==
        odml::infra::proto::TransformerParameters::
            SCALE_TYPE_RESCALE_FACTOR_INV_HEAD_DIM) {
      config.query_rescale_factor = params.has_query_rescale_factor()
                                        ? params.query_rescale_factor()
                                        : 32.0f;
    }
  } else {
    if (params.num_kv_heads() == 0 ||
        params.num_heads() == params.num_kv_heads()) {
      // If MHA, PER_DIM_SCALE is used.
      config.attention_scale_type =
          odml::infra::proto::TransformerParameters::SCALE_TYPE_PER_DIM_SCALE;
    } else {
      // If MQA or GQA, INV_SQRT_HEAD_DIM is used.
      config.attention_scale_type = odml::infra::proto::TransformerParameters::
          SCALE_TYPE_INV_SQRT_HEAD_DIM;
    }
  }

  if (params.self_attention_parameters().has_qk_norm()) {
    config.qk_norm = params.self_attention_parameters().qk_norm();
    if (params.self_attention_parameters().has_qk_norm_has_scale()) {
      config.qk_norm_has_scale =
          params.self_attention_parameters().qk_norm_has_scale();
    }
  }
  if (params.self_attention_parameters().has_vision_bidirectional_attention()) {
    config.vision_bidirectional_attention =
        params.self_attention_parameters().vision_bidirectional_attention();
  }

  config.rms_norm_scale_shift = params.rms_norm_scale_shift();

  // GeminiV2 parameters
  config.num_local_layers_per_global = params.num_local_layers_per_global();
  if (params.self_attention_parameters().has_sliding_window_size()) {
    config.sliding_window_size =
        params.self_attention_parameters().sliding_window_size();
  }
  if (params.has_ngram_vocabulary_size()) {
    config.ngram_vocabulary_size = params.ngram_vocabulary_size();
    config.load_ngrammer_embedding_dynamically =
        params.load_ngrammer_embedding_dynamically();
  }
  if (params.has_per_layer_embedding_dimension()) {
    config.per_layer_embedding_dimension =
        params.per_layer_embedding_dimension();
    config.load_per_layer_embedding_dynamically =
        params.load_per_layer_embedding_dynamically();
    if (params.has_pq_ple_with_layers()) {
      config.pq_ple_with_layers = params.pq_ple_with_layers();
    }
  }
  config.vision_tokens_num = params.vision_tokens_num();
  config.max_num_images = params.max_num_images();
  config.audio_vocab_offset = params.audio_vocab_offset();
  config.audio_vocab_extra_dim = params.audio_vocab_extra_dim();
  config.audio_input_embedding_dim = params.audio_input_embedding_dim();
  if (params.has_audio_dual_norm_soft_and_hard_tokens()) {
    config.audio_dual_norm_soft_and_hard_tokens =
        params.audio_dual_norm_soft_and_hard_tokens();
  }

  // Global RoPE parameters
  if (params.has_global_rope_wavelength()) {
    config.global_rope_wavelength = params.global_rope_wavelength();
  }
  if (params.has_global_rope_scaling()) {
    config.global_rope_scaling = params.global_rope_scaling();
  }
  if (params.has_global_rope_proportion()) {
    config.global_rope_proportion = params.global_rope_proportion();
  }

  // Media params
  if (params.has_residual_adapter_parameters()) {
    config.residual_adapter = params.residual_adapter_parameters();
  }

  // Laurel configs.
  if (params.has_laurel_config()) {
    config.laurel_rank = params.laurel_config().laurel_rank();
    config.post_laurel_norm_scale_shift =
        params.laurel_config().post_laurel_norm_scale_shift();
  }

  // Altup configs.
  if (params.has_altup_config()) {
    config.altup_config = {
        .altup_num_modalities = params.altup_config().altup_num_modalities(),
        .altup_num_inputs = params.altup_config().altup_num_inputs(),
        .altup_active_idx = params.altup_config().altup_active_idx(),
    };
  }

  // MoE configs.
  if (params.has_moe_config()) {
    config.moe_config = {
        .hidden_dimension = params.moe_config().hidden_dimension(),
        .num_experts = params.moe_config().num_experts(),
        .num_active_experts = params.moe_config().num_active_experts(),
    };
  }

  // Centroid embedder configs.
  if (params.has_centroid_embedder_config()) {
    config.centroid_embedder_config = {
        .num_centroids = params.centroid_embedder_config().num_centroids(),
        .top_k = params.centroid_embedder_config().top_k(),
    };
  }

  // MTP configs
  if (params.post_proj_size() > 0) {
    config.post_proj_size = params.post_proj_size();
  }
  if (params.has_combine_activations_with_input()) {
    config.combine_activations_with_input =
        params.combine_activations_with_input();
  }

  // KV sharing.
  if (params.kv_cache_shared_start_layer() > 0) {
    config.kv_cache_shared_start_layer = params.kv_cache_shared_start_layer();
  }

  if (params.has_gemma3_bfloat16_fix()) {
    config.gemma3_bfloat16_fix = params.gemma3_bfloat16_fix();
  }

  // Activation sparsity.
  if (params.has_activation_sparsity_stddev_multiplier()) {
    config.activation_sparsity_stddev_multiplier =
        params.activation_sparsity_stddev_multiplier();
  }

  // Whether to apply SRQ.
  if (params.has_apply_srq()) {
    config.apply_srq = params.apply_srq();
  }

  // When we have a local/global pattern and want to offset it from (L*G)*
  if (params.has_global_layer_offset()) {
    config.global_layer_offset = params.global_layer_offset();
  }

  if (params.has_relative_positional_embedding_parameters()) {
    config.relative_positional_embedding_config = {
        .min_timescale =
            params.relative_positional_embedding_parameters().min_timescale(),
        .max_timescale =
            params.relative_positional_embedding_parameters().max_timescale(),
        .block_size =
            params.relative_positional_embedding_parameters().block_size(),
        .past_horizon =
            params.relative_positional_embedding_parameters().past_horizon(),
        .future_horizon =
            params.relative_positional_embedding_parameters().future_horizon(),
    };
  }
  config.light_conv_kernel_size =
      params.light_conv_parameters().conv_kernel_size();

  if (params.has_scale_lora()) {
    config.scale_lora = params.scale_lora();
  }

  if (params.has_scale_layer()) {
    config.scale_layer = params.scale_layer();
  }

  if (params.has_use_ringbuffers()) {
    config.use_autosized_ringbuffers = params.use_ringbuffers();
  }

  config.layer_skip_start = params.layer_skip_start();
  config.num_layers_skipped = params.num_layers_skipped();
  config.matform_factor = params.matform_factor();
  return config;
}

absl::StatusOr<LlmConfig> CreateGpuConfigFromLlmParametersProto(
    const LlmParameters& llm_params, bool use_submodel) {
  LlmConfig llm_config;
  ABSL_CHECK(llm_params.has_transformer_parameters() ||
             llm_params.has_audio_encoder_parameters());
  if (llm_params.has_transformer_parameters()) {
    TransformerParameters base_params = llm_params.transformer_parameters();
    if (use_submodel) {
      base_params.MergeFrom(llm_params.submodel_transformer_parameters());
    }
    llm_config.text_decoder = CreateTransformerConfig(base_params);
    llm_config.text_decoder->vocabulary_size = llm_params.vocab_size();
  }
  if (llm_params.has_audio_encoder_parameters()) {
    llm_config.audio_encoder =
        CreateAudioEncoderConfig(llm_params.audio_encoder_parameters());
    if (!(llm_params.audio_encoder_parameters().has_projection_parameters() ^
          llm_params.audio_encoder_parameters().has_mlp_adapter_parameters())) {
      return absl::InvalidArgumentError(
          "Audio encoder must have projection params or MLP adapter params "
          "defined, and not both.");
    }
  }
  llm_config.suppressed_tokens.reserve(llm_params.suppressed_tokens_size());
  for (int token : llm_params.suppressed_tokens()) {
    llm_config.suppressed_tokens.push_back(token);
  }
  return llm_config;
}

absl::Status UpdateGpuConfigForAudio(LlmConfig& llm_config, int max_seq_length,
                                     int max_audio_seq_length,
                                     bool is_streaming, int tokens_per_frame) {
  // Ensure audio encoder is set.
  RET_CHECK(llm_config.audio_encoder.has_value());

  // Ensure valid size for max_audio_seq_length.
  const int take_every_n =
      llm_config.audio_encoder.value().feature.take_every_n;
  const int shrink_factor =
      llm_config.audio_encoder.value().feature.shrink_factor;
  RET_CHECK_EQ(max_audio_seq_length % (shrink_factor * take_every_n), 0);

  // Update audio encoder config.
  int enc_audio_tokens;
  if (is_streaming) {
    // Add tokens_per_frame to the audio encoder config.
    llm_config.audio_encoder.value().feature.tokens_per_frame =
        tokens_per_frame;

    // Need to calculate how much padding to add to each chunk.
    if (!llm_config.audio_encoder.value().feature.is_causal) {
      return absl::InvalidArgumentError(
          "Streaming is only supported for causal audio encoder.");
    }
    int pad_for_streaming = take_every_n;
    for (ml_drift::ConvConfig& conv :
         llm_config.audio_encoder.value().feature.convs) {
      pad_for_streaming =
          (pad_for_streaming * conv.stride.h) + (conv.kernel_size.h - 1);
    }
    pad_for_streaming -= take_every_n * shrink_factor;
    llm_config.audio_encoder.value().feature.pad_for_streaming =
        pad_for_streaming;
    // The audio encoder seq_size refers to the transformer layer. Each audio
    // output token is reduced by a factor of take_every_n from the seq_size of
    // the audio encoder transformer layer.
    enc_audio_tokens = take_every_n * tokens_per_frame;
  } else {
    enc_audio_tokens = max_audio_seq_length / shrink_factor;
  }
  llm_config.audio_encoder.value().transformer.cache_size =
      max_audio_seq_length / shrink_factor;
  llm_config.audio_encoder.value().transformer.sequence_size = enc_audio_tokens;

  // Update text decoder config (if exists).
  if (llm_config.text_decoder.has_value()) {
    const int dec_audio_tokens =
        max_audio_seq_length / (shrink_factor * take_every_n);
    llm_config.text_decoder->cache_size =
        std::max(128, dec_audio_tokens + max_seq_length);
  }

  return absl::OkStatus();
}

}  // namespace odml::infra::gpu
