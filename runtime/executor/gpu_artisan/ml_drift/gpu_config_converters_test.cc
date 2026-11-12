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

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/executor/gpu_artisan/ml_drift/audio_encoder.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"
#include "runtime/executor/gpu_artisan/proto/audio_encoder_params.pb.h"
#include "runtime/executor/gpu_artisan/proto/llm_params.pb.h"
#include "runtime/executor/gpu_artisan/proto/transformer_params.pb.h"
#include "runtime/executor/gpu_artisan/proto/vision_encoder_params.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace odml::infra::gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::status::StatusIs;

proto::AudioEncoderParameters MakeSampleAudioEncoderParameters() {
  proto::AudioEncoderParameters audio_params;
  audio_params.set_pad_zero_frame_to_audio(true);
  audio_params.set_vocab_size(32768);
  auto* mlp = audio_params.mutable_mlp_adapter_parameters();
  mlp->set_num_layers(2);
  mlp->set_initial_layer_norm(true);
  mlp->set_output_dim(345);

  auto* sub_sample =
      audio_params.mutable_sub_sample_conv_projection_parameters();
  sub_sample->set_input_dims(123);
  sub_sample->set_model_dims(234);
  sub_sample->set_norm_bias(true);
  sub_sample->set_is_causal(true);
  sub_sample->set_proj_bias(true);
  sub_sample->set_take_every_n(12);
  sub_sample->set_shrink_factor(99);
  auto* conv = sub_sample->add_convs();
  conv->set_channel_size(456);
  conv->mutable_kernel_size()->set_h(3);
  conv->mutable_kernel_size()->set_w(3);
  conv->mutable_stride()->set_h(2);
  conv->mutable_stride()->set_w(2);
  conv->mutable_padding()->set_h(proto::AudioEncoderParameters::SAME);
  conv->mutable_padding()->set_w(proto::AudioEncoderParameters::REVERSE_CAUSAL);
  conv->set_bias(false);

  auto* tx = audio_params.mutable_transformer_parameters();
  tx->set_num_local_layers_per_global(6);
  tx->mutable_self_attention_parameters()->set_attention_mask_type(
      proto::TransformerParameters::CAUSAL);
  tx->mutable_self_attention_parameters()->set_attention_scale_type(
      proto::TransformerParameters::SCALE_TYPE_INV_SQRT_D_MODEL_DIV_NUM_HEADS);
  tx->mutable_self_attention_parameters()->set_soft_cap_value(0.123f);
  tx->mutable_final_project_parameters()->set_soft_cap_value(0.456f);
  tx->mutable_feed_forward_parameters()->set_post_norm(
      proto::TransformerParameters::RMS_NORM);
  tx->set_skip_absolute_positional_embeddings(false);
  tx->set_num_kv_heads(1);
  tx->set_post_norm(proto::TransformerParameters::RMS_NORM);
  tx->mutable_residual_adapter_parameters()->set_bottleneck_dimension(1536);
  tx->mutable_residual_adapter_parameters()->set_where_to_interleave(
      proto::TransformerParameters::ALL);
  return audio_params;
}

TEST(GpuConfigConverterTest, CreateGpuConfigFromLlmParametersProto) {
  auto llm_params = proto::LlmParameters();
  llm_params.set_vocab_size(123);
  llm_params.mutable_transformer_parameters()->set_batch_size(456);
  llm_params.mutable_transformer_parameters()->set_max_seq_length(789);
  llm_params.mutable_transformer_parameters()->set_embedding_dim(234);
  llm_params.mutable_transformer_parameters()->set_hidden_dimension(567);
  llm_params.mutable_transformer_parameters()->set_head_dimension(891);
  llm_params.mutable_transformer_parameters()->set_num_heads(345);
  llm_params.mutable_transformer_parameters()->set_num_stacks(678);
  llm_params.mutable_transformer_parameters()
      ->mutable_self_attention_parameters()
      ->set_soft_cap_value(0.123f);
  llm_params.mutable_transformer_parameters()
      ->mutable_final_project_parameters()
      ->set_soft_cap_value(0.456f);

  llm_params.mutable_transformer_parameters()->set_num_kv_heads(912);
  llm_params.mutable_transformer_parameters()
      ->mutable_self_attention_parameters()
      ->set_attention_mask_type(proto::TransformerParameters::PREFIX);
  llm_params.mutable_transformer_parameters()
      ->mutable_feed_forward_parameters()
      ->set_post_norm(proto::TransformerParameters::NO_NORM);
  llm_params.mutable_transformer_parameters()
      ->set_skip_absolute_positional_embeddings(true);
  llm_params.mutable_transformer_parameters()
      ->mutable_self_attention_parameters()
      ->set_attention_scale_type(
          proto::TransformerParameters::SCALE_TYPE_PER_DIM_SCALE);
  llm_params.mutable_transformer_parameters()->set_num_local_layers_per_global(
      6);
  llm_params.mutable_transformer_parameters()
      ->mutable_residual_adapter_parameters()
      ->set_bottleneck_dimension(1536 * 2);
  llm_params.mutable_transformer_parameters()
      ->mutable_residual_adapter_parameters()
      ->set_where_to_interleave(proto::TransformerParameters::EVERY_OTHER_4);
  llm_params.add_suppressed_tokens(258882);
  llm_params.add_suppressed_tokens(258883);

  ASSERT_OK_AND_ASSIGN(auto gpu_config,
                       CreateGpuConfigFromLlmParametersProto(llm_params));
  ASSERT_TRUE(gpu_config.text_decoder.has_value());
  EXPECT_EQ(gpu_config.text_decoder->vocabulary_size, 123);
  EXPECT_EQ(gpu_config.text_decoder->batch_size, 456);
  EXPECT_EQ(gpu_config.text_decoder->sequence_size, 789);
  EXPECT_EQ(gpu_config.text_decoder->model_dimension, 234);
  EXPECT_EQ(gpu_config.text_decoder->hidden_dimension, 567);
  EXPECT_EQ(gpu_config.text_decoder->head_dimension, 891);
  EXPECT_EQ(gpu_config.text_decoder->number_of_heads, 345);
  EXPECT_EQ(gpu_config.text_decoder->stack_size, 678);
  EXPECT_EQ(gpu_config.text_decoder->cache_size, 789);
  EXPECT_FLOAT_EQ(gpu_config.text_decoder->attn_soft_cap_value, 0.123f);
  EXPECT_FLOAT_EQ(gpu_config.text_decoder->final_proj_soft_cap_value, 0.456f);
  EXPECT_EQ(gpu_config.text_decoder->number_of_kv_heads, 912);
  EXPECT_EQ(gpu_config.text_decoder->attention_mask_type,
            proto::TransformerParameters::PREFIX);
  EXPECT_FALSE(gpu_config.text_decoder->post_norm_ff);
  EXPECT_TRUE(gpu_config.text_decoder->skip_absolute_positional_embeddings);
  EXPECT_EQ(gpu_config.text_decoder->attention_scale_type,
            proto::TransformerParameters::SCALE_TYPE_PER_DIM_SCALE);
  EXPECT_EQ(gpu_config.text_decoder->residual_adapter->bottleneck_dimension(),
            1536 * 2);
  EXPECT_EQ(gpu_config.text_decoder->residual_adapter->where_to_interleave(),
            odml::infra::proto::TransformerParameters::EVERY_OTHER_4);
  EXPECT_EQ(gpu_config.text_decoder->sliding_window_size, std::nullopt);
  EXPECT_EQ(gpu_config.text_decoder->ngram_vocabulary_size, std::nullopt);
  EXPECT_EQ(gpu_config.text_decoder->per_layer_embedding_dimension,
            std::nullopt);
  EXPECT_EQ(gpu_config.text_decoder->num_local_layers_per_global, 6);
  EXPECT_THAT(gpu_config.suppressed_tokens, ElementsAre(258882, 258883));

  // Submodel override and audio encoder parameters.
  llm_params.mutable_submodel_transformer_parameters()->set_num_stacks(12);
  llm_params.mutable_transformer_parameters()->set_num_kv_heads(-1);
  llm_params.mutable_transformer_parameters()
      ->mutable_self_attention_parameters()
      ->set_attention_mask_type(proto::TransformerParameters::CAUSAL);
  llm_params.mutable_transformer_parameters()
      ->mutable_feed_forward_parameters()
      ->set_post_norm(proto::TransformerParameters::RMS_NORM);
  llm_params.mutable_transformer_parameters()
      ->set_skip_absolute_positional_embeddings(false);
  llm_params.mutable_transformer_parameters()
      ->mutable_self_attention_parameters()
      ->set_attention_scale_type(proto::TransformerParameters::
                                     SCALE_TYPE_INV_SQRT_D_MODEL_DIV_NUM_HEADS);
  llm_params.mutable_transformer_parameters()
      ->mutable_self_attention_parameters()
      ->set_sliding_window_size(512);
  llm_params.mutable_transformer_parameters()->set_ngram_vocabulary_size(256);
  llm_params.mutable_transformer_parameters()
      ->set_per_layer_embedding_dimension(128);
  llm_params.mutable_vision_encoder_parameters()->set_tokens_num(256);
  llm_params.mutable_vision_encoder_parameters()->set_append_eoi_embedding(
      true);
  llm_params.mutable_transformer_parameters()->set_vision_tokens_num(257);
  llm_params.mutable_transformer_parameters()->set_audio_vocab_offset(256128 *
                                                                      2);
  llm_params.mutable_transformer_parameters()->set_audio_vocab_extra_dim(128);
  llm_params.mutable_transformer_parameters()->set_audio_input_embedding_dim(
      1536);
  llm_params.mutable_transformer_parameters()
      ->set_kv_cache_shared_start_layer(20);
  *llm_params.mutable_audio_encoder_parameters() =
      MakeSampleAudioEncoderParameters();

  ASSERT_OK_AND_ASSIGN(gpu_config, CreateGpuConfigFromLlmParametersProto(
                                       llm_params, /*use_submodel=*/true));
  ASSERT_TRUE(gpu_config.text_decoder.has_value());
  EXPECT_EQ(gpu_config.text_decoder->stack_size, 12);
  EXPECT_EQ(gpu_config.text_decoder->number_of_kv_heads, 345);
  EXPECT_EQ(gpu_config.text_decoder->vision_tokens_num, 257);
  EXPECT_EQ(gpu_config.text_decoder->kv_cache_shared_start_layer, 20);

  ASSERT_TRUE(gpu_config.audio_encoder.has_value());
  EXPECT_EQ(gpu_config.audio_encoder->feature.input_dims, 123);
  EXPECT_EQ(gpu_config.audio_encoder->feature.model_dims, 234);
  ASSERT_EQ(gpu_config.audio_encoder->feature.convs.size(), 1u);
  EXPECT_EQ(gpu_config.audio_encoder->feature.convs[0].channel_size, 456);
  EXPECT_EQ(gpu_config.audio_encoder->feature.convs[0].padding.h,
            ml_drift::SubSamplePaddingType::SAME);
  EXPECT_EQ(gpu_config.audio_encoder->feature.convs[0].padding.w,
            ml_drift::SubSamplePaddingType::REVERSE_CAUSAL);
  EXPECT_EQ(gpu_config.audio_encoder->adapter.output_dim, 345);
}

TEST(GpuConfigConverterTest, UpdateGpuConfigForAudioStaticAndStreaming) {
  proto::LlmParameters llm_params;
  llm_params.set_vocab_size(321);
  llm_params.mutable_transformer_parameters()->set_max_seq_length(123);
  *llm_params.mutable_audio_encoder_parameters() =
      MakeSampleAudioEncoderParameters();

  ASSERT_OK_AND_ASSIGN(auto gpu_config,
                       CreateGpuConfigFromLlmParametersProto(llm_params));
  // shrink_factor = 99, take_every_n = 12 => product = 1188.
  ASSERT_OK(UpdateGpuConfigForAudio(gpu_config, /*max_seq_length=*/123,
                                    /*max_audio_seq_length=*/2376,
                                    /*is_streaming=*/false,
                                    /*tokens_per_frame=*/1));
  EXPECT_EQ(gpu_config.audio_encoder->transformer.sequence_size, 24);
  EXPECT_EQ(gpu_config.audio_encoder->transformer.cache_size, 24);
  EXPECT_EQ(gpu_config.text_decoder->cache_size, 128);

  ASSERT_OK(UpdateGpuConfigForAudio(gpu_config, /*max_seq_length=*/200,
                                    /*max_audio_seq_length=*/2376,
                                    /*is_streaming=*/true,
                                    /*tokens_per_frame=*/2));
  EXPECT_EQ(gpu_config.audio_encoder->feature.tokens_per_frame, 2);
  EXPECT_EQ(gpu_config.audio_encoder->transformer.sequence_size, 24);
  EXPECT_EQ(gpu_config.text_decoder->cache_size, 202);

  // Non-causal streaming must return InvalidArgumentError.
  gpu_config.audio_encoder->feature.is_causal = false;
  EXPECT_THAT(UpdateGpuConfigForAudio(gpu_config, /*max_seq_length=*/200,
                                      /*max_audio_seq_length=*/2376,
                                      /*is_streaming=*/true,
                                      /*tokens_per_frame=*/1),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Unaligned max_audio_seq_length must fail RET_CHECK_EQ.
  gpu_config.audio_encoder->feature.is_causal = true;
  EXPECT_THAT(UpdateGpuConfigForAudio(gpu_config, /*max_seq_length=*/200,
                                      /*max_audio_seq_length=*/100,
                                      /*is_streaming=*/false,
                                      /*tokens_per_frame=*/1),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(GpuConfigConverterTest, AudioEncoderAdapterOrProjectionValidation) {
  proto::LlmParameters llm_params;
  llm_params.mutable_transformer_parameters()->set_max_seq_length(123);
  *llm_params.mutable_audio_encoder_parameters() =
      MakeSampleAudioEncoderParameters();
  llm_params.mutable_audio_encoder_parameters()->clear_mlp_adapter_parameters();

  // Neither projection nor MLP adapter -> InvalidArgumentError.
  EXPECT_THAT(CreateGpuConfigFromLlmParametersProto(llm_params),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Projection only -> OK.
  llm_params.mutable_audio_encoder_parameters()
      ->mutable_projection_parameters()
      ->set_output_dim(128);
  llm_params.mutable_audio_encoder_parameters()
      ->mutable_usm_projection_parameters()
      ->set_output_dim(64);
  ASSERT_OK_AND_ASSIGN(auto cfg,
                       CreateGpuConfigFromLlmParametersProto(llm_params));
  EXPECT_EQ(ml_drift::GetOutputDim(*cfg.audio_encoder), 128);
  EXPECT_EQ(cfg.audio_encoder->usm_projection.output_dim, 64);

  // Both projection and MLP adapter -> InvalidArgumentError.
  llm_params.mutable_audio_encoder_parameters()
      ->mutable_mlp_adapter_parameters()
      ->set_output_dim(128);
  EXPECT_THAT(CreateGpuConfigFromLlmParametersProto(llm_params),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(GpuConfigConverterTest, EmptyLlmParametersReturnsInvalidArgument) {
  proto::LlmParameters empty_params;
  empty_params.set_vocab_size(32000);
  EXPECT_THAT(CreateGpuConfigFromLlmParametersProto(empty_params),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace odml::infra::gpu
