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

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/gpu_model_builder.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_tensor.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/ml_drift/audio_encoder.h"
#include "runtime/executor/gpu_artisan/ml_drift/byom/byom_factory.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm.h"
#include "runtime/executor/gpu_artisan/ml_drift/tasks/add_with_overflow.h"
#include "runtime/executor/gpu_artisan/ml_drift/tasks/apply_attention_mask.h"
#include "runtime/executor/gpu_artisan/ml_drift/tasks/apply_fire_bias.h"
#include "runtime/executor/gpu_artisan/ml_drift/tasks/flash_decoding_attention.h"
#include "runtime/executor/gpu_artisan/ml_drift/tasks/fused_rope.h"
#include "runtime/executor/gpu_artisan/ml_drift/tasks/gelu7.h"
#include "runtime/executor/gpu_artisan/ml_drift/tasks/mul_with_overflow.h"
#include "runtime/executor/gpu_artisan/ml_drift/tasks/rotqat_rotate.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/basic_gpu_tensor.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_dummy_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"
#include "runtime/executor/gpu_artisan/proto/llm_inference_engine.pb.h"
#include "runtime/executor/gpu_artisan/proto/transformer_params.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace ml_drift {
namespace {

using ::odml::infra::gpu::BasicGpuSpatialTensor;
using ::odml::infra::gpu::LlmDummyTensorLoader;
using ::odml::infra::proto::LlmModelType;
using ::testing::status::StatusIs;

TensorDescriptor MakeDesc(DataType dt, BHWC shape,
                          Layout layout = Layout::kBHWC) {
  TensorDescriptor desc(dt, TensorStorageType::kBuffer, layout);
  desc.SetBHWCShape(shape);
  return desc;
}

TransformerConfig MakeTinyTransformerConfig() {
  TransformerConfig cfg{};
  cfg.batch_size = 1;
  cfg.sequence_size = 1;
  cfg.model_dimension = 8;
  cfg.hidden_dimension = 16;
  cfg.head_dimension = 8;
  cfg.number_of_heads = 1;
  cfg.number_of_kv_heads = 1;
  cfg.vocabulary_size = 16;
  cfg.stack_size = 1;
  cfg.max_top_k = 1;
  cfg.cache_size = 8;
  return cfg;
}

bool HasGeneratedCode(const GpuInfo& gpu_info, GPUOperation* op) {
  if (op->code_.empty()) {
    if (!op->AssembleCode(gpu_info).ok()) return false;
  }
  return !op->code_.empty();
}

GpuInfo MakeWebGpuInfo() {
  GpuInfo gpu_info;
  gpu_info.gpu_api = GpuApi::kWebGpu;
  return gpu_info;
}

TEST(TasksAndByomTest, CustomTaskFactoriesConstructGpuOperations) {
  GpuInfo gpu_info = MakeWebGpuInfo();
  TensorDescriptor f16_desc = MakeDesc(DataType::kFloat16, BHWC(1, 1, 2, 8));
  TensorDescriptor f32_desc = MakeDesc(DataType::kFloat32, BHWC(1, 1, 2, 8));
  TensorDescriptor i32_desc = MakeDesc(DataType::kInt32, BHWC(1, 1, 1, 8));
  TensorDescriptor mask_desc = MakeDesc(DataType::kBool, BHWC(1, 1, 2, 8));

  GPUOperation add_op = CreateAddWithOverflow(f16_desc, /*value=*/1);
  EXPECT_TRUE(HasGeneratedCode(gpu_info, &add_op));
  GPUOperation add2_op = CreateAddWithOverflow2Input(f16_desc, f16_desc);
  EXPECT_TRUE(HasGeneratedCode(gpu_info, &add2_op));

  GPUOperation mul_op = CreateMulWithOverflow(f16_desc, /*value=*/2);
  EXPECT_TRUE(HasGeneratedCode(gpu_info, &mul_op));

  GPUOperation mask_op =
      CreateApplyAttentionMaskOp(i32_desc, mask_desc, /*sliding_window=*/4);
  EXPECT_TRUE(HasGeneratedCode(gpu_info, &mask_op));

  GPUOperation fire_op =
      CreateApplyFireBiasOp(gpu_info, i32_desc, f32_desc,
                            /*log_scale_factor=*/1.0f, /*threshold=*/16.0f);
  EXPECT_TRUE(HasGeneratedCode(gpu_info, &fire_op));

  const std::array<float, 7> anchors = {-2.0f, -1.0f, -0.5f, 0.0f,
                                        0.5f,  1.0f,  2.0f};
  const std::array<float, 7> slopes = {0.1f, 0.1f, 0.2f, 0.2f,
                                       0.2f, 0.1f, 0.1f};
  GPUOperation gelu7_op = CreateGELU7Op(f32_desc, anchors, slopes);
  EXPECT_TRUE(HasGeneratedCode(gpu_info, &gelu7_op));

  GPUOperation rope_op = CreateTransformQKVApplyRoPEOp(
      gpu_info, f32_desc, f32_desc, f32_desc, i32_desc, f32_desc, f32_desc,
      f32_desc, /*kv_heads_count=*/1, /*kv_head_group_size=*/1,
      /*sequence_size=*/2, /*min_timescale=*/1.0f, /*max_timescale=*/10000.0f);
  EXPECT_FALSE(rope_op.code_.empty());

  TransformQKVRmsNormApplyRoPEOp rms_rope_op =
      CreateTransformQKVRmsNormApplyRoPEOp(
          gpu_info, f32_desc, i32_desc, f32_desc, /*kv_heads_count=*/1,
          /*kv_head_group_size=*/1, /*sequence_size=*/2,
          /*min_timescale=*/1.0f, /*max_timescale=*/10000.0f,
          /*proportion=*/1.0f, /*k_eq_v=*/false, &f32_desc, &f32_desc,
          &f32_desc, &f32_desc, /*q_gamma=*/nullptr, /*k_gamma=*/nullptr);
  EXPECT_FALSE(rms_rope_op.code_.empty());

  TensorDescriptor rot_desc = MakeDesc(DataType::kFloat32, BHWC(1, 1, 1, 112));
  ASSERT_OK_AND_ASSIGN(
      RotqatRotateOp rotqat_op,
      CreateRotqatRotateOp(gpu_info, rot_desc, rot_desc, /*use_o7_h4=*/true,
                           /*hadamard_dim=*/std::nullopt));
  EXPECT_FALSE(rotqat_op.code_.empty());

  FlashDecodingAttentionAttributes flash_attr{
      .num_q_heads = 4,
      .num_kv_heads = 1,
      .head_size = 64,
      .cache_size = 64,
  };
  std::string flash_err;
  (void)IsFlashDecodingAttentionSupported(gpu_info, flash_attr, &flash_err);
  TensorDescriptor q_desc =
      MakeDesc(DataType::kFloat16, BHWC(1, 1, 4, 64), Layout::kHWC);
  TensorDescriptor kv_desc(DataType::kFloat16, TensorStorageType::kBuffer,
                           Layout::kLinear);
  kv_desc.SetBHWCShape(BHWC(1, 1, 1, 64 * 64));
  EXPECT_OK(ValidateFlashDecodingAttentionTensors(q_desc, kv_desc, kv_desc,
                                                  q_desc, flash_attr));
}

TEST(TasksAndByomTest, LlmHelperOperationsConstructValidShaders) {
  GpuInfo gpu_info;
  TensorDescriptor f32_desc = MakeDesc(DataType::kFloat32, BHWC(1, 1, 2, 8));
  TensorDescriptor i32_desc = MakeDesc(DataType::kInt32, BHWC(1, 1, 2, 4));
  TensorDescriptor mask_desc = MakeDesc(DataType::kBool, BHWC(1, 1, 2, 8));

  EXPECT_FALSE(CreateWriteParamsOp(i32_desc, /*params_count=*/5).code_.empty());
  EXPECT_FALSE(
      CreateCopyNthToSingleOp(gpu_info, f32_desc, f32_desc).code_.empty());
  EXPECT_FALSE(CreateCopyTokensOrPositionsOp(i32_desc).code_.empty());
  EXPECT_FALSE(CreateCopyExternalEmbeddingsOp(f32_desc).code_.empty());
  EXPECT_FALSE(CreateCopyMediaEmbeddingsOp(f32_desc, f32_desc).code_.empty());
  EXPECT_FALSE(
      CreateCopyTokensAndPositionsOp(i32_desc, i32_desc).code_.empty());
  EXPECT_FALSE(
      CreateCopyMasksOp(mask_desc, mask_desc, /*mask_width=*/8).code_.empty());
}

TEST(TasksAndByomTest, ByomBuildersConstructStackedTransformerGraphs) {
  GpuInfo gpu_info = MakeWebGpuInfo();
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kBuffer;
  std::vector<std::unique_ptr<BasicGpuSpatialTensor>> created;
  LlmDummyTensorLoader loader(LlmDummyTensorLoader::WeightsType::kFloat32);
  loader.SetDataType(DataType::kFloat32);
  loader.SetCreateTensorFn([&created](const TensorDescriptor& desc)
                               -> absl::StatusOr<GpuSpatialTensor*> {
    created.push_back(std::make_unique<BasicGpuSpatialTensor>(desc));
    return created.back().get();
  });

  TransformerConfig bad_cfg = MakeTinyTransformerConfig();
  bad_cfg.number_of_kv_heads = 0;
  EXPECT_THAT(odml_byom::CreateTransformerBuilder(
                  bad_cfg, LlmModelType::LLM_MODEL_TYPE_UNKNOWN,
                  /*kv_cache_batch_size=*/1, gpu_info, create_info, &loader),
              StatusIs(absl::StatusCode::kInvalidArgument));

  for (LlmModelType model_type : {LlmModelType::LLM_MODEL_TYPE_UNKNOWN,
                                  LlmModelType::LLM_MODEL_TYPE_FALCON_RW_1B,
                                  LlmModelType::LLM_MODEL_TYPE_STABLELM_4E1T_3B,
                                  LlmModelType::LLM_MODEL_TYPE_PHI_2}) {
    TransformerConfig cfg = MakeTinyTransformerConfig();
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<TransformerBuilder> builder,
        odml_byom::CreateTransformerBuilder(cfg, model_type,
                                            /*kv_cache_batch_size=*/1, gpu_info,
                                            create_info, &loader));
    GpuModelBuilder::TensorHandle input_h, media_h, params_h, output_h;
    std::vector<GpuModelBuilder::TensorHandle> k_cache, v_cache;
    ASSERT_OK_AND_ASSIGN(
        GpuModel model,
        builder->CreateStackedTransformerModel(
            /*sequence_size=*/1, /*batch_size=*/1, &input_h, &media_h,
            &params_h, &k_cache, &v_cache, &output_h));
    EXPECT_FALSE(model.nodes.empty());
  }
}

TEST(TasksAndByomTest, AudioEncoderBuilderConstructsGraphWithDummyWeights) {
  GpuInfo gpu_info = MakeWebGpuInfo();
  CreateGpuModelInfo create_info;
  create_info.precision = CalculationsPrecision::kF32;
  create_info.storage_type = TensorStorageType::kBuffer;
  std::vector<std::unique_ptr<BasicGpuSpatialTensor>> created;
  LlmDummyTensorLoader loader(LlmDummyTensorLoader::WeightsType::kFloat32);
  loader.SetDataType(DataType::kFloat32);
  loader.SetCreateTensorFn([&created](const TensorDescriptor& desc)
                               -> absl::StatusOr<GpuSpatialTensor*> {
    created.push_back(std::make_unique<BasicGpuSpatialTensor>(desc));
    return created.back().get();
  });

  AudioEncoderConfig audio_cfg{};
  audio_cfg.feature.input_dims = 8;
  audio_cfg.feature.model_dims = 8;
  audio_cfg.feature.is_causal = true;
  audio_cfg.feature.proj_bias = false;
  audio_cfg.feature.norm_bias = false;
  audio_cfg.feature.shrink_factor = 2;
  audio_cfg.feature.take_every_n = 1;
  audio_cfg.feature.convs.push_back(ConvConfig{
      .channel_size = 4,
      .kernel_size = HW(3, 3),
      .stride = HW(2, 2),
      .padding =
          HWPadding{SubSamplePaddingType::CAUSAL, SubSamplePaddingType::SAME},
      .bias = false,
  });
  audio_cfg.transformer = MakeTinyTransformerConfig();
  audio_cfg.transformer.sequence_size = 4;
  audio_cfg.transformer.cache_size = 4;
  audio_cfg.adapter.num_layers = 1;
  audio_cfg.adapter.output_dim = 8;
  audio_cfg.adapter.initial_layer_norm = true;

  AudioEncoderBuilder builder(audio_cfg, /*kv_cache_batch_size=*/1, gpu_info,
                              create_info, &loader);
  GpuModelBuilder::TensorHandle feat_h, mask_h, params_h, cumsum_h, signal_h,
      out_h;
  std::vector<GpuModelBuilder::TensorHandle> k_cache, v_cache, dw_cache;
  ASSERT_OK_AND_ASSIGN(GpuModel model,
                       builder.CreateAudioEncoderModel(
                           &feat_h, &mask_h, &params_h, &k_cache, &v_cache,
                           &cumsum_h, &dw_cache, &signal_h, &out_h));
  EXPECT_FALSE(model.nodes.empty());
}

}  // namespace
}  // namespace ml_drift
