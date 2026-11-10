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

#include "runtime/executor/gpu_artisan/ml_drift/gpu_model_info_generator.h"

#include <gtest/gtest.h>
#include "ml_drift/common/model_hints.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/proto/llm_inference_engine.pb.h"

namespace odml::infra::gpu {
namespace {

TEST(GpuModelInfoGeneratorTest, GenerateWithFlags) {
  // Test for F16 precision.
  ml_drift::CreateGpuModelInfo create_info = GenerateGpuModelInfo(
      /*allow_precision_loss=*/true, /*use_low_power=*/true,
      /*prefer_texture_weights=*/true, /*enable_host_mapped_pointer=*/true,
      /*enable_fast_tuning=*/true, /*enable_winograd_opt=*/true,
      /*disallow_8bit_convs=*/true);

  EXPECT_EQ(create_info.precision, ml_drift::CalculationsPrecision::kF16);
  EXPECT_TRUE(create_info.hints.Check(ml_drift::ModelHints::kLowPower));
  EXPECT_TRUE(
      create_info.hints.Check(ml_drift::ModelHints::kPreferTextureWeights));
  EXPECT_TRUE(
      create_info.hints.Check(ml_drift::ModelHints::kEnableHostMappedPointer));
  EXPECT_TRUE(create_info.hints.Check(ml_drift::ModelHints::kFastTuning));
  EXPECT_FALSE(
      create_info.hints.Check(ml_drift::ModelHints::kNoWinogradOptimizations));
  EXPECT_TRUE(
      create_info.hints.Check(ml_drift::ModelHints::kDisallow8bitConvs));

  // Test for F32 precision and disabled winograd.
  create_info = GenerateGpuModelInfo(
      /*allow_precision_loss=*/false, /*use_low_power=*/false,
      /*prefer_texture_weights=*/false, /*enable_host_mapped_pointer=*/false,
      /*enable_fast_tuning=*/false, /*enable_winograd_opt=*/false,
      /*disallow_8bit_convs=*/false);

  EXPECT_EQ(create_info.precision, ml_drift::CalculationsPrecision::kF32);
  EXPECT_TRUE(
      create_info.hints.Check(ml_drift::ModelHints::kNoWinogradOptimizations));
}

TEST(GpuModelInfoGeneratorTest, GenerateWithModelSettings) {
  proto::SessionConfig::MlDriftOptions mldrift_options;
  mldrift_options.set_use_low_power(true);
  mldrift_options.set_prefer_texture_weights(true);
  mldrift_options.set_enable_host_mapped_pointer(true);
  mldrift_options.set_enable_fast_tuning(true);
  mldrift_options.set_enable_winograd_opt(true);

  // Test for F16 precision.
  auto create_info = GenerateGpuModelInfo(
      mldrift_options, proto::SessionConfig::ACTIVATION_DATA_TYPE_F16);

  EXPECT_EQ(create_info.precision, ml_drift::CalculationsPrecision::kF16);
  EXPECT_TRUE(create_info.hints.Check(ml_drift::ModelHints::kLowPower));
  EXPECT_TRUE(
      create_info.hints.Check(ml_drift::ModelHints::kPreferTextureWeights));
  EXPECT_TRUE(
      create_info.hints.Check(ml_drift::ModelHints::kEnableHostMappedPointer));
  EXPECT_TRUE(create_info.hints.Check(ml_drift::ModelHints::kFastTuning));
  EXPECT_FALSE(
      create_info.hints.Check(ml_drift::ModelHints::kNoWinogradOptimizations));
  EXPECT_FALSE(
      create_info.hints.Check(ml_drift::ModelHints::kDisallow8bitConvs));

  // Test for F32 precision.
  create_info = GenerateGpuModelInfo(
      mldrift_options, proto::SessionConfig::ACTIVATION_DATA_TYPE_F32);

  EXPECT_EQ(create_info.precision, ml_drift::CalculationsPrecision::kF32);
}

}  // namespace
}  // namespace odml::infra::gpu
