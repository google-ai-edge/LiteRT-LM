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

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "ml_drift/common/gpu_model.h"  // from @ml_drift
#include "ml_drift/common/model_hints.h"  // from @ml_drift
#include "ml_drift/common/precision.h"  // from @ml_drift
#include "runtime/executor/gpu_artisan/proto/llm_inference_engine.pb.h"

namespace odml::infra::gpu {

ml_drift::CreateGpuModelInfo GenerateGpuModelInfo(
    bool allow_precision_loss, bool use_low_power, bool prefer_texture_weights,
    bool enable_host_mapped_pointer, bool enable_fast_tuning,
    bool enable_winograd_opt, bool disallow_8bit_convs) {
  ml_drift::CreateGpuModelInfo create_info;
  if (allow_precision_loss) {
    ABSL_LOG(INFO) << "CalculationsPrecision::F16";
    create_info.precision = ml_drift::CalculationsPrecision::kF16;
  } else {
    ABSL_LOG(INFO) << "CalculationsPrecision::F32";
    create_info.precision = ml_drift::CalculationsPrecision::kF32;
  }
  if (use_low_power) {
    ABSL_LOG(INFO) << "Low Power";
    create_info.hints.Add(ml_drift::ModelHints::kLowPower);
  }
  if (prefer_texture_weights) {
    ABSL_LOG(INFO) << "Prefer Texture Weights";
    create_info.hints.Add(ml_drift::ModelHints::kPreferTextureWeights);
  }
  if (enable_host_mapped_pointer) {
    ABSL_LOG(INFO) << "Enable Host Mapped Pointer";
    create_info.hints.Add(ml_drift::ModelHints::kEnableHostMappedPointer);
  }
  if (enable_fast_tuning) {
    ABSL_LOG(INFO) << "Fast Tuning";
    create_info.hints.Add(ml_drift::ModelHints::kFastTuning);
  }
  if (!enable_winograd_opt) {
    ABSL_LOG(INFO) << "No Winograd";
    create_info.hints.Add(ml_drift::ModelHints::kNoWinogradOptimizations);
  }
  if (disallow_8bit_convs) {
    ABSL_LOG(INFO) << "Disallow 8bit convs";
    create_info.hints.Add(ml_drift::ModelHints::kDisallow8bitConvs);
  }

  return create_info;
}

ml_drift::CreateGpuModelInfo GenerateGpuModelInfo(
    const proto::SessionConfig::MlDriftOptions& mldrift_options,
    const proto::SessionConfig::ActivationDataType& activation_data_type) {
  bool allow_precision_loss =
      activation_data_type == proto::SessionConfig::ACTIVATION_DATA_TYPE_F16;
  bool use_low_power = mldrift_options.use_low_power();
  bool prefer_texture_weights = mldrift_options.prefer_texture_weights();
  bool enable_host_mapped_pointer =
      mldrift_options.enable_host_mapped_pointer();
  bool enable_fast_tuning = mldrift_options.enable_fast_tuning();
  bool enable_winograd_opt = mldrift_options.enable_winograd_opt();
  bool disallow_8bit_convs = mldrift_options.disallow_8bit_convs();
  return GenerateGpuModelInfo(
      allow_precision_loss, use_low_power, prefer_texture_weights,
      enable_host_mapped_pointer, enable_fast_tuning, enable_winograd_opt,
      disallow_8bit_convs);
}

}  // namespace odml::infra::gpu
