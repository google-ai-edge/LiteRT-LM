// Copyright 2025 The ODML Authors.
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

#include "runtime/executor/vision_litert_compiled_model_executor.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/engine/io_types.h"
#include "runtime/executor/vision_executor_utils.h"
#if !defined(LITERT_DISABLE_NPU)
#include "litert/cc/options/litert_google_tensor_options.h"  // from @litert
#include "litert/cc/options/litert_qualcomm_options.h"  // from @litert
#endif  // !defined(LITERT_DISABLE_NPU)
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/file_util.h"
#include "runtime/util/status_macros.h"  // NOLINT

namespace litert::lm {

namespace {

// The position input tensor name for ViT encoder.
constexpr absl::string_view kPositionsXy = "positions_xy";
// The image patch input tensor name for ViT encoder.
constexpr absl::string_view kImages = "images";
// The image patch input tensor name for ViT encoder with multi-signature
// support.
constexpr absl::string_view kVisionLengthPrefix = "vision_";
// The features output tensor name for ViT encoder.
constexpr absl::string_view kFeatures = "features";
// The mask input tensor name for ViT encoder.
constexpr absl::string_view kMask = "mask";

// Set the default GPU options for the model.
absl::Status SetGpuOptions(const VisionExecutorSettings& executor_settings,
                           litert::GpuOptions& gpu_options) {
#if defined(LITERT_USE_WEBGPU_ACCELERATOR)
  gpu_options.SetBackend(GpuOptions::Backend::kWebGpu);
#endif  // defined(LITERT_USE_WEBGPU_ACCELERATOR)
  gpu_options.EnableConstantTensorSharing(true);
  if (executor_settings.GetActivationDataType().has_value()) {
    if (executor_settings.GetActivationDataType().value() ==
        ActivationDataType::FLOAT32) {
      gpu_options.SetPrecision(GpuOptions::Precision::kFp32);
    } else {
      gpu_options.SetPrecision(GpuOptions::Precision::kFp16);
    }
  } else {
    // Default to fp32 if no activation data type is specified, for backward
    // compatibility with previous launched models.
    gpu_options.SetPrecision(GpuOptions::Precision::kFp32);
  }
#if defined(__APPLE__)
  gpu_options.SetPreferTextureWeights(false);
  gpu_options.SetUseMetalArgumentBuffers(true);
#else   // !__APPLE__
  gpu_options.SetPreferTextureWeights(true);
#endif  // !__APPLE__
  gpu_options.SetMadviseOriginalSharedTensors(true);
  gpu_options.SetConvertWeightsOnGpu(true);
  return absl::OkStatus();
}

// Returns the index of the signature that should be used for the given number
// of patches. The signature is guaranteed to have at least as many tokens as
// the given number of patches, or an error status if failed.
//
// Args:
//   model: The model to get the signature index from.
//   vision_executor_properties: The vision executor properties to get the
//     patch_num_shrink_factor from.
//   num_patches: The number of patches to get the signature index for.
//
// Returns:
//   The index of the signature that should be used for the given number of
//   patches, or an error status if failed.
absl::StatusOr<int> GetVitSignatureIndex(
    const Model& model,
    const VisionExecutorProperties& vision_executor_properties,
    const int num_patches) {
  if (model.GetNumSignatures() == 1) {
    return 0;
  }
  if (!vision_executor_properties.patch_num_shrink_factor.has_value()) {
    return absl::InvalidArgumentError(
        "Multi-signature vision models require patch_num_shrink_factor to be "
        "set.");
  }
  std::optional<int> best_signature_index;
  int best_length = std::numeric_limits<int>::max();
  int max_available_length = 0;
  bool found_any_signature = false;

  const int max_num_tokens =
      num_patches / vision_executor_properties.patch_num_shrink_factor.value();

  for (int i = 0; i < model.GetNumSignatures(); ++i) {
    LITERT_ASSIGN_OR_RETURN(auto signature_name, model.GetSignature(i));
    if (absl::StartsWith(signature_name.Key(), kVisionLengthPrefix)) {
      found_any_signature = true;
      int current_length = 0;
      size_t last_underscore = signature_name.Key().find_last_of('_');
      if (last_underscore == absl::string_view::npos ||
          !absl::SimpleAtoi(signature_name.Key().substr(last_underscore + 1),
                            &current_length)) {
        return absl::InvalidArgumentError(
            absl::StrCat("Failed to parse signature name ",
                         signature_name.Key(), " to integer."));
      }

      max_available_length = std::max(max_available_length, current_length);

      if (current_length >= max_num_tokens && current_length < best_length) {
        best_length = current_length;
        best_signature_index = i;
      }
    }
  }

  if (best_signature_index.has_value()) {
    return *best_signature_index;
  }

  if (!found_any_signature) {
    return absl::InvalidArgumentError(
        absl::StrCat("No signature found with prefix ", kVisionLengthPrefix));
  }

  return absl::InvalidArgumentError(
      absl::StrCat("No signature found with prefix ", kVisionLengthPrefix,
                   " and length greater than or equal to ", max_num_tokens,
                   ". This would truncate the input image, please set "
                   "max_num_tokens to at most ",
                   max_available_length, "."));
}

}  // namespace

absl::StatusOr<
    std::unique_ptr<VisionLiteRtCompiledModelExecutor::VisionEncoder>>
VisionLiteRtCompiledModelExecutor::VisionEncoder::Create(
    Environment& env, const Model* absl_nonnull model,
    const VisionExecutorSettings& vision_executor_settings,
    const VisionExecutorProperties& vision_executor_properties) {
  auto handler = std::unique_ptr<VisionEncoder>(new VisionEncoder(
      env, model, vision_executor_settings, vision_executor_properties));
  RETURN_IF_ERROR(handler->Initialize());
  return handler;
}

absl::Status VisionLiteRtCompiledModelExecutor::VisionEncoder::Initialize() {
  // TODO(b/405424188): - Add support for NPU backends.
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  auto weight_cache_file = vision_executor_settings_.GetWeightCacheFile(
      absl::StrCat(VisionExecutorSettings::kEncoderName,
                   ExecutorSettingsBase::kXnnpackCacheSuffix),
      /*check_and_clean=*/true);
  std::string weight_cache_path = vision_executor_settings_.GetCacheDir();
  auto activation_data_type = ActivationDataType::FLOAT16;
  if (vision_executor_settings_.GetActivationDataType().has_value()) {
    activation_data_type =
        vision_executor_settings_.GetActivationDataType().value();
  }
  switch (backend_) {
    case Backend::CPU: {
      // TODO: b/403132820 - Add accelerator compilation options for XNNPACK.
      LITERT_ASSIGN_OR_RETURN(auto& cpu_options, options.GetCpuOptions());
      // Set the number of threads to 4 by default.
      cpu_options.SetNumThreads(4);
      RETURN_IF_ERROR(SetCpuCacheOptions(
          weight_cache_file,
          /*logging_prefix=*/VisionExecutorSettings::kEncoderName,
          cpu_options));
      options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
      break;
    }
    case Backend::GPU: {
      // TODO: b/403132820 - Add accelerator compilation options for ML_DRIFT.
      LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
      ASSIGN_OR_RETURN(auto model_path,
                       vision_executor_settings_.GetModelAssets().GetPath());
      absl::string_view model_basename = Basename(model_path);
      LITERT_ASSIGN_OR_RETURN(std::string metadata_id,
                                GetFileCacheIdentifier(model_path));
      std::string cache_key =
          absl::StrCat(model_basename, VisionExecutorSettings::kEncoderName,
                       "_", metadata_id);
      ABSL_LOG(INFO) << "Vision cache key: " << cache_key;
      auto program_cache_file = vision_executor_settings_.GetProgramCacheFile(
          absl::StrCat(VisionExecutorSettings::kEncoderName,
                       ExecutorSettingsBase::kMlDriftCacheSuffix),
          /*check_and_clean=*/true);
      RETURN_IF_ERROR(SetGpuOptions(vision_executor_settings_, gpu_options));
      RETURN_IF_ERROR(SetGpuCacheOptions(
          weight_cache_path, program_cache_file, vision_executor_settings_,
          cache_key,
          /*logging_prefix=*/VisionExecutorSettings::kEncoderName,
          gpu_options));
      options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
      break;
    }
#if !defined(LITERT_DISABLE_NPU)
    case Backend::NPU: {
      LITERT_ASSIGN_OR_RETURN(auto& qualcomm_options,
                              options.GetQualcommOptions());
      qualcomm_options.SetLogLevel(qualcomm::QualcommOptions::LogLevel::kOff);
      qualcomm_options.SetHtpPerformanceMode(
          qualcomm::QualcommOptions::HtpPerformanceMode::kBurst);
      LITERT_ASSIGN_OR_RETURN(auto& google_tensor_options,
                              options.GetGoogleTensorOptions());
      google_tensor_options.SetPerformanceMode(
          google_tensor::GoogleTensorOptions::PerformanceMode::kBurst);
      // TODO: yunandrew - Add support for other NPU backends.
      options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
      break;
    }
#endif  // !defined(LITERT_DISABLE_NPU)
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported encoder backend: ", backend_));
  }

  LITERT_ASSIGN_OR_RETURN(compiled_model_,
                          CompiledModel::Create(env_, model_.Get(), options));
  if (!vision_executor_properties_.patch_num_shrink_factor.has_value()) {
    // Only create input buffer at initialization for non-VIT models.
    LITERT_ASSIGN_OR_RETURN(input_buffers_,
                            compiled_model_.CreateInputBuffers(0));
    LITERT_ASSIGN_OR_RETURN(output_buffers_,
                            compiled_model_.CreateOutputBuffers(0));
  }
  return absl::OkStatus();
}

absl::StatusOr<
    std::unique_ptr<VisionLiteRtCompiledModelExecutor::VisionAdapter>>
VisionLiteRtCompiledModelExecutor::VisionAdapter::Create(
    Environment& env, const Model* absl_nonnull model,
    const VisionExecutorSettings& vision_executor_settings,
    const VisionExecutorProperties& vision_executor_properties) {
  auto handler = std::unique_ptr<VisionAdapter>(new VisionAdapter(
      env, model, vision_executor_settings, vision_executor_properties));
  RETURN_IF_ERROR(handler->Initialize());
  return handler;
}

absl::Status VisionLiteRtCompiledModelExecutor::VisionAdapter::Initialize() {
  // TODO(b/405424188): - Add support for NPU backends.
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  auto weight_cache_file = vision_executor_settings_.GetWeightCacheFile(
      absl::StrCat(VisionExecutorSettings::kAdapterName,
                   ExecutorSettingsBase::kXnnpackCacheSuffix),
      /*check_and_clean=*/true);
  std::string weight_cache_path = vision_executor_settings_.GetCacheDir();
  switch (backend_) {
    case Backend::CPU: {
      // TODO: b/403132820 - Add accelerator compilation options for XNNPACK.
      LITERT_ASSIGN_OR_RETURN(auto& cpu_options, options.GetCpuOptions());
      // Set the number of threads to 4 by default.
      cpu_options.SetNumThreads(4);
      RETURN_IF_ERROR(SetCpuCacheOptions(weight_cache_file,
                                         VisionExecutorSettings::kAdapterName,
                                         cpu_options));
      options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
      break;
    }
    case Backend::GPU: {
      // TODO: b/403132820 - Add accelerator compilation options for ML_DRIFT.
      LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
      gpu_options.EnableConstantTensorSharing(true);
      gpu_options.EnableAllowSrcQuantizedFcConvOps(true);

      gpu_options.SetPrecision(GpuOptions::Precision::kFp16);
      gpu_options.SetPreferTextureWeights(true);
      options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported adapter backend: ", backend_));
  }

  LITERT_ASSIGN_OR_RETURN(compiled_model_,
                          CompiledModel::Create(env_, model_.Get(), options));
  // This check verifies if signature 0 of the adapter model contains any
  // inputs. This is used to infer whether input buffers should be created at
  // initialization time (for single-signature models that use signature 0 by
  // default) or skipped (for multi-signature models like ViT that create
  // input buffers on-demand in `Encode` for a specific signature). This is a
  // more direct check than relying on `patch_num_shrink_factor` which was
  // previously used to detect multi-signature models.
  auto signature_or = model_.GetSignature(0);
  if (signature_or.HasValue() && !signature_or->InputNames().empty()) {
    LITERT_ASSIGN_OR_RETURN(input_buffers_,
                            compiled_model_.CreateInputBuffers(0));
    if (input_buffers_.size() != 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("The Vision Adapter model must have exactly one input "
                       "buffer but got ",
                       input_buffers_.size()));
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<VisionLiteRtCompiledModelExecutor>>
litert::lm::VisionLiteRtCompiledModelExecutor::Create(
    const VisionExecutorSettings& vision_executor_settings, Environment& env) {
  LITERT_ASSIGN_OR_RETURN(auto resources,
                          BuildLiteRtCompiledModelResources(
                              vision_executor_settings.GetModelAssets()));

  ASSIGN_OR_RETURN(auto vision_encoder_model,
                   resources->GetTFLiteModel(ModelType::kTfLiteVisionEncoder));
  if (!vision_encoder_model) {
    return absl::InternalError("Failed to build LiteRt encoder model.");
  }
  ASSIGN_OR_RETURN(auto vision_adapter_model,
                   resources->GetTFLiteModel(ModelType::kTfLiteVisionAdapter));
  if (!vision_adapter_model) {
    return absl::InternalError("Failed to build LiteRt adapter model.");
  }
  ASSIGN_OR_RETURN(
      auto vision_executor_properties,
      GetVisionExecutorPropertiesFromModelResources(*resources.get()));

  ASSIGN_OR_RETURN(
      auto vision_encoder,
      VisionEncoder::Create(env, vision_encoder_model, vision_executor_settings,
                            vision_executor_properties));

  ASSIGN_OR_RETURN(
      auto vision_adapter,
      VisionAdapter::Create(env, vision_adapter_model, vision_executor_settings,
                            vision_executor_properties));

  LITERT_ASSIGN_OR_RETURN(auto tensor_type,
                          vision_encoder_model->GetInputTensorType(0, 0));
  const auto& dimensions = tensor_type.Layout().Dimensions();
  if (dimensions.size() == 4) {
    if (dimensions[3] < 3 || dimensions[3] > 4) {
      return absl::FailedPreconditionError(
          absl::StrCat("Expected encoder input tensor to have 3 or 4 channels",
                       " but got ", dimensions[3]));
    }
  } else if (dimensions.size() != 3) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Expected encoder input tensor to have 3 or 4 dimensions, but got ",
        dimensions.size()));
  }
  auto expected_input_dimension =
      std::vector<int>(dimensions.begin(), dimensions.end());

  return absl::WrapUnique(new VisionLiteRtCompiledModelExecutor(
      vision_executor_settings, env, std::move(resources),
      std::move(vision_encoder), std::move(vision_adapter),
      expected_input_dimension, vision_executor_properties));
}

absl::StatusOr<ExecutorVisionData> VisionLiteRtCompiledModelExecutor::Encode(
    const litert::TensorBuffer& input_image_tensor) {
  LITERT_ASSIGN_OR_RETURN(
      auto output_tensor_buffers,
      vision_adapter_->GetCompiledModel().CreateOutputBuffers(0));
  if (output_tensor_buffers.size() != 1) {
    return absl::InternalError(
        absl::StrCat("The Vision Adapter model must have exactly one output "
                     "buffer but got ",
                     output_tensor_buffers.size()));
  }

  LITERT_ASSIGN_OR_RETURN(auto input_image_data,
                          ReferTensorBufferAsSpan<float>(input_image_tensor));
  LITERT_RETURN_IF_ERROR(
      vision_encoder_->GetMutableInputBuffers()[0].Write<float>(
          input_image_data));
  auto& encoder_outputs = vision_encoder_->GetMutableOutputBuffers();
  if (encoder_outputs[0].IsWebGpuMemory() ||
      encoder_outputs[0].IsMetalMemory()) {
    // For WebGPU and Metal memory, we need to create a new output buffer to
    // hold the data, otherwise we will get failed to lock TensorBuffer error on
    // the second call to `Encode`. See b/457483190
    LITERT_ASSIGN_OR_RETURN(
        encoder_outputs,
        vision_encoder_->GetCompiledModel().CreateOutputBuffers(0));
  }

  LITERT_RETURN_IF_ERROR(vision_encoder_->GetCompiledModel().Run(
      /*input_buffers=*/vision_encoder_->GetInputBuffers(),
      /*output_buffers=*/encoder_outputs));

  LITERT_RETURN_IF_ERROR(vision_adapter_->GetCompiledModel().Run(
      /*input_buffers=*/encoder_outputs,
      /*output_buffers=*/output_tensor_buffers));

  return ExecutorVisionData(std::move(output_tensor_buffers[0]),
                            /*per_layer_embeddings=*/std::nullopt);
}

absl::StatusOr<std::vector<int>>
VisionLiteRtCompiledModelExecutor::GetExpectedInputDimension() const {
  return expected_input_dimension_;
}

absl::StatusOr<ExecutorVisionData> VisionLiteRtCompiledModelExecutor::Encode(
    const absl::flat_hash_map<std::string, litert::TensorBuffer>& input_maps) {

  if (!input_maps.contains(kPositionsXy)) {
    return absl::InvalidArgumentError(
        absl::StrCat(kPositionsXy, " is not found in the input maps."));
  }
  if (!input_maps.contains(kImages)) {
    return absl::InvalidArgumentError(
        absl::StrCat(kImages, " is not found in the input maps."));
  }

  absl::flat_hash_map<absl::string_view, litert::TensorBuffer>
      encoder_input_maps;
  LITERT_ASSIGN_OR_RETURN(auto images_tensor_type,
                          input_maps.at(kImages).TensorType());
  const auto& images_dimensions = images_tensor_type.Layout().Dimensions();
  const int num_patches_from_input = images_dimensions[1];
  ASSIGN_OR_RETURN(auto encoder_signature_index,
                   GetVitSignatureIndex(vision_encoder_->GetModel(),
                                        vision_executor_properties_,
                                        num_patches_from_input));
  ASSIGN_OR_RETURN(auto adapter_signature_index,
                   GetVitSignatureIndex(vision_adapter_->GetModel(),
                                        vision_executor_properties_,
                                        num_patches_from_input));
  LITERT_ASSIGN_OR_RETURN(
      auto encoder_input_buffers,
      vision_encoder_->GetCompiledModel().CreateInputBuffers(
          encoder_signature_index));

  LITERT_ASSIGN_OR_RETURN(
      auto encoder_signature,
      vision_encoder_->GetModel().GetSignature(encoder_signature_index));
  LITERT_ASSIGN_OR_RETURN(
      auto adapter_signature,
      vision_adapter_->GetModel().GetSignature(adapter_signature_index));
  ABSL_LOG(INFO) << "encoder_signature_index: " << encoder_signature_index
                 << " name: " << encoder_signature.Key();
  ABSL_LOG(INFO) << "adapter_signature_index: " << adapter_signature_index
                 << " name: " << adapter_signature.Key();

  LITERT_ASSIGN_OR_RETURN(
      auto adapter_output_tensor_buffers,
      vision_adapter_->GetCompiledModel().CreateOutputBuffers(
          adapter_signature_index));
  if (adapter_output_tensor_buffers.size() != 1) {
    return absl::InternalError(
        absl::StrCat("The Vision Adapter model must have exactly one output "
                     "buffer but got ",
                     adapter_output_tensor_buffers.size()));
  }

  for (const auto& [key, value] : input_maps) {
    LITERT_ASSIGN_OR_RETURN(auto tensor_type, value.TensorType());
    LITERT_ASSIGN_OR_RETURN(auto input_index,
                            vision_encoder_->GetCompiledModel().FindInputIndex(
                                encoder_signature_index, key));
    encoder_input_buffers[input_index].Clear();
    if (tensor_type.ElementType() == ElementType::Float32) {
      LITERT_ASSIGN_OR_RETURN(auto input_data,
                              ReferTensorBufferAsSpan<float>(value));
      LITERT_RETURN_IF_ERROR(
          encoder_input_buffers[input_index].Write<float>(input_data));

      // Force clearing padding beyond the written data
      LITERT_ASSIGN_OR_RETURN(auto lock_and_addr,
                              litert::TensorBufferScopedLock::Create<float>(
                                  encoder_input_buffers[input_index],
                                  litert::TensorBuffer::LockMode::kWrite));
      LITERT_ASSIGN_OR_RETURN(size_t packed_size,
                              encoder_input_buffers[input_index].PackedSize());
      size_t written_size = input_data.size() * sizeof(float);
      if (packed_size > written_size) {
        std::memset(
            reinterpret_cast<uint8_t*>(lock_and_addr.second) + written_size, 0,
            packed_size - written_size);
      }
    } else if (tensor_type.ElementType() == ElementType::Int32) {
      // Initialize the position buffer to -1 since the input image tensor
      // might have different size as the encoder input tensor.
      LITERT_ASSIGN_OR_RETURN(auto encoder_tensor_type,
                              encoder_input_buffers[input_index].TensorType());
      LITERT_ASSIGN_OR_RETURN(auto position_num_elements,
                              encoder_tensor_type.Layout().NumElements());
      std::vector<int32_t> encoder_input_positions(position_num_elements, -1);
      LITERT_RETURN_IF_ERROR(encoder_input_buffers[input_index].Write<int32_t>(
          encoder_input_positions));
      LITERT_ASSIGN_OR_RETURN(auto input_data,
                              ReferTensorBufferAsSpan<int32_t>(value));
      LITERT_RETURN_IF_ERROR(
          encoder_input_buffers[input_index].Write<int32_t>(input_data));
    } else {
      return absl::InvalidArgumentError("Unsupported input tensor type");
    }
  }

  LITERT_ASSIGN_OR_RETURN(
      auto encoder_output_buffers,
      vision_encoder_->GetCompiledModel().CreateOutputBuffers(
          encoder_signature_index));

  LITERT_RETURN_IF_ERROR(vision_encoder_->GetCompiledModel().Run(
      encoder_signature_index, encoder_input_buffers, encoder_output_buffers));

  int num_patches = 0;
  auto mask_index = vision_encoder_->GetCompiledModel().FindOutputIndex(
      encoder_signature_index, kMask);

  if (!mask_index.HasValue()) {
    // If the mask is not in the encoder output, we need to estimate the number
    // of patches from the input image tensor.
    if (!vision_executor_properties_.patch_num_shrink_factor.has_value()) {
      return absl::InternalError(
          "patch_num_shrink_factor is not set in the vision executor "
          "properties.");
    }
    LITERT_ASSIGN_OR_RETURN(auto positions_tensor_type,
                            input_maps.at(kPositionsXy).TensorType());
    const int& num_patches_from_input =
        positions_tensor_type.Layout().Dimensions()[1];
    const int& patch_num_shrink_factor =
        vision_executor_properties_.patch_num_shrink_factor.value();
    // Round up the number of patches so we have at least one patch.
    num_patches = (num_patches_from_input + patch_num_shrink_factor - 1) /
                  patch_num_shrink_factor;
  } else {
    LITERT_ASSIGN_OR_RETURN(
        auto mask_tensor_type,
        encoder_output_buffers[mask_index.Value()].TensorType());
    LITERT_ASSIGN_OR_RETURN(int mask_num_elements,
                            mask_tensor_type.Layout().NumElements());
    std::vector<uint8_t> encoder_output_mask(mask_num_elements, 0);
    LITERT_RETURN_IF_ERROR(
        encoder_output_buffers[mask_index.Value()].Read<uint8_t>(
            absl::MakeSpan(encoder_output_mask)));
    num_patches = std::count(encoder_output_mask.begin(),
                             encoder_output_mask.end(), true);
  }

  LITERT_ASSIGN_OR_RETURN(auto features_index,
                          vision_encoder_->GetCompiledModel().FindOutputIndex(
                              encoder_signature_index, kFeatures));
  LITERT_ASSIGN_OR_RETURN(auto encoder_output_tensor_type,
                          encoder_output_buffers[features_index].TensorType());
  const int& encoder_output_dim =
      encoder_output_tensor_type.Layout().Dimensions()
          [encoder_output_tensor_type.Layout().Dimensions().size() - 1];
  LITERT_ASSIGN_OR_RETURN(int encoder_output_num_elements,
                          encoder_output_tensor_type.Layout().NumElements());
  std::vector<float> encoder_output_data(encoder_output_num_elements);
  LITERT_RETURN_IF_ERROR(encoder_output_buffers[features_index].Read<float>(
      absl::MakeSpan(encoder_output_data)));

  LITERT_ASSIGN_OR_RETURN(
      auto adapter_input_buffers,
      vision_adapter_->GetCompiledModel().CreateInputBuffers(
          adapter_signature_index));
  adapter_input_buffers[0].Clear();
  LITERT_RETURN_IF_ERROR(adapter_input_buffers[0].Write<float>(absl::MakeSpan(
      encoder_output_data.data(), num_patches * encoder_output_dim)));

  LITERT_RETURN_IF_ERROR(vision_adapter_->GetCompiledModel().Run(
      adapter_signature_index,
      /*input_buffers=*/adapter_input_buffers,
      /*output_buffers=*/adapter_output_tensor_buffers));

  // Create the final output tensor with the correct number of patches.
  LITERT_ASSIGN_OR_RETURN(auto adapter_output_tensor_type,
                          adapter_output_tensor_buffers[0].TensorType());
  RankedTensorType output_tensor_type(
      GetElementType<float>(),
      Layout(
          Dimensions({1, num_patches,
                      adapter_output_tensor_type.Layout().Dimensions()[2]})));
  LITERT_ASSIGN_OR_RETURN(
      auto output_tensor,
      TensorBuffer::CreateManaged(
          env_, TensorBufferType::kHostMemory, output_tensor_type,
          output_tensor_type.Layout().Dimensions()[1] *
              output_tensor_type.Layout().Dimensions()[2] * sizeof(float)));
  LITERT_ASSIGN_OR_RETURN(
      auto adapter_output_data,
      ReferTensorBufferAsSpan<float>(adapter_output_tensor_buffers[0]));

  LITERT_RETURN_IF_ERROR(output_tensor.Write<float>(adapter_output_data.subspan(
      0, num_patches * output_tensor_type.Layout().Dimensions()[2])));
  return ExecutorVisionData(std::move(output_tensor),
                            /*per_layer_embeddings=*/std::nullopt);
}

absl::StatusOr<VisionExecutorProperties>
VisionLiteRtCompiledModelExecutor::GetVisionExecutorProperties() const {
  return vision_executor_properties_;
}

}  // namespace litert::lm
