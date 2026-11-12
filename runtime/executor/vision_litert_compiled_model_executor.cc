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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/options/litert_gpu_options.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_model_types.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_stats.h"
#include "runtime/executor/vision/vision_executor_utils.h"
#if !defined(LITERT_DISABLE_NPU)
#include "litert/cc/options/litert_qualcomm_options.h"  // from @litert
#endif  // !defined(LITERT_DISABLE_NPU)
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/cancellation_latch.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision/vision_executor_settings.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/status_macros.h"  // NOLINT

namespace litert::lm {
namespace {

constexpr absl::string_view kVisionModuleName = "Vision";
constexpr absl::string_view kVisionNumImagesMetric = "Vision num images";
constexpr absl::string_view kVisionNumPatchesMetric = "Vision num patches";
constexpr absl::string_view kVisionEncoderInferenceLatency =
    "Vision encoder inference";
constexpr absl::string_view kVisionAdapterInferenceLatency =
    "Vision adapter inference";
constexpr absl::string_view kVisionEncoderMetricsPrefix = "Vision encoder ";
constexpr absl::string_view kVisionAdapterMetricsPrefix = "Vision adapter ";

// The image patch input tensor name for ViT encoder.
constexpr absl::string_view kImages = "images";
// The image patch input tensor name for ViT encoder with multi-signature
// support.
constexpr absl::string_view kVisionLengthPrefix = "vision_";
// The features output tensor name for ViT encoder.
constexpr absl::string_view kFeatures = "features";
// The mask input tensor name for ViT encoder.
constexpr absl::string_view kMask = "mask";
// The patch position input tensor name for ViT encoder.
constexpr absl::string_view kPositionsXy = "positions_xy";

// Sets the default GPU options for the vision encoder model.
absl::Status SetEncoderGpuOptions(
    const VisionExecutorSettings& executor_settings,
    litert::GpuOptions& gpu_options) {
  ABSL_RETURN_IF_ERROR(SetCommonGpuOptions(executor_settings, gpu_options));
  // The vision encoder is compiled for the GPU only, so it is always fully
  // delegated to a single delegate. This lets TFLite skip preparing the CPU
  // kernels before delegation, which otherwise allocates a CPU arena for every
  // STABLEHLO_COMPOSITE decomposition subgraph that is never released.
  gpu_options.SetHintFullyDelegatedToSingleDelegate(true);
  return absl::OkStatus();
}

// Sets the default GPU options for the vision adapter model.
void SetAdapterGpuOptions(const VisionExecutorSettings& executor_settings,
                          litert::GpuOptions& gpu_options) {
#if defined(LITERT_USE_WEBGPU_ACCELERATOR)
  gpu_options.SetBackend(GpuOptions::Backend::kWebGpu);
#endif  // defined(LITERT_USE_WEBGPU_ACCELERATOR)
  // "default" (unset) currently resolves to disabled.
  LrtSetGpuOptionsUseIrModel(gpu_options.Get(),
                             executor_settings.UseIrModel().value_or(false));
}

// Extracts the visual token length / capacity from the signature's output
// tensor dimensions.
absl::StatusOr<int> GetSignatureTokenLength(
    const ::litert::SimpleSignature& signature) {
  std::optional<::litert::RankedTensorType> output_tensor_type;
  auto features_output = signature.OutputTensorType(kFeatures);
  if (features_output.HasValue()) {
    output_tensor_type = features_output.Value();
  } else if (!signature.OutputNames().empty()) {
    LITERT_ASSIGN_OR_RETURN(output_tensor_type, signature.OutputTensorType(0));
  }

  if (output_tensor_type.has_value()) {
    const auto& dims = output_tensor_type->Layout().Dimensions();
    if (dims.size() >= 2 && dims[dims.size() - 2] > 0) {
      return dims[dims.size() - 2];
    }
  }

  return absl::InvalidArgumentError(absl::StrCat(
      "Failed to determine token length from output tensor dimensions for "
      "signature: ",
      signature.Key()));
}

// Returns whether `signature` of a multi-signature vision model can be selected
// to run an image, i.e. it has the `kVisionLengthPrefix` prefix and is one of
// `selected_signatures` (if not empty).
bool IsSelectableVitSignature(
    const ::litert::SimpleSignature& signature,
    const std::vector<std::string>& selected_signatures) {
  return absl::StartsWith(signature.Key(), kVisionLengthPrefix) &&
         (selected_signatures.empty() ||
          absl::c_linear_search(selected_signatures, signature.Key()));
}

// Returns the maximum number of image patches accepted by `signature`, i.e.
// dimension 1 of its `kImages` input, which must be a patchified
// [batch, num_patches, patch_dim] tensor as read by Encode(). As in Encode(),
// the input of a single-signature model may be unnamed, in which case input 0
// is used. Returns an error for other inputs (e.g. a [batch, height, width,
// channels] image), whose number of patches is unknown.
absl::StatusOr<int> GetSignaturePatchCapacity(
    const ::litert::SimpleSignature& signature, bool is_single_signature) {
  ::litert::Expected<::litert::RankedTensorType> images_type =
      signature.InputTensorType(kImages);
  if (!images_type.HasValue() && is_single_signature) {
    images_type = signature.InputTensorType(0);
  }
  if (!images_type.HasValue()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "No ", kImages, " input found in signature: ", signature.Key()));
  }
  const auto& dims = images_type->Layout().Dimensions();
  if (dims.size() != 3 || dims[1] <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to determine the number of patches from the ", kImages,
        " input of signature ", signature.Key(),
        ": expected a [batch, num_patches, patch_dim] tensor with a positive "
        "num_patches, got [",
        absl::StrJoin(dims, ", "), "]"));
  }
  return dims[1];
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
    const ::litert::Model& model,
    const VisionExecutorProperties& vision_executor_properties,
    const int num_patches,
    const std::vector<std::string>& selected_signatures) {
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
    LITERT_ASSIGN_OR_RETURN(auto signature, model.GetSignature(i));
    if (!IsSelectableVitSignature(signature, selected_signatures)) {
      continue;
    }
    found_any_signature = true;
    LITERT_ASSIGN_OR_RETURN(int current_length,
                            GetSignatureTokenLength(signature));

    max_available_length = std::max(max_available_length, current_length);

    if (current_length >= max_num_tokens && current_length < best_length) {
      best_length = current_length;
      best_signature_index = i;
    }
  }

  if (best_signature_index.has_value()) {
    return *best_signature_index;
  }

  if (!found_any_signature) {
    return absl::InvalidArgumentError(absl::StrCat(
        "No matching signature found with prefix ", kVisionLengthPrefix));
  }

  return absl::InvalidArgumentError(
      absl::StrCat("No signature found with prefix ", kVisionLengthPrefix,
                   " and length greater than or equal to ", max_num_tokens,
                   ". This would truncate the input image, please set "
                   "max_num_tokens to at most ",
                   max_available_length, "."));
}

// WebGPU and Metal buffers cannot be reused across calls (see b/457483190).
bool AreBuffersReusable(const std::vector<TensorBuffer>& buffers) {
  return absl::c_none_of(buffers, [](const TensorBuffer& buffer) {
    return buffer.IsWebGpuMemory() || buffer.IsMetalMemory();
  });
}

}  // namespace

absl::StatusOr<std::vector<VisionEncoderVariant>> GetVisionEncoderVariants(
    const ::litert::Model& model,
    const std::vector<std::string>& selected_signatures) {
  // Mirrors the signature selection of GetVitSignatureIndex(): the only
  // signature of a single-signature model is always used, otherwise the
  // selectable signatures are candidates.
  const bool is_single_signature = model.GetNumSignatures() == 1;
  std::vector<VisionEncoderVariant> variants;
  for (int i = 0; i < model.GetNumSignatures(); ++i) {
    LITERT_ASSIGN_OR_RETURN(::litert::SimpleSignature signature,
                            model.GetSignature(i));
    if (!is_single_signature &&
        !IsSelectableVitSignature(signature, selected_signatures)) {
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(
        int max_num_patches,
        GetSignaturePatchCapacity(signature, is_single_signature));
    ABSL_ASSIGN_OR_RETURN(int num_soft_tokens,
                          GetSignatureTokenLength(signature));
    variants.push_back({
        .signature_name = std::string(signature.Key()),
        .max_num_patches = max_num_patches,
        .num_soft_tokens = num_soft_tokens,
    });
  }
  absl::c_stable_sort(variants, [](const VisionEncoderVariant& a,
                                   const VisionEncoderVariant& b) {
    return a.num_soft_tokens < b.num_soft_tokens;
  });
  return variants;
}

Expected<std::vector<TensorBuffer>&> VisionLiteRtCompiledModelExecutor::
    SignatureBufferCache::GetOrCreateInputBuffers(CompiledModel& compiled_model,
                                                  int signature_index) {
  auto it = input_buffers_.find(signature_index);
  if (it != input_buffers_.end() && AreBuffersReusable(it->second)) {
    return it->second;
  }
  LITERT_ASSIGN_OR_RETURN(auto buffers,
                          compiled_model.CreateInputBuffers(signature_index));
  for (auto& buffer : buffers) {
    LITERT_RETURN_IF_ERROR(buffer.Clear());
  }
  return input_buffers_[signature_index] = std::move(buffers);
}

Expected<std::vector<TensorBuffer>&> VisionLiteRtCompiledModelExecutor::
    SignatureBufferCache::GetOrCreateOutputBuffers(
        CompiledModel& compiled_model, int signature_index) {
  auto it = output_buffers_.find(signature_index);
  if (it != output_buffers_.end() && AreBuffersReusable(it->second)) {
    return it->second;
  }
  LITERT_ASSIGN_OR_RETURN(auto buffers,
                          compiled_model.CreateOutputBuffers(signature_index));
  return output_buffers_[signature_index] = std::move(buffers);
}

absl::StatusOr<
    std::unique_ptr<VisionLiteRtCompiledModelExecutor::VisionSubmodel>>
VisionLiteRtCompiledModelExecutor::VisionSubmodel::CreateEncoder(
    Environment& env, const Model* absl_nonnull model,
    const VisionExecutorSettings& vision_executor_settings,
    ModelResources& resources) {
  return Create(Role::kEncoder, env, model, vision_executor_settings,
                resources);
}

absl::StatusOr<
    std::unique_ptr<VisionLiteRtCompiledModelExecutor::VisionSubmodel>>
VisionLiteRtCompiledModelExecutor::VisionSubmodel::CreateAdapter(
    Environment& env, const Model* absl_nonnull model,
    const VisionExecutorSettings& vision_executor_settings,
    ModelResources& resources) {
  return Create(Role::kAdapter, env, model, vision_executor_settings,
                resources);
}

absl::StatusOr<
    std::unique_ptr<VisionLiteRtCompiledModelExecutor::VisionSubmodel>>
VisionLiteRtCompiledModelExecutor::VisionSubmodel::Create(
    Role role, Environment& env, const Model* absl_nonnull model,
    const VisionExecutorSettings& vision_executor_settings,
    ModelResources& resources) {
  RET_CHECK_NE(model, nullptr);
  const bool is_encoder = (role == Role::kEncoder);
  const Backend backend = is_encoder
                              ? vision_executor_settings.GetEncoderBackend()
                              : vision_executor_settings.GetAdapterBackend();
  const absl::string_view cache_model_name =
      is_encoder ? VisionExecutorSettings::kEncoderName
                 : VisionExecutorSettings::kAdapterName;
  const proto::LlmMetadata::TfLiteModelType model_type =
      is_encoder ? proto::LlmMetadata::TF_LITE_VISION_ENCODER
                 : proto::LlmMetadata::TF_LITE_VISION_ADAPTER;
  const std::vector<std::string>& selected_signature_names =
      is_encoder ? vision_executor_settings.GetEncoderSelectedSignatures()
                 : vision_executor_settings.GetAdapterSelectedSignatures();
  const absl::string_view role_name = is_encoder ? "encoder" : "adapter";

  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  switch (backend) {
    case Backend::CPU: {
      // TODO: b/403132820 - Add accelerator compilation options for XNNPACK.
      auto weight_cache_file = vision_executor_settings.GetWeightCacheFile(
          absl::StrCat(cache_model_name,
                       ExecutorSettingsBase::kXnnpackCacheSuffix),
          /*check_and_clean=*/true);
      LITERT_ASSIGN_OR_RETURN(auto& cpu_options,
                              options.GetOptions<litert::CpuOptions>());
      ABSL_RETURN_IF_ERROR(SetCpuOptions(cpu_options, 4));
      ABSL_RETURN_IF_ERROR(SetCpuCacheOptions(
          weight_cache_file, /*logging_prefix=*/cache_model_name, cpu_options));
      options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
      break;
    }
    case Backend::GPU: {
      // TODO: b/403132820 - Add accelerator compilation options for ML_DRIFT.
      LITERT_ASSIGN_OR_RETURN(auto& gpu_options,
                              options.GetOptions<litert::GpuOptions>());
      ABSL_ASSIGN_OR_RETURN(
          const auto cache_files,
          GetGpuModelCacheData(vision_executor_settings, cache_model_name));
      if (is_encoder) {
        ABSL_RETURN_IF_ERROR(
            SetEncoderGpuOptions(vision_executor_settings, gpu_options));
      } else {
        SetAdapterGpuOptions(vision_executor_settings, gpu_options);
      }
      ABSL_RETURN_IF_ERROR(SetGpuCacheOptions(
          cache_files.weight_cache_file, cache_files.program_cache_file,
          cache_files.cache_key, /*logging_prefix=*/cache_model_name,
          /*cache_compiled_shaders_only=*/false, gpu_options));
      options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
      break;
    }
#if !defined(LITERT_DISABLE_NPU)
    case Backend::NPU: {
      LITERT_ASSIGN_OR_RETURN(auto& qualcomm_options,
                              options.GetOptions<qualcomm::QualcommOptions>());
      qualcomm_options.SetLogLevel(qualcomm::QualcommOptions::LogLevel::kOff);
      ABSL_RETURN_IF_ERROR(
          SetNpuOptions(vision_executor_settings.GetNpuOptions(), options));
      options.SetHardwareAccelerators(litert::HwAccelerators::kNpu |
                                      litert::HwAccelerators::kCpu);
      break;
    }
#endif  // !defined(LITERT_DISABLE_NPU)
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported ", role_name, " backend: ", backend));
  }
  ABSL_RETURN_IF_ERROR(
      SetExternalWeightOptions(resources, model_type, options));

  if (!selected_signature_names.empty()) {
    std::vector<absl::string_view> selected_signatures(
        selected_signature_names.begin(), selected_signature_names.end());
    LITERT_ASSIGN_OR_RETURN(auto& runtime_options, options.GetRuntimeOptions());
    LITERT_RETURN_IF_ERROR(
        runtime_options.SetSelectedSignatures(selected_signatures));
  }

#ifdef __EMSCRIPTEN__
  extern void SetCurrentlyCompilingModel(absl::string_view model_type_str)
      __attribute__((weak));
  if (SetCurrentlyCompilingModel) {
    SetCurrentlyCompilingModel(TfLiteModelTypeToWireString(model_type));
  }
#endif
  Expected<CompiledModel> expected_compiled_model =
      CompiledModel::Create(env, model->Get(), options);
#ifdef __EMSCRIPTEN__
  if (SetCurrentlyCompilingModel) {
    SetCurrentlyCompilingModel("");
  }
#endif
  LITERT_ASSIGN_OR_RETURN(auto compiled_model,
                          std::move(expected_compiled_model));
  auto submodel =
      absl::WrapUnique(new VisionSubmodel(*model, std::move(compiled_model)));

  // Single-signature models (non-ViT models, single-input LFM2-VL, or single-
  // signature adapters) use signature 0 on the single-buffer path, so pre-
  // create their buffers at initialization time. Multi-signature (ViT) models
  // create buffers on demand in `EncodeInputMaps` for the selected signature,
  // so signature 0 buffers should not be created in advance.
  if (model->GetNumSignatures() == 1) {
    if (is_encoder) {
      LITERT_RETURN_IF_ERROR(submodel->GetOrCreateInputBuffers(0));
      LITERT_RETURN_IF_ERROR(submodel->GetOrCreateOutputBuffers(0));
    } else {
      auto signature = model->GetSignature(0);
      if (signature.HasValue() && !signature->InputNames().empty()) {
        LITERT_RETURN_IF_ERROR(submodel->GetOrCreateInputBuffers(0));
      }
    }
  }
  return submodel;
}

absl::StatusOr<std::unique_ptr<VisionLiteRtCompiledModelExecutor>>
litert::lm::VisionLiteRtCompiledModelExecutor::Create(
    const VisionExecutorSettings& vision_executor_settings, Environment& env,
    ModelResources& resources) {
  ABSL_ASSIGN_OR_RETURN(
      auto vision_encoder_model,
      resources.GetTFLiteModel(proto::LlmMetadata::TF_LITE_VISION_ENCODER));
  if (!vision_encoder_model) {
    return absl::InternalError("Failed to build LiteRt encoder model.");
  }
  // Vision adapter is optional.
  auto vision_adapter_model =
      resources.GetTFLiteModel(proto::LlmMetadata::TF_LITE_VISION_ADAPTER);
  if (!vision_adapter_model.ok() &&
      vision_adapter_model.status().code() != absl::StatusCode::kNotFound) {
    return vision_adapter_model.status();
  }
  ABSL_ASSIGN_OR_RETURN(
      auto vision_executor_properties,
      GetVisionExecutorPropertiesFromModelResources(resources));

  ABSL_ASSIGN_OR_RETURN(
      auto vision_encoder,
      VisionSubmodel::CreateEncoder(env, vision_encoder_model,
                                    vision_executor_settings, resources));

  std::unique_ptr<VisionSubmodel> vision_adapter;
  if (vision_adapter_model.ok()) {
    ABSL_ASSIGN_OR_RETURN(
        vision_adapter,
        VisionSubmodel::CreateAdapter(env, *vision_adapter_model,
                                      vision_executor_settings, resources));
  }

  // Derive the expected input dimension for the single-tensor Encode overload.
  // Multi-signature patchified encoders (whose per-signature input dims differ)
  // are driven exclusively through the map-based Encode overload, so this is
  // only meaningful for the single-signature, single-input case. Guard against
  // patchified/multi-input shapes (e.g. images [1, L, patch_dim] where dim[2]
  // is the patch vector, not a spatial size) which would otherwise produce a
  // misleading value.
  std::vector<int> expected_input_dimension;
  LITERT_ASSIGN_OR_RETURN(auto tensor_type,
                          vision_encoder_model->GetInputTensorType(0, 0));
  auto encoder_input_names = vision_encoder_model->GetSignatureInputNames(0);
  if (vision_encoder_model->GetNumSignatures() == 1 &&
      encoder_input_names.HasValue() && encoder_input_names->size() == 1) {
    const auto& dimensions = tensor_type.Layout().Dimensions();
    if (dimensions.size() == 4) {
      if (dimensions[3] < 3 || dimensions[3] > 4) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Expected encoder input tensor to have 3 or 4 channels",
            " but got ", dimensions[3]));
      }
    } else if (dimensions.size() != 3) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Expected encoder input tensor to have 3 or 4 dimensions, but got ",
          dimensions.size()));
    }
    expected_input_dimension.assign(dimensions.begin(), dimensions.end());
  }

  return absl::WrapUnique(new VisionLiteRtCompiledModelExecutor(
      vision_executor_settings, env, /*resources=*/nullptr,
      std::move(vision_encoder), std::move(vision_adapter),
      std::move(expected_input_dimension), vision_executor_properties));
}

absl::StatusOr<std::unique_ptr<VisionLiteRtCompiledModelExecutor>>
litert::lm::VisionLiteRtCompiledModelExecutor::Create(
    const VisionExecutorSettings& vision_executor_settings, Environment& env,
    std::shared_ptr<ModelResources> resources) {
  if (resources == nullptr) {
    return absl::InvalidArgumentError("ModelResources cannot be null.");
  }
  ABSL_ASSIGN_OR_RETURN(auto executor,
                        Create(vision_executor_settings, env, *resources));
  executor->resources_ = std::move(resources);
  return executor;
}

absl::StatusOr<std::unique_ptr<VisionLiteRtCompiledModelExecutor>>
litert::lm::VisionLiteRtCompiledModelExecutor::Create(
    const VisionExecutorSettings& vision_executor_settings, Environment& env) {
  LITERT_ASSIGN_OR_RETURN(auto resources,
                          BuildLiteRtCompiledModelResources(
                              vision_executor_settings.GetModelAssets()));
  return Create(vision_executor_settings, env, std::move(resources));
}

absl::StatusOr<ExecutorVisionData> VisionLiteRtCompiledModelExecutor::Encode(
    const litert::TensorBuffer& input_image_tensor) {
  return EncodeImageTensor(input_image_tensor, /*latch=*/nullptr);
}

absl::StatusOr<ExecutorVisionData>
VisionLiteRtCompiledModelExecutor::EncodeImageTensor(
    const litert::TensorBuffer& input_image_tensor, CancellationLatch* latch) {
  ScopedLatency scoped_total_latency(latency_stats_);
  LITERT_ASSIGN_OR_RETURN(auto& encoder_input_buffers,
                          vision_encoder_->GetOrCreateInputBuffers(0));

  LITERT_ASSIGN_OR_RETURN(auto input_image_data,
                          ReferTensorBufferAsSpan<float>(input_image_tensor));
  LITERT_RETURN_IF_ERROR(
      encoder_input_buffers[0].Write<float>(input_image_data));

  if (vision_adapter_ == nullptr) {
    LITERT_ASSIGN_OR_RETURN(
        auto encoder_outputs,
        vision_encoder_->GetCompiledModel().CreateOutputBuffers(0));
    ABSL_RETURN_IF_ERROR(RunCompiledModel(
        vision_encoder_->GetCompiledModel(), /*signature_index=*/0,
        encoder_input_buffers, encoder_outputs, kVisionEncoderInferenceLatency,
        latch));
    AccumulateStat(latency_stats_, kVisionNumImagesMetric, int64_t{1});
    return ExecutorVisionData(std::move(encoder_outputs[0]),
                              /*per_layer_embeddings=*/std::nullopt);
  }

  LITERT_ASSIGN_OR_RETURN(
      auto output_tensor_buffers,
      vision_adapter_->GetCompiledModel().CreateOutputBuffers(0));
  if (output_tensor_buffers.size() != 1) {
    return absl::InternalError(
        absl::StrCat("The Vision Adapter model must have exactly one output "
                     "buffer but got ",
                     output_tensor_buffers.size()));
  }

  LITERT_ASSIGN_OR_RETURN(auto& encoder_outputs,
                          vision_encoder_->GetOrCreateOutputBuffers(0));

  ABSL_RETURN_IF_ERROR(RunCompiledModel(vision_encoder_->GetCompiledModel(),
                                        /*signature_index=*/0,
                                        encoder_input_buffers, encoder_outputs,
                                        kVisionEncoderInferenceLatency, latch));
  ABSL_RETURN_IF_ERROR(RunCompiledModel(vision_adapter_->GetCompiledModel(),
                                        /*signature_index=*/0, encoder_outputs,
                                        output_tensor_buffers,
                                        kVisionAdapterInferenceLatency, latch));

  AccumulateStat(latency_stats_, kVisionNumImagesMetric, int64_t{1});

  return ExecutorVisionData(std::move(output_tensor_buffers[0]),
                            /*per_layer_embeddings=*/std::nullopt);
}

absl::StatusOr<std::vector<VisionEncoderVariant>>
VisionLiteRtCompiledModelExecutor::GetSupportedVariants() const {
  return GetVisionEncoderVariants(
      vision_encoder_->GetModel(),
      vision_executor_settings_.GetEncoderSelectedSignatures());
}

absl::Status VisionLiteRtCompiledModelExecutor::RunCompiledModel(
    const CompiledModel& compiled_model, size_t signature_index,
    absl::Span<const litert::TensorBuffer> input_buffers,
    absl::Span<const litert::TensorBuffer> output_buffers,
    absl::string_view latency_metric, CancellationLatch* latch) {
  if (latch != nullptr && latch->IsCancelled()) {
    return absl::CancelledError("Vision encoding was cancelled.");
  }
  ScopedLatency scoped(latency_stats_, latency_metric);
  auto result =
      compiled_model.Run(signature_index, input_buffers, output_buffers);
  // An interrupted run reports a generic error.
  if (!result.HasValue() && latch != nullptr && latch->IsCancelled()) {
    return absl::CancelledError("Vision encoding was cancelled.");
  }
  LITERT_RETURN_IF_ERROR(result);
  return absl::OkStatus();
}

absl::StatusOr<ExecutorVisionData>
VisionLiteRtCompiledModelExecutor::RunCancellable(
    absl::FunctionRef<bool()> should_cancel,
    absl::FunctionRef<absl::StatusOr<ExecutorVisionData>(CancellationLatch*)>
        encode) {
  std::vector<CompiledModel*> compiled_models = {
      &vision_encoder_->GetMutableCompiledModel()};
  if (vision_adapter_ != nullptr) {
    compiled_models.push_back(&vision_adapter_->GetMutableCompiledModel());
  }
  CancellationLatch latch(should_cancel, compiled_models);
  // Check before copying any input.
  if (latch.IsCancelled()) {
    return absl::CancelledError("Vision encoding was cancelled.");
  }
  return encode(&latch);
}

absl::StatusOr<ExecutorVisionData> VisionLiteRtCompiledModelExecutor::Encode(
    const litert::TensorBuffer& input_image_tensor,
    absl::FunctionRef<bool()> should_cancel) {
  return RunCancellable(should_cancel, [&](CancellationLatch* latch) {
    return EncodeImageTensor(input_image_tensor, latch);
  });
}

absl::StatusOr<ExecutorVisionData> VisionLiteRtCompiledModelExecutor::Encode(
    const absl::flat_hash_map<std::string, litert::TensorBuffer>& input_maps,
    absl::FunctionRef<bool()> should_cancel) {
  return RunCancellable(should_cancel, [&](CancellationLatch* latch) {
    return EncodeInputMaps(input_maps, latch);
  });
}

absl::StatusOr<std::vector<int>>
VisionLiteRtCompiledModelExecutor::GetExpectedInputDimension() const {
  return expected_input_dimension_;
}

absl::StatusOr<ExecutorVisionData> VisionLiteRtCompiledModelExecutor::Encode(
    const absl::flat_hash_map<std::string, litert::TensorBuffer>& input_maps) {
  return EncodeInputMaps(input_maps, /*latch=*/nullptr);
}

absl::StatusOr<ExecutorVisionData>
VisionLiteRtCompiledModelExecutor::EncodeInputMaps(
    const absl::flat_hash_map<std::string, litert::TensorBuffer>& input_maps,
    CancellationLatch* latch) {
  ScopedLatency scoped_total_latency(latency_stats_);

  // Note: `positions_xy` is only required by transformer (ViT) encoders. Single
  // input encoders (e.g. LFM2 VL) do not provide it, so we only validate the
  // mandatory `images` tensor here and feed whatever inputs the caller provides
  // to the matching encoder signature below.
  if (!input_maps.contains(kImages)) {
    return absl::InvalidArgumentError(
        absl::StrCat(kImages, " is not found in the input maps."));
  }

  LITERT_ASSIGN_OR_RETURN(auto images_tensor_type,
                          input_maps.at(kImages).TensorType());
  const auto& images_dimensions = images_tensor_type.Layout().Dimensions();
  RET_CHECK_GE(images_dimensions.size(), 2);
  const int num_patches_from_input = images_dimensions[1];
  ABSL_ASSIGN_OR_RETURN(
      auto encoder_signature_index,
      GetVitSignatureIndex(
          vision_encoder_->GetModel(), vision_executor_properties_,
          num_patches_from_input,
          vision_executor_settings_.GetEncoderSelectedSignatures()));
  std::optional<int> adapter_signature_index;
  if (vision_adapter_ != nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        adapter_signature_index,
        GetVitSignatureIndex(
            vision_adapter_->GetModel(), vision_executor_properties_,
            num_patches_from_input,
            vision_executor_settings_.GetAdapterSelectedSignatures()));
  }

  LITERT_ASSIGN_OR_RETURN(
      auto& encoder_input_buffers,
      vision_encoder_->GetOrCreateInputBuffers(encoder_signature_index));

  LITERT_ASSIGN_OR_RETURN(
      auto encoder_signature,
      vision_encoder_->GetModel().GetSignature(encoder_signature_index));
  ABSL_VLOG(1) << "encoder_signature_index: " << encoder_signature_index
               << " name: " << encoder_signature.Key();
  if (vision_adapter_ != nullptr) {
    LITERT_ASSIGN_OR_RETURN(
        auto adapter_signature,
        vision_adapter_->GetModel().GetSignature(*adapter_signature_index));
    ABSL_VLOG(1) << "adapter_signature_index: " << *adapter_signature_index
                 << " name: " << adapter_signature.Key();
  }

  for (const auto& [key, value] : input_maps) {
    LITERT_ASSIGN_OR_RETURN(auto tensor_type, value.TensorType());
    size_t input_index = 0;
    auto input_index_or = vision_encoder_->GetCompiledModel().FindInputIndex(
        encoder_signature_index, key);
    if (input_index_or.HasValue()) {
      input_index = *input_index_or;
    } else if (vision_encoder_->GetModel().GetNumSignatures() > 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("FindInputIndex failed for signature: ", key));
    }
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
      auto& encoder_output_buffers,
      vision_encoder_->GetOrCreateOutputBuffers(encoder_signature_index));

  ABSL_RETURN_IF_ERROR(RunCompiledModel(
      vision_encoder_->GetCompiledModel(), encoder_signature_index,
      encoder_input_buffers, encoder_output_buffers,
      kVisionEncoderInferenceLatency, latch));

  int num_patches = 0;
  auto mask_index = vision_encoder_->GetCompiledModel().FindOutputIndex(
      encoder_signature_index, kMask);

  if (!mask_index.HasValue()) {
    // If the mask is not in the encoder output, we need to estimate the number
    // of patches from the input image tensor. The positions tensor is not
    // available for single-input encoders (e.g. LFM2 VL).
    if (!vision_executor_properties_.patch_num_shrink_factor.has_value()) {
      return absl::InternalError(
          "patch_num_shrink_factor is not set in the vision executor "
          "properties.");
    }
    const int patch_num_shrink_factor =
        vision_executor_properties_.patch_num_shrink_factor.value();
    // Round up the number of patches so we have at least one patch.
    num_patches = (num_patches_from_input + patch_num_shrink_factor - 1) /
                  patch_num_shrink_factor;
  } else if (const auto positions_it = input_maps.find(kPositionsXy);
             vision_adapter_ == nullptr && positions_it != input_maps.end() &&
             vision_executor_properties_.patch_num_shrink_factor.has_value()) {
    // A fused encoder (no separate adapter) emits a fixed number of soft
    // tokens per slice, and its mask marks valid *patches* rather than tokens.
    // Derive the token count from the padded positions length instead.
    LITERT_ASSIGN_OR_RETURN(auto positions_tensor_type,
                            positions_it->second.TensorType());
    const auto& positions_dims = positions_tensor_type.Layout().Dimensions();
    RET_CHECK_GE(positions_dims.size(), 2);
    const int positions_patches = positions_dims[positions_dims.size() - 2];
    const int patch_num_shrink_factor =
        vision_executor_properties_.patch_num_shrink_factor.value();
    num_patches = (positions_patches + patch_num_shrink_factor - 1) /
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
  const auto& encoder_output_dimensions =
      encoder_output_tensor_type.Layout().Dimensions();
  RET_CHECK_GE(encoder_output_dimensions.size(), 2);
  const int encoder_output_rows =
      encoder_output_dimensions[encoder_output_dimensions.size() - 2];
  const int encoder_output_dim =
      encoder_output_dimensions[encoder_output_dimensions.size() - 1];

  TensorBuffer* output_source_buffer = &encoder_output_buffers[features_index];
  int source_output_rows = encoder_output_rows;
  int output_dim = encoder_output_dim;

  if (vision_adapter_ != nullptr) {
    LITERT_ASSIGN_OR_RETURN(
        auto& adapter_input_buffers,
        vision_adapter_->GetOrCreateInputBuffers(*adapter_signature_index));
    LITERT_ASSIGN_OR_RETURN(
        auto& adapter_output_tensor_buffers,
        vision_adapter_->GetOrCreateOutputBuffers(*adapter_signature_index));
    if (adapter_output_tensor_buffers.size() != 1) {
      return absl::InternalError(
          absl::StrCat("The Vision Adapter model must have exactly one output "
                       "buffer but got ",
                       adapter_output_tensor_buffers.size()));
    }
    // Feed the adapter's `features` input from the encoder output. Any extra
    // adapter inputs are matched by name against the other encoder outputs.
    LITERT_ASSIGN_OR_RETURN(
        auto adapter_signature,
        vision_adapter_->GetModel().GetSignature(*adapter_signature_index));
    const auto& adapter_input_names = adapter_signature.InputNames();
    if (adapter_input_names.empty()) {
      return absl::InvalidArgumentError(
          "The Vision Adapter model must have at least one input.");
    }
    const auto features_input = absl::c_find(adapter_input_names, kFeatures);
    const absl::string_view features_input_name =
        features_input == adapter_input_names.end() ? adapter_input_names[0]
                                                    : *features_input;
    LITERT_ASSIGN_OR_RETURN(auto features_input_index,
                            vision_adapter_->GetCompiledModel().FindInputIndex(
                                *adapter_signature_index, features_input_name));
    LITERT_ASSIGN_OR_RETURN(const size_t encoder_output_num_elements,
                            encoder_output_tensor_type.Layout().NumElements());
    const size_t features_elements_to_copy =
        std::min(encoder_output_num_elements,
                 static_cast<size_t>(num_patches) * encoder_output_dim);
    const size_t copy_bytes = features_elements_to_copy * sizeof(float);
    TensorBuffer& adapter_features_buffer =
        adapter_input_buffers[features_input_index];
    LITERT_ASSIGN_OR_RETURN(const size_t adapter_packed_size,
                            adapter_features_buffer.PackedSize());
    RET_CHECK_LE(copy_bytes, adapter_packed_size);
    {
      LITERT_ASSIGN_OR_RETURN(
          auto dst_lock_and_addr,
          TensorBufferScopedLock::Create<float>(
              adapter_features_buffer, TensorBuffer::LockMode::kWrite));
      if (features_elements_to_copy > 0) {
        LITERT_RETURN_IF_ERROR(
            encoder_output_buffers[features_index].Read<float>(absl::MakeSpan(
                dst_lock_and_addr.second, features_elements_to_copy)));
      }
      if (copy_bytes < adapter_packed_size) {
        std::memset(
            reinterpret_cast<uint8_t*>(dst_lock_and_addr.second) + copy_bytes,
            0, adapter_packed_size - copy_bytes);
      }
    }

    for (const auto& input_name : adapter_input_names) {
      if (input_name == features_input_name) {
        continue;
      }
      auto encoder_output_index =
          vision_encoder_->GetCompiledModel().FindOutputIndex(
              encoder_signature_index, input_name);
      if (!encoder_output_index.HasValue()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Vision adapter input '", input_name,
                         "' has no matching encoder output to feed it."));
      }
      LITERT_ASSIGN_OR_RETURN(
          auto adapter_input_index,
          vision_adapter_->GetCompiledModel().FindInputIndex(
              *adapter_signature_index, input_name));
      LITERT_ASSIGN_OR_RETURN(
          auto encoder_output_span,
          ReferTensorBufferAsSpan<float>(
              encoder_output_buffers[encoder_output_index.Value()]));
      adapter_input_buffers[adapter_input_index].Clear();
      LITERT_RETURN_IF_ERROR(
          adapter_input_buffers[adapter_input_index].Write<float>(
              encoder_output_span));
    }

    ABSL_RETURN_IF_ERROR(RunCompiledModel(
        vision_adapter_->GetCompiledModel(), *adapter_signature_index,
        adapter_input_buffers, adapter_output_tensor_buffers,
        kVisionAdapterInferenceLatency, latch));

    LITERT_ASSIGN_OR_RETURN(auto adapter_output_tensor_type,
                            adapter_output_tensor_buffers[0].TensorType());
    const auto& adapter_output_dimensions =
        adapter_output_tensor_type.Layout().Dimensions();
    RET_CHECK_GE(adapter_output_dimensions.size(), 2);
    output_source_buffer = &adapter_output_tensor_buffers[0];
    source_output_rows =
        adapter_output_dimensions[adapter_output_dimensions.size() - 2];
    output_dim =
        adapter_output_dimensions[adapter_output_dimensions.size() - 1];
  }

  // The embedding size is the last dimension of the output tensor, regardless
  // of whether the model produces a 2-D ([num_tokens, embedding_size]) or 3-D
  // ([batch_size, num_tokens, embedding_size]) tensor. Rows are capped at the
  // output buffer's capacity so a longer patch/feature sequence cannot overrun
  // a fixed-length output.
  const int output_rows = std::min(num_patches, source_output_rows);
  LITERT_ASSIGN_OR_RETURN(
      auto output_tensor,
      CreateTensorBuffer<float>({1, output_rows, output_dim},
                                TensorBufferType::kHostMemory, env_));
  const size_t output_elements =
      static_cast<size_t>(output_rows) * static_cast<size_t>(output_dim);
  if (output_elements > 0) {
    LITERT_ASSIGN_OR_RETURN(auto dst_lock_and_addr,
                            TensorBufferScopedLock::Create<float>(
                                output_tensor, TensorBuffer::LockMode::kWrite));
    LITERT_RETURN_IF_ERROR(output_source_buffer->Read<float>(
        absl::MakeSpan(dst_lock_and_addr.second, output_elements)));
  }

  AccumulateStat(latency_stats_, kVisionNumImagesMetric, int64_t{1});
  AccumulateStat(latency_stats_, kVisionNumPatchesMetric,
                 static_cast<int64_t>(num_patches));

  return ExecutorVisionData(std::move(output_tensor),
                            /*per_layer_embeddings=*/std::nullopt);
}

absl::StatusOr<VisionExecutorProperties>
VisionLiteRtCompiledModelExecutor::GetVisionExecutorProperties() const {
  return vision_executor_properties_;
}

absl::Status VisionLiteRtCompiledModelExecutor::StartProfiling() {
  latency_stats_ = ExecutorStats();
  if (vision_encoder_ != nullptr) {
    StartCompiledModelMetricsCollection(vision_encoder_->GetCompiledModel());
  }
  if (vision_adapter_ != nullptr) {
    StartCompiledModelMetricsCollection(vision_adapter_->GetCompiledModel());
  }
  return absl::OkStatus();
}

absl::StatusOr<ExecutorStats>
VisionLiteRtCompiledModelExecutor::StopProfiling() {
  if (!latency_stats_.has_value()) {
    return absl::FailedPreconditionError("Profiling has not been started.");
  }
  ExecutorStats stats = *std::move(latency_stats_);
  stats.module_name = kVisionModuleName;
  latency_stats_ = std::nullopt;
  if (vision_encoder_ != nullptr) {
    CollectCompiledModelMetrics(vision_encoder_->GetCompiledModel(), stats,
                                kVisionEncoderMetricsPrefix);
  }
  if (vision_adapter_ != nullptr) {
    CollectCompiledModelMetrics(vision_adapter_->GetCompiledModel(), stats,
                                kVisionAdapterMetricsPrefix);
  }
  return stats;
}

}  // namespace litert::lm
