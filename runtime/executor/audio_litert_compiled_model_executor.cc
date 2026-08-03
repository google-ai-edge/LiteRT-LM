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

#include "runtime/executor/audio_litert_compiled_model_executor.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "runtime/components/lora_manager.h"
#include "runtime/components/model_resources.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/audio_executor_utils.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/util/convert_tensor_buffer.h"  // IWYU pragma: keep
#include "runtime/util/status_macros.h"
#include "runtime/util/tensor_buffer_util.h"
#include "tflite/delegates/xnnpack/xnnpack_delegate.h"  // from @litert
#include "tflite/types/half.h"  // from @litert

#if !defined(LITERT_DISABLE_NPU)
#include "litert/cc/options/litert_google_tensor_options.h"  // from @litert
#endif  // !defined(LITERT_DISABLE_NPU)

namespace litert::lm {
namespace {

// Set the default GPU options for the model.
absl::Status SetGpuOptions(const AudioExecutorSettings& executor_settings,
                           litert::GpuOptions& gpu_options) {
#if defined(LITERT_USE_WEBGPU_ACCELERATOR)
  gpu_options.SetBackend(GpuOptions::Backend::kWebGpu);
#endif  // defined(LITERT_USE_WEBGPU_ACCELERATOR)
  gpu_options.EnableConstantTensorSharing(true);
  // Mixed precision setting overrides the activation data type setting. The
  // underlying delegate uses fp32 precision to represent mixed precision, so we
  // set it to fp32 here.
  if (executor_settings.IsMixedPrecisionEnabled()) {
    gpu_options.SetPrecision(GpuOptions::Precision::kFp32);
  } else if (executor_settings.GetActivationDataType().has_value()) {
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
  gpu_options.SetHintFullyDelegatedToSingleDelegate(true);
  gpu_options.EnableInfiniteFloatCapping(true);
  gpu_options.SetNumStepsOfCommandBufferPreparations(2);
  gpu_options.EnableExternalTensorsMode(false);
  gpu_options.SetNumThreadsToUpload(2);
  gpu_options.SetNumThreadsToCompile(1);
  gpu_options.EnableAllowSrcQuantizedFcConvOps(false);
  return absl::OkStatus();
}

// Set the default CPU options for the model.
absl::Status SetCpuOptions(const AudioExecutorSettings& executor_settings,
                           litert::CpuOptions& cpu_options) {
  cpu_options.SetNumThreads(executor_settings.GetNumThreads());
  auto default_xnn_options = TfLiteXNNPackDelegateOptionsDefault();
  cpu_options.SetXNNPackFlags(
      default_xnn_options.flags |
      TFLITE_XNNPACK_DELEGATE_FLAG_DYNAMIC_FULLY_CONNECTED);
  return absl::OkStatus();
}

constexpr std::array<absl::string_view, 3> kAudioInputNames = {
    "audio", "src_inputs", "segment_values"};
constexpr absl::string_view kFeaturesName = "features";
constexpr absl::string_view kMaskName = "mask";
constexpr absl::string_view kMaskOutName = "mask_out";
constexpr absl::string_view kSegmentValuesName = "segment_values";
constexpr absl::string_view kSegmentMaskName = "segment_mask";
constexpr absl::string_view kPrevMaskName = "prev_mask";
constexpr absl::string_view kFeatureStatesNamePattern = "feature_state";

template <typename T>
absl::StatusOr<std::vector<T>> GetDataAsVector(TensorBuffer& tensor_buffer) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor_buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto elements, tensor_type.Layout().NumElements());
  std::vector<T> data(elements);
  LITERT_RETURN_IF_ERROR(tensor_buffer.Read<T>(absl::MakeSpan(data)));
  return data;
}

// Returns the first valid token count from the mask span.
int GetValidCount(absl::Span<const uint8_t> mask) {
  for (int i = mask.size() - 1; i >= 0; --i) {
    if (mask[i] != 0) {
      return i + 1;
    }
  }
  return 0;
}

// Returns the first valid token count from the mask tensor.
absl::StatusOr<int> GetValidCount(const TensorBuffer& mask_buffer) {
  ASSIGN_OR_RETURN(auto mask, GetDataAsVector<uint8_t>(
                                  const_cast<TensorBuffer&>(mask_buffer)));
  return GetValidCount(mask);
}

absl::Status InitializeBuffer(TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(
      auto buffer_lock_and_addr,
      TensorBufferScopedLock::Create(buffer, TensorBuffer::LockMode::kWrite));
  LITERT_ASSIGN_OR_RETURN(auto packed_size, buffer.PackedSize());
  memset(buffer_lock_and_addr.second, 0, packed_size);
  return absl::OkStatus();
}

absl::Status InitializeBuffers(std::vector<TensorBuffer>& buffers) {
  for (auto& buffer : buffers) {
    RETURN_IF_ERROR(InitializeBuffer(buffer));
  }
  return absl::OkStatus();
}

inline int CeilIntDiv(int a, int b) { return (a + b - 1) / b; }

}  // namespace

absl::Status AudioLiteRtCompiledModelExecutor::AudioEncoder::LoadLoRA(
    uint32_t lora_id, const ModelAssets& model_assets) {
  if (lora_manager_ == nullptr) {
    ASSIGN_OR_RETURN(lora_manager_,
                     LoraManager::Create(compiled_model_,
                                         /*signature_name=*/"serving_default"));
  }
  return lora_manager_->LoadLoRA(lora_id, model_assets);
}

absl::Status AudioLiteRtCompiledModelExecutor::AudioEncoder::UseLoRA(
    std::optional<uint32_t> lora_id) {
  if (lora_id.has_value()) {
    if (lora_manager_ == nullptr) {
      return absl::FailedPreconditionError(
          "LoRA manager is not initialized. Please load LoRA first.");
    }
    return lora_manager_->UseLoRA(lora_id.value());
  }

  // TODO: b/515389724 - Support unloading/clearing LoRA buffers from
  // input_buffers_map_ when LoRA is deactivated for subsequent audio calls.
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<AudioContext>> AudioStreamingContext::Clone()
    const {
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>
      new_state_buffers;
  for (auto& [name, buffer] : state_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto new_buffer, buffer.Duplicate());
    new_state_buffers[name] = std::move(new_buffer);
  }
  return std::make_unique<AudioStreamingContext>(std::move(new_state_buffers));
}

absl::StatusOr<
    std::unique_ptr<AudioLiteRtCompiledModelExecutor::AudioStaticEncoder>>
AudioLiteRtCompiledModelExecutor::AudioStaticEncoder::Create(
    const AudioExecutorSettings& executor_settings, Environment& env,
    const Model* absl_nonnull model) {
  auto handler = std::unique_ptr<AudioStaticEncoder>(
      new AudioStaticEncoder(executor_settings, env, model));
  RETURN_IF_ERROR(handler->Initialize());
  return handler;
}

absl::Status
AudioLiteRtCompiledModelExecutor::AudioStaticEncoder::Initialize() {
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  auto weight_cache_file = executor_settings_.GetWeightCacheFile(
      absl::StrCat(AudioExecutorSettings::kStaticEncoderName,
                   ExecutorSettingsBase::kXnnpackCacheSuffix),
      /*check_and_clean=*/true);
  if (executor_settings_.GetBackend() == Backend::GPU) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
    ASSIGN_OR_RETURN(
        const auto cache_files,
        GetGpuModelCacheData(executor_settings_,
                             AudioExecutorSettings::kStaticEncoderName));
    RETURN_IF_ERROR(SetGpuOptions(executor_settings_, gpu_options));
    RETURN_IF_ERROR(SetGpuCacheOptions(
        cache_files.weight_cache_file, cache_files.program_cache_file,
        cache_files.cache_key,
        /*logging_prefix=*/AudioExecutorSettings::kStaticEncoderName,
        /*cache_compiled_shaders_only=*/false, gpu_options));
    options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
  } else if (executor_settings_.GetBackend() == Backend::CPU) {
    LITERT_ASSIGN_OR_RETURN(auto& cpu_options, options.GetCpuOptions());
    RETURN_IF_ERROR(SetCpuOptions(executor_settings_, cpu_options));
    RETURN_IF_ERROR(SetCpuCacheOptions(
        weight_cache_file, AudioExecutorSettings::kEncoderName, cpu_options));

    options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
#if !defined(LITERT_DISABLE_NPU)
  } else if (executor_settings_.GetBackend() == Backend::NPU) {
    LITERT_ASSIGN_OR_RETURN(auto& google_tensor_options,
                            options.GetGoogleTensorOptions());
    google_tensor_options.SetPerformanceMode(
        google_tensor::GoogleTensorOptions::PerformanceMode::kBurst);
    options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
#endif  // !defined(LITERT_DISABLE_NPU)
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported backend for AudioStaticEncoder: ",
                     executor_settings_.GetBackend()));
  }

  LITERT_ASSIGN_OR_RETURN(compiled_model_,
                          CompiledModel::Create(env_, model_.Get(), options));
  LITERT_ASSIGN_OR_RETURN(auto signatures, model_.GetSignatures());
  if (signatures.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("The Audio Static Encoder model must have exactly one "
                     "signature but got ",
                     signatures.size()));
  }
  LITERT_ASSIGN_OR_RETURN(auto signature, model_.GetSignature(0));

  // Initialize the input buffers.
  LITERT_ASSIGN_OR_RETURN(auto input_buffers,
                          compiled_model_.CreateInputBuffers(
                              /*signature_index=*/0));
  LITERT_RETURN_IF_ERROR(InitializeBuffers(input_buffers));
  input_names_.reserve(signature.InputNames().size());
  for (int i = 0; i < signature.InputNames().size(); ++i) {
    std::string input_name = std::string(signature.InputNames()[i]);
    input_names_.push_back(input_name);
    absl::string_view input_name_view = input_names_[i];
    input_buffers_map_[input_name_view] = std::move(input_buffers[i]);
  }

  // Get pointers to specific buffers after the map is fully populated.
  std::string src_inputs_name = "";
  for (const auto& input_name : kAudioInputNames) {
    if (input_buffers_map_.contains(input_name)) {
      src_inputs_name = input_name;
      break;
    }
  }
  if (src_inputs_name.empty()) {
    return absl::InvalidArgumentError(
        "The Audio Static Encoder model must have a src_inputs or audio "
        "input buffer.");
  }
  if (input_buffers_map_.contains(kMaskName)) {
    input_mask_buffer_ = &input_buffers_map_[kMaskName];
  }
  spectrogram_buffer_ = &input_buffers_map_[src_inputs_name];

  // Initialize the output buffers.
  LITERT_ASSIGN_OR_RETURN(auto output_buffers,
                          compiled_model_.CreateOutputBuffers(
                              /*signature_index=*/0));
  LITERT_RETURN_IF_ERROR(InitializeBuffers(output_buffers));
  output_names_.reserve(signature.OutputNames().size());
  for (int i = 0; i < signature.OutputNames().size(); ++i) {
    std::string output_name = std::string(signature.OutputNames()[i]);
    output_names_.push_back(output_name);
    absl::string_view output_name_view = output_names_[i];
    output_buffers_map_[output_name_view] = std::move(output_buffers[i]);
  }
  // Get pointers to specific buffers after the map is fully populated.
  if (!output_buffers_map_.contains(kFeaturesName)) {
    return absl::InvalidArgumentError(
        "The Audio Static Encoder model must have a features output buffer.");
  }
  if (output_buffers_map_.contains(kMaskName)) {
    output_mask_buffer_ = &output_buffers_map_[kMaskName];
  } else if (output_buffers_map_.contains(kMaskOutName)) {
    output_mask_buffer_ = &output_buffers_map_[kMaskOutName];
  }
  output_features_buffer_ = &output_buffers_map_[kFeaturesName];
  return absl::OkStatus();
}

absl::Status
AudioLiteRtCompiledModelExecutor::AudioStaticEncoder::ClearInputBuffers() {
  for (auto& [input_name, input_buffer] : input_buffers_map_) {
    LITERT_ASSIGN_OR_RETURN(auto buffer_lock_and_addr,
                            TensorBufferScopedLock::Create(
                                input_buffer, TensorBuffer::LockMode::kWrite));
    LITERT_ASSIGN_OR_RETURN(auto packed_size, input_buffer.PackedSize());
    memset(buffer_lock_and_addr.second, 0, packed_size);
  }
  return absl::OkStatus();
}

absl::StatusOr<
    std::unique_ptr<AudioLiteRtCompiledModelExecutor::AudioStreamingEncoder>>
AudioLiteRtCompiledModelExecutor::AudioStreamingEncoder::Create(
    const AudioExecutorSettings& executor_settings, Environment& env,
    const Model* absl_nonnull model) {
  auto handler = std::unique_ptr<AudioStreamingEncoder>(
      new AudioStreamingEncoder(executor_settings, env, model));
  RETURN_IF_ERROR(handler->Initialize());
  return handler;
}

absl::Status
AudioLiteRtCompiledModelExecutor::AudioStreamingEncoder::Initialize() {
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  auto weight_cache_file = executor_settings_.GetWeightCacheFile(
      absl::StrCat(AudioExecutorSettings::kStreamingEncoderName,
                   ExecutorSettingsBase::kXnnpackCacheSuffix),
      /*check_and_clean=*/true);
  if (executor_settings_.GetBackend() == Backend::GPU) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
    ASSIGN_OR_RETURN(
        const auto cache_files,
        GetGpuModelCacheData(executor_settings_,
                             AudioExecutorSettings::kStreamingEncoderName));
    RETURN_IF_ERROR(SetGpuOptions(executor_settings_, gpu_options));
    RETURN_IF_ERROR(SetGpuCacheOptions(
        cache_files.weight_cache_file, cache_files.program_cache_file,
        cache_files.cache_key,
        /*logging_prefix=*/AudioExecutorSettings::kStreamingEncoderName,
        /*cache_compiled_shaders_only=*/false, gpu_options));
    options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
  } else if (executor_settings_.GetBackend() == Backend::CPU) {
    LITERT_ASSIGN_OR_RETURN(auto& cpu_options, options.GetCpuOptions());
    RETURN_IF_ERROR(SetCpuOptions(executor_settings_, cpu_options));
    RETURN_IF_ERROR(SetCpuCacheOptions(
        weight_cache_file, AudioExecutorSettings::kEncoderName, cpu_options));
    options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
#if !defined(LITERT_DISABLE_NPU)
  } else if (executor_settings_.GetBackend() == Backend::NPU) {
    LITERT_ASSIGN_OR_RETURN(auto& google_tensor_options,
                            options.GetGoogleTensorOptions());
    google_tensor_options.SetPerformanceMode(
        google_tensor::GoogleTensorOptions::PerformanceMode::kBurst);
    options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
#endif  // !defined(LITERT_DISABLE_NPU)
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported backend for AudioEncoder: ",
                     executor_settings_.GetBackend()));
  }

  LITERT_ASSIGN_OR_RETURN(compiled_model_,
                          CompiledModel::Create(env_, model_.Get(), options));
  LITERT_ASSIGN_OR_RETURN(auto signatures, model_.GetSignatures());
  if (signatures.size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The Audio Encoder model must have exactly one signature but got ",
        signatures.size()));
  }

  LITERT_ASSIGN_OR_RETURN(auto signature, model_.GetSignature(0));

  // Initialize the input buffers.
  LITERT_ASSIGN_OR_RETURN(auto input_buffers,
                          compiled_model_.CreateInputBuffers(
                              /*signature_index=*/0));
  LITERT_RETURN_IF_ERROR(InitializeBuffers(input_buffers));
  input_names_.reserve(signature.InputNames().size());
  for (int i = 0; i < signature.InputNames().size(); ++i) {
    std::string input_name = std::string(signature.InputNames()[i]);
    input_names_.push_back(input_name);
    absl::string_view input_name_view = input_names_[i];
    input_buffers_map_[input_name_view] = std::move(input_buffers[i]);
  }

  if (input_buffers_map_.contains(kSegmentMaskName)) {
    input_mask_buffer_ = &input_buffers_map_[kSegmentMaskName];
  }
  std::string src_inputs_name = "";
  for (const auto& input_name : kAudioInputNames) {
    if (input_buffers_map_.contains(input_name)) {
      src_inputs_name = input_name;
      break;
    }
  }
  if (src_inputs_name.empty()) {
    return absl::InvalidArgumentError(
        "The Audio Streaming Encoder model must have an audio input buffer "
        "with name in " +
        absl::StrJoin(kAudioInputNames, ", "));
  }
  spectrogram_buffer_ = &input_buffers_map_[src_inputs_name];

  // Initialize the output buffers.
  LITERT_ASSIGN_OR_RETURN(auto output_buffers,
                          compiled_model_.CreateOutputBuffers(
                              /*signature_index=*/0));
  LITERT_RETURN_IF_ERROR(InitializeBuffers(output_buffers));
  output_names_.reserve(signature.OutputNames().size());
  for (int i = 0; i < signature.OutputNames().size(); ++i) {
    std::string output_name = std::string(signature.OutputNames()[i]);
    output_names_.push_back(output_name);
    absl::string_view output_name_view = output_names_[i];
    output_buffers_map_[output_name_view] = std::move(output_buffers[i]);
  }
  // Get pointers to specific buffers after the map is fully populated.
  if (!output_buffers_map_.contains(kFeaturesName)) {
    return absl::InvalidArgumentError(
        "The Audio Streaming Encoder model must have a features output "
        "buffer.");
  }
  if (output_buffers_map_.contains(kMaskName)) {
    output_mask_buffer_ = &output_buffers_map_[kMaskName];
  }
  output_features_buffer_ = &output_buffers_map_[kFeaturesName];

  // Get the feature states tensor type and use it to get the overlap size.
  std::string feature_states_name =
      absl::StrCat(kFeatureStatesNamePattern, "_0");
  if (input_buffers_map_.contains(feature_states_name)) {
    LITERT_ASSIGN_OR_RETURN(
        auto feature_states_tensor_type,
        input_buffers_map_[feature_states_name].TensorType());
    // The overlap size is the number of elements in the feature states tensor,
    // which is 3 for gemma3n.
    LITERT_ASSIGN_OR_RETURN(overlap_size_,
                            feature_states_tensor_type.Layout().NumElements());
  }
  // Initialize the previous mask buffer to all ones.
  if (input_buffers_map_.contains(kPrevMaskName)) {
    LITERT_ASSIGN_OR_RETURN(auto prev_mask_type,
                            input_buffers_map_[kPrevMaskName].TensorType());
    LITERT_ASSIGN_OR_RETURN(int prev_mask_size,
                            prev_mask_type.Layout().NumElements());
    input_buffers_map_[kPrevMaskName].Write<uint8_t>(
        std::vector<uint8_t>(prev_mask_size, 1));
  }

  return absl::OkStatus();
}

absl::Status AudioLiteRtCompiledModelExecutor::AudioStreamingEncoder::
    SwapInternalStateBuffers() {
  std::vector<absl::string_view> all_input_names(input_names_.begin(),
                                                 input_names_.end());
  for (const auto& input_name : all_input_names) {
    if (output_buffers_map_.contains(input_name)) {
      std::swap(input_buffers_map_[input_name],
                output_buffers_map_[input_name]);
    }
  }
  return absl::OkStatus();
}

absl::Status
AudioLiteRtCompiledModelExecutor::AudioStreamingEncoder::ClearInputBuffers() {
  {
    LITERT_ASSIGN_OR_RETURN(
        auto buffer_lock_and_addr,
        TensorBufferScopedLock::Create(GetMutableInputSpectrogramBuffer(),
                                       TensorBuffer::LockMode::kWrite));
    LITERT_ASSIGN_OR_RETURN(auto packed_size,
                            GetInputSpectrogramBuffer().PackedSize());
    memset(buffer_lock_and_addr.second, 0, packed_size);
  }
  if (GetMutableInputMaskBuffer() != nullptr) {
    LITERT_ASSIGN_OR_RETURN(
        auto buffer_lock_and_addr,
        TensorBufferScopedLock::Create(*GetMutableInputMaskBuffer(),
                                       TensorBuffer::LockMode::kWrite));
    LITERT_ASSIGN_OR_RETURN(auto packed_size,
                            GetInputMaskBuffer()->PackedSize());
    memset(buffer_lock_and_addr.second, 0, packed_size);
  }
  return absl::OkStatus();
}

absl::Status AudioLiteRtCompiledModelExecutor::AudioStreamingEncoder::Reset() {
  for (auto& [input_name, input_buffer] : input_buffers_map_) {
    LITERT_ASSIGN_OR_RETURN(auto buffer_lock_and_addr,
                            TensorBufferScopedLock::Create(
                                input_buffer, TensorBuffer::LockMode::kWrite));
    LITERT_ASSIGN_OR_RETURN(auto packed_size, input_buffer.PackedSize());
    if (input_name == kPrevMaskName) {
      for (int i = 0; i < packed_size; ++i) {
        auto* mask_ptr = static_cast<bool*>(buffer_lock_and_addr.second);
        mask_ptr[i] = true;
      }
    } else {
      memset(buffer_lock_and_addr.second, 0, packed_size);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<AudioLiteRtCompiledModelExecutor::AudioAdapter>>
AudioLiteRtCompiledModelExecutor::AudioAdapter::Create(
    const AudioExecutorSettings& executor_settings, Environment& env,
    const Model* absl_nonnull model) {
  auto handler = std::unique_ptr<AudioAdapter>(
      new AudioAdapter(executor_settings, env, model));
  RETURN_IF_ERROR(handler->Initialize());
  return handler;
}

absl::Status AudioLiteRtCompiledModelExecutor::AudioAdapter::Initialize() {
  LITERT_ASSIGN_OR_RETURN(auto options, Options::Create());
  auto weight_cache_file = executor_settings_.GetWeightCacheFile(
      absl::StrCat(AudioExecutorSettings::kAdapterName,
                   ExecutorSettingsBase::kXnnpackCacheSuffix),
      /*check_and_clean=*/true);
  if (executor_settings_.GetBackend() == Backend::GPU) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
    ASSIGN_OR_RETURN(const auto cache_files,
                     GetGpuModelCacheData(executor_settings_,
                                          AudioExecutorSettings::kAdapterName));
    RETURN_IF_ERROR(SetGpuCacheOptions(
        cache_files.weight_cache_file, cache_files.program_cache_file,
        cache_files.cache_key,
        /*logging_prefix=*/AudioExecutorSettings::kAdapterName,
        /*cache_compiled_shaders_only=*/false, gpu_options));

    gpu_options.EnableConstantTensorSharing(true);
    gpu_options.SetPrecision(GpuOptions::Precision::kFp32);
    gpu_options.SetPreferTextureWeights(true);
#if defined(LITERT_USE_WEBGPU_ACCELERATOR)
    gpu_options.SetBackend(GpuOptions::Backend::kWebGpu);
#endif  // defined(LITERT_USE_WEBGPU_ACCELERATOR)
    options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
  } else if (executor_settings_.GetBackend() == Backend::CPU) {
    LITERT_ASSIGN_OR_RETURN(auto& cpu_options, options.GetCpuOptions());
    RETURN_IF_ERROR(SetCpuOptions(executor_settings_, cpu_options));

    RETURN_IF_ERROR(SetCpuCacheOptions(
        weight_cache_file, AudioExecutorSettings::kAdapterName, cpu_options));

    options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
#if !defined(LITERT_DISABLE_NPU)
  } else if (executor_settings_.GetBackend() == Backend::NPU) {
    LITERT_ASSIGN_OR_RETURN(auto& google_tensor_options,
                            options.GetGoogleTensorOptions());
    google_tensor_options.SetPerformanceMode(
        google_tensor::GoogleTensorOptions::PerformanceMode::kBurst);
    options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
#endif  // !defined(LITERT_DISABLE_NPU)
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported backend for AudioAdapter: ",
                     executor_settings_.GetBackend()));
  }

  LITERT_ASSIGN_OR_RETURN(compiled_model_,
                          CompiledModel::Create(env_, model_.Get(), options));
  LITERT_ASSIGN_OR_RETURN(auto signatures, model_.GetSignatures());
  if (signatures.size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The Audio Adapter model must have exactly one signature but got ",
        signatures.size()));
  }
  LITERT_ASSIGN_OR_RETURN(input_buffers_, compiled_model_.CreateInputBuffers(
                                              /*signature_index=*/0));
  if (input_buffers_.size() != 1 && input_buffers_.size() != 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The Audio Adapter model must have 1 or 2 input buffers but got ",
        input_buffers_.size()));
  }
  LITERT_ASSIGN_OR_RETURN(output_buffers_, compiled_model_.CreateOutputBuffers(
                                               /*signature_index=*/0));
  LITERT_RETURN_IF_ERROR(InitializeBuffers(input_buffers_));
  LITERT_RETURN_IF_ERROR(InitializeBuffers(output_buffers_));
  if (output_buffers_.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("The Audio Adapter model must have exactly one output "
                     "buffer but got ",
                     output_buffers_.size()));
  }

  LITERT_ASSIGN_OR_RETURN(auto signature, model_.GetSignature(0));
  for (int i = 0; i < signature.InputNames().size(); ++i) {
    if (absl::StrContains(signature.InputNames()[i], kFeaturesName)) {
      features_buffer_ = &input_buffers_[i];
    } else if (absl::StrContains(signature.InputNames()[i], kMaskName)) {
      mask_buffer_ = &input_buffers_[i];
    }
  }
  if (features_buffer_ == nullptr) {
    return absl::InvalidArgumentError(
        "The Audio Adapter model must have a features input buffer.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<AudioLiteRtCompiledModelExecutor>>
AudioLiteRtCompiledModelExecutor::Create(
    AudioExecutorSettings executor_settings, Environment& env) {
  if (executor_settings.GetMaxSequenceLength() > 0) {
    ABSL_LOG(INFO) << "Max sequence length is not used for "
                      "AudioLiteRtCompiledModelExecutor, "
                      "which can handle variable length input.";
  }
  LITERT_ASSIGN_OR_RETURN(
      auto resources,
      BuildLiteRtCompiledModelResources(executor_settings.GetModelAssets()));
  ASSIGN_OR_RETURN(auto audio_encoder_model,
                   resources->GetTFLiteModel(ModelType::kTfLiteAudioEncoderHw));
  auto audio_adapter_model_or =
      resources->GetTFLiteModel(ModelType::kTfLiteAudioAdapter);
  std::unique_ptr<AudioEncoder> audio_encoder;
  LITERT_ASSIGN_OR_RETURN(auto encoder_signature,
                          audio_encoder_model->GetSignature(0));
  LITERT_ASSIGN_OR_RETURN(
      auto executor_properties,
      GetAudioExecutorPropertiesFromModelResources(*resources));
  if (executor_properties.is_streaming_model) {
    ASSIGN_OR_RETURN(audio_encoder,
                     AudioStreamingEncoder::Create(executor_settings, env,
                                                   audio_encoder_model));
  } else {
    ASSIGN_OR_RETURN(audio_encoder,
                     AudioStaticEncoder::Create(executor_settings, env,
                                                audio_encoder_model));
  }
  std::unique_ptr<AudioAdapter> audio_adapter;
  if (audio_adapter_model_or.ok() && *audio_adapter_model_or != nullptr) {
    ASSIGN_OR_RETURN(
        audio_adapter,
        AudioAdapter::Create(executor_settings, env, *audio_adapter_model_or));
  } else {
    ABSL_LOG(INFO) << "Audio adapter model is not found. Audio encoder output "
                      "will be used directly.";
  }
  int sequence_length = 0;
  if (audio_encoder->GetInputMaskBuffer() != nullptr) {
    LITERT_ASSIGN_OR_RETURN(auto mask_tensor_type,
                            audio_encoder->GetInputMaskBuffer()->TensorType());
    LITERT_ASSIGN_OR_RETURN(sequence_length,
                            mask_tensor_type.Layout().NumElements());
  } else {
    LITERT_ASSIGN_OR_RETURN(
        auto spectrogram_tensor_type,
        audio_encoder->GetInputSpectrogramBuffer().TensorType());
    const auto& dims = spectrogram_tensor_type.Layout().Dimensions();
    if (dims.size() < 2) {
      return absl::InvalidArgumentError(
          "Spectrogram tensor must have at least 2 dimensions");
    }
    sequence_length = dims[dims.size() - 2];
  }
  LITERT_ASSIGN_OR_RETURN(
      auto spectrogram_tensor_type,
      audio_encoder->GetInputSpectrogramBuffer().TensorType());
  const int spectrogram_feature_dimensions =
      spectrogram_tensor_type.Layout().Dimensions().back();
  int audio_embedding_dimensions;
  if (audio_adapter != nullptr) {
    LITERT_ASSIGN_OR_RETURN(auto adapter_output_tensor_type,
                            audio_adapter->GetOutputBuffers()[0].TensorType());
    const auto dims = adapter_output_tensor_type.Layout().Dimensions();
    audio_embedding_dimensions = dims.back();
  } else {
    LITERT_ASSIGN_OR_RETURN(
        auto encoder_output_tensor_type,
        audio_encoder->GetOutputFeaturesBuffer().TensorType());
    const auto dims = encoder_output_tensor_type.Layout().Dimensions();
    audio_embedding_dimensions = dims.back();
  }
  const int encoder_shrinking_factor = executor_properties.audio_shrink_factor;
  if (!executor_properties.is_streaming_model) {
    if (audio_adapter != nullptr) {
      if (audio_encoder->GetOutputBuffersMap().size() !=
          audio_adapter->GetInputBuffers().size()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "The number of output buffers of the audio encoder must be equal "
            "to the number of input buffers of the audio adapter, but got ",
            audio_encoder->GetOutputBuffersMap().size(), " and ",
            audio_adapter->GetInputBuffers().size()));
      }
    }
  }

  if (audio_adapter != nullptr) {
    // Make the audio adapter take the audio encoder's mask (if available) and
    // features as input.
    if (audio_adapter->GetMaskBuffer() != nullptr) {
      if (audio_encoder->GetOutputMaskBuffer() == nullptr) {
        return absl::InvalidArgumentError(
            "Audio adapter expects a mask input, but audio encoder does not "
            "provide a mask output.");
      }
      LITERT_ASSIGN_OR_RETURN(
          auto encoder_mask_tensor,
          audio_encoder->GetOutputMaskBuffer()->Duplicate());
      *audio_adapter->GetMutableMaskBuffer() = std::move(encoder_mask_tensor);
    }

    LITERT_ASSIGN_OR_RETURN(
        auto encoder_features_type,
        audio_encoder->GetOutputFeaturesBuffer().TensorType());

    LITERT_ASSIGN_OR_RETURN(auto adapter_features_type,
                            audio_adapter->GetFeaturesBuffer().TensorType());

    if (encoder_features_type.ElementType() == litert::ElementType::Float16 &&
        adapter_features_type.ElementType() == litert::ElementType::Float32) {
      // Handled in EncodeInternal by conversion.
    } else if (encoder_features_type.ElementType() ==
               adapter_features_type.ElementType()) {
      LITERT_ASSIGN_OR_RETURN(
          auto encoder_features_tensor,
          audio_encoder->GetMutableOutputFeaturesBuffer().Duplicate());
      audio_adapter->GetMutableFeaturesBuffer() =
          std::move(encoder_features_tensor);
    } else {
      ABSL_LOG(ERROR) << "Unsupported type mismatch between audio encoder ("
                      << static_cast<int>(encoder_features_type.ElementType())
                      << ") and audio adapter ("
                      << static_cast<int>(adapter_features_type.ElementType())
                      << ")";
      return absl::InvalidArgumentError(
          "Unsupported type mismatch between audio encoder and adapter");
    }
  }
  ABSL_LOG(INFO) << "AudioLiteRtCompiledModelExecutor created with "
                    "encoder_shrinking_factor: "
                 << encoder_shrinking_factor;
  return absl::WrapUnique(new AudioLiteRtCompiledModelExecutor(
      std::move(executor_settings), std::move(executor_properties), env,
      std::move(resources), std::move(audio_encoder), std::move(audio_adapter),
      sequence_length, spectrogram_feature_dimensions,
      audio_embedding_dimensions, encoder_shrinking_factor));
}

absl::StatusOr<int> AudioLiteRtCompiledModelExecutor::EncodeInternal(
    absl::Span<float> spectrogram_tensor, absl::Span<uint8_t> spectrogram_mask,
    absl::Span<float> audio_embeddings) {
  RETURN_IF_ERROR(audio_encoder_->ClearInputBuffers());
  auto& input_buffer = audio_encoder_->GetMutableInputSpectrogramBuffer();
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, input_buffer.TensorType());
  if (tensor_type.ElementType() == litert::ElementType::Float16) {
    LITERT_ASSIGN_OR_RETURN(auto buffer_lock_and_addr,
                            TensorBufferScopedLock::Create<tflite::half>(
                                input_buffer, TensorBuffer::LockMode::kWrite));
    tflite::half* half_data = buffer_lock_and_addr.second;
    for (size_t i = 0; i < spectrogram_tensor.size(); ++i) {
      half_data[i] = static_cast<tflite::half>(spectrogram_tensor[i]);
    }
  } else {
    LITERT_RETURN_IF_ERROR(input_buffer.Write<float>(spectrogram_tensor));
  }
  if (audio_encoder_->GetMutableInputMaskBuffer() != nullptr) {
    LITERT_RETURN_IF_ERROR(
        audio_encoder_->GetMutableInputMaskBuffer()->Write<uint8_t>(
            spectrogram_mask));
  }

  auto& input_buffers_map = audio_encoder_->GetMutableInputBuffersMap();
  if (audio_encoder_->GetMutableLoraManager() != nullptr) {
    auto current_lora_id =
        audio_encoder_->GetMutableLoraManager()->GetCurrentLoRAId();
    if (current_lora_id.has_value()) {
      ASSIGN_OR_RETURN(
          auto lora_buffers,
          audio_encoder_->GetMutableLoraManager()->GetLoRABuffers());
      for (auto& [name, buffer] : lora_buffers) {
        input_buffers_map[name] = std::move(buffer);
      }
    }
  }

  LITERT_RETURN_IF_ERROR(audio_encoder_->GetMutableCompiledModel().Run(
      input_buffers_map, audio_encoder_->GetMutableOutputBuffersMap()));

  int chunk_valid_tokens = 0;
  if (audio_encoder_->GetOutputMaskBuffer() != nullptr) {
    ASSIGN_OR_RETURN(chunk_valid_tokens,
                     GetValidCount(*audio_encoder_->GetOutputMaskBuffer()));
  } else {
    int input_valid_tokens = GetValidCount(spectrogram_mask);
    chunk_valid_tokens =
        CeilIntDiv(input_valid_tokens, encoder_shrinking_factor_);
  }
  auto& encoder_output = audio_encoder_->GetMutableOutputFeaturesBuffer();

  if (audio_adapter_ != nullptr) {
    auto& adapter_input = audio_adapter_->GetMutableFeaturesBuffer();

    LITERT_ASSIGN_OR_RETURN(auto encoder_type, encoder_output.TensorType());
    LITERT_ASSIGN_OR_RETURN(auto adapter_type, adapter_input.TensorType());

    if (encoder_type.ElementType() == litert::ElementType::Float16 &&
        adapter_type.ElementType() == litert::ElementType::Float32) {
      LITERT_ASSIGN_OR_RETURN(
          auto encoder_lock_and_addr,
          TensorBufferScopedLock::Create<const tflite::half>(
              encoder_output, TensorBuffer::LockMode::kRead));
      const tflite::half* encoder_data = encoder_lock_and_addr.second;

      LITERT_ASSIGN_OR_RETURN(
          auto adapter_lock_and_addr,
          TensorBufferScopedLock::Create<float>(
              adapter_input, TensorBuffer::LockMode::kWrite));
      float* adapter_data = adapter_lock_and_addr.second;

      LITERT_ASSIGN_OR_RETURN(auto elements,
                              encoder_type.Layout().NumElements());

      for (size_t i = 0; i < elements; ++i) {
        adapter_data[i] = static_cast<float>(encoder_data[i]);
      }
    }

    LITERT_RETURN_IF_ERROR(audio_adapter_->GetMutableCompiledModel().Run(
        audio_adapter_->GetMutableInputBuffers(),
        audio_adapter_->GetMutableOutputBuffers()));

    LITERT_RETURN_IF_ERROR(
        audio_adapter_->GetMutableOutputBuffers()[0].Read<float>(
            absl::MakeSpan(audio_embeddings.data(),
                           chunk_valid_tokens * audio_embedding_dimensions_)));
  } else {
    LITERT_ASSIGN_OR_RETURN(auto encoder_type, encoder_output.TensorType());
    if (encoder_type.ElementType() == litert::ElementType::Float16) {
      LITERT_ASSIGN_OR_RETURN(
          auto encoder_lock_and_addr,
          TensorBufferScopedLock::Create<const tflite::half>(
              encoder_output, TensorBuffer::LockMode::kRead));
      const tflite::half* encoder_data = encoder_lock_and_addr.second;
      const size_t total_elements =
          chunk_valid_tokens * audio_embedding_dimensions_;
      for (size_t i = 0; i < total_elements; ++i) {
        audio_embeddings[i] = static_cast<float>(encoder_data[i]);
      }
    } else {
      LITERT_RETURN_IF_ERROR(encoder_output.Read<float>(
          absl::MakeSpan(audio_embeddings.data(),
                         chunk_valid_tokens * audio_embedding_dimensions_)));
    }
  }

  if (audio_encoder_->IsStreaming()) {
    RETURN_IF_ERROR(
        reinterpret_cast<AudioStreamingEncoder*>(audio_encoder_.get())
            ->SwapInternalStateBuffers());
  }
  return chunk_valid_tokens;
}

absl::StatusOr<ExecutorAudioData> AudioLiteRtCompiledModelExecutor::Encode(
    const TensorBuffer& spectrogram_tensor,
    const TensorBuffer& spectrogram_mask) {
  LITERT_ASSIGN_OR_RETURN(auto spectrogram_tensor_type,
                          spectrogram_tensor.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto mask_tensor_type, spectrogram_mask.TensorType());

  const auto& spec_dims = spectrogram_tensor_type.Layout().Dimensions();
  const auto& mask_dims = mask_tensor_type.Layout().Dimensions();

  if (spec_dims.size() < 2) {
    return absl::InvalidArgumentError(
        "Spectrogram tensor must have at least 2 dimensions");
  }
  if (mask_dims.empty()) {
    return absl::InvalidArgumentError(
        "Mask tensor must have at least 1 dimension");
  }

  int spec_seq_len = spec_dims[spec_dims.size() - 2];
  int spec_feature_dim = spec_dims.back();
  int mask_seq_len = mask_dims[mask_dims.size() - 1];

  if (spec_seq_len != mask_seq_len) {
    return absl::InvalidArgumentError(
        absl::StrCat("Spectrogram sequence length (", spec_seq_len,
                     ") must match mask sequence length (", mask_seq_len, ")"));
  }

  if (spec_feature_dim != spectrogram_feature_dimensions_) {
    return absl::InvalidArgumentError(
        absl::StrCat("Spectrogram feature dimension (", spec_feature_dim,
                     ") must match model expectation (",
                     spectrogram_feature_dimensions_, ")"));
  }

  ASSIGN_OR_RETURN(int input_sequence_length, GetValidCount(spectrogram_mask));
  LITERT_ASSIGN_OR_RETURN(
      auto spectrogram_host_buffer,
      GetDataAsVector<float>(const_cast<TensorBuffer&>(spectrogram_tensor)));

  int feature_dim = spectrogram_feature_dimensions_;
  std::vector<float> audio_embeddings;
  int total_valid_tokens = 0;

  // Default to non-streaming mode.
  int window_size = sequence_length_;
  int overlap_size = 0;
  int stride = window_size;

  if (audio_encoder_->IsStreaming()) {
    window_size = executor_properties_.streaming_chunk_size;
    overlap_size = executor_properties_.streaming_chunk_overlap_size;
    stride = window_size - overlap_size;
  }

  if (stride <= 0) {
    return absl::InternalError(
        "Invalid stride size (window_size <= overlap_size).");
  }

  int total_frames = input_sequence_length;
  // At least one chunk.
  int N = 1;
  if (total_frames > window_size) {
    N = CeilIntDiv(total_frames - overlap_size, stride);
  }
  int chunk_max_tokens = CeilIntDiv(window_size, encoder_shrinking_factor_);
  int max_tokens = N * chunk_max_tokens;
  audio_embeddings.resize(max_tokens * audio_embedding_dimensions_);

  LITERT_ASSIGN_OR_RETURN(
      auto spectrogram_mask_host_buffer,
      GetDataAsVector<uint8_t>(const_cast<TensorBuffer&>(spectrogram_mask)));

  int pos = 0;
  // At least one chunk is guaranteed to be processed in the first iteration,
  // and prevent the last chunk only containing overlapped information.
  while (pos == 0 || pos + overlap_size < total_frames) {
    int chunk_len = std::min(window_size, total_frames - pos);
    auto spectrogram_slice =
        absl::MakeSpan(spectrogram_host_buffer)
            .subspan(pos * feature_dim, chunk_len * feature_dim);

    auto spectrogram_mask_slice =
        absl::MakeSpan(spectrogram_mask_host_buffer).subspan(pos, chunk_len);

    int chunk_max_tokens = CeilIntDiv(chunk_len, encoder_shrinking_factor_);
    auto audio_embeddings_slice =
        absl::MakeSpan(audio_embeddings)
            .subspan(total_valid_tokens * audio_embedding_dimensions_,
                     chunk_max_tokens * audio_embedding_dimensions_);

    ASSIGN_OR_RETURN(int chunk_valid_tokens,
                     EncodeInternal(spectrogram_slice, spectrogram_mask_slice,
                                    audio_embeddings_slice));

    total_valid_tokens += chunk_valid_tokens;
    pos += stride;
  }

  // Create the final audio embeddings tensor.
  int buffer_tokens = std::max(1, total_valid_tokens);
  RankedTensorType audio_embeddings_tensor_type(
      GetElementType<float>(),
      Layout(Dimensions({1, buffer_tokens, audio_embedding_dimensions_})));
  LITERT_ASSIGN_OR_RETURN(
      auto audio_embeddings_tensor,
      TensorBuffer::CreateManaged(
          env_, TensorBufferType::kHostMemory, audio_embeddings_tensor_type,
          buffer_tokens * audio_embedding_dimensions_ * sizeof(float)));
  LITERT_RETURN_IF_ERROR(InitializeBuffer(audio_embeddings_tensor));
  LITERT_RETURN_IF_ERROR(audio_embeddings_tensor.Write<float>(
      absl::MakeSpan(audio_embeddings)
          .subspan(0, total_valid_tokens * audio_embedding_dimensions_)));
  ExecutorAudioData audio_data;
  audio_data.SetEmbeddings(std::move(audio_embeddings_tensor));
  audio_data.SetValidTokens(total_valid_tokens);
  return audio_data;
}

absl::StatusOr<ExecutorAudioData> AudioLiteRtCompiledModelExecutor::Encode(
    const TensorBuffer& spectrogram_tensor) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, spectrogram_tensor.TensorType());
  auto dimensions = tensor_type.Layout().Dimensions();
  if (dimensions.size() < 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Spectrogram tensor must have at least 2 dimensions, but got ",
        dimensions.size()));
  }
  int input_sequence_length = dimensions[dimensions.size() - 2];
  LITERT_ASSIGN_OR_RETURN(
      auto mask_tensor,
      TensorBuffer::CreateManaged(
          env_, TensorBufferType::kHostMemory,
          RankedTensorType(GetElementType<uint8_t>(),
                           Layout(Dimensions({1, input_sequence_length}))),
          input_sequence_length * sizeof(uint8_t)));
  std::vector<uint8_t> all_ones(input_sequence_length, 1);
  LITERT_RETURN_IF_ERROR(mask_tensor.Write<uint8_t>(absl::MakeSpan(all_ones)));
  return Encode(spectrogram_tensor, mask_tensor);
}

absl::StatusOr<std::unique_ptr<AudioStreamingContext>>
AudioLiteRtCompiledModelExecutor::AudioStreamingEncoder::CreateNewContext() {
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer> state_buffers;
  LITERT_ASSIGN_OR_RETURN(auto signature, compiled_model_.GetSignature(0));
  for (auto& [name, buffer] : input_buffers_map_) {
    if (name == kSegmentValuesName || name == kSegmentMaskName) {
      // Skip the segment values and mask buffers as they are not part of the
      // state.
      continue;
    }
    if (absl::c_find(kAudioInputNames, name) != kAudioInputNames.end()) {
      // Skip audio input buffers.
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(auto new_buffer, CopyTensorBuffer(env_, buffer));
    if (name == kPrevMaskName) {
      LITERT_ASSIGN_OR_RETURN(auto prev_mask_type, buffer.TensorType());
      LITERT_ASSIGN_OR_RETURN(int prev_mask_size,
                              prev_mask_type.Layout().NumElements());
      input_buffers_map_[kPrevMaskName].Write<uint8_t>(
          std::vector<uint8_t>(prev_mask_size, 1));
    } else {
      LITERT_RETURN_IF_ERROR(InitializeBuffer(new_buffer));
    }
    state_buffers[name] = std::move(new_buffer);
  }
  auto audio_streaming_context =
      std::make_unique<AudioStreamingContext>(std::move(state_buffers));
  return audio_streaming_context;
}

absl::StatusOr<std::unique_ptr<AudioStreamingContext>>
AudioLiteRtCompiledModelExecutor::AudioStreamingEncoder::CloneContext() {
  absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer> state_buffers;
  LITERT_ASSIGN_OR_RETURN(auto signature, compiled_model_.GetSignature(0));
  for (const auto& [name, buffer] : input_buffers_map_) {
    if (name == kSegmentValuesName || name == kSegmentMaskName) {
      // Skip the segment values and mask buffers as they are not part of the
      // state.
      continue;
    }
    if (absl::c_find(kAudioInputNames, name) != kAudioInputNames.end()) {
      // Skip audio input buffers.
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(auto new_buffer, CopyTensorBuffer(env_, buffer));
    state_buffers[name] = std::move(new_buffer);
  }
  auto audio_streaming_context =
      std::make_unique<AudioStreamingContext>(std::move(state_buffers));
  return audio_streaming_context;
}

absl::Status
AudioLiteRtCompiledModelExecutor::AudioStreamingEncoder::RestoreContext(
    std::unique_ptr<AudioStreamingContext> audio_streaming_context) {
  for (auto& [name, buffer] : audio_streaming_context->state_buffers()) {
    if (!input_buffers_map_.contains(name)) {
      return absl::InvalidArgumentError(
          absl::StrCat("The Audio Streaming Encoder model must have a ", name,
                       " input buffer."));
    }
    if (name == kSegmentValuesName || name == kSegmentMaskName) {
      // Skip the segment values and mask buffers as they are not part of the
      // state.
      continue;
    }

    if (input_buffers_map_[name].IsMetalMemory()) {
      // b/505373949#comment13: A temporary fix for Metal memory leak.
      LITERT_ASSIGN_OR_RETURN(auto tensor_type, buffer.TensorType());
      if (tensor_type.ElementType() == ElementType::Float32) {
        LITERT_ASSIGN_OR_RETURN(auto data_span,
                                ReferTensorBufferAsSpan<float>(buffer));
        LITERT_RETURN_IF_ERROR(
            input_buffers_map_[name].Write<float>(data_span));
      } else if (tensor_type.ElementType() == ElementType::Float16) {
        LITERT_ASSIGN_OR_RETURN(auto data_span,
                                ReferTensorBufferAsSpan<tflite::half>(buffer));
        LITERT_RETURN_IF_ERROR(
            input_buffers_map_[name].Write<tflite::half>(data_span));
      } else if (tensor_type.ElementType() == ElementType::Bool) {
        LITERT_ASSIGN_OR_RETURN(auto data_span,
                                ReferTensorBufferAsSpan<bool>(buffer));
        LITERT_RETURN_IF_ERROR(input_buffers_map_[name].Write<bool>(data_span));
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported element type for state buffer: ",
                         tensor_type.ElementType()));
      }
    } else {
      LITERT_ASSIGN_OR_RETURN(auto buffer_copy, buffer.Duplicate());
      input_buffers_map_[name] = std::move(buffer_copy);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<AudioContext>>
AudioLiteRtCompiledModelExecutor::CreateNewContext() {
  if (!audio_encoder_->IsStreaming()) {
    return absl::UnimplementedError(
        "CreateNewContext is only supported for streaming models.");
  }
  return reinterpret_cast<AudioStreamingEncoder*>(audio_encoder_.get())
      ->CreateNewContext();
}

absl::StatusOr<std::unique_ptr<AudioContext>>
AudioLiteRtCompiledModelExecutor::CloneContext() {
  if (!audio_encoder_->IsStreaming()) {
    return absl::UnimplementedError(
        "CloneContext is only supported for streaming models.");
  }
  ASSIGN_OR_RETURN(
      auto audio_encoder_context,
      reinterpret_cast<AudioStreamingEncoder*>(audio_encoder_.get())
          ->CloneContext());
  return std::move(audio_encoder_context);
}

absl::StatusOr<std::unique_ptr<AudioContext>>
AudioLiteRtCompiledModelExecutor::CloneContext(
    const AudioContext& audio_context) {
  if (!audio_encoder_->IsStreaming()) {
    return absl::UnimplementedError(
        "CloneContext is only supported for streaming models.");
  }
  const AudioStreamingContext& audio_streaming_context =
      static_cast<const AudioStreamingContext&>(audio_context);
  return audio_streaming_context.Clone();
}

absl::Status AudioLiteRtCompiledModelExecutor::RestoreContext(
    std::unique_ptr<AudioContext> audio_context) {
  if (!audio_encoder_->IsStreaming()) {
    return absl::UnimplementedError(
        "RestoreContext is only supported for streaming models.");
  }
  return reinterpret_cast<AudioStreamingEncoder*>(audio_encoder_.get())
      ->RestoreContext(std::unique_ptr<AudioStreamingContext>(
          static_cast<AudioStreamingContext*>(audio_context.release())));
}

}  // namespace litert::lm
