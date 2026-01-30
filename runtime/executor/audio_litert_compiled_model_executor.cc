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
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/cc/options/litert_cpu_options.h"  // from @litert
#include "litert/cc/options/litert_gpu_options.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/util/status_macros.h"  //NOLINT

namespace litert::lm {
namespace {

constexpr absl::string_view kFeaturesName = "features";
constexpr absl::string_view kMaskName = "mask";
constexpr absl::string_view kMaskOutName = "mask_out";
constexpr absl::string_view kSrcInputsName = "src_inputs";
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

// Returns the first valid token count from the mask tensor.
absl::StatusOr<int> GetValidCount(const TensorBuffer& mask_buffer) {
  ASSIGN_OR_RETURN(auto mask, GetDataAsVector<uint8_t>(
                                  const_cast<TensorBuffer&>(mask_buffer)));
  for (int i = mask.size() - 1; i >= 0; --i) {
    if (mask[i] != 0) {
      return i + 1;
    }
  }
  return 0;
}

absl::Status InitializeBuffers(std::vector<TensorBuffer>& buffers) {
  for (auto& buffer : buffers) {
    LITERT_ASSIGN_OR_RETURN(
        auto buffer_lock_and_addr,
        TensorBufferScopedLock::Create(buffer, TensorBuffer::LockMode::kWrite));
    LITERT_ASSIGN_OR_RETURN(auto packed_size, buffer.PackedSize());
    memset(buffer_lock_and_addr.second, 0, packed_size);
  }
  return absl::OkStatus();
}

inline int CeilIntDiv(int a, int b) { return (a + b - 1) / b; }

bool IsStreamingEncoder(const std::vector<absl::string_view>& input_names) {
  // A huristic to check if the model is a streaming model by checking if the
  // input names contain the prev_mask name.
  return std::any_of(input_names.begin(), input_names.end(),
                     [](absl::string_view input_name) {
                       return absl::StrContains(input_name, kPrevMaskName);
                     });
}

}  // namespace

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
  if (executor_settings_.GetBackend() == Backend::GPU) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
    gpu_options.EnableConstantTensorSharing(true);
    gpu_options.SetPrecision(GpuOptions::Precision::kFp32);
    gpu_options.SetPreferTextureWeights(true);
    options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
  } else if (executor_settings_.GetBackend() == Backend::CPU) {
    LITERT_ASSIGN_OR_RETURN(auto& cpu_options, options.GetCpuOptions());
    cpu_options.SetNumThreads(executor_settings_.GetNumThreads());
    options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported backend for AudioStaticEncoder: ",
                     executor_settings_.GetBackend()));
  }

  LITERT_ASSIGN_OR_RETURN(compiled_model_,
                          CompiledModel::Create(env_, model_, options));
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
  if (!input_buffers_map_.contains(kMaskName)) {
    return absl::InvalidArgumentError(
        "The Audio Static Encoder model must have a mask input buffer.");
  }
  if (!input_buffers_map_.contains(kSrcInputsName)) {
    return absl::InvalidArgumentError(
        "The Audio Static Encoder model must have a src_inputs input buffer.");
  }
  input_mask_buffer_ = &input_buffers_map_[kMaskName];
  spectrogram_buffer_ = &input_buffers_map_[kSrcInputsName];

  // Initialize the output buffers.
  LITERT_ASSIGN_OR_RETURN(auto output_buffers,
                          compiled_model_.CreateOutputBuffers(
                              /*signature_index=*/0));
  if (output_buffers.size() != 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The Audio Static Encoder model must have exactly two output "
        "buffer but got ",
        output_buffers.size()));
  }
  LITERT_RETURN_IF_ERROR(InitializeBuffers(output_buffers));
  output_names_.reserve(signature.OutputNames().size());
  for (int i = 0; i < signature.OutputNames().size(); ++i) {
    std::string output_name = std::string(signature.OutputNames()[i]);
    output_names_.push_back(output_name);
    absl::string_view output_name_view = output_names_[i];
    output_buffers_map_[output_name_view] = std::move(output_buffers[i]);
  }
  // Get pointers to specific buffers after the map is fully populated.
  if (!output_buffers_map_.contains(kMaskName) &&
      !output_buffers_map_.contains(kMaskOutName)) {
    return absl::InvalidArgumentError(
        "The Audio Static Encoder model must have a mask output buffer.");
  }
  if (!output_buffers_map_.contains(kFeaturesName)) {
    return absl::InvalidArgumentError(
        "The Audio Static Encoder model must have a features output buffer.");
  }
  output_mask_buffer_ = output_buffers_map_.contains(kMaskName)
                            ? &output_buffers_map_[kMaskName]
                            : &output_buffers_map_[kMaskOutName];
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
  if (executor_settings_.GetBackend() == Backend::GPU) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
    gpu_options.EnableConstantTensorSharing(true);
    gpu_options.SetPrecision(GpuOptions::Precision::kFp32);
    gpu_options.SetPreferTextureWeights(true);
    options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
  } else if (executor_settings_.GetBackend() == Backend::CPU) {
    LITERT_ASSIGN_OR_RETURN(auto& cpu_options, options.GetCpuOptions());
    cpu_options.SetNumThreads(executor_settings_.GetNumThreads());
    options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported backend for AudioEncoder: ",
                     executor_settings_.GetBackend()));
  }

  LITERT_ASSIGN_OR_RETURN(compiled_model_,
                          CompiledModel::Create(env_, model_, options));
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

  // Get pointers to specific buffers after the map is fully populated.
  if (!input_buffers_map_.contains(kSegmentMaskName)) {
    return absl::InvalidArgumentError(
        "The Audio Streaming Encoder model must have a segment_mask input "
        "buffer.");
  }
  if (!input_buffers_map_.contains(kSegmentValuesName)) {
    return absl::InvalidArgumentError(
        "The Audio Streaming Encoder model must have a segment_values input "
        "buffer.");
  }
  input_mask_buffer_ = &input_buffers_map_[kSegmentMaskName];
  spectrogram_buffer_ = &input_buffers_map_[kSegmentValuesName];

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
  if (!output_buffers_map_.contains(kMaskName)) {
    return absl::InvalidArgumentError(
        "The Audio Streaming Encoder model must have a mask output buffer.");
  }
  if (!output_buffers_map_.contains(kFeaturesName)) {
    return absl::InvalidArgumentError(
        "The Audio Streaming Encoder model must have a features output "
        "buffer.");
  }
  output_mask_buffer_ = &output_buffers_map_[kMaskName];
  output_features_buffer_ = &output_buffers_map_[kFeaturesName];

  // Get the feature states tensor type and use it to get the overlap size.
  std::string feature_states_name =
      absl::StrCat(kFeatureStatesNamePattern, "_0");
  if (!input_buffers_map_.contains(feature_states_name)) {
    return absl::InvalidArgumentError(
        "The Audio Streaming Encoder model must have a feature_states input "
        "buffer.");
  }
  LITERT_ASSIGN_OR_RETURN(auto feature_states_tensor_type,
                          input_buffers_map_[feature_states_name].TensorType());
  // The overlap size is the number of elements in the feature states tensor,
  // which is 3 for gemma3n.
  LITERT_ASSIGN_OR_RETURN(overlap_size_,
                          feature_states_tensor_type.Layout().NumElements());

  // Initialize the previous mask buffer to all ones.
  LITERT_ASSIGN_OR_RETURN(auto prev_mask_type,
                          input_buffers_map_[kPrevMaskName].TensorType());
  LITERT_ASSIGN_OR_RETURN(int prev_mask_size,
                          prev_mask_type.Layout().NumElements());
  input_buffers_map_[kPrevMaskName].Write<uint8_t>(
      std::vector<uint8_t>(prev_mask_size, 1));

  return absl::OkStatus();
}

void AudioLiteRtCompiledModelExecutor::AudioStreamingEncoder::
    SwapInternalStateBuffers() {
  std::vector<absl::string_view> all_input_names(input_names_.begin(),
                                                 input_names_.end());
  for (const auto& input_name : all_input_names) {
    if (output_buffers_map_.contains(input_name)) {
      std::swap(input_buffers_map_[input_name],
                output_buffers_map_[input_name]);
    }
  }
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
  {
    LITERT_ASSIGN_OR_RETURN(
        auto buffer_lock_and_addr,
        TensorBufferScopedLock::Create(GetMutableInputMaskBuffer(),
                                       TensorBuffer::LockMode::kWrite));
    LITERT_ASSIGN_OR_RETURN(auto packed_size,
                            GetInputMaskBuffer().PackedSize());
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
  if (executor_settings_.GetBackend() == Backend::GPU) {
    LITERT_ASSIGN_OR_RETURN(auto& gpu_options, options.GetGpuOptions());
    gpu_options.EnableConstantTensorSharing(true);
    gpu_options.SetPrecision(GpuOptions::Precision::kFp32);
    gpu_options.SetPreferTextureWeights(true);
    options.SetHardwareAccelerators(litert::HwAccelerators::kGpu);
  } else if (executor_settings_.GetBackend() == Backend::CPU) {
    LITERT_ASSIGN_OR_RETURN(auto& cpu_options, options.GetCpuOptions());
    cpu_options.SetNumThreads(executor_settings_.GetNumThreads());
    options.SetHardwareAccelerators(litert::HwAccelerators::kCpu);
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported backend for AudioAdapter: ",
                     executor_settings_.GetBackend()));
  }

  LITERT_ASSIGN_OR_RETURN(compiled_model_,
                          CompiledModel::Create(env_, model_, options));
  LITERT_ASSIGN_OR_RETURN(auto signatures, model_.GetSignatures());
  if (signatures.size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The Audio Adapter model must have exactly one signature but got ",
        signatures.size()));
  }
  LITERT_ASSIGN_OR_RETURN(input_buffers_, compiled_model_.CreateInputBuffers(
                                              /*signature_index=*/0));
  if (input_buffers_.size() != 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The Audio Adapter model must have exactly two input buffer but got ",
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
  if (mask_buffer_ == nullptr) {
    return absl::InvalidArgumentError(
        "The Audio Adapter model must have a mask input buffer.");
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
  ASSIGN_OR_RETURN(auto audio_adapter_model,
                   resources->GetTFLiteModel(ModelType::kTfLiteAudioAdapter));
  std::unique_ptr<AudioEncoder> audio_encoder;
  LITERT_ASSIGN_OR_RETURN(auto encoder_signature,
                          audio_encoder_model->GetSignature(0));
  const bool is_streaming_encoder =
      IsStreamingEncoder(encoder_signature.InputNames());
  if (is_streaming_encoder) {
    ASSIGN_OR_RETURN(audio_encoder,
                     AudioStreamingEncoder::Create(executor_settings, env,
                                                   audio_encoder_model));
  } else {
    ASSIGN_OR_RETURN(audio_encoder,
                     AudioStaticEncoder::Create(executor_settings, env,
                                                audio_encoder_model));
  }
  LITERT_ASSIGN_OR_RETURN(
      auto audio_adapter,
      AudioAdapter::Create(executor_settings, env, audio_adapter_model));
  const auto& tmp = audio_encoder->GetInputMaskBuffer();
  LITERT_ASSIGN_OR_RETURN(auto mask_tensor_type, tmp.TensorType());
  LITERT_ASSIGN_OR_RETURN(int sequence_length,
                          mask_tensor_type.Layout().NumElements());
  LITERT_ASSIGN_OR_RETURN(
      auto spectrogram_tensor_type,
      audio_encoder->GetInputSpectrogramBuffer().TensorType());
  const int spectrogram_feature_dimensions =
      spectrogram_tensor_type.Layout().Dimensions().back();
  LITERT_ASSIGN_OR_RETURN(auto adapter_output_tensor_type,
                          audio_adapter->GetOutputBuffers()[0].TensorType());
  const auto dims = adapter_output_tensor_type.Layout().Dimensions();
  const int audio_embedding_dimensions = dims.back();
  const int output_sequence_length = dims[dims.size() - 2];
  if (sequence_length % output_sequence_length != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The sequence length of the audio encoder must be divisible by the "
        "output sequence length of the audio adapter, but got ",
        sequence_length, " and ", output_sequence_length));
  }
  int encoder_shrinking_factor = 1;
  if (!is_streaming_encoder) {
    if (audio_encoder->GetOutputBuffersMap().size() !=
        audio_adapter->GetInputBuffers().size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The number of output buffers of the audio encoder must be equal to "
          "the number of input buffers of the audio adapter, but got ",
          audio_encoder->GetOutputBuffersMap().size(), " and ",
          audio_adapter->GetInputBuffers().size()));
    }
    encoder_shrinking_factor = sequence_length / output_sequence_length;
  } else {
    // shrinking factor is 16 for gemma3n audio streaming.
    encoder_shrinking_factor =
        (sequence_length -
         reinterpret_cast<AudioStreamingEncoder*>(audio_encoder.get())
             ->GetOverlapSize()) /
        output_sequence_length;
  }

  // Make the audio adapter take the audio encoder's mask and features as input.
  LITERT_ASSIGN_OR_RETURN(auto encoder_mask_tensor,
                          audio_encoder->GetOutputMaskBuffer().Duplicate());
  audio_adapter->GetMutableInputBuffers()[0] = std::move(encoder_mask_tensor);
  LITERT_ASSIGN_OR_RETURN(
      auto encoder_features_tensor,
      audio_encoder->GetMutableOutputFeaturesBuffer().Duplicate());
  audio_adapter->GetMutableInputBuffers()[1] =
      std::move(encoder_features_tensor);
  ABSL_LOG(INFO) << "AudioLiteRtCompiledModelExecutor created with "
                    "encoder_shrinking_factor: "
                 << encoder_shrinking_factor;
  return absl::WrapUnique(new AudioLiteRtCompiledModelExecutor(
      std::move(executor_settings), env, std::move(resources),
      std::move(audio_encoder), std::move(audio_adapter), sequence_length,
      spectrogram_feature_dimensions, audio_embedding_dimensions,
      encoder_shrinking_factor, is_streaming_encoder));
}

absl::StatusOr<int> AudioLiteRtCompiledModelExecutor::EncodeInternal(
    absl::Span<float> spectrogram_tensor, absl::Span<uint8_t> spectrogram_mask,
    absl::Span<float> audio_embeddings) {
  RETURN_IF_ERROR(audio_encoder_->ClearInputBuffers());
  LITERT_RETURN_IF_ERROR(
      audio_encoder_->GetMutableInputSpectrogramBuffer().Write<float>(
          spectrogram_tensor));
  LITERT_RETURN_IF_ERROR(
      audio_encoder_->GetMutableInputMaskBuffer().Write<uint8_t>(
          spectrogram_mask));
  LITERT_RETURN_IF_ERROR(audio_encoder_->GetMutableCompiledModel().Run(
      audio_encoder_->GetMutableInputBuffersMap(),
      audio_encoder_->GetMutableOutputBuffersMap()));
  ASSIGN_OR_RETURN(int chunk_valid_tokens,
                   GetValidCount(audio_encoder_->GetOutputMaskBuffer()));
  LITERT_RETURN_IF_ERROR(audio_adapter_->GetMutableCompiledModel().Run(
      audio_adapter_->GetMutableInputBuffers(),
      audio_adapter_->GetMutableOutputBuffers()));
  LITERT_RETURN_IF_ERROR(
      audio_adapter_->GetMutableOutputBuffers()[0].Read<float>(
          absl::MakeSpan(audio_embeddings.data(),
                         chunk_valid_tokens * audio_embedding_dimensions_)));
  if (is_streaming_) {
    reinterpret_cast<AudioStreamingEncoder*>(audio_encoder_.get())
        ->SwapInternalStateBuffers();
  }
  return chunk_valid_tokens;
}

absl::StatusOr<ExecutorAudioData> AudioLiteRtCompiledModelExecutor::Encode(
    const TensorBuffer& spectrogram_tensor,
    const TensorBuffer& spectrogram_mask) {
  ASSIGN_OR_RETURN(int input_sequence_length, GetValidCount(spectrogram_mask));
  LITERT_ASSIGN_OR_RETURN(
      auto spectrogram_host_buffer,
      GetDataAsVector<float>(const_cast<TensorBuffer&>(spectrogram_tensor)));
  LITERT_ASSIGN_OR_RETURN(
      auto spectrogram_mask_host_buffer,
      GetDataAsVector<uint8_t>(const_cast<TensorBuffer&>(spectrogram_mask)));

  std::vector<float> audio_embeddings(input_sequence_length *
                                      audio_embedding_dimensions_);
  // Chunk the spectrogram into smaller pieces and encode them one by one.
  int total_valid_tokens = 0;
  int pos = 0;
  while (pos < input_sequence_length) {
    int end = std::min(pos + sequence_length_, input_sequence_length);
    auto spectrogram_host_buffer_slice =
        absl::MakeSpan(spectrogram_host_buffer)
            .subspan(pos * spectrogram_feature_dimensions_,
                     (end - pos) * spectrogram_feature_dimensions_);
    auto spectrogram_mask_host_buffer_slice =
        absl::MakeSpan(spectrogram_mask_host_buffer).subspan(pos, end - pos);
    auto audio_embeddings_slice =
        absl::MakeSpan(audio_embeddings)
            .subspan(CeilIntDiv(pos, encoder_shrinking_factor_) *
                         audio_embedding_dimensions_,
                     CeilIntDiv(end - pos, encoder_shrinking_factor_) *
                         audio_embedding_dimensions_);
    ASSIGN_OR_RETURN(int chunk_valid_tokens,
                     EncodeInternal(spectrogram_host_buffer_slice,
                                    spectrogram_mask_host_buffer_slice,
                                    audio_embeddings_slice));
    total_valid_tokens += chunk_valid_tokens;
    pos = end;
  }

  // Create the final audio embeddings tensor.
  RankedTensorType audio_embeddings_tensor_type(
      GetElementType<float>(),
      Layout(Dimensions({1, total_valid_tokens, audio_embedding_dimensions_})));
  LITERT_ASSIGN_OR_RETURN(
      auto audio_embeddings_tensor,
      TensorBuffer::CreateManaged(env_, TensorBufferType::kHostMemory,
                                  audio_embeddings_tensor_type,
                                  audio_embeddings.size() * sizeof(float)));
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

}  // namespace litert::lm
