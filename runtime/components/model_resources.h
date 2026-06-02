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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_H_

// All the loaded model resources the executor needs to hold to avoid the model
// being destroyed.
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/tokenizer.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

enum class ModelType {
  kUnknown = 0,              // Placeholder for uninitialized model type.
  kTfLitePrefillDecode = 1,  // The base model is used for prefill and decode.
  kTfLiteEmbedder = 2,
  kTfLitePerLayerEmbedder = 3,
  kTfLiteAux = 4,
  kTfLiteAudioFrontend = 9,  // Audio frontend model is weight-less Short-Time
                             // Fourier Transform (STFT) to convert audio to
                             // spectrogram.
  kTfLiteAudioEncoderHw = 5,
  kTfLiteAudioAdapter = 10,
  kTfLiteEndOfAudio = 6,
  kTfLiteVisionAdapter = 7,
  kTfLiteEndOfVision = 12,  // The end of vision token model.
  kTfLiteVisionEncoder = 8,
  kArtisanTextDecoder = 11,  // The text decoder model for the artisan gpu.
  kTfLiteMtpDrafter = 13,    // The MTP drafter model.
  kTfLiteMtpAux = 14,        // The MTP auxiliary model.
};

// Utility function to convert a string to ModelType. It's case insensitive.
inline absl::StatusOr<ModelType> StringToModelType(
    absl::string_view model_type_str) {
  const std::string lower_case_model_type_str =
      absl::AsciiStrToLower(model_type_str);
  if (lower_case_model_type_str == "tf_lite_prefill_decode") {
    return ModelType::kTfLitePrefillDecode;
  } else if (lower_case_model_type_str == "tf_lite_embedder") {
    return ModelType::kTfLiteEmbedder;
  } else if (lower_case_model_type_str == "tf_lite_per_layer_embedder") {
    return ModelType::kTfLitePerLayerEmbedder;
  } else if (lower_case_model_type_str == "tf_lite_aux") {
    return ModelType::kTfLiteAux;
  } else if (lower_case_model_type_str == "tf_lite_audio_frontend") {
    return ModelType::kTfLiteAudioFrontend;
  } else if (lower_case_model_type_str == "tf_lite_audio_encoder_hw") {
    return ModelType::kTfLiteAudioEncoderHw;
  } else if (lower_case_model_type_str == "tf_lite_audio_adapter") {
    return ModelType::kTfLiteAudioAdapter;
  } else if (lower_case_model_type_str == "tf_lite_end_of_audio") {
    return ModelType::kTfLiteEndOfAudio;
  } else if (lower_case_model_type_str == "tf_lite_vision_adapter") {
    return ModelType::kTfLiteVisionAdapter;
  } else if (lower_case_model_type_str == "tf_lite_end_of_vision") {
    return ModelType::kTfLiteEndOfVision;
  } else if (lower_case_model_type_str == "tf_lite_vision_encoder") {
    return ModelType::kTfLiteVisionEncoder;
  } else if (lower_case_model_type_str == "tf_lite_artisan_text_decoder") {
    return ModelType::kArtisanTextDecoder;
  } else if (lower_case_model_type_str == "tf_lite_mtp_drafter") {
    return ModelType::kTfLiteMtpDrafter;
  } else if (lower_case_model_type_str == "tf_lite_mtp_aux") {
    return ModelType::kTfLiteMtpAux;
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unknown model type: ", model_type_str));
  }
}

// Utility function to convert a ModelType to string.
inline std::string ModelTypeToString(ModelType model_type) {
  switch (model_type) {
    case ModelType::kTfLitePrefillDecode:
      return "TF_LITE_PREFILL_DECODE";
    case ModelType::kTfLiteEmbedder:
      return "TF_LITE_EMBEDDER";
    case ModelType::kTfLitePerLayerEmbedder:
      return "TF_LITE_PER_LAYER_EMBEDDER";
    case ModelType::kTfLiteAux:
      return "TF_LITE_AUX";
    case ModelType::kTfLiteAudioFrontend:
      return "TF_LITE_AUDIO_FRONTEND";
    case ModelType::kTfLiteAudioEncoderHw:
      return "TF_LITE_AUDIO_ENCODER_HW";
    case ModelType::kTfLiteAudioAdapter:
      return "TF_LITE_AUDIO_ADAPTER";
    case ModelType::kTfLiteEndOfAudio:
      return "TF_LITE_END_OF_AUDIO";
    case ModelType::kTfLiteVisionAdapter:
      return "TF_LITE_VISION_ADAPTER";
    case ModelType::kTfLiteEndOfVision:
      return "TF_LITE_END_OF_VISION";
    case ModelType::kTfLiteVisionEncoder:
      return "TF_LITE_VISION_ENCODER";
    case ModelType::kArtisanTextDecoder:
      return "TF_LITE_ARTISAN_TEXT_DECODER";
    case ModelType::kTfLiteMtpDrafter:
      return "TF_LITE_MTP_DRAFTER";
    case ModelType::kTfLiteMtpAux:
      return "TF_LITE_MTP_AUX";
    case ModelType::kUnknown:
      return "UNKNOWN";
    default:
      return "INVALID";
  }
}

// ModelResources is an interface that manages all the loaded model resources
// that need to be hold to avoid the model being destroyed. It provides a way
// to load the models in a lazy way.
// Basically, it will create the models when they are actually used. Before the
// Get*() functions are called, the models are not created yet. And once the
// models are created, they will be re-used for all the following calls.
//
// It's not thread-safe.
class ModelResources {
 public:
  virtual ~ModelResources() = default;

  // Returns the litert model. We will create the model if it is not created
  // yet. And the model is created from memory mapped file, so physical memory
  // is only allocated when the model is actually used.
  virtual absl::StatusOr<const litert::Model*> GetTFLiteModel(
      ModelType model_type) = 0;

  // Returns the TFLite model buffer. Note that the returned string_view is
  // valid only until the ModelResources is destroyed.
  // When there is no model for the given model type, it will return an error
  // status.
  // Prefer to use GetTFLiteModel() if possible, as this function will leave
  // the model lifecycle management to the caller.
  virtual absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      ModelType model_type) = 0;

  // Returns the reference to the ScopedFile. This is used for the getting the
  // external weights that should not be mmapped into the memory.
  virtual absl::StatusOr<std::reference_wrapper<ScopedFile>>
  GetScopedFile() = 0;

  // Returns the section start offset and end offset.
  virtual absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      ModelType model_type) = 0;

  // Returns the TFLite model backend constraint. When there is no constraint
  // for the given model type, it will return an nullopt.
  virtual std::optional<std::string> GetTFLiteModelBackendConstraint(
      ModelType model_type) = 0;

  // Returns the TFLite model prefer activation type. When there is no
  // prefer activation type for the given model type, it will return an
  // nullopt.
  virtual std::optional<std::string> GetTFLiteModelPreferActivationType(
      ModelType model_type) = 0;

  // Builds a tokenizer instance from the model and returns it.
  virtual absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() = 0;

  // Returns the llm metadata.
  virtual absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_H_
