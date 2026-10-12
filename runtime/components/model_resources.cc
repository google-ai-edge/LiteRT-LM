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

#include "runtime/components/model_resources.h"

#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/proto/asr_metadata.pb.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "runtime/proto/tts_metadata.pb.h"

namespace litert::lm {

std::string TfLiteModelTypeToWireString(
    proto::LlmMetadata::TfLiteModelType model_type) {
  return absl::AsciiStrToLower(
      proto::LlmMetadata::TfLiteModelType_Name(model_type));
}

std::string TfLiteModelTypeToWireString(
    proto::EmbeddingMetadata::TfLiteModelType model_type) {
  return absl::AsciiStrToLower(
      proto::EmbeddingMetadata::TfLiteModelType_Name(model_type));
}

std::string TfLiteModelTypeToWireString(
    proto::TtsMetadata::TfLiteModelType model_type) {
  return absl::AsciiStrToLower(
      proto::TtsMetadata::TfLiteModelType_Name(model_type));
}

std::string TfLiteModelTypeToWireString(
    proto::AsrMetadata::TfLiteModelType model_type) {
  return absl::AsciiStrToLower(
      proto::AsrMetadata::TfLiteModelType_Name(model_type));
}

std::string TfLiteModelTypeToWireString(
    proto::Text2ImageMetadata::TfLiteModelType model_type) {
  return absl::AsciiStrToLower(
      proto::Text2ImageMetadata::TfLiteModelType_Name(model_type));
}

std::string TfLiteModelTypeToWireString(ModelType model_type) {
  return absl::AsciiStrToLower(ModelTypeToString(model_type));
}

absl::StatusOr<proto::LlmMetadata::TfLiteModelType>
ModelTypeToLlmTfLiteModelType(ModelType model_type) {
  switch (model_type) {
    case ModelType::kUnknown:
      return proto::LlmMetadata::TF_LITE_MODEL_TYPE_UNSPECIFIED;
    case ModelType::kTfLitePrefillDecode:
      return proto::LlmMetadata::TF_LITE_PREFILL_DECODE;
    case ModelType::kTfLiteEmbedder:
      return proto::LlmMetadata::TF_LITE_EMBEDDER;
    case ModelType::kTfLitePerLayerEmbedder:
      return proto::LlmMetadata::TF_LITE_PER_LAYER_EMBEDDER;
    case ModelType::kTfLiteAux:
      return proto::LlmMetadata::TF_LITE_AUX;
    case ModelType::kTfLiteAudioFrontend:
      return proto::LlmMetadata::TF_LITE_AUDIO_FRONTEND;
    case ModelType::kTfLiteAudioEncoderHw:
      return proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW;
    case ModelType::kTfLiteAudioAdapter:
      return proto::LlmMetadata::TF_LITE_AUDIO_ADAPTER;
    case ModelType::kTfLiteEndOfAudio:
      return proto::LlmMetadata::TF_LITE_END_OF_AUDIO;
    case ModelType::kTfLiteVisionAdapter:
      return proto::LlmMetadata::TF_LITE_VISION_ADAPTER;
    case ModelType::kTfLiteEndOfVision:
      return proto::LlmMetadata::TF_LITE_END_OF_VISION;
    case ModelType::kTfLiteVisionEncoder:
      return proto::LlmMetadata::TF_LITE_VISION_ENCODER;
    case ModelType::kArtisanTextDecoder:
      return proto::LlmMetadata::TF_LITE_ARTISAN_TEXT_DECODER;
    case ModelType::kTfLiteMtpDrafter:
      return proto::LlmMetadata::TF_LITE_MTP_DRAFTER;
    case ModelType::kTfLiteMtpAux:
      return proto::LlmMetadata::TF_LITE_MTP_AUX;
    case ModelType::kTfLiteTextEncoder:
      // The text encoder belongs to the embedding capability.
      break;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("ModelType ", ModelTypeToString(model_type),
                   " has no LlmMetadata::TfLiteModelType counterpart."));
}

absl::StatusOr<ModelType> LlmTfLiteModelTypeToModelType(
    proto::LlmMetadata::TfLiteModelType model_type) {
  switch (model_type) {
    case proto::LlmMetadata::TF_LITE_MODEL_TYPE_UNSPECIFIED:
      return ModelType::kUnknown;
    case proto::LlmMetadata::TF_LITE_PREFILL_DECODE:
      return ModelType::kTfLitePrefillDecode;
    case proto::LlmMetadata::TF_LITE_EMBEDDER:
      return ModelType::kTfLiteEmbedder;
    case proto::LlmMetadata::TF_LITE_PER_LAYER_EMBEDDER:
      return ModelType::kTfLitePerLayerEmbedder;
    case proto::LlmMetadata::TF_LITE_AUX:
      return ModelType::kTfLiteAux;
    case proto::LlmMetadata::TF_LITE_AUDIO_FRONTEND:
      return ModelType::kTfLiteAudioFrontend;
    case proto::LlmMetadata::TF_LITE_AUDIO_ENCODER_HW:
      return ModelType::kTfLiteAudioEncoderHw;
    case proto::LlmMetadata::TF_LITE_AUDIO_ADAPTER:
      return ModelType::kTfLiteAudioAdapter;
    case proto::LlmMetadata::TF_LITE_END_OF_AUDIO:
      return ModelType::kTfLiteEndOfAudio;
    case proto::LlmMetadata::TF_LITE_VISION_ADAPTER:
      return ModelType::kTfLiteVisionAdapter;
    case proto::LlmMetadata::TF_LITE_END_OF_VISION:
      return ModelType::kTfLiteEndOfVision;
    case proto::LlmMetadata::TF_LITE_VISION_ENCODER:
      return ModelType::kTfLiteVisionEncoder;
    case proto::LlmMetadata::TF_LITE_ARTISAN_TEXT_DECODER:
      return ModelType::kArtisanTextDecoder;
    case proto::LlmMetadata::TF_LITE_MTP_DRAFTER:
      return ModelType::kTfLiteMtpDrafter;
    case proto::LlmMetadata::TF_LITE_MTP_AUX:
      return ModelType::kTfLiteMtpAux;
    default:
      break;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unrecognized LlmMetadata::TfLiteModelType: ",
                   static_cast<int>(model_type)));
}

absl::StatusOr<proto::EmbeddingMetadata::TfLiteModelType>
ModelTypeToEmbeddingTfLiteModelType(ModelType model_type) {
  switch (model_type) {
    case ModelType::kUnknown:
      return proto::EmbeddingMetadata::TF_LITE_MODEL_TYPE_UNSPECIFIED;
    case ModelType::kTfLiteTextEncoder:
      return proto::EmbeddingMetadata::TF_LITE_TEXT_ENCODER;
    case ModelType::kTfLiteEmbedder:
      return proto::EmbeddingMetadata::TF_LITE_EMBEDDER;
    case ModelType::kTfLitePerLayerEmbedder:
      return proto::EmbeddingMetadata::TF_LITE_PER_LAYER_EMBEDDER;
    case ModelType::kTfLiteVisionEncoder:
      return proto::EmbeddingMetadata::TF_LITE_VISION_ENCODER;
    case ModelType::kTfLiteVisionAdapter:
      return proto::EmbeddingMetadata::TF_LITE_VISION_ADAPTER;
    case ModelType::kTfLiteEndOfVision:
      return proto::EmbeddingMetadata::TF_LITE_END_OF_VISION;
    case ModelType::kTfLiteAudioEncoderHw:
      return proto::EmbeddingMetadata::TF_LITE_AUDIO_ENCODER_HW;
    case ModelType::kTfLiteAudioAdapter:
      return proto::EmbeddingMetadata::TF_LITE_AUDIO_ADAPTER;
    default:
      break;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("ModelType ", ModelTypeToString(model_type),
                   " has no EmbeddingMetadata::TfLiteModelType counterpart."));
}

absl::StatusOr<ModelType> EmbeddingTfLiteModelTypeToModelType(
    proto::EmbeddingMetadata::TfLiteModelType model_type) {
  switch (model_type) {
    case proto::EmbeddingMetadata::TF_LITE_MODEL_TYPE_UNSPECIFIED:
      return ModelType::kUnknown;
    case proto::EmbeddingMetadata::TF_LITE_TEXT_ENCODER:
      return ModelType::kTfLiteTextEncoder;
    case proto::EmbeddingMetadata::TF_LITE_EMBEDDER:
      return ModelType::kTfLiteEmbedder;
    case proto::EmbeddingMetadata::TF_LITE_PER_LAYER_EMBEDDER:
      return ModelType::kTfLitePerLayerEmbedder;
    case proto::EmbeddingMetadata::TF_LITE_VISION_ENCODER:
      return ModelType::kTfLiteVisionEncoder;
    case proto::EmbeddingMetadata::TF_LITE_VISION_ADAPTER:
      return ModelType::kTfLiteVisionAdapter;
    case proto::EmbeddingMetadata::TF_LITE_END_OF_VISION:
      return ModelType::kTfLiteEndOfVision;
    case proto::EmbeddingMetadata::TF_LITE_AUDIO_ENCODER_HW:
      return ModelType::kTfLiteAudioEncoderHw;
    case proto::EmbeddingMetadata::TF_LITE_AUDIO_ADAPTER:
      return ModelType::kTfLiteAudioAdapter;
    default:
      break;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unrecognized EmbeddingMetadata::TfLiteModelType: ",
                   static_cast<int>(model_type)));
}

absl::StatusOr<const litert::Model*> ModelResources::GetTFLiteModel(
    absl::string_view model_type_str) {
  auto model_type = StringToModelType(model_type_str);
  if (model_type.ok()) {
    return GetTFLiteModel(*model_type);
  }
  return absl::UnimplementedError(
      "GetTFLiteModel(absl::string_view) is not implemented.");
}

absl::StatusOr<absl::string_view> ModelResources::GetTFLiteModelBuffer(
    absl::string_view model_type_str) {
  auto model_type = StringToModelType(model_type_str);
  if (model_type.ok()) {
    return GetTFLiteModelBuffer(*model_type);
  }
  return absl::UnimplementedError(
      "GetTFLiteModelBuffer(absl::string_view) is not implemented.");
}

std::vector<std::string> ModelResources::GetGenericBinaryDataNames() const {
  return {};
}

}  // namespace litert::lm
