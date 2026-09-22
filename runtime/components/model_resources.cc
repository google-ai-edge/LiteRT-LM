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
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/proto/asr_metadata.pb.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
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

std::string TfLiteModelTypeToWireString(ModelType model_type) {
  return absl::AsciiStrToLower(ModelTypeToString(model_type));
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
