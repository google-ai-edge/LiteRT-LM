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

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/speech2text_metadata.pb.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "runtime/proto/text2speech_metadata.pb.h"

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
    proto::Text2SpeechMetadata::TfLiteModelType model_type) {
  return absl::AsciiStrToLower(
      proto::Text2SpeechMetadata::TfLiteModelType_Name(model_type));
}

std::string TfLiteModelTypeToWireString(
    proto::Speech2TextMetadata::TfLiteModelType model_type) {
  return absl::AsciiStrToLower(
      proto::Speech2TextMetadata::TfLiteModelType_Name(model_type));
}

std::string TfLiteModelTypeToWireString(
    proto::Text2ImageMetadata::TfLiteModelType model_type) {
  return absl::AsciiStrToLower(
      proto::Text2ImageMetadata::TfLiteModelType_Name(model_type));
}

std::string TfLiteModelTypeToWireString(
    proto::Any2AnyMetadata::TfLiteModelType model_type) {
  return absl::AsciiStrToLower(
      proto::Any2AnyMetadata::TfLiteModelType_Name(model_type));
}

absl::StatusOr<proto::LlmMetadata::TfLiteModelType> ParseLlmTfLiteModelType(
    absl::string_view model_type_str) {
  proto::LlmMetadata::TfLiteModelType model_type;
  if (!proto::LlmMetadata::TfLiteModelType_Parse(
          absl::AsciiStrToUpper(model_type_str), &model_type)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unknown model type: ", model_type_str));
  }
  return model_type;
}

bool ModelResources::HasTFLiteModel(absl::string_view model_type_str) {
  auto model_type = ParseLlmTfLiteModelType(model_type_str);
  if (model_type.ok()) {
    return HasTFLiteModel(*model_type);
  }
  return false;
}

absl::StatusOr<const litert::Model*> ModelResources::GetTFLiteModel(
    absl::string_view model_type_str) {
  auto model_type = ParseLlmTfLiteModelType(model_type_str);
  if (model_type.ok()) {
    return GetTFLiteModel(*model_type);
  }
  return absl::UnimplementedError(
      "GetTFLiteModel(absl::string_view) is not implemented.");
}

absl::StatusOr<absl::string_view> ModelResources::GetTFLiteModelBuffer(
    absl::string_view model_type_str) {
  auto model_type = ParseLlmTfLiteModelType(model_type_str);
  if (model_type.ok()) {
    return GetTFLiteModelBuffer(*model_type);
  }
  return absl::UnimplementedError(
      "GetTFLiteModelBuffer(absl::string_view) is not implemented.");
}

absl::StatusOr<std::pair<size_t, size_t>>
ModelResources::GetWeightsSectionOffset(absl::string_view model_type_str) {
  auto model_type = ParseLlmTfLiteModelType(model_type_str);
  if (model_type.ok()) {
    return GetWeightsSectionOffset(*model_type);
  }
  return absl::UnimplementedError(
      "GetWeightsSectionOffset(absl::string_view) is not implemented.");
}

absl::StatusOr<FileRegion> ModelResources::GetTFLiteModelSectionFileRegion(
    absl::string_view model_type_str) {
  auto model_type = ParseLlmTfLiteModelType(model_type_str);
  if (model_type.ok()) {
    return GetTFLiteModelSectionFileRegion(*model_type);
  }
  return absl::UnimplementedError(
      "GetTFLiteModelSectionFileRegion(absl::string_view) is not "
      "implemented.");
}

std::optional<std::string> ModelResources::GetTFLiteModelBackendConstraint(
    absl::string_view model_type_str) {
  auto model_type = ParseLlmTfLiteModelType(model_type_str);
  if (model_type.ok()) {
    return GetTFLiteModelBackendConstraint(*model_type);
  }
  return std::nullopt;
}

std::optional<std::string> ModelResources::GetTFLiteModelPreferActivationType(
    absl::string_view model_type_str) {
  auto model_type = ParseLlmTfLiteModelType(model_type_str);
  if (model_type.ok()) {
    return GetTFLiteModelPreferActivationType(*model_type);
  }
  return std::nullopt;
}

absl::StatusOr<std::unique_ptr<Tokenizer>> ModelResources::GetTokenizer(
    proto::LlmMetadata::TfLiteModelType model_type) {
  if (model_type == proto::LlmMetadata::TF_LITE_PREFILL_DECODE) {
    return GetTokenizer();
  }
  return absl::UnimplementedError(
      absl::StrCat("GetTokenizer for model type ",
                   proto::LlmMetadata::TfLiteModelType_Name(model_type),
                   " is not implemented."));
}

absl::StatusOr<std::unique_ptr<Tokenizer>> ModelResources::GetTokenizer(
    absl::string_view model_type_str) {
  auto model_type = ParseLlmTfLiteModelType(model_type_str);
  if (model_type.ok()) {
    return GetTokenizer(*model_type);
  }
  return absl::UnimplementedError(absl::StrCat(
      "GetTokenizer for model type ", model_type_str, " is not implemented."));
}

absl::StatusOr<const Tokenizer*> ModelResources::GetOrCreateTokenizer(
    proto::LlmMetadata::TfLiteModelType model_type) {
  return absl::UnimplementedError(
      absl::StrCat("GetOrCreateTokenizer for model type ",
                   proto::LlmMetadata::TfLiteModelType_Name(model_type),
                   " is not implemented."));
}

absl::StatusOr<const Tokenizer*> ModelResources::GetOrCreateTokenizer(
    absl::string_view model_type_str) {
  auto model_type = ParseLlmTfLiteModelType(model_type_str);
  if (model_type.ok()) {
    return GetOrCreateTokenizer(*model_type);
  }
  return absl::UnimplementedError(
      absl::StrCat("GetOrCreateTokenizer for model type ", model_type_str,
                   " is not implemented."));
}

std::vector<std::string> ModelResources::GetGenericBinaryDataNames() const {
  return {};
}

const absl::flat_hash_map<std::string, absl::Span<const std::byte>>*
ModelResources::GetWeightInMemoryMap(
    proto::LlmMetadata::TfLiteModelType model_type) const {
  return nullptr;
}

const absl::flat_hash_map<std::string, absl::Span<const std::byte>>*
ModelResources::GetWeightInMemoryMap(absl::string_view model_type_str) const {
  auto model_type = ParseLlmTfLiteModelType(model_type_str);
  if (model_type.ok()) {
    return GetWeightInMemoryMap(*model_type);
  }
  return nullptr;
}

}  // namespace litert::lm
