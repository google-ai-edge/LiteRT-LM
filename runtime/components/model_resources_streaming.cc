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

#include "runtime/components/model_resources_streaming.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "flatbuffers/verifier.h"  // from @flatbuffers
#include "litert/c/litert_tensor_buffer_types.h"  // from @litert
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/data_stream.h"
#include "runtime/util/scoped_file.h"
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::lm {
namespace {

// Returns the names of the external buffer groups declared by the serialized
// TFLite model in `buffer`, or an empty vector if it declares none or is not a
// valid TFLite model.
std::vector<std::string> GetExternalBufferGroupNames(
    const std::vector<char>& buffer) {
  std::vector<std::string> names;
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size());
  if (!tflite::VerifyModelBuffer(verifier)) {
    return names;
  }
  const tflite::Model* model = tflite::GetModel(buffer.data());
  if (model->external_buffer_groups() == nullptr) {
    return names;
  }
  for (const tflite::ExternalBufferGroup* group :
       *model->external_buffer_groups()) {
    if (group != nullptr && group->name() != nullptr) {
      names.push_back(group->name()->str());
    }
  }
  return names;
}

}  // namespace

bool ModelResourcesStreaming::HasTFLiteModel(
    proto::LlmMetadata::TfLiteModelType model_type) {
  return HasTFLiteModel(TfLiteModelTypeToWireString(model_type));
}

bool ModelResourcesStreaming::HasTFLiteModel(absl::string_view model_type_str) {
  const std::string key = absl::AsciiStrToLower(model_type_str);
  if (auto it = models_.find(key);
      it != models_.end() && it->second != nullptr) {
    return true;
  }
  auto buf_it = model_buffers_.find(key);
  return buf_it != model_buffers_.end() && !buf_it->second.empty();
}

absl::StatusOr<const litert::Model*> ModelResourcesStreaming::GetTFLiteModel(
    proto::LlmMetadata::TfLiteModelType model_type) {
  return GetTFLiteModel(TfLiteModelTypeToWireString(model_type));
}

absl::StatusOr<const litert::Model*> ModelResourcesStreaming::GetTFLiteModel(
    absl::string_view model_type_str) {
  const std::string key = absl::AsciiStrToLower(model_type_str);
  auto it = models_.find(key);
  if (it != models_.end() && it->second != nullptr) {
    return it->second.get();
  }

  auto buf_it = model_buffers_.find(key);
  if (buf_it == model_buffers_.end() || buf_it->second.empty()) {
    return absl::NotFoundError(
        absl::StrCat("Model buffer not found for type: ", key));
  }

  auto expected_model =
      litert::Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
          reinterpret_cast<uint8_t*>(buf_it->second.data()),
          buf_it->second.size()));
  if (!expected_model.HasValue()) {
    return absl::InternalError(
        absl::StrCat("Failed to create model from buffer for type: ", key));
  }
  auto model = std::make_unique<litert::Model>(std::move(*expected_model));
  const auto* model_ptr = model.get();
  models_[key] = std::move(model);
  return model_ptr;
}

absl::StatusOr<absl::string_view> ModelResourcesStreaming::GetTFLiteModelBuffer(
    proto::LlmMetadata::TfLiteModelType model_type) {
  return GetTFLiteModelBuffer(TfLiteModelTypeToWireString(model_type));
}

absl::StatusOr<absl::string_view> ModelResourcesStreaming::GetTFLiteModelBuffer(
    absl::string_view model_type_str) {
  const std::string key = absl::AsciiStrToLower(model_type_str);
  auto it = model_buffers_.find(key);
  if (it == model_buffers_.end() || it->second.empty()) {
    return absl::NotFoundError(
        absl::StrCat("Model buffer not found for type: ", key));
  }
  return absl::string_view(it->second.data(), it->second.size());
}

absl::StatusOr<std::reference_wrapper<ScopedFile>>
ModelResourcesStreaming::GetScopedFile() {
  return absl::UnimplementedError("GetScopedFile not implemented.");
}

absl::StatusOr<std::pair<size_t, size_t>>
ModelResourcesStreaming::GetWeightsSectionOffset(
    proto::LlmMetadata::TfLiteModelType model_type) {
  return absl::UnimplementedError("GetWeightsSectionOffset not implemented.");
}

absl::StatusOr<std::pair<size_t, size_t>>
ModelResourcesStreaming::GetWeightsSectionOffset(
    absl::string_view model_type_str) {
  return absl::UnimplementedError("GetWeightsSectionOffset not implemented.");
}

std::optional<std::string>
ModelResourcesStreaming::GetTFLiteModelBackendConstraint(
    proto::LlmMetadata::TfLiteModelType model_type) {
  return GetTFLiteModelBackendConstraint(
      TfLiteModelTypeToWireString(model_type));
}

std::optional<std::string>
ModelResourcesStreaming::GetTFLiteModelBackendConstraint(
    absl::string_view model_type_str) {
  auto it = backend_constraints_.find(absl::AsciiStrToLower(model_type_str));
  if (it != backend_constraints_.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::optional<std::string>
ModelResourcesStreaming::GetTFLiteModelPreferActivationType(
    proto::LlmMetadata::TfLiteModelType model_type) {
  return GetTFLiteModelPreferActivationType(
      TfLiteModelTypeToWireString(model_type));
}

std::optional<std::string>
ModelResourcesStreaming::GetTFLiteModelPreferActivationType(
    absl::string_view model_type_str) {
  auto it =
      prefer_activation_types_.find(absl::AsciiStrToLower(model_type_str));
  if (it != prefer_activation_types_.end()) {
    return it->second;
  }
  return std::nullopt;
}

absl::StatusOr<std::unique_ptr<Tokenizer>>
ModelResourcesStreaming::GetTokenizer() {
  return absl::UnimplementedError("GetTokenizer not implemented.");
}

absl::StatusOr<const proto::LlmMetadata*>
ModelResourcesStreaming::GetLlmMetadata() {
  if (llm_metadata_.has_value()) {
    return &*llm_metadata_;
  }
  return absl::NotFoundError("LlmMetadata not set.");
}

absl::StatusOr<const proto::EmbeddingMetadata*>
ModelResourcesStreaming::GetEmbeddingMetadata() {
  if (embedding_metadata_.has_value()) {
    return &*embedding_metadata_;
  }
  return absl::NotFoundError("EmbeddingMetadata not set.");
}

absl::StatusOr<const proto::ExecutorMetadata*>
ModelResourcesStreaming::GetExecutorMetadata() {
  if (executor_metadata_.has_value()) {
    return &*executor_metadata_;
  }
  return absl::NotFoundError("ExecutorMetadata not set.");
}

absl::StatusOr<FileRegion>
ModelResourcesStreaming::GetTFLiteModelSectionFileRegion(
    proto::LlmMetadata::TfLiteModelType model_type) {
  return absl::UnimplementedError(
      "GetTFLiteModelSectionFileRegion not implemented.");
}

absl::StatusOr<FileRegion>
ModelResourcesStreaming::GetTFLiteModelSectionFileRegion(
    absl::string_view model_type_str) {
  return absl::UnimplementedError(
      "GetTFLiteModelSectionFileRegion not implemented.");
}

const absl::flat_hash_map<std::string, absl::Span<const std::byte>>*
ModelResourcesStreaming::GetWeightInMemoryMap(
    proto::LlmMetadata::TfLiteModelType model_type) const {
  return GetWeightInMemoryMap(TfLiteModelTypeToWireString(model_type));
}

const absl::flat_hash_map<std::string, absl::Span<const std::byte>>*
ModelResourcesStreaming::GetWeightInMemoryMap(
    absl::string_view model_type_str) const {
  auto it =
      weights_in_memory_per_model_.find(absl::AsciiStrToLower(model_type_str));
  if (it != weights_in_memory_per_model_.end()) {
    return &it->second;
  }
  return nullptr;
}

void ModelResourcesStreaming::SetTFLiteModelBackendConstraint(
    absl::string_view model_type_str, std::string backend_constraint) {
  backend_constraints_[absl::AsciiStrToLower(model_type_str)] =
      std::move(backend_constraint);
}

void ModelResourcesStreaming::SetTFLiteModelPreferActivationType(
    absl::string_view model_type_str, std::string prefer_activation_type) {
  prefer_activation_types_[absl::AsciiStrToLower(model_type_str)] =
      std::move(prefer_activation_type);
}

void ModelResourcesStreaming::SetModelBuffer(absl::string_view model_type_str,
                                             std::vector<char> buffer) {
  model_buffers_[absl::AsciiStrToLower(model_type_str)] = std::move(buffer);
}

absl::Status ModelResourcesStreaming::SetWeightsFromStream(
    absl::string_view model_type_str, DataStream& stream, size_t size) {
  const std::string key = absl::AsciiStrToLower(model_type_str);
  auto& storage = weights_storage_[key];
  // LiteRT requires host memory tensor buffers to be aligned to
  // LITERT_HOST_MEMORY_BUFFER_ALIGNMENT.
  storage.resize(size + LITERT_HOST_MEMORY_BUFFER_ALIGNMENT);
  uintptr_t addr = reinterpret_cast<uintptr_t>(storage.data());
  uintptr_t aligned_addr = (addr + LITERT_HOST_MEMORY_BUFFER_ALIGNMENT - 1) &
                           ~uintptr_t(LITERT_HOST_MEMORY_BUFFER_ALIGNMENT - 1);
  uint8_t* aligned_ptr = reinterpret_cast<uint8_t*>(aligned_addr);

  constexpr size_t kChunkSize = 1024 * 1024 * 16;  // 16MB chunk
  size_t remaining = size;
  size_t offset = 0;
  while (remaining > 0) {
    size_t read_size = std::min(remaining, kChunkSize);
    ABSL_RETURN_IF_ERROR(stream.ReadAndDiscard(
        reinterpret_cast<char*>(aligned_ptr + offset), offset, read_size));
    remaining -= read_size;
    offset += read_size;
  }
  auto span = absl::MakeConstSpan(
      reinterpret_cast<const std::byte*>(aligned_ptr), size);
  weights_in_memory_per_model_[key][kModelWeightGroupName] = span;

  // Expose the weights to the other models that reference this model's
  // section (e.g. prefill/decode reading a tied embedding table stored once in
  // the embedder's section of a deduplicated model file). On web, this lets
  // their WebGPU upload read the shared weights from host memory.
  const std::string shared_group_name =
      absl::StrCat(kModelWeightGroupName, kWeightGroupModelTypeSeparator, key);
  for (const auto& [model_key, buffer] : model_buffers_) {
    if (model_key == key) {
      continue;
    }
    for (const std::string& group_name : GetExternalBufferGroupNames(buffer)) {
      if (absl::AsciiStrToLower(group_name) == shared_group_name) {
        weights_in_memory_per_model_[model_key][group_name] = span;
      }
    }
  }
  return absl::OkStatus();
}

void ModelResourcesStreaming::ReleaseWeights(absl::string_view model_type_str) {
  const std::string key = absl::AsciiStrToLower(model_type_str);
  // Drop the spans before the storage they point into, so that no dangling
  // span is observable at any point. This includes the spans other models
  // hold over this model's weights.
  weights_in_memory_per_model_.erase(key);
  const std::string shared_group_name =
      absl::StrCat(kModelWeightGroupName, kWeightGroupModelTypeSeparator, key);
  for (auto& [model_key, weight_map] : weights_in_memory_per_model_) {
    absl::erase_if(weight_map, [&shared_group_name](const auto& entry) {
      return absl::AsciiStrToLower(entry.first) == shared_group_name;
    });
  }
  weights_storage_.erase(key);
}

}  // namespace litert::lm
