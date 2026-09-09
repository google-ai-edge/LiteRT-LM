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
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_tensor_buffer_types.h"  // from @litert
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/data_stream.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

absl::StatusOr<const litert::Model*> ModelResourcesStreaming::GetTFLiteModel(
    ModelType model_type) {
  auto it = models_.find(model_type);
  if (it != models_.end() && it->second != nullptr) {
    return it->second.get();
  }

  auto buf_it = model_buffers_.find(model_type);
  if (buf_it == model_buffers_.end()) {
    return absl::NotFoundError(absl::StrCat("Model buffer not found for type: ",
                                            static_cast<int>(model_type)));
  }

  auto expected_model =
      litert::Model::CreateFromBuffer(litert::BufferRef<uint8_t>(
          reinterpret_cast<uint8_t*>(buf_it->second.data()),
          buf_it->second.size()));
  if (!expected_model.HasValue()) {
    return absl::InternalError(
        absl::StrCat("Failed to create model from buffer for type: ",
                     static_cast<int>(model_type)));
  }
  auto model = std::make_unique<litert::Model>(std::move(*expected_model));
  const auto* model_ptr = model.get();
  models_[model_type] = std::move(model);
  return model_ptr;
}

absl::StatusOr<absl::string_view> ModelResourcesStreaming::GetTFLiteModelBuffer(
    ModelType model_type) {
  auto it = model_buffers_.find(model_type);
  if (it == model_buffers_.end()) {
    return absl::NotFoundError(absl::StrCat("Model buffer not found for type: ",
                                            static_cast<int>(model_type)));
  }
  return absl::string_view(it->second.data(), it->second.size());
}

absl::StatusOr<std::reference_wrapper<ScopedFile>>
ModelResourcesStreaming::GetScopedFile() {
  return absl::UnimplementedError("GetScopedFile not implemented.");
}

absl::StatusOr<std::pair<size_t, size_t>>
ModelResourcesStreaming::GetWeightsSectionOffset(ModelType model_type) {
  return absl::UnimplementedError("GetWeightsSectionOffset not implemented.");
}

std::optional<std::string>
ModelResourcesStreaming::GetTFLiteModelBackendConstraint(ModelType model_type) {
  return std::nullopt;
}

std::optional<std::string>
ModelResourcesStreaming::GetTFLiteModelPreferActivationType(
    ModelType model_type) {
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
  return absl::UnimplementedError("GetExecutorMetadata not implemented.");
}

absl::StatusOr<FileRegion>
ModelResourcesStreaming::GetTFLiteModelSectionFileRegion(ModelType model_type) {
  return absl::UnimplementedError(
      "GetTFLiteModelSectionFileRegion not implemented.");
}

const absl::flat_hash_map<std::string, absl::Span<const std::byte>>*
ModelResourcesStreaming::GetWeightInMemoryMap(ModelType model_type) const {
  auto it = weights_in_memory_per_model_.find(model_type);
  if (it != weights_in_memory_per_model_.end()) {
    return &it->second;
  }
  return nullptr;
}

absl::Status ModelResourcesStreaming::SetWeightsFromStream(ModelType model_type,
                                                           DataStream& stream,
                                                           size_t size) {
  auto& storage = weights_storage_[model_type];
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
  weights_in_memory_per_model_[model_type]["tflite_weights"] = span;
  return absl::OkStatus();
}

}  // namespace litert::lm
