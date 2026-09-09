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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_STREAMING_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_STREAMING_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/data_stream.h"
#include "runtime/util/scoped_file.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

// Implementation of ModelResources for streaming model loading (e.g. from
// .litertlm container streams). Lazily constructs models from in-memory
// buffers, holds LLM or Embedding metadata, and manages both file-based
// (ScopedFile) and in-memory external weights for submodels.
class ModelResourcesStreaming : public ModelResources {
 public:
  // Creates an empty ModelResourcesStreaming instance.
  ModelResourcesStreaming() = default;

  // Creates a ModelResourcesStreaming instance initialized with LLM metadata.
  explicit ModelResourcesStreaming(const proto::LlmMetadata& llm_metadata)
      : llm_metadata_(llm_metadata) {}

  // Creates a ModelResourcesStreaming instance initialized with Embedding
  // metadata.
  explicit ModelResourcesStreaming(
      const proto::EmbeddingMetadata& embedding_metadata)
      : embedding_metadata_(embedding_metadata) {}

  ~ModelResourcesStreaming() override = default;

  // Returns the litert::Model for the specified `model_type`.
  // Lazily creates and caches the model from the in-memory flatbuffer stored
  // in `model_buffers_` if not already created. Returns `NotFoundError` if no
  // model buffer has been set for `model_type`.
  // ModelResourcesStreaming retains ownership of the returned model, which
  // remains valid until this instance is destroyed.
  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      ModelType model_type) override;

  // Returns a string_view over the raw TFLite model flatbuffer bytes for the
  // specified `model_type`, or `NotFoundError` if no buffer has been set.
  // The returned view remains valid for the lifetime of this instance.
  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      ModelType model_type) override;

  // Always returns `UnimplementedError` because streamed models do not own a
  // backing ScopedFile on disk.
  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override;

  // Always returns `UnimplementedError` because weights section file offsets
  // do not apply to streamed models.
  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      ModelType model_type) override;

  // Returns `std::nullopt` as backend constraints are not currently stored in
  // ModelResourcesStreaming.
  std::optional<std::string> GetTFLiteModelBackendConstraint(
      ModelType model_type) override;

  // Returns `std::nullopt` as activation type preferences are not currently
  // stored in ModelResourcesStreaming.
  std::optional<std::string> GetTFLiteModelPreferActivationType(
      ModelType model_type) override;

  // Always returns `UnimplementedError` as tokenizer loading is handled
  // separately from streamed model resources.
  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override;

  // Returns the LlmMetadata if set, or `NotFoundError` if not available.
  // The returned pointer remains valid for the lifetime of this instance.
  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override;

  // Returns the EmbeddingMetadata if set, or `NotFoundError` if not available.
  // The returned pointer remains valid for the lifetime of this instance.
  absl::StatusOr<const proto::EmbeddingMetadata*> GetEmbeddingMetadata()
      override;

  // Always returns `UnimplementedError` as ExecutorMetadata is not stored in
  // ModelResourcesStreaming.
  absl::StatusOr<const proto::ExecutorMetadata*> GetExecutorMetadata() override;

  // Always returns `UnimplementedError` because streamed models do not have
  // file regions on disk.
  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      ModelType model_type) override;

  // Returns a pointer to the map of external weight section names to 64-byte
  // aligned in-memory byte spans for `model_type`, or `nullptr` if no weights
  // have been loaded into memory for that model type.
  const absl::flat_hash_map<std::string, absl::Span<const std::byte>>*
  GetWeightInMemoryMap(ModelType model_type) const override;

  // Sets the LLM metadata for this streaming resource instance.
  void SetLlmMetadata(const proto::LlmMetadata& llm_metadata) {
    llm_metadata_ = llm_metadata;
  }

  // Sets the Embedding metadata for this streaming resource instance.
  void SetEmbeddingMetadata(proto::EmbeddingMetadata embedding_metadata) {
    embedding_metadata_ = std::move(embedding_metadata);
  }

  // Sets the flatbuffer model data for a given `model_type`. This stores the
  // raw flatbuffer in memory so that `GetTFLiteModel` can lazily construct the
  // `litert::Model` without relying on a persistent stream or file.
  void SetModelBuffer(ModelType model_type, std::vector<char> buffer) {
    model_buffers_[model_type] = std::move(buffer);
  }

  // Reads `size` bytes from `stream` into a 64-byte aligned host memory buffer
  // (`weights_storage_`) and maps it under the `"tflite_weights"` section key
  // in `weights_in_memory_per_model_[model_type]`.
  // The 64-byte alignment satisfies LiteRT/XNNPACK host tensor memory alignment
  // requirements (`LITERT_HOST_MEMORY_BUFFER_ALIGNMENT`).
  absl::Status SetWeightsFromStream(ModelType model_type, DataStream& stream,
                                    size_t size);

  // Convenience helper to stream external weights for the per-layer embedder
  // submodel (`ModelType::kTfLitePerLayerEmbedder`).
  absl::Status SetPerLayerWeightsFromStream(DataStream& stream, size_t size) {
    return SetWeightsFromStream(ModelType::kTfLitePerLayerEmbedder, stream,
                                size);
  }

  // Convenience helper to stream external weights for the token embedder
  // submodel (`ModelType::kTfLiteEmbedder`).
  absl::Status SetEmbedderWeightsFromStream(DataStream& stream, size_t size) {
    return SetWeightsFromStream(ModelType::kTfLiteEmbedder, stream, size);
  }

  // Convenience helper to stream external weights for vision encoder submodels.
  absl::Status SetVisionWeightsFromStream(ModelType model_type,
                                          DataStream& stream, size_t size) {
    return SetWeightsFromStream(model_type, stream, size);
  }

  // Convenience helper to stream external weights for the vision adapter
  // submodel (`ModelType::kTfLiteVisionAdapter`).
  absl::Status SetVisionAdapterWeightsFromStream(DataStream& stream,
                                                 size_t size) {
    return SetWeightsFromStream(ModelType::kTfLiteVisionAdapter, stream, size);
  }

  // Convenience helper to stream external weights for audio encoder submodels.
  absl::Status SetAudioWeightsFromStream(ModelType model_type,
                                         DataStream& stream, size_t size) {
    return SetWeightsFromStream(model_type, stream, size);
  }

 private:
  std::optional<proto::LlmMetadata> llm_metadata_;
  std::optional<proto::EmbeddingMetadata> embedding_metadata_;
  // In-memory buffers for model flatbuffers.
  // These are cached from the input stream so that 'GetTFLiteModel' can load
  // models without relying on the original stream. This does not include
  // external weights.
  std::unordered_map<ModelType, std::vector<char>> model_buffers_;
  std::unordered_map<ModelType, std::unique_ptr<litert::Model>> models_;

  // In-memory host storage for model weights. Byte-aligned to
  // LITERT_HOST_MEMORY_BUFFER_ALIGNMENT.
  absl::flat_hash_map<ModelType, std::vector<uint8_t>> weights_storage_;

  // In-memory map of external weight section names to 64-byte aligned byte
  // spans in `weights_storage_`. We have spans into weights_storage_ rather
  // than storing the weights directly in this map to guarantee 64-byte
  // alignment.
  absl::flat_hash_map<
      ModelType, absl::flat_hash_map<std::string, absl::Span<const std::byte>>>
      weights_in_memory_per_model_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_STREAMING_H_
