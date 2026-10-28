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
#include <type_traits>
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

  using ModelResources::GetTFLiteModel;
  using ModelResources::GetTFLiteModelBackendConstraint;
  using ModelResources::GetTFLiteModelBuffer;
  using ModelResources::GetTFLiteModelPreferActivationType;
  using ModelResources::GetTFLiteModelSectionFileRegion;
  using ModelResources::GetWeightInMemoryMap;
  using ModelResources::GetWeightsSectionOffset;
  using ModelResources::HasTFLiteModel;

  bool HasTFLiteModel(proto::LlmMetadata::TfLiteModelType model_type) override;
  bool HasTFLiteModel(absl::string_view model_type_str) override;

  // Returns the litert::Model for the specified `model_type`.
  // Lazily creates and caches the model from the in-memory flatbuffer stored
  // in `model_buffers_` if not already created. Returns `NotFoundError` if no
  // model buffer has been set for `model_type`.
  // ModelResourcesStreaming retains ownership of the returned model, which
  // remains valid until this instance is destroyed.
  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      proto::LlmMetadata::TfLiteModelType model_type) override;
  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      absl::string_view model_type_str) override;

  // Returns a string_view over the raw TFLite model flatbuffer bytes for the
  // specified `model_type`, or `NotFoundError` if no buffer has been set.
  // The returned view remains valid for the lifetime of this instance.
  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      proto::LlmMetadata::TfLiteModelType model_type) override;
  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      absl::string_view model_type_str) override;

  // Always returns `UnimplementedError` because streamed models do not own a
  // backing ScopedFile on disk.
  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override;

  // Always returns `UnimplementedError` because weights section file offsets
  // do not apply to streamed models.
  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      proto::LlmMetadata::TfLiteModelType model_type) override;
  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      absl::string_view model_type_str) override;

  // Returns the backend constraint for `model_type` if set, or `std::nullopt`.
  std::optional<std::string> GetTFLiteModelBackendConstraint(
      proto::LlmMetadata::TfLiteModelType model_type) override;
  std::optional<std::string> GetTFLiteModelBackendConstraint(
      absl::string_view model_type_str) override;

  // Returns the preferred activation type for `model_type` if set, or
  // `std::nullopt`.
  std::optional<std::string> GetTFLiteModelPreferActivationType(
      proto::LlmMetadata::TfLiteModelType model_type) override;
  std::optional<std::string> GetTFLiteModelPreferActivationType(
      absl::string_view model_type_str) override;

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

  // Returns the ExecutorMetadata if set, or `NotFoundError` if not available.
  // The returned pointer remains valid for the lifetime of this instance.
  absl::StatusOr<const proto::ExecutorMetadata*> GetExecutorMetadata() override;

  // Always returns `UnimplementedError` because streamed models do not have
  // file regions on disk.
  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      proto::LlmMetadata::TfLiteModelType model_type) override;
  absl::StatusOr<FileRegion> GetTFLiteModelSectionFileRegion(
      absl::string_view model_type_str) override;

  // Returns a pointer to the map of external weight section names to 64-byte
  // aligned in-memory byte spans for `model_type`, or `nullptr` if no weights
  // have been loaded into memory for that model type.
  const absl::flat_hash_map<std::string, absl::Span<const std::byte>>*
  GetWeightInMemoryMap(
      proto::LlmMetadata::TfLiteModelType model_type) const override;
  const absl::flat_hash_map<std::string, absl::Span<const std::byte>>*
  GetWeightInMemoryMap(absl::string_view model_type_str) const override;

  // Sets the LLM metadata for this streaming resource instance.
  void SetLlmMetadata(const proto::LlmMetadata& llm_metadata) {
    llm_metadata_ = llm_metadata;
  }

  // Sets the Embedding metadata for this streaming resource instance.
  void SetEmbeddingMetadata(proto::EmbeddingMetadata embedding_metadata) {
    embedding_metadata_ = std::move(embedding_metadata);
  }

  // Sets the Executor metadata for this streaming resource instance.
  void SetExecutorMetadata(proto::ExecutorMetadata executor_metadata) {
    executor_metadata_ = std::move(executor_metadata);
  }

  // Sets the backend constraint for a given model type wire string.
  //
  // args:
  //   model_type_str: The model type wire string (e.g. "tf_lite_embedder").
  //   backend_constraint: The backend constraint for the model.
  void SetTFLiteModelBackendConstraint(absl::string_view model_type_str,
                                       std::string backend_constraint);

  // Sets the backend constraint for a given capability-scoped Protobuf model
  // type enum (e.g. proto::LlmMetadata::TF_LITE_PREFILL_DECODE).
  template <typename TfLiteModelTypeT,
            typename = std::enable_if_t<kIsTfLiteModelType<TfLiteModelTypeT>>>
  void SetTFLiteModelBackendConstraint(TfLiteModelTypeT model_type,
                                       std::string backend_constraint) {
    SetTFLiteModelBackendConstraint(TfLiteModelTypeToWireString(model_type),
                                    std::move(backend_constraint));
  }

  // Sets the preferred activation type for a given model type wire string.
  //
  // args:
  //   model_type_str: The model type wire string (e.g. "tf_lite_embedder").
  //   prefer_activation_type: The preferred activation type for the model.
  void SetTFLiteModelPreferActivationType(absl::string_view model_type_str,
                                          std::string prefer_activation_type);

  // Sets the preferred activation type for a given capability-scoped Protobuf
  // model type enum.
  template <typename TfLiteModelTypeT,
            typename = std::enable_if_t<kIsTfLiteModelType<TfLiteModelTypeT>>>
  void SetTFLiteModelPreferActivationType(TfLiteModelTypeT model_type,
                                          std::string prefer_activation_type) {
    SetTFLiteModelPreferActivationType(TfLiteModelTypeToWireString(model_type),
                                       std::move(prefer_activation_type));
  }

  // Sets the flatbuffer model data for a given model type wire string. This
  // stores the raw flatbuffer in memory so that `GetTFLiteModel` can lazily
  // construct the `litert::Model` without relying on a persistent stream or
  // file.
  //
  // args:
  //   model_type_str: The model type wire string (e.g. "tf_lite_embedder").
  //   buffer: The raw TFLite flatbuffer bytes.
  void SetModelBuffer(absl::string_view model_type_str,
                      std::vector<char> buffer);

  // Sets the flatbuffer model data for a given capability-scoped Protobuf
  // model type enum.
  template <typename TfLiteModelTypeT,
            typename = std::enable_if_t<kIsTfLiteModelType<TfLiteModelTypeT>>>
  void SetModelBuffer(TfLiteModelTypeT model_type, std::vector<char> buffer) {
    SetModelBuffer(TfLiteModelTypeToWireString(model_type), std::move(buffer));
  }

  // Reads `size` bytes from `stream` into a 64-byte aligned host memory buffer
  // (`weights_storage_`) and maps it under the `"tflite_weights"` section key
  // in `weights_in_memory_per_model_[model_type_str]`.
  // The 64-byte alignment satisfies LiteRT/XNNPACK host tensor memory alignment
  // requirements (`LITERT_HOST_MEMORY_BUFFER_ALIGNMENT`).
  //
  // The buffer is also mapped for every other model whose flatbuffer (set
  // with `SetModelBuffer`) references this model's section through the
  // external buffer group `"tflite_weights:<model_type_str>"`, so models of a
  // deduplicated model file can share weights stored once.
  //
  // args:
  //   model_type_str: The model type wire string (e.g. "tf_lite_embedder").
  //   stream: The stream to read the weights from.
  //   size: The number of bytes to read.
  // returns:
  //   OkStatus on success, or the error from reading the stream.
  absl::Status SetWeightsFromStream(absl::string_view model_type_str,
                                    DataStream& stream, size_t size);

  // Streams external weights for a given capability-scoped Protobuf model type
  // enum. See SetWeightsFromStream(absl::string_view, DataStream&, size_t).
  template <typename TfLiteModelTypeT,
            typename = std::enable_if_t<kIsTfLiteModelType<TfLiteModelTypeT>>>
  absl::Status SetWeightsFromStream(TfLiteModelTypeT model_type,
                                    DataStream& stream, size_t size) {
    return SetWeightsFromStream(TfLiteModelTypeToWireString(model_type), stream,
                                size);
  }

  // Releases the host memory holding the external weights of the model type
  // wire string `model_type_str`, and the spans over it returned by
  // `GetWeightInMemoryMap`, including those mapped for other models that share
  // these weights. Does nothing if no weights are stored for it.
  //
  // Only call this once you know the weights will not be read again.
  //
  // args:
  //   model_type_str: The model type wire string (e.g. "tf_lite_embedder").
  void ReleaseWeights(absl::string_view model_type_str);

  // Releases the external weights of a capability-scoped Protobuf model type
  // enum. See ReleaseWeights(absl::string_view).
  template <typename TfLiteModelTypeT,
            typename = std::enable_if_t<kIsTfLiteModelType<TfLiteModelTypeT>>>
  void ReleaseWeights(TfLiteModelTypeT model_type) {
    ReleaseWeights(TfLiteModelTypeToWireString(model_type));
  }

 private:
  std::optional<proto::LlmMetadata> llm_metadata_;
  std::optional<proto::EmbeddingMetadata> embedding_metadata_;
  std::optional<proto::ExecutorMetadata> executor_metadata_;
  // All per-model maps below are keyed by the lowercase model type wire
  // string (e.g. "tf_lite_embedder").
  absl::flat_hash_map<std::string, std::string> backend_constraints_;
  absl::flat_hash_map<std::string, std::string> prefer_activation_types_;
  // In-memory buffers for model flatbuffers.
  // These are cached from the input stream so that 'GetTFLiteModel' can load
  // models without relying on the original stream. This does not include
  // external weights.
  absl::flat_hash_map<std::string, std::vector<char>> model_buffers_;
  absl::flat_hash_map<std::string, std::unique_ptr<litert::Model>> models_;

  // In-memory host storage for model weights. Byte-aligned to
  // LITERT_HOST_MEMORY_BUFFER_ALIGNMENT.
  absl::flat_hash_map<std::string, std::vector<uint8_t>> weights_storage_;

  // In-memory map of external weight section names to 64-byte aligned byte
  // spans in `weights_storage_`. We have spans into weights_storage_ rather
  // than storing the weights directly in this map to guarantee 64-byte
  // alignment.
  absl::flat_hash_map<
      std::string,
      absl::flat_hash_map<std::string, absl::Span<const std::byte>>>
      weights_in_memory_per_model_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_STREAMING_H_
