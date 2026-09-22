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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LITERT_LM_LOADER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LITERT_LM_LOADER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_read.h"

namespace litert::lm {

inline constexpr uint64_t kLitertLmHeaderMaxSize = 16 * 1024;

// Each buffer is keyed by the data type as the major key and the model type
// wire string as the optional secondary key when the data type is TFLiteModel
// or TFLiteWeights, or name when the data type is GenericBinaryData.
struct BufferKey {
  schema::AnySectionDataType data_type;
  std::optional<std::string> model_type;

  // Constructor for common cases (no model_type needed)
  explicit BufferKey(schema::AnySectionDataType type);

  // Constructor for string-keyed TFLiteModel, TFLiteWeights, SP_Tokenizer,
  // HF_Tokenizer_Zlib, or GenericBinaryData case
  explicit BufferKey(schema::AnySectionDataType type,
                     absl::string_view model_type_str);

  // Constructor for legacy ModelType enum
  explicit BufferKey(schema::AnySectionDataType type, ModelType model_type);

  // Equality operator (REQUIRED for std::unordered_map, good for std::map)
  bool operator==(const BufferKey& other) const;
};

// The hint for the TfLite models, that is used to validate the model settings
// and choose the appropriate backend and activation type.
struct TfLiteSectionHint {
  std::optional<std::string> backend_constraint = std::nullopt;
  std::optional<std::string> prefer_activation_type = std::nullopt;
};

// Extracts the BufferKey and backend constraint from the section metadata.
absl::StatusOr<std::pair<BufferKey, TfLiteSectionHint>>
ExtractBufferKeyAndTfLiteSectionHint(const schema::SectionObject* section);

// Hash function for BufferKey
struct BufferKeyHash {
  size_t operator()(const BufferKey& k) const;
};

// A class to load the Litert LM model from the .litertlm file. The loader will
// read the model header from and map the sections to the section buffers.
class LitertLmLoader {
 public:
  // Creates a LitertLmLoader from the model file. The loader will read the
  // model header from and map the sections to the section buffers.
  static absl::StatusOr<std::unique_ptr<LitertLmLoader>> Create(
      ScopedFile model_file);

  // Creates a LitertLmLoader from a shared ScopedFile.
  static absl::StatusOr<std::unique_ptr<LitertLmLoader>> Create(
      std::shared_ptr<ScopedFile> shared_scoped_file);

  // Creates a LitertLmLoader from an already memory-mapped model file.
  // This is useful when the file is managed externally.
  //
  // `scoped_file` is optional and does not affect how sections are loaded:
  // sections are still served directly out of `memory_mapped_model_file`. It
  // only backs `GetScopedFile()`, which callers need when a section has to be
  // referenced by file descriptor rather than by pointer - most notably
  // external weights, which LiteRT loads via
  // `Options::SetExternalWeightScopedFile`. Pass it whenever a path or
  // descriptor for the same file is available.
  static absl::StatusOr<std::unique_ptr<LitertLmLoader>> Create(
      std::shared_ptr<MemoryMappedFile> memory_mapped_model_file,
      std::shared_ptr<ScopedFile> scoped_file = nullptr);

  // Returns the tokenizer section buffer for the SentencePiece tokenizer
  // for a given ModelType or wire string.
  std::optional<litert::BufferRef<uint8_t>> GetSentencePieceTokenizer(
      ModelType model_type = ModelType::kTfLitePrefillDecode);
  std::optional<litert::BufferRef<uint8_t>> GetSentencePieceTokenizer(
      absl::string_view model_type_str);

  // Returns the tokenizer section buffer for the HuggingFace tokenizer
  // for a given ModelType or wire string.
  std::optional<litert::OwningBufferRef<uint8_t>> GetHuggingFaceTokenizer(
      ModelType model_type = ModelType::kTfLitePrefillDecode);
  std::optional<litert::OwningBufferRef<uint8_t>> GetHuggingFaceTokenizer(
      absl::string_view model_type_str);

  // Returns the TFLite model section buffer.
  litert::BufferRef<uint8_t> GetTFLiteModel(ModelType model_type);
  litert::BufferRef<uint8_t> GetTFLiteModel(absl::string_view model_type_str);
  template <typename TfLiteModelTypeT,
            typename = std::enable_if_t<std::is_enum_v<TfLiteModelTypeT>>>
  litert::BufferRef<uint8_t> GetTFLiteModel(TfLiteModelTypeT model_type) {
    return GetTFLiteModel(TfLiteModelTypeToWireString(model_type));
  }

  litert::BufferRef<uint8_t> GetTFLiteWeights(ModelType model_type);
  litert::BufferRef<uint8_t> GetTFLiteWeights(absl::string_view model_type_str);

  // Returns the TFLite model backend constraint.
  // If not found, returns std::nullopt.
  std::optional<std::string> GetTFLiteModelBackendConstraint(
      ModelType model_type);
  std::optional<std::string> GetTFLiteModelBackendConstraint(
      absl::string_view model_type_str);

  // Returns the TFLite model section buffer's prefer activation type.
  // If not found, returns std::nullopt.
  std::optional<std::string> GetTFLiteModelPreferActivationType(
      ModelType model_type);
  std::optional<std::string> GetTFLiteModelPreferActivationType(
      absl::string_view model_type_str);

  // Returns the tokenizer section buffer.
  litert::BufferRef<uint8_t> GetLlmMetadata() {
    return GetSectionBuffer(
               BufferKey(schema::AnySectionDataType_LlmMetadataProto))
        .value();
  }

  // Returns the executor metadata section buffer.
  std::optional<litert::BufferRef<uint8_t>> GetExecutorMetadata() {
    return GetSectionBuffer(
        BufferKey(schema::AnySectionDataType_ExecutorMetadataProto));
  }

  // Returns the embedding metadata section buffer. If not found, returns
  // std::nullopt.
  std::optional<litert::BufferRef<uint8_t>> GetEmbeddingMetadata() {
    return GetSectionBuffer(
        BufferKey(schema::AnySectionDataType_EmbeddingMetadataProto));
  }

  // Returns the TTS metadata section buffer. If not found, returns
  // std::nullopt.
  std::optional<litert::BufferRef<uint8_t>> GetTtsMetadata();

  // Returns the ASR metadata section buffer. If not found, returns
  // std::nullopt.
  std::optional<litert::BufferRef<uint8_t>> GetAsrMetadata();

  // Returns a GenericBinaryData section buffer matching the given name.
  std::optional<litert::BufferRef<uint8_t>> GetGenericBinaryData(
      absl::string_view name);

  // Returns the names of all named GenericBinaryData sections in the model
  // container.
  std::vector<std::string> GetGenericBinaryDataNames() const;

  absl::StatusOr<std::pair<size_t, size_t>> GetSectionLocation(
      BufferKey buffer_key) const;

  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile();

  absl::StatusOr<std::shared_ptr<ScopedFile>> GetSharedScopedFile();

 private:
  explicit LitertLmLoader(ScopedFile model_file)
      : model_source_(std::make_shared<ScopedFile>(std::move(model_file))) {}

  explicit LitertLmLoader(std::shared_ptr<ScopedFile> shared_scoped_file)
      : model_source_(std::move(shared_scoped_file)) {}

  explicit LitertLmLoader(
      std::shared_ptr<MemoryMappedFile> memory_mapped_model_file,
      std::shared_ptr<ScopedFile> scoped_file = nullptr)
      : model_source_(std::move(memory_mapped_model_file)),
        scoped_file_(std::move(scoped_file)) {}
  // Initializes the LitertLmLoader. Includes reading the model header and
  // recording the section locations for on-demand loading later.
  absl::Status Initialize();
  absl::Status MapSection(BufferKey buffer_key, uint64_t begin_offset,
                          uint64_t end_offset)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(section_buffers_mutex_);
  // Returns the section buffer for the given buffer key. Will map the section
  // if it has not been mapped yet. If not found, returns std::nullopt.
  std::optional<litert::BufferRef<uint8_t>> GetSectionBuffer(
      BufferKey buffer_key) ABSL_LOCKS_EXCLUDED(section_buffers_mutex_);

  // The model file to be loaded, can be either a ScopedFile or a
  // memory-mapped file.
  std::variant<std::shared_ptr<ScopedFile>, std::shared_ptr<MemoryMappedFile>>
      model_source_;

  // An optional descriptor for the same file as `model_source_`, set only when
  // `model_source_` holds a MemoryMappedFile. Sections are never read through
  // it; it exists so `GetScopedFile()` can still hand out a descriptor for
  // features that require one (e.g. external weights).
  std::shared_ptr<ScopedFile> scoped_file_;

  // The header of the model file. Use this to understand what sections are
  // available and their offsets.
  schema::LitertlmHeader header_;

  // The section locations in the model file. This is populated during
  // initialization and later used to map the section buffers to the section
  // memory mapped files on-demand.
  ::std::unordered_map<
      BufferKey, std::pair</*begin_offset*/ uint64_t, /*end_offset=*/uint64_t>,
      BufferKeyHash>
      section_locations_;

  absl::Mutex section_buffers_mutex_;
  // The section memory mapped files - stored here to ensure they are not
  // unmapped while in use. On Windows, these MemoryMappedFiles may contain more
  // than the current section's data because Windows has a data alignment of
  // 64KB but the LiteRT LM file has a 16KB alignment.
  ::std::unordered_map<BufferKey, std::unique_ptr<MemoryMappedFile>,
                       BufferKeyHash>
      section_memory_mapped_files_ ABSL_GUARDED_BY(section_buffers_mutex_);
  // The section buffers. Unlike the section_memory_mapped_files_, these
  // buffers point to only the data of the each section, even on Windows.
  ::std::unordered_map<BufferKey, litert::BufferRef<uint8_t>, BufferKeyHash>
      section_buffers_ ABSL_GUARDED_BY(section_buffers_mutex_);

  // Map of all the sections' section info.
  ::std::unordered_map<BufferKey, TfLiteSectionHint, BufferKeyHash>
      section_hints_map_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LITERT_LM_LOADER_H_
