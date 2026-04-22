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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
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
// as the optional secondary key when the data type is TFLiteModel or
// TFLiteWeights.
struct BufferKey {
  schema::AnySectionDataType data_type;
  std::optional<ModelType>
      model_type;  // This can be nullopt for data types
                   // other than TFLiteModel or TFLiteWeights!

  // Constructor for common cases (no ModelType needed)
  explicit BufferKey(schema::AnySectionDataType type)
      : data_type(type), model_type(std::nullopt) {}

  // Constructor for TFLiteModel or TFLiteWeights case
  explicit BufferKey(schema::AnySectionDataType type, ModelType model_type)
      : data_type(type), model_type(model_type) {
    if (type != schema::AnySectionDataType_TFLiteModel &&
        type != schema::AnySectionDataType_TFLiteWeights) {
      ABSL_LOG(ERROR) << "ModelType should only be provided for TFLiteModel or "
                         "TFLiteWeights";
    }
  }

  // Equality operator (REQUIRED for std::unordered_map, good for std::map)
  bool operator==(const BufferKey& other) const {
    return data_type == other.data_type && model_type == other.model_type;
  }
};

// Extracts the BufferKey and backend constraint from the section metadata.
absl::StatusOr<std::pair<BufferKey, std::optional<std::string>>>
ExtractBufferKeyAndBackendConstraint(const schema::SectionObject* section);

// Hash function for BufferKey
struct BufferKeyHash {
  size_t operator()(const BufferKey& k) const {
    size_t h1 = std::hash<schema::AnySectionDataType>{}(k.data_type);
    size_t h2 = 0;
    if (k.model_type.has_value()) {
      h2 = std::hash<ModelType>{}(k.model_type.value());
    }
    // A simple hash combine. For more robust hashing, consider
    // boost::hash_combine
    return h1 ^ (h2 << 1);
  }
};

// A class to load the Litert LM model from the .litertlm file. The loader will
// read the model header from and map the sections to the section buffers.
class LitertLmLoader {
 public:
  // Creates a LitertLmLoader from the model file. The loader will read the
  // model header from and map the sections to the section buffers.
  static absl::StatusOr<std::unique_ptr<LitertLmLoader>> Create(
      ScopedFile model_file);

  // Creates a LitertLmLoader from an already memory-mapped model file.
  // This is useful when the file is managed externally.
  static absl::StatusOr<std::unique_ptr<LitertLmLoader>> Create(
      std::shared_ptr<MemoryMappedFile> memory_mapped_model_file);

  // Returns the tokenizer section buffer for the SentencePiece tokenizer.
  // If not found, returns std::nullopt.
  std::optional<litert::BufferRef<uint8_t>> GetSentencePieceTokenizer() {
    return GetSectionBuffer(BufferKey(schema::AnySectionDataType_SP_Tokenizer));
  }

  // Returns the tokenizer section buffer for the HuggingFace tokenizer.
  // If not found, returns std::nullopt.
  std::optional<litert::OwningBufferRef<uint8_t>> GetHuggingFaceTokenizer();

  // Returns the TFLite model section buffer.
  litert::BufferRef<uint8_t> GetTFLiteModel(ModelType model_type) {
    auto optional_section_buffer = GetSectionBuffer(
        BufferKey(schema::AnySectionDataType_TFLiteModel, model_type));
    if (optional_section_buffer.has_value()) {
      return optional_section_buffer.value();
    }
    ABSL_LOG(WARNING) << "TFLite model for type: "
                      << ModelTypeToString(model_type)
                      << " not found. Skipping.";
    return litert::BufferRef<uint8_t>();
  };

  litert::BufferRef<uint8_t> GetTFLiteWeights(ModelType model_type) {
    auto optional_section_buffer = GetSectionBuffer(
        BufferKey(schema::AnySectionDataType_TFLiteWeights, model_type));
    if (optional_section_buffer.has_value()) {
      return optional_section_buffer.value();
    }
    ABSL_LOG(WARNING) << "TFLite weights for type: "
                      << ModelTypeToString(model_type)
                      << " not found. Skipping.";
    return litert::BufferRef<uint8_t>();
  };

  // Returns the TFLite model section buffer.
  std::optional<std::string> GetTFLiteModelBackendConstraint(
      ModelType model_type) {
    if (section_backend_constraint_.contains(
            BufferKey(schema::AnySectionDataType_TFLiteModel, model_type))) {
      return section_backend_constraint_[BufferKey(
          schema::AnySectionDataType_TFLiteModel, model_type)];
    }
    ABSL_LOG(WARNING) << "TFLite model type: " << ModelTypeToString(model_type)
                      << " not found for backend constraints. Skipping.";
    return std::nullopt;
  };

  // Returns the tokenizer section buffer.
  litert::BufferRef<uint8_t> GetLlmMetadata() {
    return GetSectionBuffer(
               BufferKey(schema::AnySectionDataType_LlmMetadataProto))
        .value();
  }

  absl::StatusOr<std::pair<size_t, size_t>> GetSectionLocation(
      BufferKey buffer_key) const;

  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile();

 private:
  explicit LitertLmLoader(ScopedFile model_file)
      : model_source_(std::move(model_file)) {}

  explicit LitertLmLoader(
      std::shared_ptr<MemoryMappedFile> memory_mapped_model_file)
      : model_source_(std::move(memory_mapped_model_file)) {}
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
  std::variant<ScopedFile, std::shared_ptr<MemoryMappedFile>> model_source_;

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

  // Map of all the sections' metadata, for now, focusing on the backend
  // constraints
  ::std::unordered_map<BufferKey, std::string, BufferKeyHash>
      section_backend_constraint_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LITERT_LM_LOADER_H_
