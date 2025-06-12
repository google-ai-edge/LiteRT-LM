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
#include <unordered_map>
#include <utility>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "schema/core/litertlm_header_schema_generated.h"

namespace litert::lm {

// Each buffer is keyed by the data type as the major key and the model type
// as the optional secondary key when the data type is TFLITE_MODEL_DATA.
struct BufferKey {
  schema::AnySectionDataType data_type;
  std::optional<ModelType> model_type;  // This can be nullopt for data types
                                        // other than TFLITE_MODEL_DATA!

  // Constructor for common cases (no ModelType needed)
  explicit BufferKey(schema::AnySectionDataType type)
      : data_type(type), model_type(std::nullopt) {}

  // Constructor for TFLITE_MODEL_DATA case
  explicit BufferKey(schema::AnySectionDataType type,
                     ModelType model_type)
      : data_type(type), model_type(model_type) {
    // Optional: Add an assertion here if 'type' MUST be TFLITE_MODEL_DATA for
    // model_t to be set
    ABSL_CHECK(type == schema::AnySectionDataType_TFLiteModel &&
               "ModelType should only be provided for TFLITE_MODEL_DATA");
  }

  // Equality operator (REQUIRED for std::unordered_map, good for std::map)
  bool operator==(const BufferKey& other) const {
    return data_type == other.data_type && model_type == other.model_type;
  }
};

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
  explicit LitertLmLoader(ScopedFile model_file)
      : model_file_(std::move(model_file)) {
    ABSL_CHECK_OK(Initialize());
  }
  // Returns the tokenizer section buffer for the SentencePiece tokenizer.
  std::optional<litert::BufferRef<uint8_t>> GetSentencePieceTokenizer() {
    if (!section_buffers_.contains(
            BufferKey(schema::AnySectionDataType_SP_Tokenizer))) {
      return std::nullopt;
    }
    return section_buffers_[BufferKey(
        schema::AnySectionDataType_SP_Tokenizer)];
  }

  // Returns the tokenizer section buffer for the HuggingFace tokenizer.
  std::optional<litert::BufferRef<uint8_t>> GetHuggingFaceTokenizer() {
    if (!section_buffers_.contains(
            BufferKey(schema::AnySectionDataType_HF_Tokenizer_Zlib))) {
      return std::nullopt;
    }
    if (this->hf_tokenizer_data_.empty()) {
      ABSL_LOG(ERROR) << " HuggingFace tokenizer data already cleared.";
      return std::nullopt;
    }
    return section_buffers_[BufferKey(
        schema::AnySectionDataType_HF_Tokenizer_Zlib)];
  }

  // Clears the HuggingFace tokenizer json data to remove the JSON data from
  // memory.
  void ClearHuggingFaceTokenizerJson() { this->hf_tokenizer_data_.clear(); }

  // Returns the TFLite model section buffer.
  litert::BufferRef<uint8_t> GetTFLiteModel(ModelType model_type) {
    return section_buffers_[BufferKey(
        schema::AnySectionDataType_TFLiteModel, model_type)];
  };

  // Returns the tokenizer section buffer.
  litert::BufferRef<uint8_t> GetLlmMetadata() {
    return section_buffers_[BufferKey(
        schema::AnySectionDataType_LlmMetadataProto)];
  }

 private:
  // Initializes the LitertLmLoader. Includes reading the model header and
  // mapping the sections to the section buffers.
  absl::Status Initialize();
  // Maps the sections to the section buffers.
  absl::Status MapSections();
  // The model file to be loaded.
  ScopedFile model_file_;
  // The model_file_ mapped to a MemoryMappedFile.
  ::std::unique_ptr<MemoryMappedFile> memory_mapped_file_;
  // The HuggingFace tokenizer JSON data while needed.
  std::vector<uint8_t> hf_tokenizer_data_;

  // TODO (b/413793273): Add the extra names to the key to differentiate
  // between the TFLite models.
  ::std::unordered_map<BufferKey, BufferRef<uint8_t>, BufferKeyHash>
      section_buffers_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LITERT_LM_LOADER_H_
