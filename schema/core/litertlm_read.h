#ifndef THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_READ_H_
#define THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_READ_H_

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/memory_mapped_file.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "sentencepiece_processor.h"  // from @sentencepiece
#include "tflite/model_builder.h"  // from @litert

namespace litert {

namespace lm {

namespace schema {

using litert::lm::proto::LlmMetadata;

struct LitertlmHeader {
  std::unique_ptr<uint8_t[]> buffer;
  const LiteRTLMMetaData* metadata;
  uint32_t major_version = 0;
  uint32_t minor_version = 0;
  uint32_t patch_version = 0;

  // Default constructor
  LitertlmHeader() : buffer(nullptr), metadata(nullptr) {}

  // Constructor that takes ownership of the buffer.
  explicit LitertlmHeader(std::unique_ptr<uint8_t[]>&& buffer_)
      : buffer(std::move(buffer_)) {
    reset(std::move(buffer_));
  }

  // Disable copy constructor and assignment operator to prevent incorrect
  // copying.
  LitertlmHeader(const LitertlmHeader&) = delete;
  LitertlmHeader& operator=(const LitertlmHeader&) = delete;

  // Add a move constructor and move assignment operator for proper transfer of
  // ownership
  LitertlmHeader(LitertlmHeader&& other) noexcept
      : buffer(std::move(other.buffer)), metadata(other.metadata) {
    other.metadata = nullptr;
  }

  // reset function
  void reset(std::unique_ptr<uint8_t[]>&& buffer_) {
    buffer = std::move(buffer_);
    if (buffer) {
      metadata = GetLiteRTLMMetaData(buffer.get());
    } else {
      metadata = nullptr;
    }
  }

  ~LitertlmHeader() {
    // No need to delete, unique_ptr handles it.
  }
};

// Reads the LiteRTLM file header starting at `data`. It is assumed
// that this function can read up to `length` bytes starting at `data`.
//
// Args:
//   data: The pointer to some buffer we can read from.
//   length: The number of bytes that it is valid to read from starting at
//           `data`
//   header: The LitertlmHeader struct with populated schema data.
//
// Returns:
//   absl::OkStatus() if the file was read successfully, or an error status
//   otherwise.
absl::Status ReadHeaderFromLiteRTLM(void* data, size_t length,
                                    LitertlmHeader* header);

// Reads the LiteRTLM file from the given path and populates a header
// data structure (allocating and owning data for the header).
//
// Args:
//   litertlm_path: The path to the LiteRTLM file.
//   header: The LitertlmHeader struct with populated schema data.
//
// Returns:
//   absl::OkStatus() if the file was read successfully, or an error status
//   otherwise.
absl::Status ReadHeaderFromLiteRTLM(const std::string& litertlm_path,
                                    LitertlmHeader* header);

// Reads the LiteRTLM file from the given istream and populates a header
// data structure (allocating and owning data for the header).
//
// Args:
//   litertlm_stream: The input stream to the LiteRTLM file.
//   header: The LitertlmHeader struct with populated schema data.
//
// Returns:
//   absl::OkStatus() if the file was read successfully, or an error status
//   otherwise.
absl::Status ReadHeaderFromLiteRTLM(std::istream& litertlm_stream,
                                    LitertlmHeader* header);

// Read a TF Lite file from the specified section in the LiteRT-LM file.
// Returns InvalidArgumentError if no TFLite is found in that section.
// TFLite libraries expect a caller-provided buffer, so the convention
// here is: we read the TF Lite at the given section from the LiteRT-LM
// file. Upon return, the caller provided mapped_file will be the holder
// of the mmapped buffer (assigned to the passed object via move operations).
absl::Status ReadTFLiteFileFromSection(
    const std::string& litertlm_path, int section_idx,
    std::unique_ptr<tflite::FlatBufferModel>* tflite_model,
    std::unique_ptr<MemoryMappedFile>* mapped_file);

// Read a TF Lite file from the specified section in the LiteRT-LM file.
// Returns InvalidArgumentError if no TFLite is found in that section.
// In this version, the FlatBufferModel creates and owns an Allocation
// that holds the memory for the TFLite model.
absl::Status ReadTFLiteFileFromSection(
    const std::string& litertlm_path, int section_idx,
    std::unique_ptr<tflite::FlatBufferModel>* tflite_model);

// Read any TF Lite from the file (convenience function if the caller knows
// that only 1 TF Lite file exists in the LiteRT-LM file). This function will
// not return an error if there are more than 1 TF Lite sections.
// See above for semantics of the mapped_file.
absl::Status ReadAnyTFLiteFile(
    const std::string& litertlm_path,
    std::unique_ptr<tflite::FlatBufferModel>* tflite_model,
    std::unique_ptr<MemoryMappedFile>* mapped_file);

// Read any TF Lite from the file (convenience function if the caller knows
// that only 1 TF Lite file exists in the LiteRT-LM file). This function will
// not return an error if there are more than 1 TF Lite sections.
// Same semantics as above (FlatBufferModel creates and owns the allocation).
absl::Status ReadAnyTFLiteFile(
    const std::string& litertlm_path,
    std::unique_ptr<tflite::FlatBufferModel>* tflite_model);

// Read a LlmMetadata from the specified section in the LiteRT-LM file.
// Returns InvalidArgumentError if no LlmMetadata are found in that section.
absl::Status ReadLlmMetadataFromSection(const std::string& litertlm_path,
                                        int section_idx,
                                        LlmMetadata* llm_metadata);

// Read any LlmMetadata from the file (convenience function if the caller
// knows that only 1 LlmMetadata proto exists in the LiteRT-LM file).
absl::Status ReadAnyLlmMetadata(const std::string& litertlm_path,
                                LlmMetadata* llm_metadata);

// Read a SP Tokenizer from the specified section in the LiteRT-LM file.
// Returns InvalidArgumentError if no SP Tokenizer is found in that section.
absl::Status ReadSPTokenizerFromSection(
    const std::string& litertlm_path, int section_idx,
    sentencepiece::SentencePieceProcessor* sp_proc);

// Read any SP Tokenizer from the file (convenience function if the caller knows
// that only 1 SP Tokenizer exists in the LiteRT-LM file).
absl::Status ReadAnySPTokenizer(const std::string& litertlm_path,
                                sentencepiece::SentencePieceProcessor* sp_proc);

// Read binary data from the specified section in the LiteRT-LM file.
// Returns InvalidArgumentError if binary data is not found in that section.
absl::Status ReadBinaryDataFromSection(const std::string& litertlm_path,
                                       int section_idx, std::string* data);

// Read any binary data from the file (convenience function if the caller knows
// that only 1 binary data block exists in the LiteRT-LM file).
absl::Status ReadAnyBinaryData(const std::string& litertlm_path,
                               std::string* data);
}  // end namespace schema
}  // end namespace lm
}  // end namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_READ_H_
