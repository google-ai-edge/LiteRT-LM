#include "schema/core/litertlm_read.h"

#include <zlib.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // NOLINT
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_utils.h"
#include "sentencepiece_processor.h"  // from @sentencepiece
#include "tflite/model_builder.h"  // from @litert
#include "zconf.h"  // from @zlib

namespace litert {
namespace lm {
namespace schema {

using litert::lm::proto::LlmMetadata;

absl::Status ReadHeaderFromLiteRTLM(std::istream& litertlm_stream,
                                    LitertlmHeader* header) {
  // 0. Read magic number and version.
  char magic_number[8];
  litertlm_stream.read(magic_number, 8);
  if (litertlm_stream.gcount() != 8 ||
      std::string(magic_number, 8) != "LITERTLM") {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid magic number or failed to read: %s",
                        std::string(magic_number, litertlm_stream.gcount())));
  }

  litertlm_stream.read(reinterpret_cast<char*>(&header->major_version),
                       sizeof(uint32_t));
  litertlm_stream.read(reinterpret_cast<char*>(&header->minor_version),
                       sizeof(uint32_t));
  litertlm_stream.read(reinterpret_cast<char*>(&header->patch_version),
                       sizeof(uint32_t));

  if (!litertlm_stream) {
    return absl::InternalError("Failed to read version bytes.");
  }

  // If major version doesn't match our current major version,
  // bail out for now
  if (header->major_version != LITERTLM_MAJOR_VERSION) {
    return absl::UnimplementedError(
        absl::StrFormat("Unimplemented Error: This reader doesn't support "
                        "version %d, expected version %d.",
                        header->major_version, LITERTLM_MAJOR_VERSION));
  }

  // 1. Skip 4 bytes of padding.
  litertlm_stream.ignore(4);
  if (!litertlm_stream) {
    return absl::InternalError("Failed to skip padding after version.");
  }

  // 2. Read the header end offset.
  uint64_t header_end_offset;
  litertlm_stream.read(reinterpret_cast<char*>(&header_end_offset),
                       sizeof(uint64_t));
  if (!litertlm_stream) {
    return absl::InternalError("Failed to read header end offset.");
  }


  // Calculate the header size.
  std::streampos current_position = litertlm_stream.tellg();
  if (current_position == -1) {
    return absl::InternalError("Failed to get current stream position.");
  }
  // Ensure header_end_offset is greater than or equal to current_position
  if (header_end_offset < static_cast<uint64_t>(current_position)) {
    return absl::InvalidArgumentError(
        "Invalid header end offset: smaller than current position.");
  }
  uint64_t header_size =
      header_end_offset - static_cast<uint64_t>(current_position);

  // 3. Read the header data into a buffer.
  auto header_buffer = std::make_unique<uint8_t[]>(header_size);

  litertlm_stream.read(reinterpret_cast<char*>(header_buffer.get()),
                       header_size);
  if (!litertlm_stream) {
    return absl::InternalError("Failed to read header data.");
  }

  header->reset(std::move(header_buffer));
  return absl::OkStatus();
}

// The public function that takes a file path.
absl::Status ReadHeaderFromLiteRTLM(const std::string& litertlm_path,
                                    LitertlmHeader* header) {
  std::ifstream input_file_stream(litertlm_path, std::ios::binary);
  if (!input_file_stream.is_open()) {
    return absl::InternalError(
        absl::StrFormat("Could not open file: %s", litertlm_path));
  }

  absl::Status status = ReadHeaderFromLiteRTLM(input_file_stream, header);

  return status;
}

// The public function that takes a pointer and a length.
absl::Status ReadHeaderFromLiteRTLM(void* data, std::size_t length,
                                    LitertlmHeader* header) {
  char* char_data = static_cast<char*>(data);
  // Create a streambuf instance based on the given buffer info.
  MemoryStreamBuf sbuf(char_data, length);
  // Create an istream using the custom streambuf.
  std::istream input_stream(&sbuf);

  absl::Status status = ReadHeaderFromLiteRTLM(input_stream, header);
  // Cleanup of the streambuf and istream is automatic upon exit.
  return status;
}

template <AnySectionDataType SectionT, typename T, typename... Args>
absl::Status ReadValueTFromSection(
    const std::string& litertlm_path, int section_idx, T* data,
    std::function<absl::Status(const std::string&, uint64_t, uint64_t, T*,
                               Args...)>
        read_section_into_t,
    Args&&... additional_args) {
  LitertlmHeader header;

  // Read the header information.
  RETURN_IF_ERROR(ReadHeaderFromLiteRTLM(litertlm_path, &header)); // NOLINT

  auto sections = header.metadata->section_metadata()->objects();
  // Check if the section_idx is valid.
  if (section_idx < 0 || section_idx >= sections->size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid section index: %d, num sections = %d",
                        section_idx, sections->size()));
  }

  const SectionObject* section = sections->Get(section_idx);

  // Verify that the section type is correct.
  if (section->data_type() != SectionT) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Section %d is not the expected type.  It is: %s, expected %d",
        section_idx, AnySectionDataTypeToString(section->data_type()),
        static_cast<int>(SectionT)));
  }

  // Calculate the size of the data.
  size_t end_offset = section->end_offset();
  size_t begin_offset = section->begin_offset();
  size_t data_size = end_offset - begin_offset;
  if (data_size == 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Section %d has zero size.", section_idx));
  }

  return read_section_into_t(litertlm_path, begin_offset, end_offset, data,
                             std::forward<Args>(additional_args)...);
}

// Function to read TFLite model data from a section.
absl::Status ReadSectionIntoTFLite(
    const std::string& litertlm_path, uint64_t begin_offset,
    uint64_t end_offset, std::unique_ptr<tflite::FlatBufferModel>* tflite_model,
    std::unique_ptr<MemoryMappedFile>* mapped_file) {
  size_t model_size = end_offset - begin_offset;

  // Create the scoped file
  auto model_file = lm::ScopedFile::Open(litertlm_path);

  litert::lm::ScopedFile::PlatformFile platform_file =
      model_file.value().file();

  absl::StatusOr<std::unique_ptr<MemoryMappedFile>> mmap_status =
      litert::lm::MemoryMappedFile::Create(platform_file, begin_offset,
                                           model_size, "section");
  if (!mmap_status.ok()) {
    ABSL_LOG(ERROR) << "Failed to create memory-mapped file: "
                    << mmap_status.status();
    return absl::InternalError("Failed to create memory-mapped file");
  }

  *mapped_file = std::move(*mmap_status);
  *tflite_model = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>((*mapped_file)->data()),
      (*mapped_file)->length());
  return absl::OkStatus();
}

// Function to read LlmMetadata from a section.
absl::Status ReadSectionIntoLlmMetadata(const std::string& litertlm_path,
                                        uint64_t begin_offset,
                                        uint64_t end_offset,
                                        LlmMetadata* llm_metadata) {
  // Create an ifstream from the file.
  std::ifstream input_file_stream(litertlm_path, std::ios::binary);
  if (!input_file_stream.is_open()) {
    return absl::InternalError(
        absl::StrFormat("Could not open file: %s", litertlm_path));
  }
  input_file_stream.seekg(begin_offset);

  size_t size = end_offset - begin_offset;
  std::unique_ptr<char[]> buffer(new char[size]);
  input_file_stream.read(buffer.get(), size);
  if (!input_file_stream) {
    return absl::InternalError(
        absl::StrFormat("Could not read %d bytes from stream.", size));
  }
  llm_metadata->ParseFromArray(buffer.get(), size);
  return absl::OkStatus();
}

// Function to read a SP Tokenizer from a section.
absl::Status ReadSectionIntoSPTokenizer(
    const std::string& litertlm_path, uint64_t begin_offset,
    uint64_t end_offset, sentencepiece::SentencePieceProcessor* sp_proc) {
  // Create an ifstream from the file.
  std::ifstream input_file_stream(litertlm_path, std::ios::binary);
  if (!input_file_stream.is_open()) {
    return absl::InternalError(
        absl::StrFormat("Could not open file: %s", litertlm_path));
  }
  input_file_stream.seekg(begin_offset);

  size_t size = end_offset - begin_offset;
  std::unique_ptr<char[]> buffer(new char[size]);
  input_file_stream.read(buffer.get(), size);
  if (!input_file_stream) {
    return absl::InternalError(
        absl::StrFormat("Could not read %d bytes from stream.", size));
  }
  absl::string_view buffer_view(buffer.get(), size);
  return sp_proc->LoadFromSerializedProto(buffer_view);
}

absl::Status ReadSectionIntoBinaryData(const std::string& litertlm_path,
                                       uint64_t begin_offset,
                                       uint64_t end_offset,
                                       std::vector<uint8_t>* data) {
  // Create an ifstream from the file.
  std::ifstream input_file_stream(litertlm_path, std::ios::binary);
  if (!input_file_stream.is_open()) {
    return absl::InternalError(
        absl::StrFormat("Could not open file: %s", litertlm_path));
  }
  input_file_stream.seekg(begin_offset);

  size_t size = end_offset - begin_offset;
  data->resize(size);
  input_file_stream.read(reinterpret_cast<char*>(data->data()), size);
  if (!input_file_stream) {
    return absl::InternalError(
        absl::StrFormat("Could not read %d bytes from stream.", size));
  }
  return absl::OkStatus();
}

absl::Status DecompressData(uint8_t* compressed_data,
                            size_t compressed_data_length,
                            std::vector<uint8_t>* output) {
  // The first uint64_t bytes contain the compressed data size. Initialize the
  // uncompressed buffer.
  if (compressed_data_length < sizeof(uint64_t)) {
    return absl::InternalError("Data too short to contain compressed size.");
  }
  size_t uncompressed_buffer_size;
  std::memcpy(&uncompressed_buffer_size, compressed_data, sizeof(uint64_t));
  output->resize(uncompressed_buffer_size);

  // Decompress the data.
  int result = uncompress(
      output->data(), &uncompressed_buffer_size,
      reinterpret_cast<const Bytef*>(compressed_data + sizeof(uint64_t)),
      compressed_data_length - sizeof(uint64_t));
  switch (result) {
    case Z_OK:
      return absl::OkStatus();
    case Z_BUF_ERROR:
      return absl::InternalError("Output buffer was not large enough.");
    case Z_MEM_ERROR:
      return absl::InternalError("Not enough memory to decompress.");
    case Z_DATA_ERROR:
      return absl::InternalError("Invalid or incomplete compressed data.");
    default:
      return absl::InternalError("Unknown decompression error " +
                                 std::to_string(result));
  }
}

absl::Status ReadSectionIntoHfTokenizerJsonData(
    const std::string& litertlm_path, uint64_t begin_offset,
    uint64_t end_offset, std::string* output) {
  std::vector<uint8_t> compressed_data;
  RETURN_IF_ERROR(ReadSectionIntoBinaryData(litertlm_path, // NOLINT
                                            begin_offset,
                                            end_offset, &compressed_data));

  std::vector<uint8_t> uncompressed_data;
  RETURN_IF_ERROR(DecompressData(compressed_data.data(), // NOLINT
                                 compressed_data.size(),
                                 &uncompressed_data));
  output->assign(reinterpret_cast<char*>(uncompressed_data.data()),
                 uncompressed_data.size());

  return absl::OkStatus();
}

absl::Status ReadTFLiteFileFromSection(
    const std::string& litertlm_path, int section_idx,
    std::unique_ptr<tflite::FlatBufferModel>* tflite_model,
    std::unique_ptr<MemoryMappedFile>* mapped_file) {
  return ReadValueTFromSection<AnySectionDataType_TFLiteModel,
                               std::unique_ptr<tflite::FlatBufferModel>,
                               std::unique_ptr<MemoryMappedFile>*>(
      litertlm_path, section_idx, tflite_model,
      std::function<absl::Status(const std::string&, uint64_t, uint64_t,
                                 std::unique_ptr<tflite::FlatBufferModel>*,
                                 std::unique_ptr<MemoryMappedFile>*)>(
          ReadSectionIntoTFLite),
      std::forward<std::unique_ptr<MemoryMappedFile>*>(mapped_file));
}

absl::Status ReadLlmMetadataFromSection(const std::string& litertlm_path,
                                        int section_idx,
                                        LlmMetadata* llm_metadata) {
  return ReadValueTFromSection<AnySectionDataType_LlmMetadataProto,
                               LlmMetadata>(
      litertlm_path, section_idx, llm_metadata,
      std::function<absl::Status(const std::string&, uint64_t, uint64_t,
                                 LlmMetadata*)>(ReadSectionIntoLlmMetadata));
}

absl::Status ReadSPTokenizerFromSection(
    const std::string& litertlm_path, int section_idx,
    sentencepiece::SentencePieceProcessor* sp_proc) {
  return ReadValueTFromSection<AnySectionDataType_SP_Tokenizer,
                               sentencepiece::SentencePieceProcessor>(
      litertlm_path, section_idx, sp_proc,
      std::function<absl::Status(const std::string&, uint64_t, uint64_t,
                                 sentencepiece::SentencePieceProcessor*)>(
          ReadSectionIntoSPTokenizer));
}

absl::Status ReadHfTokenizerJsonFromSection(const std::string& litertlm_path,
                                            int section_idx,
                                            std::string* tokenizer_json) {
  return ReadValueTFromSection<AnySectionDataType_HF_Tokenizer_Zlib,
                               std::string>(
      litertlm_path, section_idx, tokenizer_json,
      std::function<absl::Status(const std::string&, uint64_t, uint64_t,
                                 std::string*)>(
          ReadSectionIntoHfTokenizerJsonData));
}

absl::Status ReadBinaryDataFromSection(const std::string& litertlm_path,
                                       int section_idx,
                                       std::vector<uint8_t>* data) {
  return ReadValueTFromSection<AnySectionDataType_GenericBinaryData,
                               std::vector<uint8_t>>(
      litertlm_path, section_idx, data,
      std::function<absl::Status(const std::string&, uint64_t, uint64_t,
                                 std::vector<uint8_t>*)>(
          ReadSectionIntoBinaryData));
}

template <AnySectionDataType SectionT, typename T, typename... Args>
absl::Status ReadAnyT(
    const std::string& litertlm_path, T* data,
    std::function<absl::Status(const std::string&, int, T*, Args...)>
        read_data_from_section,
    Args&&... additional_args) {
  LitertlmHeader header;

  // Read the header information.
  RETURN_IF_ERROR(ReadHeaderFromLiteRTLM(litertlm_path, &header)); // NOLINT

  // Search for the first section with the specified type.
  auto sections = header.metadata->section_metadata()->objects();
  int section_index = -1;
  for (size_t i = 0; i < sections->size(); ++i) {
    const SectionObject* section = sections->Get(i);
    if (section->data_type() == SectionT) {
      section_index = static_cast<int>(i);
      break;
    }
  }

  if (section_index == -1) {
    return absl::NotFoundError("No matching section found in the file.");
  }

  // Read the data from the found section.
  return read_data_from_section(litertlm_path, section_index, data,
                                std::forward<Args>(additional_args)...);
}

// Instantiation of ReadAnyT for TFLite models.
absl::Status ReadAnyTFLiteFile(
    const std::string& litertlm_path,
    std::unique_ptr<tflite::FlatBufferModel>* tflite_model,
    std::unique_ptr<MemoryMappedFile>* mapped_file) {
  return ReadAnyT<AnySectionDataType_TFLiteModel,
                  std::unique_ptr<tflite::FlatBufferModel>,
                  std::unique_ptr<MemoryMappedFile>*>(
      litertlm_path, tflite_model,
      std::function<absl::Status(
          const std::string&, int, std::unique_ptr<tflite::FlatBufferModel>*,
          std::unique_ptr<MemoryMappedFile>*)>(ReadTFLiteFileFromSection),
      std::forward<std::unique_ptr<MemoryMappedFile>*>(mapped_file));
}

// Instantiation of ReadAnyT for LlmMetadata.
absl::Status ReadAnyLlmMetadata(const std::string& litertlm_path,
                                LlmMetadata* llm_metadata) {
  return ReadAnyT<AnySectionDataType_LlmMetadataProto, LlmMetadata>(
      litertlm_path, llm_metadata,
      std::function<absl::Status(const std::string&, int, LlmMetadata*)>(
          ReadLlmMetadataFromSection));
}

// Instantiation of ReadAnyT for LlmMetadata.
absl::Status ReadAnySPTokenizer(
    const std::string& litertlm_path,
    sentencepiece::SentencePieceProcessor* sp_proc) {
  return ReadAnyT<AnySectionDataType_SP_Tokenizer,
                  sentencepiece::SentencePieceProcessor>(
      litertlm_path, sp_proc,
      std::function<absl::Status(const std::string&, int,
                                 sentencepiece::SentencePieceProcessor*)>(
          ReadSPTokenizerFromSection));
}

// Read any binary data from the file (convenience function if the caller knows
// that only 1 binary data block exists in the LiteRT-LM file).
absl::Status ReadAnyBinaryData(const std::string& litertlm_path,
                               std::vector<uint8_t>* data) {
  return ReadAnyT<AnySectionDataType_GenericBinaryData, std::vector<uint8_t>>(
      litertlm_path, data,
      std::function<absl::Status(const std::string&, int,
                                 std::vector<uint8_t>*)>(
          ReadBinaryDataFromSection));
}

absl::Status ReadAnyHfTokenizerJson(const std::string& litertlm_path,
                                    std::string* tokenizer_json) {
  return ReadAnyT<AnySectionDataType_HF_Tokenizer_Zlib, std::string>(
      litertlm_path, tokenizer_json,
      std::function<absl::Status(const std::string&, int, std::string*)>(
          ReadHfTokenizerJsonFromSection));
}

}  // namespace schema
}  // namespace lm
}  // namespace litert
