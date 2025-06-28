#include "schema/core/litertlm_export.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_section.h"
#include "schema/core/litertlm_utils.h"
#include "runtime/util/status_macros.h" //NOLINT

namespace litert {
namespace lm {
namespace schema {

constexpr int kHeaderBeginByteOffset = 32;
constexpr int kHeaderEndLocationByteOffset = 24;
constexpr int kBlockSize = 16 * 1024;

absl::Status WriteHeader(
    flatbuffers::FlatBufferBuilder& builder, std::ostream& output_stream,
    const std::vector<KVPair>& system_metadata_map,
    const std::vector<std::vector<KVPair>>& section_items_maps,
    const std::vector<std::pair<uint64_t, uint64_t>>& section_offsets,
    const std::vector<AnySectionDataType>& section_types) {
  auto system_metadata_offset =
      CreateSystemMetadata(builder, builder.CreateVector(system_metadata_map));

  // All Section Object data.
  std::vector<flatbuffers::Offset<SectionObject>> section_objects_vector;
  for (std::size_t i = 0; i < section_types.size(); ++i) {
    auto section_object = CreateSectionObject(
        builder, builder.CreateVector(section_items_maps[i]),
        section_offsets[i].first, section_offsets[i].second, section_types[i]);
    section_objects_vector.push_back(section_object);
  }

  auto section_metadata_offset = CreateSectionMetadata(
      builder, builder.CreateVector(section_objects_vector));

  // Finish created LiteRTLMMetaData.
  auto root_offset = CreateLiteRTLMMetaData(builder, system_metadata_offset,
                                            section_metadata_offset);
  builder.Finish(root_offset);

  uint8_t* buffer = builder.GetBufferPointer();
  size_t size = builder.GetSize();
  ABSL_DLOG(INFO) << "Header size is: " << size;

  output_stream.write(reinterpret_cast<const char*>(buffer), size);
  output_stream.flush();
  if (!output_stream.good()) {
    return absl::Status(absl::StatusCode::kInternal,
                        "Error writing header to output stream.");
  }
  return absl::OkStatus();
}
absl::Status WriteZeroPad(std::ostream& output_stream, uint64_t num_bytes) {
  std::vector<char> padding(num_bytes, 0);  // Create a buffer of zeros
  output_stream.write(padding.data(), num_bytes);
  output_stream.flush();
  if (!output_stream.good()) {
    return absl::Status(absl::StatusCode::kInternal,
                        "Error zero pad to output stream.");
  }
  return absl::OkStatus();
}

absl::Status PadUntilNextPageBlock(std::ostream& output_file,
                                   size_t block_size) {
  std::streampos current_position = output_file.tellp();
  size_t bytes_written = static_cast<size_t>(current_position);
  size_t required_size = (bytes_written + block_size - 1) / block_size *
                         block_size;  // Calculate the next multiple of 16k
  if (bytes_written < required_size) {
    size_t padding_needed = required_size - bytes_written;
    RETURN_IF_ERROR(WriteZeroPad(output_file, padding_needed));
  }
  return absl::OkStatus();
}

absl::Status MakeLiteRTLMFromSections(
    flatbuffers::FlatBufferBuilder& builder,
    const std::vector<std::unique_ptr<SectionStreamBase>>& sections,
    const std::vector<AnySectionDataType>& section_types,
    const std::vector<KVPair>& system_metadata_map,
    const std::vector<std::vector<KVPair>>& section_items_maps,
    const std::string& out_path) {
  // ** Validation **
  if (sections.empty()) {
    ABSL_LOG(ERROR) << "Input sections list is empty.";
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Input sections list is empty.");
  }
  if (sections.size() != section_types.size() ||
      sections.size() != section_items_maps.size()) {
    ABSL_LOG(ERROR) << "sections, section_types, and section_items_maps must "
                       "have the same size.";
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "sections, section_types, and section_items_maps must "
                        "have the same size.");
  }

  // ** Open an std::ostream for binary file writing **
  std::ofstream output_file(out_path, std::ios::binary);
  if (!output_file.is_open()) {
    ABSL_LOG(ERROR) << "Could not open output file: " << out_path;
    return absl::Status(
        absl::StatusCode::kInternal,
        absl::StrFormat("Could not open output file: %s", out_path));
  }

  // ** 0. Write magic bytes and symver version. **
  output_file.write("LITERTLM", 8);
  output_file.write(reinterpret_cast<const char*>(&LITERTLM_MAJOR_VERSION),
                    sizeof(uint32_t));
  output_file.write(reinterpret_cast<const char*>(&LITERTLM_MINOR_VERSION),
                    sizeof(uint32_t));
  output_file.write(reinterpret_cast<const char*>(&LITERTLM_PATCH_VERSION),
                    sizeof(uint32_t));

  // ** 1. Write zero pad until offset kBlockSize. **
  RETURN_IF_ERROR(PadUntilNextPageBlock(output_file, kBlockSize));

  // ** 2. Write the sections. **
  std::vector<std::pair<uint64_t, uint64_t>> section_offsets;
  for (size_t i = 0; i < sections.size(); ++i) {
    RETURN_IF_ERROR(sections[i]->Prepare());
    std::streampos start_byte_offset = output_file.tellp();  // capture start
    std::istream& strm = sections[i]->GetStream();
    output_file << strm.rdbuf();
    std::streampos end_byte_offset = output_file.tellp();  // capture end
    section_offsets.push_back(
        std::make_pair(static_cast<uint64_t>(start_byte_offset),
                       static_cast<uint64_t>(end_byte_offset)));
    RETURN_IF_ERROR(sections[i]->Finalize());
    RETURN_IF_ERROR(PadUntilNextPageBlock(output_file, kBlockSize));
  }

  // ** 3. Write the header. **
  output_file.seekp(kHeaderBeginByteOffset, std::ios::beg);

  RETURN_IF_ERROR(WriteHeader(builder, output_file, system_metadata_map,
                              section_items_maps, section_offsets,
                              section_types));
  std::streampos header_end_pos = output_file.tellp();
  uint64_t header_end_offset = static_cast<uint64_t>(header_end_pos);
  ABSL_DLOG(INFO) << "Header End Offset is " << header_end_offset;

  // ** 4. Check if header exceeds 16k boundary**
  if (header_end_offset > kBlockSize) {
    // TODO(413978412): support headers > 16KB in this header writer.
    ABSL_LOG(ERROR) << "Header size exceeds 16KB limit.";
    return absl::Status(absl::StatusCode::kInternal,
                        "Header size exceeds 16KB limit.");
  }
  // ** 5. Finally, write the header end offset. **
  output_file.seekp(kHeaderEndLocationByteOffset, std::ios::beg);
  output_file.write(reinterpret_cast<const char*>(&header_end_offset),
                    sizeof(uint64_t));

  output_file.close();
  if (!output_file.good()) {
    return absl::Status(absl::StatusCode::kInternal,
                        "Error writing LiteRT-LM file.");
  }
  return absl::OkStatus();
}

}  // namespace schema
}  // namespace lm
}  // namespace litert
