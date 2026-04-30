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

#include "runtime/util/litert_lm_loader.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_read.h"

namespace litert::lm {

namespace {
// Utility function to Creates a memory-mapped file from a ScopedFile.
absl::StatusOr<std::unique_ptr<MemoryMappedFile>> CreateMemoryMapFromScopedFile(
    litert::lm::ScopedFile& scoped_file, uint64_t offset = 0,
    uint64_t size = 0) {
  if (!scoped_file.IsValid()) {
    return absl::InvalidArgumentError("Invalid ScopedFile provided.");
  }
  litert::lm::ScopedFile::PlatformFile platform_file = scoped_file.file();
  // For a read-only memory-mapped file:
  // TODO - b/454926463: Add support for different keys for more optimal loading
  // on Windows.
  return litert::lm::MemoryMappedFile::Create(platform_file, offset, size,
                                              "whole");
}

}  // namespace

absl::StatusOr<std::pair<BufferKey, std::optional<std::string>>>
ExtractBufferKeyAndBackendConstraint(const schema::SectionObject* section) {
  auto items = section->items();
  BufferKey buffer_key(section->data_type());
  std::optional<std::string> backend_constraint;
  // Extract the specific model type from the section items KeyValuePairs.
  if ((section->data_type() == schema::AnySectionDataType_TFLiteModel ||
       section->data_type() == schema::AnySectionDataType_TFLiteWeights) &&
      items != nullptr) {
    bool found_model_type = false;
    std::string model_type;
    for (size_t j = 0; j < items->size(); ++j) {
      auto item = items->Get(j);
      if (item->key() &&
          absl::AsciiStrToLower(item->key()->str()) == "model_type" &&
          item->value()) {
        found_model_type = true;
        model_type = *(item->value_as_StringValue()->value());
      }
      if (item->key() &&
          absl::AsciiStrToLower(item->key()->str()) == "backend_constraint" &&
          item->value()) {
        backend_constraint = *(item->value_as_StringValue()->value());
      }
    }
    if (found_model_type) {
      ABSL_LOG(INFO) << "model_type: " << model_type;
      ASSIGN_OR_RETURN(ModelType model_type_enum,
                       StringToModelType(model_type));
      buffer_key = BufferKey(section->data_type(), model_type_enum);
    } else {
      ABSL_LOG(WARNING) << "model_type not found, use kTfLitePrefillDecode";
      // For backward compatibility, we will use the default model type if
      // model_type is not found.
      buffer_key =
          BufferKey(section->data_type(), ModelType::kTfLitePrefillDecode);
    }
  }
  return std::make_pair(buffer_key, backend_constraint);
}

absl::Status LitertLmLoader::MapSection(BufferKey buffer_key,
                                        uint64_t begin_offset,
                                        uint64_t end_offset) {
  uint8_t* data = nullptr;
  if (std::holds_alternative<std::shared_ptr<MemoryMappedFile>>(
          model_source_)) {
    // If the loader was initialized with an existing memory-mapped file, the
    // entire file content is already mapped into memory. We can access any
    // section by adding its begin_offset to the base pointer of the mapped
    // region. Unlike mmap offsets, pointer arithmetic within an
    // already-mapped region does not require page alignment, so no
    // alignment_gap is needed here.
    data = static_cast<uint8_t*>(
               std::get<std::shared_ptr<MemoryMappedFile>>(model_source_)
                   ->data()) +
           begin_offset;
  } else {
    // If the begin offset is not aligned to the required platform alignment, we
    // need to map the section starting a bit earlier so that the data is
    // aligned.
    auto& model_file = std::get<ScopedFile>(model_source_);
    size_t alignment = MemoryMappedFile::GetOffsetAlignment();
    uint64_t alignment_gap = begin_offset % alignment;
    uint64_t aligned_begin_offset = begin_offset - alignment_gap;

    uint64_t aligned_section_size = end_offset - aligned_begin_offset;
    ASSIGN_OR_RETURN(
        section_memory_mapped_files_[buffer_key],
        CreateMemoryMapFromScopedFile(model_file, aligned_begin_offset,
                                      aligned_section_size));
    auto& memory_mapped_file = section_memory_mapped_files_[buffer_key];

    // The section buffer that is stored should point to the section data only,
    // not include the alignment gap.
    data = static_cast<uint8_t*>(memory_mapped_file->data()) + alignment_gap;
  }

  uint64_t section_size = end_offset - begin_offset;
  section_buffers_[buffer_key] = BufferRef<uint8_t>(data, section_size);

  return absl::OkStatus();
}

absl::StatusOr<std::reference_wrapper<ScopedFile>>
LitertLmLoader::GetScopedFile() {
  if (std::holds_alternative<ScopedFile>(model_source_)) {
    return std::get<ScopedFile>(model_source_);
  }
  return absl::InvalidArgumentError(
      "Model source is not a ScopedFile, cannot get ScopedFile.");
}

absl::StatusOr<std::unique_ptr<LitertLmLoader>> LitertLmLoader::Create(
    ScopedFile model_file) {
  auto loader = absl::WrapUnique(new LitertLmLoader(std::move(model_file)));
  RETURN_IF_ERROR(loader->Initialize());
  return std::move(loader);
}

absl::StatusOr<std::unique_ptr<LitertLmLoader>> LitertLmLoader::Create(
    std::shared_ptr<MemoryMappedFile> memory_mapped_model_file) {
  auto loader =
      absl::WrapUnique(new LitertLmLoader(std::move(memory_mapped_model_file)));
  RETURN_IF_ERROR(loader->Initialize());
  return std::move(loader);
}

absl::Status LitertLmLoader::Initialize() {
  ABSL_LOG(INFO) << "LitertLmLoader::Initialize";

  // Map the header of the model file.
  uint64_t model_file_size;
  uint64_t header_size;
  void* header_data;
  std::unique_ptr<MemoryMappedFile> header_memory_mapped_file;
  if (std::holds_alternative<std::shared_ptr<MemoryMappedFile>>(
          model_source_)) {
    auto& memory_mapped_model_file =
        std::get<std::shared_ptr<MemoryMappedFile>>(model_source_);
    model_file_size = memory_mapped_model_file->length();
    header_size = std::min(kLitertLmHeaderMaxSize, model_file_size);
    header_data = memory_mapped_model_file->data();
  } else {
    auto& model_file = std::get<ScopedFile>(model_source_);
    ASSIGN_OR_RETURN(model_file_size, model_file.GetSize());
    header_size = std::min(kLitertLmHeaderMaxSize, model_file_size);
    ASSIGN_OR_RETURN(header_memory_mapped_file,
                     CreateMemoryMapFromScopedFile(model_file, /*offset=*/0,
                                                   /*size=*/header_size));
    header_data = header_memory_mapped_file->data();
  }
  ABSL_LOG(INFO) << "mmap_status is ok";

  // Read the header information.
  absl::Status status =
      ReadHeaderFromLiteRTLM(header_data, header_size, &header_);
  ABSL_LOG(INFO) << "status: " << status;
  ABSL_LOG(INFO) << "major_version: " << header_.major_version;
  ABSL_LOG(INFO) << "minor_version: " << header_.minor_version;
  ABSL_LOG(INFO) << "patch_version: " << header_.patch_version;
  if (!status.ok()) {
    return status;
  }

  // Loop through the sections and record the section locations.
  auto sections = header_.metadata->section_metadata()->objects();
  for (size_t i = 0; i < sections->size(); ++i) {
    const schema::SectionObject* section = sections->Get(i);
    ASSIGN_OR_RETURN(auto key_and_constraint,
                     ExtractBufferKeyAndBackendConstraint(section));
    BufferKey buffer_key = key_and_constraint.first;
    if (key_and_constraint.second.has_value() &&
        !key_and_constraint.second->empty()) {
      section_backend_constraint_[buffer_key] = *key_and_constraint.second;
      ABSL_LOG(INFO) << "section_backend_constraint: "
                     << *key_and_constraint.second;
    }
    if (section->begin_offset() > section->end_offset()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Section %d has invalid offsets: begin_offset (%d) > "
                          "end_offset (%d).",
                          i, section->begin_offset(), section->end_offset()));
    }
    section_locations_[buffer_key] =
        std::make_pair(section->begin_offset(), section->end_offset());

    ABSL_LOG(INFO) << "section_index: " << i;
    ABSL_LOG(INFO) << "section_data_type: "
                   << EnumNameAnySectionDataType(section->data_type());
    ABSL_LOG(INFO) << "section_begin_offset: " << section->begin_offset();
    ABSL_LOG(INFO) << "section_end_offset: " << section->end_offset();
  }
  return absl::OkStatus();
}

std::optional<litert::BufferRef<uint8_t>> LitertLmLoader::GetSectionBuffer(
    BufferKey buffer_key) {
  {
    absl::ReaderMutexLock lock(&section_buffers_mutex_);
    auto section_buffer_it = section_buffers_.find(buffer_key);
    if (section_buffer_it != section_buffers_.end()) {
      return section_buffer_it->second;
    }
  }

  absl::MutexLock lock(&section_buffers_mutex_);
  // Check again in case another thread has mapped it.
  auto section_buffer_it = section_buffers_.find(buffer_key);
  if (section_buffer_it != section_buffers_.end()) {
    return section_buffer_it->second;
  }

  auto section_location_it = section_locations_.find(buffer_key);
  if (section_location_it == section_locations_.end()) {
    ABSL_LOG(WARNING) << "Section not found: " << buffer_key.data_type;
    return std::nullopt;
  }

  // If we have not already mapped this section, map it now.
  auto [offset_begin, offset_end] = section_location_it->second;
  absl::Status status = MapSection(buffer_key, offset_begin, offset_end);
  if (!status.ok()) {
    ABSL_LOG(WARNING) << "Failed to map section: " << status;
    return std::nullopt;
  }
  // Return a BufferRef to the mapped section.
  return section_buffers_[buffer_key];
}

absl::StatusOr<std::pair<size_t, size_t>> LitertLmLoader::GetSectionLocation(
    BufferKey buffer_key) const{
  auto section_location_it = section_locations_.find(buffer_key);
  if (section_location_it == section_locations_.end()) {
    return absl::NotFoundError("Section not found.");
  }
  return section_location_it->second;
}

std::optional<litert::OwningBufferRef<uint8_t>>
LitertLmLoader::GetHuggingFaceTokenizer() {
  auto optional_section_buffer =
      GetSectionBuffer(BufferKey(schema::AnySectionDataType_HF_Tokenizer_Zlib));
  if (!optional_section_buffer.has_value()) {
    return std::nullopt;
  }
  const auto& section = optional_section_buffer.value();

  std::vector<uint8_t> hf_tokenizer_data;
  auto status = schema::DecompressData(section.Data(), section.Size(),
                                       &hf_tokenizer_data);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to decompress HuggingFace tokenizer data: "
                    << status;
    return std::nullopt;
  }

  return OwningBufferRef<uint8_t>{
      static_cast<const uint8_t*>(hf_tokenizer_data.data()),
      hf_tokenizer_data.size()};
}

}  // namespace litert::lm
