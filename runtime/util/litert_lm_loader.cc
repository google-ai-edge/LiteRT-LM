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

#include "absl/hash/hash.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
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
  // Use an empty key so each mapping is unnamed/anonymous. Using a shared
  // name like "whole" causes OpenFileMappingA() in the Windows implementation
  // (memory_mapped_file_win.cc) to return a *previously created* mapping
  // object that is bound to the first file ever mapped with that name. The
  // result is that loading a second model file in the same process silently
  // maps the bytes of the first file (the file handle and size are correct,
  // but the bytes come from the wrong file). Anonymous mappings have no
  // such cross-file aliasing.
  return litert::lm::MemoryMappedFile::Create(platform_file, offset, size,
                                              /*key=*/"");
}

bool IsValidTfLiteModelTypeWireString(absl::string_view model_type_str) {
  const std::string upper = absl::AsciiStrToUpper(model_type_str);
  proto::LlmMetadata::TfLiteModelType llm_type;
  if (proto::LlmMetadata::TfLiteModelType_Parse(upper, &llm_type) &&
      llm_type != proto::LlmMetadata::TF_LITE_MODEL_TYPE_UNSPECIFIED) {
    return true;
  }
  proto::EmbeddingMetadata::TfLiteModelType emb_type;
  if (proto::EmbeddingMetadata::TfLiteModelType_Parse(upper, &emb_type) &&
      emb_type != proto::EmbeddingMetadata::TF_LITE_MODEL_TYPE_UNSPECIFIED) {
    return true;
  }
  proto::TtsMetadata::TfLiteModelType tts_type;
  if (proto::TtsMetadata::TfLiteModelType_Parse(upper, &tts_type) &&
      tts_type != proto::TtsMetadata::TF_LITE_MODEL_TYPE_UNSPECIFIED) {
    return true;
  }
  proto::AsrMetadata::TfLiteModelType asr_type;
  if (proto::AsrMetadata::TfLiteModelType_Parse(upper, &asr_type) &&
      asr_type != proto::AsrMetadata::TF_LITE_MODEL_TYPE_UNSPECIFIED) {
    return true;
  }
  return StringToModelType(model_type_str).ok();
}

}  // namespace

BufferKey::BufferKey(schema::AnySectionDataType type)
    : data_type(type), model_type(std::nullopt) {}

BufferKey::BufferKey(schema::AnySectionDataType type,
                     absl::string_view model_type_str)
    : data_type(type),
      model_type(type == schema::AnySectionDataType_GenericBinaryData
                     ? (model_type_str.empty()
                            ? std::nullopt
                            : std::optional<std::string>(model_type_str))
                     : std::optional<std::string>(
                           absl::AsciiStrToLower(model_type_str))) {
  if (type != schema::AnySectionDataType_TFLiteModel &&
      type != schema::AnySectionDataType_TFLiteWeights &&
      type != schema::AnySectionDataType_SP_Tokenizer &&
      type != schema::AnySectionDataType_HF_Tokenizer_Zlib &&
      type != schema::AnySectionDataType_GenericBinaryData) {
    ABSL_LOG(ERROR) << "model_type should only be provided for TFLiteModel, "
                       "TFLiteWeights, SP_Tokenizer, HF_Tokenizer_Zlib, or "
                       "GenericBinaryData";
  }
}

BufferKey::BufferKey(schema::AnySectionDataType type, ModelType model_type_enum)
    : BufferKey(type, TfLiteModelTypeToWireString(model_type_enum)) {}

bool BufferKey::operator==(const BufferKey& other) const {
  return data_type == other.data_type && model_type == other.model_type;
}

size_t BufferKeyHash::operator()(const BufferKey& k) const {
  size_t h1 = std::hash<schema::AnySectionDataType>{}(k.data_type);
  size_t h2 = 0;
  if (k.model_type.has_value()) {
    h2 = absl::HashOf(*k.model_type);
  }
  return h1 ^ (h2 << 1);
}

absl::StatusOr<std::pair<BufferKey, TfLiteSectionHint>>
ExtractBufferKeyAndTfLiteSectionHint(const schema::SectionObject* section) {
  auto items = section->items();
  BufferKey buffer_key(section->data_type());
  TfLiteSectionHint section_hint;
  // Extract the specific model type from the section items KeyValuePairs.
  if ((section->data_type() == schema::AnySectionDataType_TFLiteModel ||
       section->data_type() == schema::AnySectionDataType_TFLiteWeights ||
       section->data_type() == schema::AnySectionDataType_SP_Tokenizer ||
       section->data_type() == schema::AnySectionDataType_HF_Tokenizer_Zlib) &&
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
        section_hint.backend_constraint =
            *(item->value_as_StringValue()->value());
      }
      if (item->key() &&
          absl::AsciiStrToLower(item->key()->str()) ==
              "prefer_activation_type" &&
          item->value()) {
        section_hint.prefer_activation_type =
            *(item->value_as_StringValue()->value());
      }
    }
    if (found_model_type) {
      ABSL_VLOG(1) << "model_type: " << model_type;
      if (!IsValidTfLiteModelTypeWireString(model_type)) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Unknown model type: %s", model_type));
      }
      buffer_key = BufferKey(section->data_type(), model_type);
    } else {
      if (section->data_type() == schema::AnySectionDataType_TFLiteModel ||
          section->data_type() == schema::AnySectionDataType_TFLiteWeights) {
        ABSL_LOG(WARNING)
            << "model_type not found, use tf_lite_prefill_decode";
        // For backward compatibility, we will use the default model type if
        // model_type is not found.
        buffer_key =
            BufferKey(section->data_type(), "tf_lite_prefill_decode");
      }
    }
  } else if (section->data_type() ==
                 schema::AnySectionDataType_GenericBinaryData &&
             items != nullptr) {
    for (size_t j = 0; j < items->size(); ++j) {
      auto item = items->Get(j);
      if (item->key() && item->key()->str() == "name" && item->value() &&
          item->value_type() == schema::VData_StringValue &&
          item->value_as_StringValue()->value() != nullptr) {
        buffer_key = BufferKey(section->data_type(),
                               item->value_as_StringValue()->value()->str());
        break;
      }
    }
  }
  return std::make_pair(buffer_key, section_hint);
}

absl::Status LitertLmLoader::MapSection(BufferKey buffer_key,
                                        uint64_t begin_offset,
                                        uint64_t end_offset) {
  std::string section_name =
      schema::EnumNameAnySectionDataType(buffer_key.data_type);
  if (buffer_key.model_type.has_value()) {
    absl::StrAppend(&section_name, ":",
                    ModelTypeToString(*buffer_key.model_type));
  }
  ABSL_LOG(INFO) << "LitertLmLoader::MapSection: Mapping section ["
                 << section_name << "] begin_offset=" << begin_offset
                 << ", end_offset=" << end_offset
                 << " (size: " << (end_offset - begin_offset) << " bytes)";

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
    auto& model_file = *std::get<std::shared_ptr<ScopedFile>>(model_source_);
    size_t alignment = MemoryMappedFile::GetOffsetAlignment();
    uint64_t alignment_gap = begin_offset % alignment;
    uint64_t aligned_begin_offset = begin_offset - alignment_gap;

    uint64_t aligned_section_size = end_offset - aligned_begin_offset;
    ABSL_ASSIGN_OR_RETURN(
        section_memory_mapped_files_[buffer_key],
        CreateMemoryMapFromScopedFile(model_file, aligned_begin_offset,
                                      aligned_section_size));
    auto& memory_mapped_file = section_memory_mapped_files_[buffer_key];
    ABSL_LOG(INFO)
        << "LitertLmLoader::MapSection: Created MemoryMappedFile for key ["
        << section_name << "]. Total section_memory_mapped_files_ entries: "
        << section_memory_mapped_files_.size();

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
  if (std::holds_alternative<std::shared_ptr<ScopedFile>>(model_source_)) {
    return *std::get<std::shared_ptr<ScopedFile>>(model_source_);
  }
  if (scoped_file_ != nullptr) {
    return *scoped_file_;
  }
  return absl::InvalidArgumentError(
      "Model source is not a ScopedFile, cannot get ScopedFile.");
}

absl::StatusOr<std::shared_ptr<ScopedFile>>
LitertLmLoader::GetSharedScopedFile() {
  if (std::holds_alternative<std::shared_ptr<ScopedFile>>(model_source_)) {
    return std::get<std::shared_ptr<ScopedFile>>(model_source_);
  }
  if (scoped_file_ != nullptr) {
    return scoped_file_;
  }
  return absl::InvalidArgumentError(
      "Model source is not a ScopedFile, cannot get ScopedFile.");
}

absl::StatusOr<std::unique_ptr<LitertLmLoader>> LitertLmLoader::Create(
    ScopedFile model_file) {
  auto loader = absl::WrapUnique(new LitertLmLoader(std::move(model_file)));
  ABSL_RETURN_IF_ERROR(loader->Initialize());
  return std::move(loader);
}

absl::StatusOr<std::unique_ptr<LitertLmLoader>> LitertLmLoader::Create(
    std::shared_ptr<ScopedFile> shared_scoped_file) {
  auto loader =
      absl::WrapUnique(new LitertLmLoader(std::move(shared_scoped_file)));
  ABSL_RETURN_IF_ERROR(loader->Initialize());
  return std::move(loader);
}

absl::StatusOr<std::unique_ptr<LitertLmLoader>> LitertLmLoader::Create(
    std::shared_ptr<MemoryMappedFile> memory_mapped_model_file,
    std::shared_ptr<ScopedFile> scoped_file) {
  auto loader = absl::WrapUnique(new LitertLmLoader(
      std::move(memory_mapped_model_file), std::move(scoped_file)));
  ABSL_RETURN_IF_ERROR(loader->Initialize());
  return std::move(loader);
}

absl::Status LitertLmLoader::Initialize() {
  ABSL_VLOG(1) << "LitertLmLoader::Initialize";

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
    auto& model_file = *std::get<std::shared_ptr<ScopedFile>>(model_source_);
    ABSL_ASSIGN_OR_RETURN(model_file_size, model_file.GetSize());
    header_size = std::min(kLitertLmHeaderMaxSize, model_file_size);
    ABSL_ASSIGN_OR_RETURN(
        header_memory_mapped_file,
        CreateMemoryMapFromScopedFile(model_file, /*offset=*/0,
                                      /*size=*/header_size));
    header_data = header_memory_mapped_file->data();
  }
  ABSL_VLOG(1) << "mmap_status is ok";

  // Read the header information.
  absl::Status status =
      ReadHeaderFromLiteRTLM(header_data, header_size, &header_);
  ABSL_VLOG(1) << "status: " << status;
  ABSL_VLOG(1) << "major_version: " << header_.major_version;
  ABSL_VLOG(1) << "minor_version: " << header_.minor_version;
  ABSL_VLOG(1) << "patch_version: " << header_.patch_version;
  if (!status.ok()) {
    return status;
  }

  // Loop through the sections and record the section locations.
  auto sections = header_.metadata->section_metadata()->objects();
  for (size_t i = 0; i < sections->size(); ++i) {
    const schema::SectionObject* section = sections->Get(i);
    ABSL_ASSIGN_OR_RETURN(auto key_and_section_hint,
                          ExtractBufferKeyAndTfLiteSectionHint(section));
    BufferKey buffer_key = key_and_section_hint.first;
    const auto& section_hint = key_and_section_hint.second;
    section_hints_map_[buffer_key] = section_hint;

    if (section_hint.backend_constraint.has_value()) {
      ABSL_VLOG(1) << "section_backend_constraint: "
                   << *section_hint.backend_constraint;
    }
    if (section_hint.prefer_activation_type.has_value()) {
      ABSL_VLOG(1) << "section_prefer_activation_type: "
                   << *section_hint.prefer_activation_type;
    }

    if (section->begin_offset() > section->end_offset()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Section %d has invalid offsets: begin_offset (%d) > "
                          "end_offset (%d).",
                          i, section->begin_offset(), section->end_offset()));
    }
    section_locations_[buffer_key] =
        std::make_pair(section->begin_offset(), section->end_offset());

    std::string section_name =
        schema::EnumNameAnySectionDataType(section->data_type());
    if (buffer_key.model_type.has_value()) {
      absl::StrAppend(&section_name, ":",
                      ModelTypeToString(*buffer_key.model_type));
    }
    ABSL_LOG(INFO) << "LitertLmLoader::Initialize: Registered section ["
                   << section_name << "] offset=[" << section->begin_offset()
                   << ", " << section->end_offset() << "] (size: "
                   << (section->end_offset() - section->begin_offset())
                   << " bytes)";

    ABSL_VLOG(1) << "section_index: " << i;
    ABSL_VLOG(1) << "section_data_type: "
                 << EnumNameAnySectionDataType(section->data_type());
    ABSL_VLOG(1) << "section_begin_offset: " << section->begin_offset();
    ABSL_VLOG(1) << "section_end_offset: " << section->end_offset();
  }
  return absl::OkStatus();
}

std::optional<litert::BufferRef<uint8_t>> LitertLmLoader::GetSectionBuffer(
    BufferKey buffer_key) {
  {
    absl::ReaderMutexLock lock(section_buffers_mutex_);
    auto section_buffer_it = section_buffers_.find(buffer_key);
    if (section_buffer_it != section_buffers_.end()) {
      return section_buffer_it->second;
    }
  }

  absl::MutexLock lock(section_buffers_mutex_);
  // Check again in case another thread has mapped it.
  auto section_buffer_it = section_buffers_.find(buffer_key);
  if (section_buffer_it != section_buffers_.end()) {
    return section_buffer_it->second;
  }

  auto section_location_it = section_locations_.find(buffer_key);
  if (section_location_it == section_locations_.end()) {
    ABSL_LOG(WARNING) << "Section not found: "
                      << schema::EnumNameAnySectionDataType(
                             buffer_key.data_type);
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
    BufferKey buffer_key) const {
  auto section_location_it = section_locations_.find(buffer_key);
  if (section_location_it == section_locations_.end()) {
    return absl::NotFoundError("Section not found.");
  }
  return section_location_it->second;
}

std::optional<litert::BufferRef<uint8_t>>
LitertLmLoader::GetSentencePieceTokenizer(ModelType model_type) {
  return GetSentencePieceTokenizer(TfLiteModelTypeToWireString(model_type));
}

std::optional<litert::BufferRef<uint8_t>>
LitertLmLoader::GetSentencePieceTokenizer(absl::string_view model_type_str) {
  std::string key = absl::AsciiStrToLower(model_type_str);
  auto buf = GetSectionBuffer(
      BufferKey(schema::AnySectionDataType_SP_Tokenizer, key));
  if (!buf.has_value() && key == "tf_lite_prefill_decode") {
    buf = GetSectionBuffer(BufferKey(schema::AnySectionDataType_SP_Tokenizer));
  }
  return buf;
}

std::optional<litert::OwningBufferRef<uint8_t>>
LitertLmLoader::GetHuggingFaceTokenizer(ModelType model_type) {
  return GetHuggingFaceTokenizer(TfLiteModelTypeToWireString(model_type));
}

std::optional<litert::OwningBufferRef<uint8_t>>
LitertLmLoader::GetHuggingFaceTokenizer(absl::string_view model_type_str) {
  std::string key = absl::AsciiStrToLower(model_type_str);
  auto optional_section_buffer = GetSectionBuffer(
      BufferKey(schema::AnySectionDataType_HF_Tokenizer_Zlib, key));
  if (!optional_section_buffer.has_value() &&
      key == "tf_lite_prefill_decode") {
    optional_section_buffer = GetSectionBuffer(
        BufferKey(schema::AnySectionDataType_HF_Tokenizer_Zlib));
  }
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

litert::BufferRef<uint8_t> LitertLmLoader::GetTFLiteModel(
    ModelType model_type) {
  return GetTFLiteModel(TfLiteModelTypeToWireString(model_type));
}

litert::BufferRef<uint8_t> LitertLmLoader::GetTFLiteModel(
    absl::string_view model_type_str) {
  auto optional_section_buffer = GetSectionBuffer(
      BufferKey(schema::AnySectionDataType_TFLiteModel, model_type_str));
  if (optional_section_buffer.has_value()) {
    return optional_section_buffer.value();
  }
  ABSL_LOG(WARNING) << "TFLite model for type: " << model_type_str
                    << " not found. Skipping.";
  return litert::BufferRef<uint8_t>();
}

litert::BufferRef<uint8_t> LitertLmLoader::GetTFLiteWeights(
    ModelType model_type) {
  return GetTFLiteWeights(TfLiteModelTypeToWireString(model_type));
}

litert::BufferRef<uint8_t> LitertLmLoader::GetTFLiteWeights(
    absl::string_view model_type_str) {
  auto optional_section_buffer = GetSectionBuffer(
      BufferKey(schema::AnySectionDataType_TFLiteWeights, model_type_str));
  if (optional_section_buffer.has_value()) {
    return optional_section_buffer.value();
  }
  ABSL_LOG(WARNING) << "TFLite weights for type: " << model_type_str
                    << " not found. Skipping.";
  return litert::BufferRef<uint8_t>();
}

std::optional<std::string> LitertLmLoader::GetTFLiteModelBackendConstraint(
    ModelType model_type) {
  return GetTFLiteModelBackendConstraint(
      TfLiteModelTypeToWireString(model_type));
}

std::optional<std::string> LitertLmLoader::GetTFLiteModelBackendConstraint(
    absl::string_view model_type_str) {
  BufferKey key(schema::AnySectionDataType_TFLiteModel, model_type_str);
  auto it = section_hints_map_.find(key);
  if (it != section_hints_map_.end()) {
    return it->second.backend_constraint;
  }
  ABSL_LOG(WARNING) << "TFLite model type: " << model_type_str
                    << " not found for backend constraints. Skipping.";
  return std::nullopt;
}

std::optional<std::string> LitertLmLoader::GetTFLiteModelPreferActivationType(
    ModelType model_type) {
  return GetTFLiteModelPreferActivationType(
      TfLiteModelTypeToWireString(model_type));
}

std::optional<std::string> LitertLmLoader::GetTFLiteModelPreferActivationType(
    absl::string_view model_type_str) {
  BufferKey key(schema::AnySectionDataType_TFLiteModel, model_type_str);
  auto it = section_hints_map_.find(key);
  if (it != section_hints_map_.end()) {
    return it->second.prefer_activation_type;
  }
  ABSL_LOG(WARNING)
      << "TFLite model type: " << model_type_str
      << " not found for prefer activation type. Use system's "
         "default backend activation type. System's default activation "
         "type for Text decoder is fp16. Vision encoder and audio encoder "
         "default is fp32.";
  return std::nullopt;
}

std::optional<litert::BufferRef<uint8_t>> LitertLmLoader::GetTtsMetadata() {
  return GetSectionBuffer(
      BufferKey(schema::AnySectionDataType_TtsMetadataProto));
}

std::optional<litert::BufferRef<uint8_t>> LitertLmLoader::GetAsrMetadata() {
  return GetSectionBuffer(
      BufferKey(schema::AnySectionDataType_AsrMetadataProto));
}

std::optional<litert::BufferRef<uint8_t>> LitertLmLoader::GetGenericBinaryData(
    absl::string_view name) {
  return GetSectionBuffer(
      BufferKey(schema::AnySectionDataType_GenericBinaryData, name));
}

std::vector<std::string> LitertLmLoader::GetGenericBinaryDataNames() const {
  std::vector<std::string> result;
  for (const auto& [key, location] : section_locations_) {
    if (key.data_type == schema::AnySectionDataType_GenericBinaryData &&
        key.model_type.has_value()) {
      result.push_back(*key.model_type);
    }
  }
  return result;
}

}  // namespace litert::lm
