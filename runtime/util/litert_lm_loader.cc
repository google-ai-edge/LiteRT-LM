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
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/util/memory_mapped_file.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // NOLINT
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_read.h"

namespace litert::lm {

namespace {
// Utility function to Creates a memory-mapped file from a ScopedFile.
absl::StatusOr<std::unique_ptr<MemoryMappedFile>> CreateMemoryMapFromScopedFile(
    litert::lm::ScopedFile& scoped_file) {
  if (!scoped_file.IsValid()) {
    return absl::InvalidArgumentError("Invalid ScopedFile provided.");
  }
  litert::lm::ScopedFile::PlatformFile platform_file = scoped_file.file();
  // For a read-only memory-mapped file:
  return litert::lm::MemoryMappedFile::Create(platform_file, 0, 0, "whole");
}

constexpr uint64_t kLitertLmHeaderMaxSize = 16 * 1024;

}  // namespace

absl::Status LitertLmLoader::MapSections() {
  schema::LitertlmHeader header;
  // Read the header information.
  absl::Status status = ReadHeaderFromLiteRTLM(
      memory_mapped_file_->data(),
      std::min(kLitertLmHeaderMaxSize, memory_mapped_file_->length()), &header);
  ABSL_LOG(INFO) << "status: " << status;
  ABSL_LOG(INFO) << "major_version: " << header.major_version;
  ABSL_LOG(INFO) << "minor_version: " << header.minor_version;
  ABSL_LOG(INFO) << "patch_version: " << header.patch_version;

  if (!status.ok()) {
    return status;
  }

  bool tokenizer_found = false;

  // Loop through the sections and map them to the section buffers.
  auto sections = header.metadata->section_metadata()->objects();
  for (size_t i = 0; i < sections->size(); ++i) {
    const schema::SectionObject* section = sections->Get(i);
    auto items = section->items();
    BufferKey buffer_key(section->data_type());
    BufferRef<uint8_t> buffer_ref;

    // Extract the specific model type from the section items KeyValuePairs.
    if (section->data_type() ==
        schema::AnySectionDataType_TFLiteModel) {
      bool found_model_type = false;
      std::string model_type;
      for (size_t j = 0; j < items->size(); ++j) {
        auto item = items->Get(j);
        if (item->key() &&
            absl::AsciiStrToLower(item->key()->str()) == "model_type" &&
            item->value()) {
          found_model_type = true;
          model_type = *(item->value_as_StringValue()->value());
          break;
        }
      }
      if (found_model_type) {
        ABSL_LOG(INFO) << "model_type: " << model_type;
        buffer_key = BufferKey(section->data_type(),
                               StringToModelType(model_type).value());
      } else {
        ABSL_LOG(WARNING) << "model_type not found, use kTfLitePrefillDecode";
        // For backward compatibility, we will use the default model type if
        // model_type is not found.
        buffer_key =
            BufferKey(section->data_type(), ModelType::kTfLitePrefillDecode);
      }
      buffer_ref =
          BufferRef<uint8_t>(static_cast<uint8_t*>(memory_mapped_file_->data()),
                             section->end_offset(), section->begin_offset());
    } else if (section->data_type() ==
               schema::AnySectionDataType_SP_Tokenizer) {
#if DISABLE_SENTENCEPIECE_TOKENIZER
      continue;
#else
      if (tokenizer_found) {
        // Only load the first tokenizer that is supported, which allows users
        // to specify a preferred tokenizer.
        continue;
      }
      tokenizer_found = true;
      buffer_ref =
          BufferRef<uint8_t>(static_cast<uint8_t*>(memory_mapped_file_->data()),
                             section->end_offset(), section->begin_offset());
#endif  // DISABLE_SENTENCEPIECE_TOKENIZER
    } else if (section->data_type() ==
               schema::AnySectionDataType_HF_Tokenizer_Zlib) {
#if DISABLE_HUGGINGFACE_TOKENIZER
      continue;
#else
      if (tokenizer_found) {
        // Only load the first tokenizer that is supported, which allows users
        // to specify a preferred tokenizer.
        continue;
      }
      tokenizer_found = true;
      RETURN_IF_ERROR(schema::DecompressData(  // NOLINT
          static_cast<uint8_t*>(memory_mapped_file_->data()) +
              section->begin_offset(),
          section->end_offset() - section->begin_offset(),
          &this->hf_tokenizer_data_));
      buffer_ref = BufferRef<uint8_t>(this->hf_tokenizer_data_.data(),
                                      this->hf_tokenizer_data_.size(), 0);

#endif  // DISABLE_HUGGINGFACE_TOKENIZER
    } else {
      buffer_ref =
          BufferRef<uint8_t>(static_cast<uint8_t*>(memory_mapped_file_->data()),
                             section->end_offset(), section->begin_offset());
    }
    section_buffers_[buffer_key] = std::move(buffer_ref);
    ABSL_LOG(INFO) << "section_index: " << i;
    ABSL_LOG(INFO) << "section_data_type: "
                   << EnumNameAnySectionDataType(section->data_type());
    ABSL_LOG(INFO) << "section_begin_offset: " << section->begin_offset();
    ABSL_LOG(INFO) << "section_end_offset: " << section->end_offset();
  }
  return absl::OkStatus();
}

absl::Status LitertLmLoader::Initialize() {
  ABSL_LOG(INFO) << "LitertLmLoader::Initialize";

  absl::StatusOr<std::unique_ptr<MemoryMappedFile>> mmap_status =
      CreateMemoryMapFromScopedFile(model_file_);

  if (mmap_status.ok()) {
    memory_mapped_file_ = std::move(mmap_status).value();
    ABSL_LOG(INFO) << "mmap_status is ok";
    ABSL_LOG(INFO) << "length: " << memory_mapped_file_->length();
  } else {
    ABSL_LOG(ERROR) << "Failed to create memory-mapped file: "
                    << mmap_status.status();
  }

  ABSL_CHECK_OK(MapSections());

  return absl::OkStatus();
}

}  // namespace litert::lm
