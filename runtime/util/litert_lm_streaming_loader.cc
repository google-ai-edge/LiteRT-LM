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

#include "runtime/util/litert_lm_streaming_loader.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_read.h"

namespace litert::lm {

absl::Status LitertLmStreamingLoader::LoadHeader() {
  if (header_loaded_) {
    return absl::OkStatus();
  }

  HeaderPreamble header_preamble;
  RETURN_IF_ERROR(data_stream_->ReadAndDiscard(&header_preamble, 0,
                                               sizeof(HeaderPreamble)));

  // Check the magic number.
  std::string magic_str(header_preamble.magic, 8);
  if (magic_str != "LITERTLM") {
    return absl::InternalError(absl::StrCat(
        "Invalid magic number. Expected 'LITERTLM', got '", magic_str, "'"));
  }

  // Sanity check: The header end offset should be less than the max size.
  if (header_preamble.header_end_offset > kLitertLmHeaderMaxSize) {
    return absl::InternalError(absl::StrCat("Header end offset is too large: ",
                                            header_preamble.header_end_offset));
  }

  // Ensure header size is at least the preamble size.
  if (header_preamble.header_end_offset < sizeof(HeaderPreamble)) {
    return absl::InternalError(absl::StrCat("Header end offset is too small: ",
                                            header_preamble.header_end_offset));
  }

  // Allocate buffer for the entire header.
  size_t header_size = header_preamble.header_end_offset;
  std::unique_ptr<char[]> header_data(new char[header_size]);

  // Copy the preamble into the buffer.
  memcpy(header_data.get(), &header_preamble, sizeof(HeaderPreamble));

  // Read the rest of the header from the stream.
  size_t remaining_size = header_size - sizeof(HeaderPreamble);
  if (remaining_size > 0) {
    RETURN_IF_ERROR(
        data_stream_->ReadAndDiscard(header_data.get() + sizeof(HeaderPreamble),
                                     sizeof(HeaderPreamble), remaining_size));
  }

  // Read the LiteRTLM header from the buffer.
  RETURN_IF_ERROR(schema::ReadHeaderFromLiteRTLM(
      static_cast<void*>(header_data.get()), header_size, header_.get()));

  // Record the section metadata.
  auto objects = header_->metadata->section_metadata()->objects();
  ordered_section_info_.reserve(objects->size());
  for (size_t i = 0; i < objects->size(); ++i) {
    const schema::SectionObject* section = objects->Get(i);
    ASSIGN_OR_RETURN(auto key_and_section_hint,
                     ExtractBufferKeyAndTfLiteSectionHint(section));
    ordered_section_info_.push_back(
        {section, key_and_section_hint.first,
         key_and_section_hint.second.backend_constraint, nullptr, header_});
  }

  // Sort by section begin offset since this is the order the stream will
  // provide them to us.
  std::sort(ordered_section_info_.begin(), ordered_section_info_.end(),
            [](const SectionInfo& a, const SectionInfo& b) {
              return a.section->begin_offset() < b.section->begin_offset();
            });

  header_loaded_ = true;
  return absl::OkStatus();
}

absl::StatusOr<std::optional<SectionInfo>>
LitertLmStreamingLoader::GetNextSection() {
  if (!header_loaded_) {
    RETURN_IF_ERROR(LoadHeader());
  }

  if (next_section_index_ >= ordered_section_info_.size()) {
    return std::nullopt;
  }

  SectionInfo& section_info = ordered_section_info_[next_section_index_];
  if (section_info.section->begin_offset() >
      section_info.section->end_offset()) {
    return absl::InternalError(absl::StrCat(
        "Section begin offset (", section_info.section->begin_offset(),
        ") is greater than end offset (", section_info.section->end_offset(),
        ")"));
  }

  // Open the substream.
  ASSIGN_OR_RETURN(
      auto sub_stream,
      data_stream_->OpenSubStream(section_info.section->begin_offset(),
                                  section_info.section->end_offset() -
                                      section_info.section->begin_offset()));

  // Create a SectionInfo to return. This will be returned by move,
  // allowing the caller to manage the lifetime of the substream.
  SectionInfo return_val = {
      section_info.section,
      section_info.buffer_key,
      section_info.backend_constraint,
      std::move(sub_stream),
      section_info.header,
  };

  next_section_index_++;
  return std::move(return_val);
}

}  // namespace litert::lm
