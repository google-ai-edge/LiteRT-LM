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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LITERT_LM_STREAMING_LOADER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LITERT_LM_STREAMING_LOADER_H_

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
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/util/data_stream.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_read.h"

namespace litert::lm {

// The preamble of the Litert LM header. Everything up to but excluding the
// flatbuffer that contains the section metadata.
struct HeaderPreamble {
  char magic[8];  // "LITERTLM"
  uint32_t major_version;
  uint32_t minor_version;
  uint32_t patch_version;
  char padding[4];  // Padding for 8-byte alignment.
  // Absolute position of the end of the flatbuffer in the stream. Flatbuffer
  // starts immediately after this.
  uint64_t header_end_offset;
};

// Information about a section in the Litert LM model.
// Should not outlive the LitertLmStreamingLoader.
struct SectionInfo {
  const schema::SectionObject* section;
  BufferKey buffer_key;
  std::optional<std::string> backend_constraint;
  std::optional<std::string> prefer_activation_type;
  std::shared_ptr<DataStream> data_stream;

  std::weak_ptr<schema::LitertlmHeader> header;
};

// A streaming loader that reads a .litertlm file from a data stream and returns
// the sections of the file one by one.
// This approach avoids loading the entire model file into memory at once,
// making it suitable for large models on platforms that do not support
// memory-mapping files.
class LitertLmStreamingLoader {
 public:
  explicit LitertLmStreamingLoader(std::shared_ptr<DataStream> data_stream)
      : data_stream_(std::move(data_stream)),
        header_(std::make_shared<schema::LitertlmHeader>()) {}

  // Loads the header from the stream.
  // If preserve_stream is true, uses ReadAndPreserve so the header remains in
  // the stream for subsequent readers.
  // Returns an error if the header cannot be loaded.
  // Optional. If not called, LoadHeader() will be called within
  // GetNextSection().
  absl::Status LoadHeader(bool preserve_stream = false);

  // Returns whether the model contains a section with the given buffer key.
  // Must be called after LoadHeader().
  bool HasSection(const BufferKey& key) const {
    return section_hints_map_.find(key) != section_hints_map_.end();
  }

  // Returns the next section from the stream.
  // Returns std::nullopt if there are no more sections to read.
  absl::StatusOr<std::optional<SectionInfo>> GetNextSection();

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

 private:
  // The parent data stream from which sections are created.
  std::shared_ptr<DataStream> data_stream_;
  // The flatbuffer header of the model file that describes the sections.
  std::shared_ptr<schema::LitertlmHeader> header_;
  // The sections of the model file in the order they are provided by the
  // stream.
  std::vector<SectionInfo> ordered_section_info_;
  // Map of all the sections' section hints.
  std::unordered_map<BufferKey, TfLiteSectionHint, BufferKeyHash>
      section_hints_map_;
  // The index of the next section to be returned.
  size_t next_section_index_ = 0;
  bool header_loaded_ = false;
};

}  // namespace litert::lm
#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LITERT_LM_STREAMING_LOADER_H_
