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
  std::unique_ptr<DataStream> data_stream;

  std::weak_ptr<schema::LitertlmHeader> header;
};

// Reads the Litert LM header from the given data stream.
// If preserve is true, the read data will remain in the stream.
// If preserve is false, the data will be discarded from the stream.
absl::StatusOr<std::unique_ptr<schema::LitertlmHeader>>
ReadHeaderFromDataStream(DataStream* data_stream, bool preserve);

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
  // Returns an error if the header cannot be loaded.
  // Optional. If not called, LoadHeader() will be called within
  // GetNextSection().
  absl::Status LoadHeader();

  // Returns the next section from the stream.
  // Returns std::nullopt if there are no more sections to read.
  absl::StatusOr<std::optional<SectionInfo>> GetNextSection();

  // Returns all sections info
  const std::vector<SectionInfo>& GetOrderedSectionInfo() const {
    return ordered_section_info_;
  }

 private:
  // The parent data stream from which sections are created.
  std::shared_ptr<DataStream> data_stream_;
  // The flatbuffer header of the model file that describes the sections.
  std::shared_ptr<schema::LitertlmHeader> header_;
  // The sections of the model file in the order they are provided by the
  // stream.
  std::vector<SectionInfo> ordered_section_info_;
  // The index of the next section to be returned.
  size_t next_section_index_ = 0;
  bool header_loaded_ = false;
};

}  // namespace litert::lm
#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LITERT_LM_STREAMING_LOADER_H_
