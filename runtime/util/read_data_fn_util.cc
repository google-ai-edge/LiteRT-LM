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

#include "runtime/util/read_data_fn_util.h"

#include <cstdint>
#include <cstdlib>
#include <memory>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/executor/gpu_artisan/utils/model_data.h"
#include "runtime/util/data_stream.h"

namespace litert::lm {

namespace oi = ::odml::infra;

odml::infra::llm_utils::ModelData::ReadDataFn MakeReadDataFn(
    std::shared_ptr<DataStream> section_data_stream, uint64_t section_size) {
  return [section_data_stream, section_size](uint64_t offset, uint64_t size,
                                             int mode) -> void* {
    char* buffer = static_cast<char*>(malloc(size));
    if (buffer == nullptr) {
      ABSL_LOG(ERROR) << "Failed to allocate buffer of size " << size;
      return nullptr;
    }

    absl::Status read_result;
    if (mode == oi::llm_utils::ModelData::ReadMode::KEEP) {
      read_result = section_data_stream->ReadAndPreserve(buffer, offset, size);
    } else if (mode == oi::llm_utils::ModelData::ReadMode::DISCARD) {
      read_result = section_data_stream->ReadAndDiscard(buffer, offset, size);
    } else if (mode == oi::llm_utils::ModelData::ReadMode::DISCARD_ALL) {
      read_result = section_data_stream->Discard(0, section_size);
    }

    if (!read_result.ok()) {
      ABSL_LOG(ERROR) << "Failed to read from stream: " << read_result;
      free(buffer);
      return nullptr;
    }

    return buffer;
  };
}

}  // namespace litert::lm
