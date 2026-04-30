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

#include "schema/capabilities/speculative_decoding.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <ios>
#include <istream>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "runtime/util/status_macros.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_read.h"

namespace litert::lm::schema::capabilities {

absl::StatusOr<bool> HasSpeculativeDecodingSupport(
    std::istream& litertlm_stream) {
  litertlm_stream.seekg(0);
  LitertlmHeader header;
  RETURN_IF_ERROR(ReadHeaderFromLiteRTLM(litertlm_stream, &header));

  const std::vector<std::string> speculative_decoding_model_types = {
      "tf_lite_mtp_drafter"};

  const LiteRTLMMetaData* litertlm_metadata = header.metadata;
  RET_CHECK_NE(litertlm_metadata, nullptr);
  const litert::lm::schema::SectionMetadata* section_metadata_obj =
      litertlm_metadata->section_metadata();
  RET_CHECK_NE(section_metadata_obj, nullptr);
  auto section_objects = section_metadata_obj->objects();
  RET_CHECK_NE(section_objects, nullptr);
  for (int i = 0; i < section_objects->size(); ++i) {
    const litert::lm::schema::SectionObject* section_object =
        section_objects->Get(i);
    RET_CHECK_NE(section_object, nullptr);
    if (section_object->data_type() == AnySectionDataType_TFLiteModel) {
      const auto* items = section_object->items();
      RET_CHECK_NE(items, nullptr);
      for (size_t j = 0; j < items->size(); ++j) {
        const KeyValuePair* item = items->Get(j);
        RET_CHECK_NE(item, nullptr);
        const auto* key = item->key();
        RET_CHECK_NE(key, nullptr);
        if (key->string_view() == "model_type") {
          const auto* value = item->value_as_StringValue();
          RET_CHECK_NE(value, nullptr);
          const auto* value_string = value->value();
          RET_CHECK_NE(value_string, nullptr);
          if (std::find(speculative_decoding_model_types.begin(),
                        speculative_decoding_model_types.end(),
                        value_string->string_view()) !=
              speculative_decoding_model_types.end()) {
            return true;
          }
        }
      }
    }
  }

  return false;
}

absl::StatusOr<bool> HasSpeculativeDecodingSupport(
    const std::string& litertlm_path) {
  std::ifstream input_file_stream(litertlm_path, std::ios::binary);
  if (!input_file_stream.is_open()) {
    return absl::InternalError(
        absl::StrFormat("Could not open file: %s", litertlm_path));
  }
  return HasSpeculativeDecodingSupport(input_file_stream);
}

}  // namespace litert::lm::schema::capabilities
