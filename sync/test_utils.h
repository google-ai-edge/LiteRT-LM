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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SYNC_TEST_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_SYNC_TEST_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "sync/sha256.h"

namespace litert::lm::sync {

// Section specification used by `CreateTestModel` to build synthetic
// `.litertlm` container files in tests.
struct TestSectionSpec {
  schema::AnySectionDataType data_type;
  std::string model_type;
  std::string payload;
  size_t begin_offset;
};

// Writes a valid `.litertlm` container file at `path` with the given sections
// and optional per-section SHA-256 metadata, returning `path`.
inline std::string CreateTestModel(const std::string& path,
                                   const std::vector<TestSectionSpec>& sections,
                                   bool include_sha256 = true) {
  flatbuffers::FlatBufferBuilder builder;

  std::vector<flatbuffers::Offset<schema::SectionObject>> section_objs;
  for (const auto& sec : sections) {
    std::vector<flatbuffers::Offset<schema::KeyValuePair>> items;
    if (include_sha256) {
      std::string sha = ComputeSha256Hex(sec.payload);
      items.push_back(schema::CreateKeyValuePair(builder, "sha256", sha));
    }
    if (!sec.model_type.empty()) {
      items.push_back(
          schema::CreateKeyValuePair(builder, "model_type", sec.model_type));
    }

    size_t end_offset = sec.begin_offset + sec.payload.size();
    section_objs.push_back(schema::CreateSectionObject(
        builder, builder.CreateVector(items), sec.begin_offset, end_offset,
        sec.data_type));
  }

  auto section_metadata_offset = schema::CreateSectionMetadata(
      builder, builder.CreateVector(section_objs));
  auto root_offset =
      schema::CreateLiteRTLMMetaData(builder, 0, section_metadata_offset);
  builder.Finish(root_offset);

  uint8_t* fb_buf = builder.GetBufferPointer();
  size_t fb_size = builder.GetSize();

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write("LITERTLM", 8);
  uint32_t major = 1, minor = 9, patch = 0, pad = 0;
  out.write(reinterpret_cast<const char*>(&major), 4);
  out.write(reinterpret_cast<const char*>(&minor), 4);
  out.write(reinterpret_cast<const char*>(&patch), 4);
  out.write(reinterpret_cast<const char*>(&pad), 4);

  uint64_t header_end_offset = 32 + fb_size;
  out.write(reinterpret_cast<const char*>(&header_end_offset), 8);
  out.write(reinterpret_cast<const char*>(fb_buf), fb_size);

  for (const auto& sec : sections) {
    if (static_cast<uint64_t>(out.tellp()) < sec.begin_offset) {
      size_t fill = sec.begin_offset - out.tellp();
      std::string zeroes(fill, '\0');
      out.write(zeroes.data(), fill);
    }
    out.seekp(sec.begin_offset);
    out.write(sec.payload.data(), sec.payload.size());
  }
  out.close();

  return path;
}

}  // namespace litert::lm::sync

#endif  // THIRD_PARTY_ODML_LITERT_LM_SYNC_TEST_UTILS_H_
