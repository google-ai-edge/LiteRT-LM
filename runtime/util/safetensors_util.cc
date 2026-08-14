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

#include "runtime/util/safetensors_util.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "litert/cc/litert_element_type.h"  // from @litert

namespace litert::lm {
namespace {

// Internal 8-byte aligned JSON serialization keys prescribed by the
// HuggingFace Safetensors format standard.
constexpr absl::string_view kDtypeKey = "dtype";
constexpr absl::string_view kShapeKey = "shape";
constexpr absl::string_view kDataOffsetsKey = "data_offsets";
constexpr absl::string_view kMetadataKey = "__metadata__";

// Maps a SafetensorsDtype enum to its canonical Safetensors JSON string
// representation.
absl::string_view ToString(SafetensorsDtype dtype) {
  switch (dtype) {
    case SafetensorsDtype::kF32:
      return "F32";
    case SafetensorsDtype::kF16:
      return "F16";
    case SafetensorsDtype::kBFloat16:
      return "BF16";
    case SafetensorsDtype::kInt32:
      return "I32";
    case SafetensorsDtype::kInt64:
      return "I64";
    case SafetensorsDtype::kInt8:
      return "I8";
    case SafetensorsDtype::kUInt8:
      return "U8";
    case SafetensorsDtype::kBool:
      return "BOOL";
  }
  return "F32";
}

}  // namespace

SafetensorsDtypeInfo GetSafetensorsDtypeInfo(
    ::litert::ElementType element_type) {
  switch (element_type) {
    case ::litert::ElementType::Float32:
      return {.dtype = SafetensorsDtype::kF32, .element_size = sizeof(float)};
    case ::litert::ElementType::Float16:
      return {.dtype = SafetensorsDtype::kF16, .element_size = 2};
    case ::litert::ElementType::BFloat16:
      return {.dtype = SafetensorsDtype::kBFloat16, .element_size = 2};
    case ::litert::ElementType::Int32:
      return {.dtype = SafetensorsDtype::kInt32,
              .element_size = sizeof(int32_t)};
    case ::litert::ElementType::Int64:
      return {.dtype = SafetensorsDtype::kInt64,
              .element_size = sizeof(int64_t)};
    case ::litert::ElementType::Int8:
      return {.dtype = SafetensorsDtype::kInt8, .element_size = sizeof(int8_t)};
    case ::litert::ElementType::UInt8:
      return {.dtype = SafetensorsDtype::kUInt8,
              .element_size = sizeof(uint8_t)};
    case ::litert::ElementType::Bool:
      return {.dtype = SafetensorsDtype::kBool, .element_size = sizeof(bool)};
    default:
      return {.dtype = SafetensorsDtype::kDefault,
              .element_size = sizeof(float)};
  }
}

std::string BuildSafetensorsHeader(
    absl::string_view tensor_name, SafetensorsDtype dtype,
    const std::vector<int64_t>& shape, size_t total_bytes,
    const absl::flat_hash_map<std::string, std::string>& metadata) {
  nlohmann::ordered_json header;
  header[std::string(tensor_name)] = {
      {std::string(kDtypeKey), std::string(ToString(dtype))},
      {std::string(kShapeKey), shape},
      {std::string(kDataOffsetsKey), {0, total_bytes}},
  };
  if (!metadata.empty()) {
    header[std::string(kMetadataKey)] = metadata;
  }

  std::string json_header = header.dump();
  size_t remainder = json_header.size() % kSafetensorsHeaderAlignment;
  if (remainder != 0) {
    json_header.append(kSafetensorsHeaderAlignment - remainder, ' ');
  }
  return json_header;
}

}  // namespace litert::lm
