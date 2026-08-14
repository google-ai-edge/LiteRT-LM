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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_SAFETENSORS_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_SAFETENSORS_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert

namespace litert::lm {

// Safetensors header alignment requirement (8 bytes).
inline constexpr size_t kSafetensorsHeaderAlignment = 8;

// Authoritative filesystem extension for HuggingFace Safetensors serialization
// artifacts.
inline constexpr absl::string_view kSafetensorsExtension = ".safetensors";

// Enumeration of standard HuggingFace Safetensors primitive data-types.
enum class SafetensorsDtype {
  kF32,
  kF16,
  kBFloat16,
  kInt32,
  kInt64,
  kInt8,
  kUInt8,
  kBool,
  kDefault = kF32,
};

// LiteRT ElementType mapping metadata.
struct SafetensorsDtypeInfo {
  SafetensorsDtype dtype = SafetensorsDtype::kDefault;
  size_t element_size = 0;
};

// Maps LiteRT ElementType to Safetensors dtype string and element byte size.
SafetensorsDtypeInfo GetSafetensorsDtypeInfo(
    ::litert::ElementType element_type);

// Builds the HuggingFace Safetensors 8-byte aligned JSON header.
// Supports generic metadata dictionary without coupling to specific domain
// fields.
std::string BuildSafetensorsHeader(
    absl::string_view tensor_name, SafetensorsDtype dtype,
    const std::vector<int64_t>& shape, size_t total_bytes,
    const absl::flat_hash_map<std::string, std::string>& metadata = {});

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_SAFETENSORS_UTIL_H_
