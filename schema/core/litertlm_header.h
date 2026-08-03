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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_HEADER_H_
#define THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_HEADER_H_

#include <cstdint>
#include <string>

#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "schema/core/litertlm_header_schema_generated.h"

namespace litert {

namespace lm {

namespace schema {

// LiteRT-LM File Format Version uses Semantic Version Rules (SemVer):
// MAJOR version: increments for incompatible API changes.
// MINOR version: increments on added functionality in a backward
//                compatible manner.
// PATCH version: increments on backward compatible bug fixes.
constexpr uint32_t LITERTLM_MAJOR_VERSION = 1;
constexpr uint32_t LITERTLM_MINOR_VERSION = 5;
constexpr uint32_t LITERTLM_PATCH_VERSION = 0;

// Alias for a fully constructed KeyValuePair for LiteRTLM metadata.
// Users of the CreateKeyValuePair function (see below) will get
// back one of these during the creation of their metadata
// data structures.
using KVPair = ::flatbuffers::Offset<KeyValuePair>;

template <typename T>
struct ValueTypeTraits {
  using SchemaType = T;
};

template <>
struct ValueTypeTraits<uint8_t> {
  using SchemaType = UInt8;
};
template <>
struct ValueTypeTraits<int8_t> {
  using SchemaType = Int8;
};
template <>
struct ValueTypeTraits<uint16_t> {
  using SchemaType = UInt16;
};
template <>
struct ValueTypeTraits<int16_t> {
  using SchemaType = Int16;
};
template <>
struct ValueTypeTraits<uint32_t> {
  using SchemaType = UInt32;
};
template <>
struct ValueTypeTraits<int32_t> {
  using SchemaType = Int32;
};
template <>
struct ValueTypeTraits<float> {
  using SchemaType = Float32;
};
template <>
struct ValueTypeTraits<bool> {
  using SchemaType = Bool;
};
template <>
struct ValueTypeTraits<uint64_t> {
  using SchemaType = UInt64;
};
template <>
struct ValueTypeTraits<int64_t> {
  using SchemaType = Int64;
};

template <typename T>
KVPair CreateKeyValuePair(flatbuffers::FlatBufferBuilder& builder,
                          const std::string& key, const T& value) {
  auto key_offset = builder.CreateString(key);

  flatbuffers::Offset<void> value_offset;
  VData value_type;

  if constexpr (std::is_same_v<T, uint8_t>) {
    value_offset = CreateUInt8(builder, value).Union();
    value_type = VData::VData_UInt8;
  } else if constexpr (std::is_same_v<T, int8_t>) {
    value_offset = CreateInt8(builder, value).Union();
    value_type = VData::VData_Int8;
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    value_offset = CreateUInt16(builder, value).Union();
    value_type = VData::VData_UInt16;
  } else if constexpr (std::is_same_v<T, int16_t>) {
    value_offset = CreateInt16(builder, value).Union();
    value_type = VData::VData_Int16;
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    value_offset = CreateUInt32(builder, value).Union();
    value_type = VData::VData_UInt32;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    value_offset = CreateInt32(builder, value).Union();
    value_type = VData::VData_Int32;
  } else if constexpr (std::is_same_v<T, float>) {
    value_offset = CreateFloat32(builder, value).Union();
    value_type = VData::VData_Float32;
  } else if constexpr (std::is_same_v<T, bool>) {
    value_offset = CreateBool(builder, value).Union();
    value_type = VData::VData_Bool;
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    value_offset = CreateUInt64(builder, value).Union();
    value_type = VData::VData_UInt64;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    value_offset = CreateInt64(builder, value).Union();
    value_type = VData::VData_Int64;
  }

  KeyValuePairBuilder kvp_builder(builder);
  kvp_builder.add_key(key_offset);
  kvp_builder.add_value(value_offset);
  kvp_builder.add_value_type(value_type);
  return kvp_builder.Finish();
}

template <>
inline KVPair CreateKeyValuePair(flatbuffers::FlatBufferBuilder& builder,
                                 const std::string& key,
                                 const std::string& value) {
  auto key_offset = builder.CreateString(key);
  // NB: The StringValue object *must* be created before the
  // KeyValuePairBuilder.
  auto value_offset = CreateStringValue(builder, builder.CreateString(value));
  KeyValuePairBuilder kvp_builder(builder);
  kvp_builder.add_key(key_offset);
  kvp_builder.add_value(value_offset.Union());
  kvp_builder.add_value_type(VData::VData_StringValue);
  return kvp_builder.Finish();
}

template <>
inline KVPair CreateKeyValuePair(
    flatbuffers::FlatBufferBuilder& builder, const std::string& key,
    const flatbuffers::Offset<StringValue>& value) {
  auto key_offset = builder.CreateString(key);
  KeyValuePairBuilder kvp_builder(builder);
  kvp_builder.add_key(key_offset);
  kvp_builder.add_value(value.Union());
  kvp_builder.add_value_type(VData::VData_StringValue);
  return kvp_builder.Finish();
}

}  // end namespace schema
}  // end namespace lm
}  // end namespace litert

#endif  // THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CORE_LITERTLM_HEADER_H_
