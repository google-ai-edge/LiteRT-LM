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

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "litert/cc/litert_element_type.h"  // from @litert

namespace litert::lm {
namespace {

TEST(SafetensorsUtilTest, GetSafetensorsDtypeInfoMapsAllSupportedTypes) {
  auto f32_info = GetSafetensorsDtypeInfo(::litert::ElementType::Float32);
  EXPECT_EQ(f32_info.dtype, SafetensorsDtype::kF32);
  EXPECT_EQ(f32_info.element_size, sizeof(float));

  auto f16_info = GetSafetensorsDtypeInfo(::litert::ElementType::Float16);
  EXPECT_EQ(f16_info.dtype, SafetensorsDtype::kF16);
  EXPECT_EQ(f16_info.element_size, 2);

  auto bf16_info = GetSafetensorsDtypeInfo(::litert::ElementType::BFloat16);
  EXPECT_EQ(bf16_info.dtype, SafetensorsDtype::kBFloat16);
  EXPECT_EQ(bf16_info.element_size, 2);

  auto i32_info = GetSafetensorsDtypeInfo(::litert::ElementType::Int32);
  EXPECT_EQ(i32_info.dtype, SafetensorsDtype::kInt32);
  EXPECT_EQ(i32_info.element_size, sizeof(int32_t));

  auto i64_info = GetSafetensorsDtypeInfo(::litert::ElementType::Int64);
  EXPECT_EQ(i64_info.dtype, SafetensorsDtype::kInt64);
  EXPECT_EQ(i64_info.element_size, sizeof(int64_t));

  auto i8_info = GetSafetensorsDtypeInfo(::litert::ElementType::Int8);
  EXPECT_EQ(i8_info.dtype, SafetensorsDtype::kInt8);
  EXPECT_EQ(i8_info.element_size, sizeof(int8_t));

  auto u8_info = GetSafetensorsDtypeInfo(::litert::ElementType::UInt8);
  EXPECT_EQ(u8_info.dtype, SafetensorsDtype::kUInt8);
  EXPECT_EQ(u8_info.element_size, sizeof(uint8_t));

  auto bool_info = GetSafetensorsDtypeInfo(::litert::ElementType::Bool);
  EXPECT_EQ(bool_info.dtype, SafetensorsDtype::kBool);
  EXPECT_EQ(bool_info.element_size, sizeof(bool));
}

TEST(SafetensorsUtilTest, BuildSafetensorsHeaderWithMetadata) {
  std::vector<int64_t> shape = {1, 16, 64};
  absl::flat_hash_map<std::string, std::string> metadata = {
      {"signature", "decode"},
      {"step", "5"},
  };
  std::string header = BuildSafetensorsHeader(
      "post_query", SafetensorsDtype::kF32, shape,
      /*total_bytes=*/4096, metadata);

  EXPECT_EQ(header.size() % kSafetensorsHeaderAlignment, 0);

  auto json_obj = nlohmann::json::parse(header);
  ASSERT_TRUE(json_obj.contains("post_query"));
  EXPECT_EQ(json_obj["post_query"]["dtype"], "F32");
  EXPECT_EQ(json_obj["post_query"]["shape"], (std::vector<int64_t>{1, 16, 64}));
  EXPECT_EQ(json_obj["post_query"]["data_offsets"],
            (std::vector<size_t>{0, 4096}));

  ASSERT_TRUE(json_obj.contains("__metadata__"));
  EXPECT_EQ(json_obj["__metadata__"]["signature"], "decode");
  EXPECT_EQ(json_obj["__metadata__"]["step"], "5");
}

TEST(SafetensorsUtilTest, BuildSafetensorsHeaderWithoutMetadata) {
  std::vector<int64_t> shape = {2, 8};
  std::string header = BuildSafetensorsHeader(
      "embeddings", SafetensorsDtype::kF16, shape, /*total_bytes=*/32);

  EXPECT_EQ(header.size() % kSafetensorsHeaderAlignment, 0);

  auto json_obj = nlohmann::json::parse(header);
  ASSERT_TRUE(json_obj.contains("embeddings"));
  EXPECT_EQ(json_obj["embeddings"]["dtype"], "F16");
  EXPECT_EQ(json_obj["embeddings"]["shape"], (std::vector<int64_t>{2, 8}));
  EXPECT_FALSE(json_obj.contains("__metadata__"));
}

}  // namespace
}  // namespace litert::lm
