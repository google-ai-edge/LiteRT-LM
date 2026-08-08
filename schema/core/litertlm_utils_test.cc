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

#include "schema/core/litertlm_utils.h"

#include <gtest/gtest.h>
#include "schema/core/litertlm_header_schema_generated.h"

namespace litert::lm::schema {
namespace {

TEST(LitertLmUtilsTest, AnySectionDataTypeToString) {
  EXPECT_EQ(AnySectionDataTypeToString(AnySectionDataType_NONE),
            "AnySectionDataType_NONE");
  EXPECT_EQ(AnySectionDataTypeToString(AnySectionDataType_Deprecated),
            "AnySectionDataType_Deprecated");
  EXPECT_EQ(AnySectionDataTypeToString(AnySectionDataType_TFLiteModel),
            "AnySectionDataType_TFLiteModel");
  EXPECT_EQ(AnySectionDataTypeToString(AnySectionDataType_SP_Tokenizer),
            "AnySectionDataType_SP_Tokenizer");
  EXPECT_EQ(AnySectionDataTypeToString(AnySectionDataType_LlmMetadataProto),
            "AnySectionDataType_LlmMetadataProto");
  EXPECT_EQ(AnySectionDataTypeToString(AnySectionDataType_GenericBinaryData),
            "AnySectionDataType_GenericBinaryData");
  EXPECT_EQ(AnySectionDataTypeToString(AnySectionDataType_HF_Tokenizer_Zlib),
            "AnySectionDataType_HF_Tokenizer_Zlib");
  EXPECT_EQ(AnySectionDataTypeToString(AnySectionDataType_TFLiteWeights),
            "AnySectionDataType_TFLiteWeights");
  EXPECT_EQ(
      AnySectionDataTypeToString(AnySectionDataType_EmbeddingMetadataProto),
      "AnySectionDataType_EmbeddingMetadataProto");
}

}  // namespace
}  // namespace litert::lm::schema
