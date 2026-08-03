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

#include "schema/core/litertlm_utils.h"

#include <string>

#include "schema/core/litertlm_header_schema_generated.h"

namespace litert {
namespace lm {
namespace schema {

std::string AnySectionDataTypeToString(AnySectionDataType value) {
  switch (value) {
    case AnySectionDataType_NONE:
      return "AnySectionDataType_NONE";
    case AnySectionDataType_Deprecated:
      return "AnySectionDataType_Deprecated";
    case AnySectionDataType_TFLiteModel:
      return "AnySectionDataType_TFLiteModel";
    case AnySectionDataType_SP_Tokenizer:
      return "AnySectionDataType_SP_Tokenizer";
    case AnySectionDataType_LlmMetadataProto:
      return "AnySectionDataType_LlmMetadataProto";
    case AnySectionDataType_GenericBinaryData:
      return "AnySectionDataType_GenericBinaryData";
    case AnySectionDataType_HF_Tokenizer_Zlib:
      return "AnySectionDataType_HF_Tokenizer_Zlib";
    default:
      // Handle cases for MIN/MAX or potentially invalid values.
      return "Unknown AnySectionDataType value";
  }
}

}  // namespace schema
}  // namespace lm
}  // namespace litert
