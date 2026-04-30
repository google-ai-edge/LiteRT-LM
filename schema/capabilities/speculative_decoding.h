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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CAPABILITIES_SPECULATIVE_DECODING_H_
#define THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CAPABILITIES_SPECULATIVE_DECODING_H_

#include <istream>
#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl

namespace litert::lm::schema::capabilities {

// Returns true if the LiteRT-LM file has speculative decoding support.
//
// Args:
//   litertlm_stream: The input stream to the LiteRT-LM file.
//
// Returns:
//   True if the LiteRT-LM file has speculative decoding support, false
//   otherwise.
absl::StatusOr<bool> HasSpeculativeDecodingSupport(
    std::istream& litertlm_stream);

// Returns true if the LiteRT-LM file has speculative decoding support.
//
// Args:
//   litertlm_path: The path to the LiteRT-LM file.
//
// Returns:
//   True if the LiteRT-LM file has speculative decoding support, false
//   otherwise.
absl::StatusOr<bool> HasSpeculativeDecodingSupport(
    const std::string& litertlm_path);

}  // namespace litert::lm::schema::capabilities

#endif  // THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CAPABILITIES_SPECULATIVE_DECODING_H_
