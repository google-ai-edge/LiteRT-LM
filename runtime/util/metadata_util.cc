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

#include "runtime/util/metadata_util.h"

#include <string>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/proto/llm_metadata.pb.h"

namespace litert::lm {

absl::StatusOr<proto::LlmMetadata> ExtractOrConvertLlmMetadata(
    absl::string_view string_view) {
  proto::LlmMetadata llm_metadata;
  if (!llm_metadata.ParseFromString(string_view) ||
      !llm_metadata.has_start_token()) {
      return absl::InvalidArgumentError("Failed to parse LlmMetadata.");
  }
  ABSL_LOG(INFO) << "The llm metadata: " << llm_metadata.DebugString();
  return llm_metadata;
}

}  // namespace litert::lm
