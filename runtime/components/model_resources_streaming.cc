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

#include "runtime/components/model_resources_streaming.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/tokenizer.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

absl::StatusOr<const litert::Model*> ModelResourcesStreaming::GetTFLiteModel(
    ModelType model_type) {
  return absl::UnimplementedError("Not implemented.");
}

absl::StatusOr<absl::string_view> ModelResourcesStreaming::GetTFLiteModelBuffer(
    ModelType model_type) {
  return absl::UnimplementedError("Not implemented.");
}

absl::StatusOr<std::reference_wrapper<ScopedFile>>
ModelResourcesStreaming::GetScopedFile() {
  return absl::UnimplementedError("Not implemented.");
}

absl::StatusOr<std::pair<size_t, size_t>>
ModelResourcesStreaming::GetWeightsSectionOffset(ModelType model_type) {
  return absl::UnimplementedError("Not implemented.");
}

std::optional<std::string>
ModelResourcesStreaming::GetTFLiteModelBackendConstraint(ModelType model_type) {
  return std::nullopt;
}

absl::StatusOr<std::unique_ptr<Tokenizer>>
ModelResourcesStreaming::GetTokenizer() {
  return absl::UnimplementedError("Not implemented.");
}

absl::StatusOr<const proto::LlmMetadata*>
ModelResourcesStreaming::GetLlmMetadata() {
  return absl::UnimplementedError("Not implemented.");
}

}  // namespace litert::lm
