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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_STREAMING_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_STREAMING_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/tokenizer.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

// A dummy implementation of ModelResources that is used when streaming model
// weights instead of loading them from model resources. Most implementation
// methods return unimplemented errors as streaming weights does not support
// random access within the model file.
class ModelResourcesStreaming : public ModelResources {
 public:
  ModelResourcesStreaming() = default;

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      ModelType model_type) override;

  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      ModelType model_type) override;

  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override;

  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      ModelType model_type) override;

  std::optional<std::string> GetTFLiteModelBackendConstraint(
      ModelType model_type) override;

  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override;

  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_STREAMING_H_
