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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_TASK_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_TASK_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/scoped_file.h"

namespace litert::lm {

// Model resources for the task model.
class ModelResourcesTask : public ModelResources {
 public:
  static absl::StatusOr<std::unique_ptr<ModelResources>> Create(
      std::unique_ptr<ModelAssetBundleResources> model_asset_bundle_resources);

  absl::StatusOr<const litert::Model*> GetTFLiteModel(
      ModelType model_type) override;
  absl::StatusOr<absl::string_view> GetTFLiteModelBuffer(
      ModelType model_type) override;
  std::optional<std::string> GetTFLiteModelBackendConstraint(
      ModelType model_type) override {
    // Task model does not support backend constraint.
    return std::nullopt;
  };
  std::optional<std::string> GetTFLiteModelPreferActivationType(
      ModelType model_type) override {
    // Task model does not support prefer activation type.
    return std::nullopt;
  };
  absl::StatusOr<std::unique_ptr<Tokenizer>> GetTokenizer() override;
  absl::StatusOr<const proto::LlmMetadata*> GetLlmMetadata() override;
  absl::StatusOr<std::reference_wrapper<ScopedFile>> GetScopedFile() override {
    return absl::UnimplementedError(
        "GetScopedFile is not implemented for Task model.");
  }
  absl::StatusOr<std::pair<size_t, size_t>> GetWeightsSectionOffset(
      ModelType model_type) override {
    return absl::UnimplementedError(
        "GetWeightsSectionOffset is not implemented for Task model.");
  }

 private:
  explicit ModelResourcesTask(
      std::unique_ptr<ModelAssetBundleResources> model_asset_bundle_resources)
      : model_asset_bundle_resources_(std::move(model_asset_bundle_resources)) {
  }

  absl::flat_hash_map<ModelType, std::shared_ptr<litert::Model>> model_map_;
  std::unique_ptr<proto::LlmMetadata> llm_metadata_;

  // The model asset bundle resources produced by reading task bundle. Not null
  // only when the model is provided through .task format. If the model is
  // retrieved from this resource, releasing this resource will also invalidate
  // the model.
  std::unique_ptr<ModelAssetBundleResources> model_asset_bundle_resources_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_TASK_H_
