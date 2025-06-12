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

#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/model_asset_bundle_resources.h"

namespace litert::lm {

// Model resources for the task model.
class ModelResourcesTask : public ModelResources {
 public:
  static absl::StatusOr<std::unique_ptr<ModelResources>> Create(
      std::unique_ptr<ModelAssetBundleResources> model_asset_bundle_resources);

  absl::StatusOr<std::shared_ptr<litert::Model>> GetTFLiteModel(
      ModelType model_type) override;

  absl::StatusOr<std::shared_ptr<Tokenizer>> GetTokenizer() override;

  absl::StatusOr<std::shared_ptr<proto::LlmMetadata>> GetLlmMetadata() override;

 private:
  explicit ModelResourcesTask(
      std::unique_ptr<ModelAssetBundleResources> model_asset_bundle_resources)
      : model_asset_bundle_resources_(std::move(model_asset_bundle_resources)) {
  }

  std::unordered_map<ModelType, std::shared_ptr<litert::Model>> model_map_;
  std::shared_ptr<SentencePieceTokenizer> tokenizer_;
  std::shared_ptr<proto::LlmMetadata> llm_metadata_;

  // The model asset bundle resources produced by reading task bundle. Not null
  // only when the model is provided through .task format. If the model is
  // retrieved from this resource, releasing this resource will also invalidate
  // the model.
  std::unique_ptr<ModelAssetBundleResources> model_asset_bundle_resources_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_TASK_H_
