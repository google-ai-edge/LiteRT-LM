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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_LITERT_LM_GOOGLE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_LITERT_LM_GOOGLE_H_

#include <memory>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/components/model_data_provider.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/executor/gpu_artisan/utils/model_data.h"
#include "runtime/util/litert_lm_loader.h"
#include "tflite/model_builder.h"  // from @litert

namespace litert::lm {

// Model resources for the litert lm model.
class ModelResourcesLitertLmGoogle : public ModelResourcesLitertLm,
                                     public ModelDataProvider {
 public:
  static absl::StatusOr<std::unique_ptr<ModelResourcesLitertLmGoogle>> Create(
      std::unique_ptr<LitertLmLoader> litert_lm_loader,
      bool enable_file_backed_model_loading = false);

  // ModelData is used for loading the weights of the hand-written model (i.e.
  // Artisan Executor). This function is responsible for creating the ModelData
  // loader and return the shared_ptr to it. The main owner of the ModelData is
  // this ModelResourcesLitertLmGoogle class similar to other assets.
  absl::StatusOr<std::shared_ptr<odml::infra::llm_utils::ModelData>>
  GetSharedArtisanModeldata() override;

 private:
  explicit ModelResourcesLitertLmGoogle(
      std::unique_ptr<LitertLmLoader> litert_lm_loader,
      bool enable_file_backed_model_loading);

  std::shared_ptr<tflite::FlatBufferModel> artisan_model_;
  std::shared_ptr<odml::infra::llm_utils::ModelData> artisan_model_data_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_LITERT_LM_GOOGLE_H_
