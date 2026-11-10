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

#include "runtime/components/model_resources_litert_lm_google.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/executor/gpu_artisan/utils/model_data.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/status_macros.h"  // NOLINT
#include "schema/core/litertlm_header_schema_generated.h"
#include "tflite/model_builder.h"  // from @litert

namespace litert::lm {
namespace {

using ::odml::infra::llm_utils::ModelData;

}  // namespace

absl::StatusOr<std::unique_ptr<ModelResourcesLitertLmGoogle>>
ModelResourcesLitertLmGoogle::Create(
    std::unique_ptr<LitertLmLoader> litert_lm_loader,
    bool enable_file_backed_model_loading) {
  return absl::WrapUnique(new ModelResourcesLitertLmGoogle(
      std::move(litert_lm_loader), enable_file_backed_model_loading));
}

ModelResourcesLitertLmGoogle::ModelResourcesLitertLmGoogle(
    std::unique_ptr<LitertLmLoader> litert_lm_loader,
    bool enable_file_backed_model_loading)
    : ModelResourcesLitertLm(std::move(litert_lm_loader),
                             enable_file_backed_model_loading) {}

absl::StatusOr<std::shared_ptr<odml::infra::llm_utils::ModelData>>
ModelResourcesLitertLmGoogle::GetSharedArtisanModeldata() {
  if (artisan_model_data_ != nullptr) {
    return artisan_model_data_;
  }
  auto shared_file = litert_lm_loader_->GetSharedScopedFile();
  if (shared_file.ok()) {
    // If the model scoped file could be shared, we can create a model data
    // loading the layers weights on demand, instead of loading the entire model
    // into memory once and for all.
    ASSIGN_OR_RETURN(auto section_location,
                     litert_lm_loader_->GetSectionLocation(BufferKey(
                         schema::AnySectionDataType_TFLiteModel,
                         proto::LlmMetadata::TF_LITE_ARTISAN_TEXT_DECODER)));
    ASSIGN_OR_RETURN(
        artisan_model_data_,
        ModelData::Create(*shared_file, section_location.first,
                          section_location.second - section_location.first));
  } else {
    litert::BufferRef<uint8_t> buffer_ref = litert_lm_loader_->GetTFLiteModel(
        proto::LlmMetadata::TF_LITE_ARTISAN_TEXT_DECODER);
    artisan_model_ = tflite::FlatBufferModel::BuildFromBuffer(
        reinterpret_cast<const char*>(buffer_ref.Data()), buffer_ref.Size());
    if (artisan_model_ == nullptr) {
      return absl::DataLossError("Artisan model data failed to be created.");
    }
    auto model_data = ModelData::Create(artisan_model_);
    if (!model_data.ok()) {
      return absl::DataLossError("Artisan model data failed to be created.");
    }
    artisan_model_data_ = *std::move(model_data);
  }
  return artisan_model_data_;
}

}  // namespace litert::lm
