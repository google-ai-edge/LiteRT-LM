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

#include "runtime/components/model_resources_task.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_buffer_ref.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/util/metadata_util.h"
#include "runtime/util/model_asset_bundle_resources.h"
#include "runtime/util/status_macros.h"  // NOLINT

namespace litert::lm {

using ::litert::BufferRef;
using ::litert::Model;

// static
absl::StatusOr<std::unique_ptr<ModelResources>> ModelResourcesTask::Create(
    std::unique_ptr<ModelAssetBundleResources> model_asset_bundle_resources) {
  auto model_resources = absl::WrapUnique(
      new ModelResourcesTask(std::move(model_asset_bundle_resources)));

  return model_resources;
}

absl::StatusOr<std::shared_ptr<litert::Model>>
ModelResourcesTask::GetTFLiteModel(ModelType model_type) {
  if (model_map_.find(model_type) != model_map_.end()) {
    return model_map_[model_type];
  }
  std::string model_file = litert::lm::ModelTypeToString(model_type);
  auto buffer = model_asset_bundle_resources_->GetFile(model_file);
  if (!buffer.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to get model file: ", model_file,
                     "for model type: ", model_type));
  }
  ABSL_LOG(INFO) << "litert model size: " << buffer->size();
  auto buffer_ref = BufferRef<uint8_t>(buffer->data(), buffer->size());
  LITERT_ASSIGN_OR_RETURN(auto model, Model::CreateFromBuffer(buffer_ref));
  model_map_[model_type] = std::make_shared<Model>(std::move(model));
  return model_map_[model_type];
}

absl::StatusOr<std::shared_ptr<Tokenizer>> ModelResourcesTask::GetTokenizer() {
  if (tokenizer_ != nullptr) {
    return tokenizer_;
  }
  ASSIGN_OR_RETURN(auto string_view,  // NOLINT
                   model_asset_bundle_resources_->GetFile("TOKENIZER_MODEL"));
  ASSIGN_OR_RETURN(auto tokenizer,  // NOLINT
                   SentencePieceTokenizer::CreateFromBuffer(string_view));
  tokenizer_ = std::move(tokenizer);
  return tokenizer_;
}

absl::StatusOr<std::shared_ptr<proto::LlmMetadata>>
ModelResourcesTask::GetLlmMetadata() {
  if (llm_metadata_ != nullptr) {
    return llm_metadata_;
  }
  ASSIGN_OR_RETURN(auto string_view,  // NOLINT
                   model_asset_bundle_resources_->GetFile("METADATA"));
  ASSIGN_OR_RETURN(auto llm_metadata,  // NOLINT
                   ExtractOrConvertLlmMetadata(string_view));

  llm_metadata_ = std::make_shared<proto::LlmMetadata>(std::move(llm_metadata));

  ABSL_LOG(INFO) << "The llm metadata: " << llm_metadata_->DebugString();
  return llm_metadata_;
};

}  // namespace litert::lm
