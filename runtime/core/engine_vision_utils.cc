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

#include "runtime/core/engine_vision_utils.h"

#include <optional>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/model_signature_utils.h"
#include "runtime/proto/llm_metadata.pb.h"

namespace litert::lm {
namespace {

std::optional<int> GetVisionTokensPerImageFromMetadata(
    const proto::LlmMetadata* llm_metadata) {
  if (llm_metadata == nullptr || !llm_metadata->has_llm_model_type()) {
    return std::nullopt;
  }
  const auto& model_type = llm_metadata->llm_model_type();
  int max_num_patches = 0;
  int pooling_kernel_size = 1;
  if (model_type.has_gemma4()) {
    max_num_patches = model_type.gemma4().max_num_patches();
    pooling_kernel_size = model_type.gemma4().pooling_kernel_size() > 0
                              ? model_type.gemma4().pooling_kernel_size()
                              : 3;
  } else if (model_type.has_generic_model()) {
    max_num_patches = model_type.generic_model().max_num_patches();
    if (model_type.generic_model().pooling_kernel_size() > 0) {
      pooling_kernel_size = model_type.generic_model().pooling_kernel_size();
    }
  } else if (model_type.has_lfm2()) {
    max_num_patches = model_type.lfm2().max_num_patches();
    pooling_kernel_size = model_type.lfm2().pooling_kernel_size() > 0
                              ? model_type.lfm2().pooling_kernel_size()
                              : 2;
  }
  if (max_num_patches <= 0) {
    return std::nullopt;
  }
  const int patch_num_shrink_factor = pooling_kernel_size * pooling_kernel_size;
  return max_num_patches / patch_num_shrink_factor;
}

}  // namespace

absl::Status MaybeSelectVisionSignatures(
    EngineSettings& engine_settings, ModelResources& model_resources,
    const proto::LlmMetadata* absl_nullable llm_metadata) {
  if (!engine_settings.GetVisionExecutorSettings().has_value()) {
    return absl::OkStatus();
  }
  std::optional<int> vision_token_limit;
  if (engine_settings.GetMaxVisionTokensPerImage().has_value()) {
    const int max_vision_tokens_per_image =
        *engine_settings.GetMaxVisionTokensPerImage();
    if (max_vision_tokens_per_image <= 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("max_vision_tokens_per_image must be positive, got: ",
                       max_vision_tokens_per_image));
    }
    if (GetVisionTokensPerImageFromMetadata(llm_metadata).has_value()) {
      vision_token_limit = max_vision_tokens_per_image;
    }
  } else {
    vision_token_limit = GetVisionTokensPerImageFromMetadata(llm_metadata);
    if (vision_token_limit.has_value() && *vision_token_limit > 0) {
      engine_settings.SetMaxVisionTokensPerImage(*vision_token_limit);
    }
  }

  if (!vision_token_limit.has_value() || *vision_token_limit <= 0) {
    return absl::OkStatus();
  }
#ifdef __EMSCRIPTEN__
  // On web, load a single vision signature by default. Each extra signature
  // takes hundreds of MB to GBs of heap on GPU, and loading all of them
  // overflows the 4GB wasm32 address space. Users can still lower the min to
  // load more signatures.
  if (!engine_settings.GetMinVisionTokensPerImage().has_value()) {
    engine_settings.SetMinVisionTokensPerImage(*vision_token_limit);
  }
#endif  // __EMSCRIPTEN__
  const std::optional<int> min_vision_tokens_per_image =
      engine_settings.GetMinVisionTokensPerImage();

  ABSL_ASSIGN_OR_RETURN(
      auto vision_sig_info,
      SelectVisionEncoderSignatures(model_resources, *vision_token_limit,
                                    min_vision_tokens_per_image));
  engine_settings.GetMutableVisionExecutorSettings()
      ->SetEncoderSelectedSignatures(vision_sig_info.signature_names);

  ABSL_ASSIGN_OR_RETURN(
      auto adapter_sig_info,
      SelectVisionAdapterSignatures(model_resources, *vision_token_limit,
                                    min_vision_tokens_per_image));
  if (adapter_sig_info.has_value()) {
    engine_settings.GetMutableVisionExecutorSettings()
        ->SetAdapterSelectedSignatures(adapter_sig_info->signature_names);
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
