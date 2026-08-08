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

#include "runtime/executor/vision_executor_utils.h"

#include <optional>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {

constexpr absl::string_view kFeatures = "features";

absl::StatusOr<VisionExecutorProperties>
GetVisionExecutorPropertiesFromModelResources(ModelResources& model_resources) {
  VisionExecutorProperties properties;
  auto vision_adapter_model_or =
      model_resources.GetTFLiteModel(ModelType::kTfLiteVisionAdapter);
  const Model* vision_adapter_model = nullptr;
  if (vision_adapter_model_or.ok()) {
    vision_adapter_model = vision_adapter_model_or.value();
  } else if (vision_adapter_model_or.status().code() !=
             absl::StatusCode::kNotFound) {
    return vision_adapter_model_or.status();
  }

  ABSL_ASSIGN_OR_RETURN(
      auto vision_encoder_model,
      model_resources.GetTFLiteModel(ModelType::kTfLiteVisionEncoder));

  if (vision_adapter_model != nullptr) {
    LITERT_ASSIGN_OR_RETURN(auto adapter_output_tensor_type,
                            vision_adapter_model->GetOutputTensorType(0, 0));
    if (adapter_output_tensor_type.Layout().Dimensions().size() < 2) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The adapter output tensor has invalid dimensions: ",
          adapter_output_tensor_type.Layout().Dimensions().size()));
    }
    properties.num_tokens_per_image =
        adapter_output_tensor_type.Layout().Dimensions()
            [adapter_output_tensor_type.Layout().Dimensions().size() - 2];
  } else {
    LITERT_ASSIGN_OR_RETURN(auto encoder_signature,
                            vision_encoder_model->GetSignature(0));
    auto features_output_or = encoder_signature.OutputTensorType(kFeatures);
    std::optional<RankedTensorType> encoder_output_tensor_type;
    if (features_output_or.HasValue()) {
      encoder_output_tensor_type = features_output_or.Value();
    } else {
      LITERT_ASSIGN_OR_RETURN(auto fallback_output,
                              encoder_signature.OutputTensorType(0));
      encoder_output_tensor_type = fallback_output;
    }
    if (encoder_output_tensor_type->Layout().Dimensions().size() < 2) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The encoder output tensor has invalid dimensions: ",
          encoder_output_tensor_type->Layout().Dimensions().size()));
    }
    properties.num_tokens_per_image =
        encoder_output_tensor_type->Layout().Dimensions()
            [encoder_output_tensor_type->Layout().Dimensions().size() - 2];
  }

  LITERT_ASSIGN_OR_RETURN(auto encoder_input_names,
                          vision_encoder_model->GetSignatureInputNames(0));
  // Deduce the patch shrink factor from the encoder's image input tensor.
  // Transformer (ViT) encoders expose two inputs (image patches +
  // positions_xy), while single input encoders (e.g. LFM2 VL) expose only the
  // image patches tensor. In both cases the image patches tensor is the first
  // input, so this works as long as the encoder has at least one input.
  if (!encoder_input_names.empty()) {
    // The image patches tensor has shape [batch_size, num_patches, patch_dim],
    // so the second-to-last dimension is the number of input patches.
    LITERT_ASSIGN_OR_RETURN(auto encoder_input_tensor_type,
                            vision_encoder_model->GetInputTensorType(0, 0));
    properties.patch_num_shrink_factor =
        encoder_input_tensor_type.Layout().Dimensions()
            [encoder_input_tensor_type.Layout().Dimensions().size() - 2] /
        properties.num_tokens_per_image;
  }
  return properties;
}

}  // namespace litert::lm
