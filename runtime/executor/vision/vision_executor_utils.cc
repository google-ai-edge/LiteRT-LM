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

#include "runtime/executor/vision/vision_executor_utils.h"

#include <cstddef>
#include <optional>

#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/engine/io_types.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

constexpr absl::string_view kFeatures = "features";
constexpr absl::string_view kImages = "images";
constexpr absl::string_view kPositionsXy = "positions_xy";
constexpr absl::string_view kVisionEncoderSignaturePrefix = "vision_";
constexpr absl::string_view kVisionAdapterSignaturePrefix = "vision_adapter_";

absl::StatusOr<int> GetAdapterSignatureTokenLength(const Model& adapter_model,
                                                   size_t signature_index) {
  LITERT_ASSIGN_OR_RETURN(
      auto adapter_output_tensor_type,
      adapter_model.GetOutputTensorType(signature_index, 0));
  const auto& dims = adapter_output_tensor_type.Layout().Dimensions();
  if (dims.size() < 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("The adapter output tensor has invalid dimensions: ",
                     dims.size()));
  }
  return dims[dims.size() - 2];
}

absl::StatusOr<int> GetEncoderSignatureTokenLength(const Model& encoder_model,
                                                   size_t signature_index) {
  LITERT_ASSIGN_OR_RETURN(auto encoder_signature,
                          encoder_model.GetSignature(signature_index));
  auto features_output_or = encoder_signature.OutputTensorType(kFeatures);
  std::optional<RankedTensorType> encoder_output_tensor_type;
  if (features_output_or.HasValue()) {
    encoder_output_tensor_type = features_output_or.Value();
  } else {
    LITERT_ASSIGN_OR_RETURN(auto fallback_output,
                            encoder_signature.OutputTensorType(0));
    encoder_output_tensor_type = fallback_output;
  }
  const auto& dims = encoder_output_tensor_type->Layout().Dimensions();
  if (dims.size() < 2) {
    return absl::InvalidArgumentError(
        absl::StrCat("The encoder output tensor has invalid dimensions: ",
                     dims.size()));
  }
  return dims[dims.size() - 2];
}

std::optional<absl::string_view> ExtractEncoderBudgetSuffix(
    absl::string_view key) {
  if (absl::StartsWith(key, kVisionEncoderSignaturePrefix) &&
      !absl::StartsWith(key, kVisionAdapterSignaturePrefix)) {
    return key.substr(kVisionEncoderSignaturePrefix.size());
  }
  return std::nullopt;
}

std::optional<absl::string_view> ExtractAdapterBudgetSuffix(
    absl::string_view key) {
  if (absl::StartsWith(key, kVisionAdapterSignaturePrefix)) {
    return key.substr(kVisionAdapterSignaturePrefix.size());
  }
  return std::nullopt;
}

struct SelectedVisionSignaturePair {
  size_t encoder_signature_index = 0;
  int num_tokens_per_image = 0;
};

// Selects the encoder signature (and matching adapter signature when present)
// representing the maximum visual token budget. Multi-signature `.tflite`
// exports do not guarantee subgraph/signature positional order across the
// encoder and adapter models, so signatures are paired by name
// (`vision_<budget>` <-> `vision_adapter_<budget>`, or identical keys) rather
// than assuming index 0 of both models describe the same budget.
absl::StatusOr<SelectedVisionSignaturePair> SelectMaxBudgetSignaturePair(
    const Model& vision_encoder_model, const Model* vision_adapter_model) {
  const size_t num_encoder_sigs = vision_encoder_model.GetNumSignatures();
  const size_t num_adapter_sigs =
      vision_adapter_model != nullptr ? vision_adapter_model->GetNumSignatures()
                                      : 0;

  if (num_encoder_sigs <= 1 && num_adapter_sigs <= 1) {
    int num_tokens = 0;
    if (vision_adapter_model != nullptr) {
      ABSL_ASSIGN_OR_RETURN(
          num_tokens, GetAdapterSignatureTokenLength(*vision_adapter_model, 0));
    } else {
      ABSL_ASSIGN_OR_RETURN(
          num_tokens, GetEncoderSignatureTokenLength(vision_encoder_model, 0));
    }
    return SelectedVisionSignaturePair{
        .encoder_signature_index = 0,
        .num_tokens_per_image = num_tokens,
    };
  }

  if (vision_adapter_model == nullptr) {
    bool has_prefixed_sig = false;
    for (size_t i = 0; i < num_encoder_sigs; ++i) {
      LITERT_ASSIGN_OR_RETURN(auto sig, vision_encoder_model.GetSignature(i));
      if (ExtractEncoderBudgetSuffix(sig.Key()).has_value()) {
        has_prefixed_sig = true;
        break;
      }
    }
    std::optional<SelectedVisionSignaturePair> best;
    for (size_t i = 0; i < num_encoder_sigs; ++i) {
      LITERT_ASSIGN_OR_RETURN(auto sig, vision_encoder_model.GetSignature(i));
      if (has_prefixed_sig &&
          !ExtractEncoderBudgetSuffix(sig.Key()).has_value()) {
        continue;
      }
      ABSL_ASSIGN_OR_RETURN(
          int tokens, GetEncoderSignatureTokenLength(vision_encoder_model, i));
      if (!best.has_value() || tokens > best->num_tokens_per_image) {
        best = SelectedVisionSignaturePair{
            .encoder_signature_index = i,
            .num_tokens_per_image = tokens,
        };
      }
    }
    if (!best.has_value()) {
      return absl::InvalidArgumentError(
          "No valid vision encoder signature found.");
    }
    return *best;
  }

  // Build lookup tables for encoder signatures by budget suffix (`vision_<N>`)
  // and by exact signature key.
  absl::flat_hash_map<absl::string_view, size_t> encoder_by_suffix;
  absl::flat_hash_map<absl::string_view, size_t> encoder_by_key;
  for (size_t i = 0; i < num_encoder_sigs; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto sig, vision_encoder_model.GetSignature(i));
    encoder_by_key.emplace(sig.Key(), i);
    if (auto suffix = ExtractEncoderBudgetSuffix(sig.Key());
        suffix.has_value()) {
      encoder_by_suffix.emplace(*suffix, i);
    }
  }

  std::optional<SelectedVisionSignaturePair> best_matched;
  for (size_t j = 0; j < num_adapter_sigs; ++j) {
    LITERT_ASSIGN_OR_RETURN(auto adapter_sig,
                            vision_adapter_model->GetSignature(j));
    ABSL_ASSIGN_OR_RETURN(
        int adapter_tokens,
        GetAdapterSignatureTokenLength(*vision_adapter_model, j));

    std::optional<size_t> matched_encoder_idx;
    if (auto suffix = ExtractAdapterBudgetSuffix(adapter_sig.Key());
        suffix.has_value()) {
      if (auto it = encoder_by_suffix.find(*suffix);
          it != encoder_by_suffix.end()) {
        matched_encoder_idx = it->second;
      }
    }
    if (!matched_encoder_idx.has_value()) {
      if (auto it = encoder_by_key.find(adapter_sig.Key());
          it != encoder_by_key.end()) {
        matched_encoder_idx = it->second;
      }
    }
    // Single-signature adapter paired with multi-signature encoder: match by
    // token count when signature names do not carry a shared suffix.
    if (!matched_encoder_idx.has_value() && num_adapter_sigs == 1) {
      for (size_t i = 0; i < num_encoder_sigs; ++i) {
        auto enc_tokens =
            GetEncoderSignatureTokenLength(vision_encoder_model, i);
        if (enc_tokens.ok() && *enc_tokens == adapter_tokens) {
          matched_encoder_idx = i;
          break;
        }
      }
    }

    if (matched_encoder_idx.has_value() &&
        (!best_matched.has_value() ||
         adapter_tokens > best_matched->num_tokens_per_image)) {
      best_matched = SelectedVisionSignaturePair{
          .encoder_signature_index = *matched_encoder_idx,
          .num_tokens_per_image = adapter_tokens,
      };
    }
  }

  if (!best_matched.has_value()) {
    return absl::InvalidArgumentError(
        "No matching signature names found between vision encoder and vision "
        "adapter.");
  }
  return *best_matched;
}

}  // namespace

absl::StatusOr<VisionExecutorProperties>
GetVisionExecutorPropertiesFromModelResources(ModelResources& model_resources) {
  VisionExecutorProperties properties;
  auto vision_adapter_model_or = model_resources.GetTFLiteModel(
      proto::LlmMetadata::TF_LITE_VISION_ADAPTER);
  const Model* vision_adapter_model = nullptr;
  if (vision_adapter_model_or.ok()) {
    vision_adapter_model = vision_adapter_model_or.value();
  } else if (vision_adapter_model_or.status().code() !=
             absl::StatusCode::kNotFound) {
    return vision_adapter_model_or.status();
  }

  ABSL_ASSIGN_OR_RETURN(auto vision_encoder_model,
                        model_resources.GetTFLiteModel(
                            proto::LlmMetadata::TF_LITE_VISION_ENCODER));

  ABSL_ASSIGN_OR_RETURN(
      const SelectedVisionSignaturePair selected,
      SelectMaxBudgetSignaturePair(*vision_encoder_model,
                                   vision_adapter_model));
  const size_t encoder_sig_idx = selected.encoder_signature_index;
  properties.num_tokens_per_image = selected.num_tokens_per_image;

  LITERT_ASSIGN_OR_RETURN(
      auto encoder_input_names,
      vision_encoder_model->GetSignatureInputNames(encoder_sig_idx));
  // Deduce patch_num_shrink_factor, i.e. the number of input patches consumed
  // per output vision token.
  //
  // - Transformer (ViT) encoders paired with a separate adapter expose the
  //   image patches as `images` (or input 0), so the factor is input_patches /
  //   num_tokens_per_image.
  // - Single input encoders (e.g. LFM2 VL) expose only the image patches
  //   tensor, so the same formula applies.
  // - Fused encoders without an adapter that emit a fixed number of soft
  //   tokens (e.g. MiniCPM-V) do not scale their output with the number of
  //   valid patches. For those, the positions_xy capacity is the quantity the
  //   data processor has to pad the patch count up to.
  if (!encoder_input_names.empty()) {
    if (properties.num_tokens_per_image <= 0) {
      return absl::FailedPreconditionError(
          "Vision encoder num_tokens_per_image must be positive.");
    }
    const auto positions_input =
        absl::c_find(encoder_input_names, kPositionsXy);
    if (vision_adapter_model == nullptr &&
        positions_input != encoder_input_names.end()) {
      // positions_xy is not necessarily input 1, so look it up by name. Fused
      // graphs may take extra inputs (e.g. vit_positions).
      LITERT_ASSIGN_OR_RETURN(
          auto positions_tensor_type,
          vision_encoder_model->GetInputTensorType(
              encoder_sig_idx,
              static_cast<int>(positions_input - encoder_input_names.begin())));
      const auto& dims = positions_tensor_type.Layout().Dimensions();
      if (dims.size() < 2) {
        return absl::FailedPreconditionError(
            "positions_xy tensor must have at least 2 dimensions.");
      }
      const int shrink_factor =
          dims[dims.size() - 2] / properties.num_tokens_per_image;
      if (shrink_factor <= 0) {
        return absl::FailedPreconditionError(
            "Fused vision encoder positions capacity is smaller than "
            "num_tokens_per_image.");
      }
      properties.patch_num_shrink_factor = shrink_factor;
    } else {
      // The image patches tensor has shape [batch_size, num_patches,
      // patch_dim], so the second-to-last dimension is the number of input
      // patches. Look up `images` by name when present so input ordering within
      // a signature does not affect the calculation, falling back to input 0.
      const auto images_input = absl::c_find(encoder_input_names, kImages);
      const int images_input_idx =
          images_input != encoder_input_names.end()
              ? static_cast<int>(images_input - encoder_input_names.begin())
              : 0;
      LITERT_ASSIGN_OR_RETURN(
          auto encoder_input_tensor_type,
          vision_encoder_model->GetInputTensorType(encoder_sig_idx,
                                                   images_input_idx));
      const auto& dims = encoder_input_tensor_type.Layout().Dimensions();
      if (dims.size() < 2) {
        return absl::FailedPreconditionError(
            "Vision encoder input tensor must have at least 2 dimensions.");
      }
      properties.patch_num_shrink_factor =
          dims[dims.size() - 2] / properties.num_tokens_per_image;
    }
  }
  return properties;
}

}  // namespace litert::lm
