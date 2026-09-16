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

#include "runtime/executor/model_signature_utils.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

constexpr absl::string_view kEmbeddingsName = "embeddings";
constexpr absl::string_view kFeaturesName = "features";

template <typename DimensionsT>
std::vector<int> ToIntVector(const DimensionsT& dimensions) {
  std::vector<int> shape;
  shape.reserve(dimensions.size());
  for (auto dim : dimensions) {
    shape.push_back(static_cast<int>(dim));
  }
  return shape;
}

// Reduces `candidates` (ascending by length, non-empty) to at most
// `max_num_signatures` entries, preserving order.
//
// The last candidate is always kept: it is the one that bounds the longest
// input the executor will accept, so dropping it would silently shrink the
// model's usable input length rather than just costing padding. The remaining
// budget is spread over the other candidates by splitting them into equal
// sized buckets and taking the middle of each, which keeps the retained
// lengths roughly evenly spaced instead of clustering at one end. With
// 128/256/512/1024/2048 available, a budget of 2 keeps 512 and 2048, and a
// budget of 3 keeps 256, 1024, and 2048.
std::vector<const SignatureInfo*> ThinCandidates(
    const std::vector<const SignatureInfo*>& candidates,
    std::optional<int> max_num_signatures) {
  const int num_candidates = static_cast<int>(candidates.size());
  if (!max_num_signatures.has_value() ||
      *max_num_signatures >= num_candidates) {
    return candidates;
  }

  // Number of slots left for candidates other than the last one, and the
  // number of candidates competing for them.
  const int num_kept = *max_num_signatures - 1;
  const int num_remaining = num_candidates - 1;

  std::vector<const SignatureInfo*> result;
  result.reserve(*max_num_signatures);
  for (int i = 0; i < num_kept; ++i) {
    // Midpoint of the i-th bucket: floor((i + 0.5) * num_remaining / num_kept),
    // computed without floating point.
    const int index = ((2 * i + 1) * num_remaining) / (2 * num_kept);
    result.push_back(candidates[index]);
  }
  result.push_back(candidates.back());
  return result;
}

absl::StatusOr<std::vector<SignatureInfo>> GetTextEncoderSignatures(
    const litert::Model& model) {
  std::vector<SignatureInfo> signature_infos;
  const size_t num_signatures = model.GetNumSignatures();

  for (size_t i = 0; i < num_signatures; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto signature, model.GetSignature(i));
    SignatureInfo info;
    info.signature_name = std::string(signature.Key());

    if (signature.InputNames().empty()) {
      continue;
    }

    std::optional<::litert::RankedTensorType> embeddings_tensor_type;
    for (auto name : signature.InputNames()) {
      if (absl::StrContains(name, kEmbeddingsName)) {
        LITERT_ASSIGN_OR_RETURN(embeddings_tensor_type,
                                signature.InputTensorType(name));
        break;
      }
    }

    if (!embeddings_tensor_type.has_value()) {
      LITERT_ASSIGN_OR_RETURN(embeddings_tensor_type,
                              signature.InputTensorType(0));
    }

    info.input_shape =
        ToIntVector(embeddings_tensor_type->Layout().Dimensions());

    if (embeddings_tensor_type->Layout().Rank() == 3) {
      // [batch_size, seq_len, embed_dim]
      info.length = embeddings_tensor_type->Layout().Dimensions()[1];
    } else if (embeddings_tensor_type->Layout().Rank() == 2) {
      // [seq_len, embed_dim]
      info.length = embeddings_tensor_type->Layout().Dimensions()[0];
    } else if (embeddings_tensor_type->Layout().Rank() == 1) {
      // [seq_len]
      info.length = embeddings_tensor_type->Layout().Dimensions()[0];
    }

    if (!signature.OutputNames().empty()) {
      LITERT_ASSIGN_OR_RETURN(auto output_type, signature.OutputTensorType(0));
      info.output_shape = ToIntVector(output_type.Layout().Dimensions());
    }

    signature_infos.push_back(std::move(info));
  }

  if (signature_infos.empty()) {
    return absl::NotFoundError("No text encoder signatures found in model.");
  }
  return signature_infos;
}

absl::StatusOr<std::vector<SignatureInfo>> GetVisionModelSignatures(
    const litert::Model& model) {
  std::vector<SignatureInfo> signature_infos;
  const size_t num_signatures = model.GetNumSignatures();

  for (size_t i = 0; i < num_signatures; ++i) {
    LITERT_ASSIGN_OR_RETURN(auto signature, model.GetSignature(i));
    SignatureInfo info;
    info.signature_name = std::string(signature.Key());

    if (!signature.InputNames().empty()) {
      LITERT_ASSIGN_OR_RETURN(auto input_type, signature.InputTensorType(0));
      info.input_shape = ToIntVector(input_type.Layout().Dimensions());
    }

    std::optional<::litert::RankedTensorType> output_tensor_type;
    auto features_output = signature.OutputTensorType(kFeaturesName);
    if (features_output.HasValue()) {
      output_tensor_type = features_output.Value();
    } else if (!signature.OutputNames().empty()) {
      LITERT_ASSIGN_OR_RETURN(output_tensor_type,
                              signature.OutputTensorType(0));
    }

    if (output_tensor_type.has_value()) {
      info.output_shape =
          ToIntVector(output_tensor_type->Layout().Dimensions());
      const auto& dims = output_tensor_type->Layout().Dimensions();
      if (dims.size() >= 2) {
        info.length = dims[dims.size() - 2];
      }
    }

    signature_infos.push_back(std::move(info));
  }

  if (signature_infos.empty()) {
    return absl::NotFoundError("No vision signatures found in model.");
  }
  return signature_infos;
}

}  // namespace

absl::StatusOr<std::vector<SignatureInfo>> GetAvailableSignatures(
    const litert::Model& model, ModelType model_type) {
  switch (model_type) {
    case ModelType::kTfLiteTextEncoder:
      return GetTextEncoderSignatures(model);
    case ModelType::kTfLiteVisionEncoder:
    case ModelType::kTfLiteVisionAdapter:
      return GetVisionModelSignatures(model);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported model type for signature inspection: ",
                       ModelTypeToString(model_type)));
  }
}

absl::StatusOr<std::vector<SignatureInfo>> GetAvailableSignatures(
    ModelResources& resources, ModelType model_type) {
  LITERT_ASSIGN_OR_RETURN(const litert::Model* model,
                          resources.GetTFLiteModel(model_type));
  if (model == nullptr) {
    return absl::NotFoundError(absl::StrCat("Model not found in resources: ",
                                            ModelTypeToString(model_type)));
  }
  return GetAvailableSignatures(*model, model_type);
}

absl::StatusOr<SelectedTextSignaturesInfo> SelectSignaturesByCapacity(
    const std::vector<SignatureInfo>& signatures,
    std::optional<int> target_capacity, std::optional<int> min_capacity,
    std::optional<int> max_num_signatures) {
  if (target_capacity.has_value() && *target_capacity <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Target capacity must be positive, got: ", *target_capacity));
  }
  if (min_capacity.has_value()) {
    if (*min_capacity < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "min_capacity must be non-negative, got: ", *min_capacity));
    }
    if (target_capacity.has_value() && *min_capacity > *target_capacity) {
      return absl::InvalidArgumentError(absl::StrCat(
          "min_capacity (", *min_capacity,
          ") cannot be greater than target_capacity (", *target_capacity, ")"));
    }
  }
  if (max_num_signatures.has_value() && *max_num_signatures <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "max_num_signatures must be positive, got: ", *max_num_signatures));
  }
  // A cap on its own is a selection intent too: callers that only care about
  // how many signatures get compiled (the web bindings, which bound wasm
  // address space without knowing the model's length range) set it and leave
  // both capacity bounds unset. Only a call that constrains nothing at all is
  // meaningless.
  if (!target_capacity.has_value() && !min_capacity.has_value() &&
      !max_num_signatures.has_value()) {
    return absl::InvalidArgumentError(
        "At least one of target_capacity, min_capacity, or max_num_signatures "
        "must be set.");
  }
  if (signatures.empty()) {
    return absl::NotFoundError("No signatures found in model.");
  }

  // Sort signatures by length ascending.
  std::vector<SignatureInfo> sorted_signatures = signatures;
  std::sort(sorted_signatures.begin(), sorted_signatures.end(),
            [](const SignatureInfo& a, const SignatureInfo& b) {
              return a.length < b.length;
            });

  std::vector<const SignatureInfo*> candidates;
  for (const auto& sig : sorted_signatures) {
    if (min_capacity.has_value() && sig.length < *min_capacity) {
      continue;
    }
    candidates.push_back(&sig);
    if (target_capacity.has_value() && sig.length >= *target_capacity) {
      break;
    }
  }

  if (candidates.empty()) {
    return absl::NotFoundError("No signatures could be selected.");
  }

  SelectedTextSignaturesInfo result;
  for (const SignatureInfo* sig :
       ThinCandidates(candidates, max_num_signatures)) {
    result.signature_names.push_back(sig->signature_name);
    result.signature_lengths.push_back(sig->length);
  }
  result.max_signature_length = result.signature_lengths.back();

  if (target_capacity.has_value() &&
      *target_capacity > result.max_signature_length) {
    return absl::InvalidArgumentError(
        absl::StrCat("Requested target capacity (", *target_capacity,
                     ") exceeds maximum available signature length (",
                     result.max_signature_length, ")."));
  }

  return result;
}

absl::StatusOr<SelectedTextSignaturesInfo> SelectTextEncoderSignatures(
    const std::vector<SignatureInfo>& signatures,
    std::optional<int> max_input_length, std::optional<int> min_input_length,
    std::optional<int> max_num_signatures) {
  if (max_input_length.has_value() && *max_input_length <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "max_input_length must be positive, got: ", *max_input_length));
  }
  if (min_input_length.has_value() && *min_input_length < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "min_input_length must be non-negative, got: ", *min_input_length));
  }
  if (signatures.empty()) {
    return absl::NotFoundError("No text encoder signatures found in model.");
  }
  return SelectSignaturesByCapacity(signatures, max_input_length,
                                    min_input_length, max_num_signatures);
}

absl::StatusOr<SelectedTextSignaturesInfo> SelectTextEncoderSignatures(
    ModelResources& resources, std::optional<int> max_input_length,
    std::optional<int> min_input_length,
    std::optional<int> max_num_signatures) {
  LITERT_ASSIGN_OR_RETURN(
      auto signatures,
      GetAvailableSignatures(resources, ModelType::kTfLiteTextEncoder));
  return SelectTextEncoderSignatures(signatures, max_input_length,
                                     min_input_length, max_num_signatures);
}

absl::StatusOr<SelectedTextSignaturesInfo> SelectVisionEncoderSignatures(
    const std::vector<SignatureInfo>& signatures, int vision_tokens_per_image) {
  if (vision_tokens_per_image <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("vision_tokens_per_image must be positive, got: ",
                     vision_tokens_per_image));
  }
  if (signatures.empty()) {
    return absl::NotFoundError("No vision encoder signatures found in model.");
  }
  return SelectSignaturesByCapacity(signatures, vision_tokens_per_image);
}

absl::StatusOr<SelectedTextSignaturesInfo> SelectVisionEncoderSignatures(
    ModelResources& resources, int vision_tokens_per_image) {
  LITERT_ASSIGN_OR_RETURN(
      auto signatures,
      GetAvailableSignatures(resources, ModelType::kTfLiteVisionEncoder));
  return SelectVisionEncoderSignatures(signatures, vision_tokens_per_image);
}

absl::StatusOr<std::optional<SelectedTextSignaturesInfo>>
SelectVisionAdapterSignatures(const std::vector<SignatureInfo>& signatures,
                              int vision_tokens_per_image) {
  if (signatures.empty()) {
    return std::nullopt;
  }
  if (vision_tokens_per_image <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("vision_tokens_per_image must be positive, got: ",
                     vision_tokens_per_image));
  }
  return SelectSignaturesByCapacity(signatures, vision_tokens_per_image);
}

absl::StatusOr<std::optional<SelectedTextSignaturesInfo>>
SelectVisionAdapterSignatures(ModelResources& resources,
                              int vision_tokens_per_image) {
  auto adapter_signatures =
      GetAvailableSignatures(resources, ModelType::kTfLiteVisionAdapter);
  if (!adapter_signatures.ok() || adapter_signatures->empty()) {
    return std::nullopt;
  }
  return SelectVisionAdapterSignatures(*adapter_signatures,
                                       vision_tokens_per_image);
}

int GetSignatureInputLength(const SignatureInfo& signature) {
  if (signature.input_shape.empty()) {
    return 0;
  }
  return signature.input_shape.size() >= 2 ? signature.input_shape[1]
                                           : signature.input_shape[0];
}

}  // namespace litert::lm
