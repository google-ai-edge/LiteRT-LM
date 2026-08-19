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
#include "absl/types/span.h"  // from @com_google_absl
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

}  // namespace litert::lm
