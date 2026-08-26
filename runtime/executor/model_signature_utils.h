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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MODEL_SIGNATURE_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MODEL_SIGNATURE_UTILS_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/model_resources.h"

namespace litert::lm {

// Represents metadata about a model signature including its name, capacity,
// and tensor shapes.
struct SignatureInfo {
  std::string signature_name;
  // Sequence length (tokens) for text encoders, or visual token count for
  // vision models.
  int length = 0;
  std::vector<int> input_shape;
  std::vector<int> output_shape;
};

// Result of text encoder signature selection.
struct SelectedTextSignaturesInfo {
  // The names of the loaded signatures.
  std::vector<std::string> signature_names;
  // The sequence lengths of the loaded signatures.
  std::vector<int> signature_lengths;
  // The length of the longest loaded signature in the model.
  int max_signature_length = 0;
};

// Result of vision signature selection.
struct SelectedVisionSignatureInfo {
  // The names of the selected vision encoder signatures.
  std::vector<std::string> signature_names;
  // The sequence lengths / visual token counts of the selected vision encoder
  // signatures.
  std::vector<int> signature_lengths;
  // The length of the longest selected vision encoder signature.
  int max_signature_length = 0;
  // The names of the selected vision adapter signatures, if configured.
  std::vector<std::string> adapter_signature_names;
  // The sequence lengths / token counts of the selected vision adapter
  // signatures.
  std::vector<int> adapter_signature_lengths;
  // The maximum number of patches configured in the embedding metadata.
  int max_num_patches = 0;
};

// Inspects the given model and extracts all available signatures with their
// corresponding capacities and dimensions for the specified ModelType.
//
// Supported ModelTypes:
// - ModelType::kTfLiteTextEncoder
// - ModelType::kTfLiteVisionEncoder
// - ModelType::kTfLiteVisionAdapter
absl::StatusOr<std::vector<SignatureInfo>> GetAvailableSignatures(
    const litert::Model& model, ModelType model_type);

// Convenience overload that retrieves the model from ModelResources.
absl::StatusOr<std::vector<SignatureInfo>> GetAvailableSignatures(
    ModelResources& resources, ModelType model_type);

// Selects signatures to load based on the target capacity. Loads signatures for
// all lengths up to `target_capacity`, plus the smallest signature that can
// accommodate `target_capacity`. If `target_capacity` exceeds all available
// signatures, loads all of them. Returns metadata describing the loaded
// signatures.
absl::StatusOr<SelectedTextSignaturesInfo> SelectSignaturesByCapacity(
    const std::vector<SignatureInfo>& signatures, int target_capacity);

// Selects the text encoder signatures to load based on the expected maximum
// input length. Loads signatures for all lengths up to `max_input_length`,
// plus the smallest signature that can accommodate `max_input_length`. If
// `max_input_length` exceeds all available signatures, loads all of them.
absl::StatusOr<SelectedTextSignaturesInfo> SelectTextEncoderSignatures(
    const std::vector<SignatureInfo>& signatures, int max_input_length);

// Convenience overload that retrieves text encoder signatures from
// ModelResources.
absl::StatusOr<SelectedTextSignaturesInfo> SelectTextEncoderSignatures(
    ModelResources& resources, int max_input_length);

// Selects the vision encoder signatures to load based on
// `vision_tokens_per_image`. Loads signatures for all lengths up to
// `vision_tokens_per_image`, plus the smallest signature that can accommodate
// `vision_tokens_per_image`. If `vision_tokens_per_image` exceeds all
// available signatures, loads all of them.
absl::StatusOr<SelectedTextSignaturesInfo> SelectVisionEncoderSignatures(
    const std::vector<SignatureInfo>& signatures, int vision_tokens_per_image);

// Convenience overload that retrieves vision encoder signatures from
// ModelResources.
absl::StatusOr<SelectedTextSignaturesInfo> SelectVisionEncoderSignatures(
    ModelResources& resources, int vision_tokens_per_image);

// Selects the vision adapter signatures to load based on
// `vision_tokens_per_image`. Loads signatures for all lengths up to
// `vision_tokens_per_image`, plus the smallest signature that can accommodate
// `vision_tokens_per_image`. If `vision_tokens_per_image` exceeds all
// available signatures, loads all of them. Returns nullopt if no adapter
// signatures are present.
absl::StatusOr<std::optional<SelectedTextSignaturesInfo>>
SelectVisionAdapterSignatures(const std::vector<SignatureInfo>& signatures,
                              int vision_tokens_per_image);

// Convenience overload that retrieves vision adapter signatures from
// ModelResources.
absl::StatusOr<std::optional<SelectedTextSignaturesInfo>>
SelectVisionAdapterSignatures(ModelResources& resources,
                              int vision_tokens_per_image);

// Extracts the primary input dimension (number of patches or sequence length)
// from the signature's input shape.
int GetSignatureInputLength(const SignatureInfo& signature);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_MODEL_SIGNATURE_UTILS_H_
