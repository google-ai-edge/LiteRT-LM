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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_H_

// All the loaded model resources the executor needs to hold to avoid the model
// being destroyed.
#include <memory>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_model.h"  // from @litert
#include "runtime/components/tokenizer.h"
#include "runtime/proto/llm_metadata.pb.h"

namespace litert::lm {

enum class ModelType {
  kUnknown = 0,              // Placeholder for uninitialized model type.
  kTfLitePrefillDecode = 1,  // The base model is used for prefill and decode.
  kTfLiteEmbedder = 2,
  kTfLitePerLayerEmbedder = 3,
  kTfLiteAux = 4,
};

// Utility function to convert a string to ModelType. It's case insensitive.
inline absl::StatusOr<ModelType> StringToModelType(
    absl::string_view model_type_str) {
  const std::string lower_case_model_type_str =
      absl::AsciiStrToLower(model_type_str);
  if (lower_case_model_type_str == "tf_lite_prefill_decode") {
    return ModelType::kTfLitePrefillDecode;
  } else if (lower_case_model_type_str == "tf_lite_embedder") {
    return ModelType::kTfLiteEmbedder;
  } else if (lower_case_model_type_str == "tf_lite_per_layer_embedder") {
    return ModelType::kTfLitePerLayerEmbedder;
  } else if (lower_case_model_type_str == "tf_lite_aux") {
    return ModelType::kTfLiteAux;
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unknown model type: ", model_type_str));
  }
}

// Utility function to convert a ModelType to string.
inline std::string ModelTypeToString(ModelType model_type) {
  switch (model_type) {
    case ModelType::kTfLitePrefillDecode:
      return "TF_LITE_PREFILL_DECODE";
    case ModelType::kTfLiteEmbedder:
      return "TF_LITE_EMBEDDER";
    case ModelType::kTfLitePerLayerEmbedder:
      return "TF_LITE_PER_LAYER_EMBEDDER";
    case ModelType::kTfLiteAux:
      return "TF_LITE_AUX";
    case ModelType::kUnknown:
      return "UNKNOWN";
    default:
      return "INVALID";
  }
}

// ModelResources is an interface that manages all the loaded model resources
// that need to be hold to avoid the model being destroyed. So this object's
// lifetime need to be longer than the models it manages, like the tokenizer
// and the litert model. It provides a way to load the models in a lazy way.
// Basically, it will create the models when they are actually used. Before the
// Get*() functions are called, the models are not created yet. And once the
// models are created, they will be re-used for all the following calls.
class ModelResources {
 public:
  virtual ~ModelResources() = default;

  // Returns the litert model. We will create the model if it is not created
  // yet. And the model is created from memory mapped file, so physical memory
  // is only allocated when the model is actually used.
  virtual absl::StatusOr<std::shared_ptr<litert::Model>> GetTFLiteModel(
      ModelType model_type) = 0;
  // Returns the tokenizer.
  virtual absl::StatusOr<std::shared_ptr<Tokenizer>> GetTokenizer() = 0;

  // Returns the llm metadata.
  virtual absl::StatusOr<std::shared_ptr<proto::LlmMetadata>>
  GetLlmMetadata() = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_MODEL_RESOURCES_H_
