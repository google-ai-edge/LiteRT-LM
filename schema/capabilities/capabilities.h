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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CAPABILITIES_CAPABILITIES_H_
#define THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CAPABILITIES_CAPABILITIES_H_

#include <istream>
#include <optional>
#include <ostream>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm::schema::capabilities {

// Input and output modalities supported by the model.
enum class Modality {
  kText = 0,
  kVision = 1,
  kAudio = 2,
  kVideo = 3,
};

// Represents the set of supported input/output modalities.
struct SupportedModalities {
  bool text = false;
  bool vision = false;
  bool audio = false;
  bool video = false;
};

// The strategy used by the sampler to pick the next token.
enum class SamplerType {
  kUnspecified = 0,
  kTopK = 1,
  kTopP = 2,
  kGreedy = 3,
};

// Parameters for the decoding strategy.
struct SamplerParameters {
  SamplerType type = SamplerType::kUnspecified;
  int k = 0;
  float p = 0.0;
  float temperature = 0.0;
};

// Extracted capabilities and configurations for Large Language Models (LLM).
struct LlmInferenceCapability {
  // Input modalities supported by the model (e.g. Text, Vision, Audio, Video).
  SupportedModalities input_modalities;

  // Output modalities supported by the model (typically just Text).
  SupportedModalities output_modalities;

  // Whether the model supports function calling / tool usage.
  bool supports_function_calling = false;

  // Whether the model supports thinking / internal reasoning steps.
  bool supports_thinking = false;

  // Whether the model supports speculative decoding (requires an auxiliary
  // drafter).
  bool supports_speculative_decoding = false;

  // Maximum vision token budget for multimodal inputs. Defaults to -1 if not
  // defined.
  int max_vision_token_budget = -1;

  // Default sampler parameters for the model.
  SamplerParameters default_sampler_params;
};

// Container for overall model metadata and capabilities.
// This wrapper is used to support future non-LLM model types (like Embeddings
// or Classifiers) by adding new optional capability fields without breaking the
// API.
struct ModelCapabilities {
  // LLM capabilities and configuration parameters, if available.
  std::optional<LlmInferenceCapability> llm_capability;
};

// Inspects the given LiteRT-LM file stream.
absl::StatusOr<ModelCapabilities> InspectModel(std::istream& litertlm_stream);

// Inspects the given LiteRT-LM file path.
absl::StatusOr<ModelCapabilities> InspectModel(
    absl::string_view litertlm_path);

std::ostream& operator<<(std::ostream& os,
                         const SupportedModalities& modalities);
std::ostream& operator<<(std::ostream& os,
                         const LlmInferenceCapability& llm_cap);
std::ostream& operator<<(std::ostream& os,
                         const ModelCapabilities& capabilities);

}  // namespace litert::lm::schema::capabilities

#endif  // THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CAPABILITIES_CAPABILITIES_H_
