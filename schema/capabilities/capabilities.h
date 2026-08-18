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

#include <cstdint>
#include <istream>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/proto/sampler_params.pb.h"
#include "schema/capabilities/speculative_decoding.h"  // IWYU pragma: export

namespace litert::lm::schema::capabilities {

enum class Modality {
  kText = 0,
  kVision = 1,
  kAudio = 2,
};

struct LlmInferenceCapability {
  std::vector<Modality> input_modalities;
  std::vector<Modality> output_modalities;
  bool supports_function_calling = false;
  bool supports_thinking = false;
  bool supports_speculative_decoding = false;
  int32_t max_context_length = 0;
  std::optional<proto::SamplerParameters> default_sampler_params;
  std::vector<std::string> supported_backends;
  std::vector<int32_t> supported_vision_resolutions;
};

struct ModelMetadataInfo {
  std::string model_class;
  std::string tf_hub_model_id;
  std::string min_litertlm_version;
  std::optional<LlmInferenceCapability> llm_capability;
};

// Inspects the given LiteRT-LM file stream.
absl::StatusOr<ModelMetadataInfo> InspectModel(std::istream& litertlm_stream);

// Inspects the given LiteRT-LM file path.
absl::StatusOr<ModelMetadataInfo> InspectModel(
    const std::string& litertlm_path);

}  // namespace litert::lm::schema::capabilities

#endif  // THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CAPABILITIES_CAPABILITIES_H_
