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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_MODEL_METADATA_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_MODEL_METADATA_H_

#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/proto/asr_metadata.pb.h"

namespace litert::omni::asr {

// TODO: Remove AsrEngineConfig and PopulateConfigFromMetadataJson once all
// clients move to Omni API.
struct AsrEngineConfig {
  std::string model_url;
  std::string tokenizer_url;
  lm::proto::AsrMetadata metadata;
};

// Populates `AsrEngineConfig` for `model_name` from `OmniRegistry`.
absl::Status PopulateConfigFromMetadataJson(absl::string_view model_name,
                                            absl::string_view json_str,
                                            AsrEngineConfig& config);

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_MODEL_METADATA_H_
