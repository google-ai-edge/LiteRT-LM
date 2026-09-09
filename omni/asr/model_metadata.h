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

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/asr/asr_engine.h"

namespace litert::omni::asr {

// Returns the compile-time embedded model_metadata.json content.
absl::string_view GetEmbeddedModelMetadataJson();

// Populates AsrEngineConfig for the given model_name from JSON metadata.
// If json_str is empty, uses GetEmbeddedModelMetadataJson().
absl::Status PopulateConfigFromMetadataJson(absl::string_view model_name,
                                            absl::string_view json_str,
                                            AsrEngineConfig& config);

// Returns an AsrEngineConfig populated from JSON metadata for model_name.
// If json_str is empty, uses GetEmbeddedModelMetadataJson().
absl::StatusOr<AsrEngineConfig> GetConfigFromMetadataJson(
    absl::string_view model_name, absl::string_view json_str = "");

}  // namespace litert::omni::asr

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ASR_MODEL_METADATA_H_
