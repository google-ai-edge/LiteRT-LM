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

#ifndef THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CAPABILITIES_CAPABILITIES_C_H_
#define THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CAPABILITIES_CAPABILITIES_C_H_

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque struct representing a loaded LiteRT-LM file for capability checks.
typedef struct LiteRtLmLoadedFile LiteRtLmLoadedFile;

// Loads a LiteRT-LM file from the given path for capability queries.
// Returns NULL if the file cannot be opened.
LiteRtLmLoadedFile* litert_lm_loaded_file_create(const char* litertlm_path);

// Deletes a loaded LiteRT-LM file.
void litert_lm_loaded_file_delete(LiteRtLmLoadedFile* loaded_file);

// Returns true if the loaded LiteRT-LM file supports speculative decoding.
bool litert_lm_loaded_file_has_speculative_decoding_support(
    LiteRtLmLoadedFile* loaded_file);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_SCHEMA_CAPABILITIES_CAPABILITIES_C_H_
