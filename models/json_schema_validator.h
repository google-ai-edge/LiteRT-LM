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

#ifndef THIRD_PARTY_ODML_LITERT_LM_MODELS_JSON_SCHEMA_VALIDATOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_MODELS_JSON_SCHEMA_VALIDATOR_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json

namespace litert::lm {

// Validates `instance` against a JSON Schema document `schema`.
// Supports the JSON Schema keywords used by chat_template_input_schema.json:
// `type`, `properties`, `required`, `additionalProperties`, `items`, `enum`,
// `anyOf`, and `oneOf`.
// Returns absl::OkStatus() if valid, or absl::InvalidArgumentError describing
// the JSONPath and reason for the violation.
absl::Status ValidateJsonSchema(const nlohmann::ordered_json& instance,
                                const nlohmann::ordered_json& schema);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_MODELS_JSON_SCHEMA_VALIDATOR_H_
