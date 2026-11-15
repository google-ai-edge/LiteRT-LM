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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_TOOL_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_TOOL_UTILS_H_

#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json

namespace litert::lm {

// Extracts the tools array from a tools JSON object or array.
// If the input is an object, checks common wrapper keys ("oneOf", "one_of",
// "anyOf", "any_of", "tools") for an array. Otherwise, returns the input JSON.
const nlohmann::ordered_json& GetToolsArray(
    const nlohmann::ordered_json& tools);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_TOOL_UTILS_H_
