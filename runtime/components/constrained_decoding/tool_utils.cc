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

#include "runtime/components/constrained_decoding/tool_utils.h"

#include "nlohmann/json.hpp"  // from @nlohmann_json

namespace litert::lm {

const nlohmann::ordered_json& GetToolsArray(
    const nlohmann::ordered_json& tools) {
  if (tools.is_object()) {
    for (const char* key : {"oneOf", "one_of", "anyOf", "any_of", "tools"}) {
      auto it = tools.find(key);
      if (it != tools.end() && it->is_array()) {
        return *it;
      }
    }
  }
  return tools;
}

}  // namespace litert::lm
