// Copyright 2025 The Google AI Edge Authors.
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

#include "runtime/components/tool_use/parser_common.h"

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/tool_use/rust/parsers.rs.h"

namespace litert::lm {

nlohmann::ordered_json ConvertJsonValue(const JsonValue& json_value) {
  if (json_value.is_null()) {
    return nlohmann::ordered_json(nullptr);
  } else if (json_value.is_bool()) {
    return nlohmann::ordered_json(json_value.get_bool());
  } else if (json_value.is_number()) {
    // Check for an integer first: is_number() is also true for integers, and
    // routing them through the f64 accessor would serialize `1000` as
    // `1000.0`, which strict consumers reject when the schema says "integer".
    if (json_value.is_integer()) {
      return nlohmann::ordered_json(json_value.get_integer());
    }
    return nlohmann::ordered_json(json_value.get_number());
  } else if (json_value.is_string()) {
    return nlohmann::ordered_json(json_value.get_string());
  } else if (json_value.is_array()) {
    nlohmann::ordered_json array = nlohmann::ordered_json::array();
    for (const JsonValue& json_value : json_value.array_get()) {
      array.push_back(ConvertJsonValue(json_value));
    }
    return array;
  } else if (json_value.is_object()) {
    nlohmann::ordered_json object = nlohmann::ordered_json::object();
    for (const auto& key : json_value.object_keys()) {
      absl::string_view key_str(key.data(), key.size());
      object[key_str] = ConvertJsonValue(*json_value.object_get(key));
    }
    return object;
  }
  return nlohmann::ordered_json();
}

}  // namespace litert::lm
