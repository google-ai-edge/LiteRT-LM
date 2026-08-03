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

#include "runtime/components/tool_use/fc_tool_format_utils.h"

#include <sstream>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/ascii.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

absl::StatusOr<std::string> FormatObjectAsFc(
    const nlohmann::ordered_json& object, absl::string_view escape_tag) {
  if (!object.is_object()) {
    return absl::InvalidArgumentError(
        std::string("Object must be a JSON object. Got: ") +
        object.type_name());
  }
  std::stringstream ss;
  ss << "{";
  int count = 0;
  for (const auto& [key, value] : object.items()) {
    ASSIGN_OR_RETURN(std::string value_str, FormatValueAsFc(value, escape_tag));
    ss << key << ":" << value_str;
    count += 1;
    if (count < object.size()) {
      ss << ",";
    }
  }
  ss << "}";
  return ss.str();
}

absl::StatusOr<std::string> FormatArrayAsFc(const nlohmann::ordered_json& array,
                                            absl::string_view escape_tag) {
  if (!array.is_array()) {
    return absl::InvalidArgumentError(
        std::string("Array must be a JSON array. Got ") + array.type_name());
  }
  std::stringstream ss;
  ss << "[";
  int count = 0;
  for (const auto& element : array) {
    ASSIGN_OR_RETURN(std::string element_str,
                     FormatValueAsFc(element, escape_tag));
    ss << element_str;
    count += 1;
    if (count < array.size()) {
      ss << ",";
    }
  }
  ss << "]";
  return ss.str();
}

nlohmann::ordered_json UppercaseTypes(const nlohmann::ordered_json& object) {
  nlohmann::ordered_json new_object = object;
  if (new_object.is_object()) {
    for (auto it = new_object.begin(); it != new_object.end(); ++it) {
      if (it.key() == "type" && it.value().is_string()) {
        std::string type_str = it.value().get<std::string>();
        if (type_str == "string" || type_str == "number" ||
            type_str == "integer" || type_str == "object" ||
            type_str == "array" || type_str == "boolean" ||
            type_str == "null") {
          it.value() = absl::AsciiStrToUpper(type_str);
        }
      }
      // Recursively apply to nested objects and arrays.
      if (it.value().is_object() || it.value().is_array()) {
        it.value() = UppercaseTypes(it.value());
      }
    }
  } else if (new_object.is_array()) {
    for (auto& element : new_object) {
      // Recursively apply to elements that are objects or arrays.
      if (element.is_object() || element.is_array()) {
        element = UppercaseTypes(element);
      }
    }
  }
  return new_object;
}

}  // namespace

absl::StatusOr<std::string> FormatValueAsFc(const nlohmann::ordered_json& value,
                                            absl::string_view escape_tag) {
  std::stringstream ss;
  if (value.is_null()) {
    ss << "null";
  } else if (value.is_string()) {
    ss << escape_tag << value.get<std::string>() << escape_tag;
  } else if (value.is_number()) {
    ss << value.dump();
  } else if (value.is_boolean()) {
    ss << value.dump();
  } else if (value.is_object()) {
    ASSIGN_OR_RETURN(std::string object_str,
                     FormatObjectAsFc(value, escape_tag));
    ss << object_str;
  } else if (value.is_array()) {
    ASSIGN_OR_RETURN(std::string array_str, FormatArrayAsFc(value, escape_tag));
    ss << array_str;
  } else {
    return absl::InvalidArgumentError(
        std::string(
            "Value is not a supported type. Supported types are null, string, "
            "number, boolean, object, array. Got: ") +
        value.type_name());
  }
  return ss.str();
}

absl::StatusOr<std::string> FormatToolAsFc(const nlohmann::ordered_json& tool,
                                           absl::string_view escape_tag) {
  if (!tool.is_object()) {
    return absl::InvalidArgumentError(
        std::string("Tool must be a JSON object. Got: ") + tool.type_name());
  }

  const nlohmann::ordered_json& function =
      tool.contains("function") ? tool["function"] : tool;

  if (!function.contains("name") || !function["name"].is_string()) {
    return absl::InvalidArgumentError(
        "Tool name is required and must be a string.");
  }

  nlohmann::ordered_json fields = UppercaseTypes(function);
  fields.erase("name");
  std::stringstream ss;
  ASSIGN_OR_RETURN(std::string fields_str,
                   FormatObjectAsFc(fields, escape_tag));
  ss << "declaration:" << function["name"].get<std::string>() << fields_str;
  return ss.str();
}

}  // namespace litert::lm
