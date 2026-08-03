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

#include "runtime/components/tool_use/python_tool_format_utils.h"

#include <sstream>
#include <string>

#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/util/status_macros.h"

namespace litert::lm {
namespace {

absl::StatusOr<std::string> FormatObjectAsPythonDict(
    const nlohmann::ordered_json& object) {
  if (!object.is_object()) {
    return absl::InvalidArgumentError("Object must be a JSON object.");
  }
  std::stringstream ss;
  ss << "{";
  int count = 0;
  for (const auto& [key, value] : object.items()) {
    ss << "\"" << key << "\"" << ": ";
    ss << FormatValueAsPython(value);
    count += 1;
    if (count < object.size()) {
      ss << ", ";
    }
  }
  ss << "}";
  return ss.str();
}

absl::StatusOr<std::string> FormatObjectAsPythonInstance(
    absl::string_view name, const nlohmann::ordered_json& object) {
  if (!object.is_object()) {
    return absl::InvalidArgumentError("Object must be a JSON object.");
  }
  std::stringstream ss;
  ss << name << "(";
  int count = 0;
  for (const auto& [key, value] : object.items()) {
    ss << key << "=";
    ASSIGN_OR_RETURN(std::string formatted_value, FormatValueAsPython(value));
    ss << formatted_value;
    count += 1;
    if (count < object.size()) {
      ss << ", ";
    }
  }
  ss << ")";
  return ss.str();
}

absl::StatusOr<std::string> FormatArrayAsPython(
    const nlohmann::ordered_json& array) {
  if (!array.is_array()) {
    return absl::InvalidArgumentError("Array must be a JSON array.");
  }
  std::stringstream ss;
  ss << "[";
  int count = 0;
  for (const auto& element : array) {
    ss << FormatValueAsPython(element);
    count += 1;
    if (count < array.size()) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

std::string FormatParameterType(absl::string_view key,
                                const nlohmann::ordered_json& schema,
                                bool is_required) {
  std::stringstream ss;
  std::string type = schema.value("type", "");

  if (type == "boolean") {
    ss << "bool";
  } else if (type == "integer") {
    ss << "int";
  } else if (type == "number") {
    ss << "float";
  } else if (type == "string") {
    ss << "str";
  } else if (type == "array") {
    if (schema.contains("items") && schema["items"].is_object()) {
      ss << "list[" << FormatParameterType(key, schema["items"], true) << "]";
    } else {
      ss << "list[Any]";
    }
  } else if (type == "object") {
    ss << "dict";
  } else {
    ss << "Any";
  }

  if (!is_required) {
    ss << " | None = None";
  }

  return ss.str();
}

std::string GenerateDocstring(const nlohmann::ordered_json& tool) {
  std::stringstream ss;

  if (tool.contains("description")) {
    ss << tool["description"].get<std::string>() << "\n";
  }

  // Generate argument descriptions.
  if (tool.contains("parameters") &&
      tool["parameters"].contains("properties")) {
    ss << "\n  Args:\n";
    for (const auto& [key, value] : tool["parameters"]["properties"].items()) {
      ss << "    " << key;

      if (value.contains("description")) {
        ss << ": " << value["description"].get<std::string>() << "\n";
      }
    }
  }

  return ss.str();
}

}  // namespace

absl::StatusOr<std::string> FormatValueAsPython(
    const nlohmann::ordered_json& value) {
  std::stringstream ss;
  if (value.is_null()) {
    ss << "None";
  } else if (value.is_string()) {
    ss << "\"" << value.get<std::string>() << "\"";
  } else if (value.is_number()) {
    ss << value.dump();
  } else if (value.is_boolean()) {
    ss << (value.get<bool>() ? "True" : "False");
  } else if (value.is_object()) {
    if (value.contains("type") && value["type"].is_string()) {
      nlohmann::ordered_json kwargs = value;
      kwargs.erase("type");
      ss << FormatObjectAsPythonInstance(value["type"].get<std::string>(),
                                         kwargs);
    } else {
      ss << FormatObjectAsPythonDict(value);
    }
  } else if (value.is_array()) {
    ss << FormatArrayAsPython(value);
  } else {
    return absl::InvalidArgumentError("Value is not a supported type.");
  }
  return ss.str();
}

absl::StatusOr<std::string> FormatToolAsPython(
    const nlohmann::ordered_json& tool) {
  if (!tool.is_object()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Tool must be a JSON object but got: ", tool.type_name()));
  }

  const nlohmann::ordered_json& function =
      tool.contains("function") ? tool["function"] : tool;

  if (!function.contains("name")) {
    return absl::InvalidArgumentError("Tool name is required.");
  }

  std::stringstream ss;
  ss << "def " << function["name"].get<std::string>() << "(";

  if (function.contains("parameters") &&
      function["parameters"].contains("properties")) {
    ss << "\n";
    const nlohmann::ordered_json required_params = function["parameters"].value(
        "required", nlohmann::ordered_json::array());
    absl::flat_hash_set<std::string> required(required_params.begin(),
                                              required_params.end());
    int count = 0;
    for (const auto& [key, value] :
         function["parameters"]["properties"].items()) {
      const bool is_required = required.contains(key);
      ss << "    " << key << ": ";
      ss << FormatParameterType(key, value, is_required);
      ss << ",";
      if (++count < function["parameters"]["properties"].size()) {
        ss << "\n";
      }
    }
    ss << "\n";
  }

  ss << ") -> dict:\n";

  std::string docstring = GenerateDocstring(function);
  if (!docstring.empty()) {
    ss << "  \"\"\"";
    ss << docstring;
    ss << "  \"\"\"\n";
  }

  return ss.str();
}

}  // namespace litert::lm
