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

#include "models/json_schema_validator.h"

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json

namespace litert::lm {
namespace {

using json = nlohmann::ordered_json;

absl::string_view JsonTypeName(const json& value) {
  if (value.is_object()) return "object";
  if (value.is_array()) return "array";
  if (value.is_string()) return "string";
  if (value.is_boolean()) return "boolean";
  if (value.is_number_integer()) return "integer";
  if (value.is_number()) return "number";
  if (value.is_null()) return "null";
  return "unknown";
}

bool MatchesSingleType(const json& value, absl::string_view expected_type) {
  if (expected_type == "object") return value.is_object();
  if (expected_type == "array") return value.is_array();
  if (expected_type == "string") return value.is_string();
  if (expected_type == "boolean") return value.is_boolean();
  if (expected_type == "integer") return value.is_number_integer();
  if (expected_type == "number") return value.is_number();
  if (expected_type == "null") return value.is_null();
  return false;
}

struct Validator {
  static absl::Status ValidateType(const json& instance, const json& type_spec,
                                   absl::string_view path) {
    if (type_spec.is_string()) {
      const std::string& expected_type =
          type_spec.get_ref<const std::string&>();
      if (!MatchesSingleType(instance, expected_type)) {
        return absl::InvalidArgumentError(
            absl::StrCat(path, ": expected type '", expected_type, "', got '",
                         JsonTypeName(instance), "'"));
      }
      return absl::OkStatus();
    }
    if (type_spec.is_array()) {
      std::vector<absl::string_view> expected_types;
      expected_types.reserve(type_spec.size());
      for (const json& entry : type_spec) {
        if (!entry.is_string()) {
          return absl::InvalidArgumentError(
              absl::StrCat(path, ": schema 'type' array must contain strings"));
        }
        const std::string& expected_type = entry.get_ref<const std::string&>();
        if (MatchesSingleType(instance, expected_type)) {
          return absl::OkStatus();
        }
        expected_types.push_back(expected_type);
      }
      return absl::InvalidArgumentError(absl::StrCat(
          path, ": expected type in [", absl::StrJoin(expected_types, ", "),
          "], got '", JsonTypeName(instance), "'"));
    }
    return absl::InvalidArgumentError(
        absl::StrCat(path, ": schema 'type' must be a string or array"));
  }

  static absl::Status ValidateEnum(const json& instance,
                                   const json& enum_values,
                                   absl::string_view path) {
    if (!enum_values.is_array()) {
      return absl::InvalidArgumentError(
          absl::StrCat(path, ": schema 'enum' must be an array"));
    }
    for (const json& allowed : enum_values) {
      if (instance == allowed) {
        return absl::OkStatus();
      }
    }
    return absl::InvalidArgumentError(
        absl::StrCat(path, ": value ", instance.dump(), " is not in enum ",
                     enum_values.dump()));
  }

  static absl::Status ValidateAnyOf(const json& instance,
                                    const json& subschemas,
                                    absl::string_view path) {
    if (!subschemas.is_array()) {
      return absl::InvalidArgumentError(
          absl::StrCat(path, ": schema 'anyOf' must be an array"));
    }
    std::vector<std::string> errors;
    for (const json& subschema : subschemas) {
      absl::Status status = Validate(instance, subschema, path);
      if (status.ok()) {
        return absl::OkStatus();
      }
      errors.push_back(std::string(status.message()));
    }
    return absl::InvalidArgumentError(
        absl::StrCat(path, ": does not match any schema in anyOf (",
                     absl::StrJoin(errors, "; "), ")"));
  }

  static absl::Status ValidateOneOf(const json& instance,
                                    const json& subschemas,
                                    absl::string_view path) {
    if (!subschemas.is_array()) {
      return absl::InvalidArgumentError(
          absl::StrCat(path, ": schema 'oneOf' must be an array"));
    }
    int match_count = 0;
    std::vector<std::string> errors;
    for (const json& subschema : subschemas) {
      absl::Status status = Validate(instance, subschema, path);
      if (status.ok()) {
        ++match_count;
      } else {
        errors.push_back(std::string(status.message()));
      }
    }
    if (match_count == 0) {
      return absl::InvalidArgumentError(
          absl::StrCat(path, ": does not match any schema in oneOf (",
                       absl::StrJoin(errors, "; "), ")"));
    }
    if (match_count > 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          path, ": matched ", match_count, " schemas in oneOf (expected 1)"));
    }
    return absl::OkStatus();
  }

  static absl::Status ValidateObject(const json& instance, const json& schema,
                                     absl::string_view path) {
    if (const auto req_it = schema.find("required"); req_it != schema.end()) {
      if (!req_it->is_array()) {
        return absl::InvalidArgumentError(
            absl::StrCat(path, ": schema 'required' must be an array"));
      }
      for (const json& key : *req_it) {
        if (!key.is_string()) {
          return absl::InvalidArgumentError(absl::StrCat(
              path, ": schema 'required' entries must be strings"));
        }
        const std::string& key_name = key.get_ref<const std::string&>();
        if (!instance.contains(key_name)) {
          return absl::InvalidArgumentError(absl::StrCat(
              path, ": missing required property '", key_name, "'"));
        }
      }
    }

    const json* properties = nullptr;
    if (const auto props_it = schema.find("properties");
        props_it != schema.end()) {
      if (!props_it->is_object()) {
        return absl::InvalidArgumentError(
            absl::StrCat(path, ": schema 'properties' must be an object"));
      }
      properties = &(*props_it);
    }

    const json* additional_properties = nullptr;
    if (const auto add_it = schema.find("additionalProperties");
        add_it != schema.end()) {
      if (!add_it->is_boolean() && !add_it->is_object()) {
        return absl::InvalidArgumentError(absl::StrCat(
            path,
            ": schema 'additionalProperties' must be a boolean or object"));
      }
      additional_properties = &(*add_it);
    }

    for (const auto& [key, value] : instance.items()) {
      const std::string child_path = absl::StrCat(path, ".", key);
      if (properties != nullptr) {
        if (const auto prop_it = properties->find(key);
            prop_it != properties->end()) {
          if (absl::Status status = Validate(value, *prop_it, child_path);
              !status.ok()) {
            return status;
          }
          continue;
        }
      }
      if (additional_properties != nullptr) {
        if (additional_properties->is_boolean()) {
          if (!additional_properties->get<bool>()) {
            return absl::InvalidArgumentError(absl::StrCat(
                path, ": unexpected additional property '", key, "'"));
          }
        } else {
          if (absl::Status status =
                  Validate(value, *additional_properties, child_path);
              !status.ok()) {
            return status;
          }
        }
      }
    }
    return absl::OkStatus();
  }

  static absl::Status ValidateArray(const json& instance,
                                    const json& item_schema,
                                    absl::string_view path) {
    if (!item_schema.is_object()) {
      return absl::InvalidArgumentError(
          absl::StrCat(path, ": schema 'items' must be an object"));
    }
    for (std::size_t i = 0; i < instance.size(); ++i) {
      const std::string item_path = absl::StrCat(path, "[", i, "]");
      if (absl::Status status = Validate(instance[i], item_schema, item_path);
          !status.ok()) {
        return status;
      }
    }
    return absl::OkStatus();
  }

  static absl::Status Validate(const json& instance, const json& schema,
                               absl::string_view path) {
    if (!schema.is_object()) {
      return absl::InvalidArgumentError(
          absl::StrCat(path, ": schema must be a JSON object"));
    }
    if (const auto it = schema.find("type"); it != schema.end()) {
      if (absl::Status status = ValidateType(instance, *it, path);
          !status.ok()) {
        return status;
      }
    }
    if (const auto it = schema.find("enum"); it != schema.end()) {
      if (absl::Status status = ValidateEnum(instance, *it, path);
          !status.ok()) {
        return status;
      }
    }
    if (const auto it = schema.find("anyOf"); it != schema.end()) {
      if (absl::Status status = ValidateAnyOf(instance, *it, path);
          !status.ok()) {
        return status;
      }
    }
    if (const auto it = schema.find("oneOf"); it != schema.end()) {
      if (absl::Status status = ValidateOneOf(instance, *it, path);
          !status.ok()) {
        return status;
      }
    }
    if (instance.is_object()) {
      if (absl::Status status = ValidateObject(instance, schema, path);
          !status.ok()) {
        return status;
      }
    }
    if (instance.is_array()) {
      if (const auto it = schema.find("items"); it != schema.end()) {
        if (absl::Status status = ValidateArray(instance, *it, path);
            !status.ok()) {
          return status;
        }
      }
    }
    return absl::OkStatus();
  }
};

}  // namespace

absl::Status ValidateJsonSchema(const nlohmann::ordered_json& instance,
                                const nlohmann::ordered_json& schema) {
  return Validator::Validate(instance, schema, "$");
}

}  // namespace litert::lm
