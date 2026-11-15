// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation

/// Response format for constrained decoding.
///
/// Currently supports JSON Schema and Regex.
public struct ResponseFormat: Equatable {
  /// The type of constraint.
  public enum FormatType: Int {
    case regex = 1
    case jsonObject = 2
  }

  /// The type of constraint.
  public let type: FormatType

  /// The schema (for JSON_OBJECT) or pattern (for REGEX) string.
  public let schemaOrPattern: String

  private init(type: FormatType, schemaOrPattern: String) {
    self.type = type
    self.schemaOrPattern = schemaOrPattern
  }

  /// Creates a JSON Schema response format.
  ///
  /// - Parameter schema: The JSON schema as a JSON string.
  /// - Throws: `LiteRTLMError` if the schema is not a valid JSON string.
  public static func json(schema: String) throws -> ResponseFormat {
    guard let data = schema.data(using: .utf8),
      (try? JSONSerialization.jsonObject(with: data, options: [])) != nil
    else {
      throw LiteRTLMError.config(.invalidJsonSchema(schema))
    }
    return ResponseFormat(type: .jsonObject, schemaOrPattern: schema)
  }

  /// Creates a JSON Schema response format.
  ///
  /// - Parameter schema: The JSON schema as a Dictionary.
  /// - Throws: `LiteRTLMError` if the schema cannot be serialized to JSON.
  public static func json(schema: [String: Any]) throws -> ResponseFormat {
    guard JSONSerialization.isValidJSONObject(schema) else {
      throw LiteRTLMError.config(.invalidJsonSchema("Not a valid JSON object"))
    }
    let data = try JSONSerialization.data(withJSONObject: schema, options: [])
    guard let schemaString = String(data: data, encoding: .utf8) else {
      throw LiteRTLMError.config(.invalidJsonSchema("Failed to convert data to string"))
    }
    return ResponseFormat(type: .jsonObject, schemaOrPattern: schemaString)
  }

  /// Creates a Regex response format.
  ///
  /// - Parameter pattern: The regex pattern string.
  public static func regex(pattern: String) -> ResponseFormat {
    return ResponseFormat(type: .regex, schemaOrPattern: pattern)
  }
}
