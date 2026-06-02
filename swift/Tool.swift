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

/// A protocol that defines what types can be used as parameters in a LiteRT-LM tool.
public protocol ToolParameterValue: Codable {
  /// Returns the JSON schema representation for this type.
  static func getJsonSchema() -> [String: Any]
}

extension String: ToolParameterValue {
  public static func getJsonSchema() -> [String: Any] { ["type": "string"] }
}
extension Int: ToolParameterValue {
  public static func getJsonSchema() -> [String: Any] { ["type": "integer"] }
}
extension Bool: ToolParameterValue {
  public static func getJsonSchema() -> [String: Any] { ["type": "boolean"] }
}
extension Double: ToolParameterValue {
  public static func getJsonSchema() -> [String: Any] { ["type": "number"] }
}
extension Float: ToolParameterValue {
  public static func getJsonSchema() -> [String: Any] { ["type": "number"] }
}
extension Optional: ToolParameterValue where Wrapped: ToolParameterValue {
  public static func getJsonSchema() -> [String: Any] {
    var schema = Wrapped.getJsonSchema()
    schema["nullable"] = true
    return schema
  }
}
extension Array: ToolParameterValue where Element: ToolParameterValue {
  public static func getJsonSchema() -> [String: Any] {
    ["type": "array", "items": Element.getJsonSchema()]
  }
}

/// A non-generic protocol allowing `ToolManager` to inspect `ToolParam` properties at runtime.
protocol ToolParamProtocol {
  var description: String { get }
  var isRequired: Bool { get }

  /// Type-erased access to the underlying storage (default value), used for reflection.
  /// This allows `ToolManager` to read default values without knowing the specific generic `Value` type.
  var wrappedValueAny: Any? { get }

  func getFullSchema() -> [String: Any]
}

/// A property wrapper that defines a parameter for a LiteRT-LM tool.
///
/// Use this property wrapper to mark properties of a `Tool` as parameters that can be passed to
/// the tool. The `description` will be used to generate the tool's schema.
///
/// Example:
/// ```
/// // Example of required parameters:
/// @ToolParam(description: "The title of the event.")
/// var title: String
/// @ToolParam(description: "The date and time of the event in format YYYY-MM-DDTHH:MM:SS.")
/// var dateTime: String
///
/// // Example of a parameter with a default value:
/// @ToolParam(description: "The duration of the event in minutes. Default is 30.")
/// var durationMinutes: Int = 30
///
/// // Example of an optional parameter:
/// @ToolParam(description: "The topic of the event.")
/// var topic: String?
/// ```
@propertyWrapper
public struct ToolParam<Value: ToolParameterValue>: Decodable, ToolParamProtocol {

  /// Holds the actual value of the parameter.
  private var storage: Value?

  /// A description of the parameter. This is used to to describe the parameter in the schema.
  public let description: String

  /// Whether the parameter has a default value provided by the developer when declaring the tool.
  private let hasDefaultValue: Bool

  /// Initializes a parameter with an optional default value.
  public init(wrappedValue: Value? = nil, description: String) {
    self.storage = wrappedValue
    self.description = description
    self.hasDefaultValue = wrappedValue != nil
  }

  /// The value of the parameter.
  ///
  /// Returns the stored value if available (from a default or decoded input).
  /// Returns `nil` if no value is present and the type `Value` is Optional.
  public var wrappedValue: Value {
    get {
      // If we have data, return it.
      if let value = storage {
        return value
      }

      // If storage is empty, check if it's an Optional.
      if let nilValue = Optional<Any>.none as? Value {
        return nilValue
      }

      // It's not optional and we have no data. Throw an error.
      fatalError("ToolParam of type \(Value.self) was not set and has no default value.")
    }
    set { storage = newValue }
  }

  /// Provides type-erased access to the underlying storage of the parameter.
  ///
  /// Unlike `wrappedValue`, this property returns `nil` if no value is set, instead of crashing.
  /// This is essential for `ToolManager` to safely inspect default values via reflection (Mirror)
  /// before the decoding process occurs.
  var wrappedValueAny: Any? {
    return storage
  }

  func getFullSchema() -> [String: Any] {
    var schema = Value.getJsonSchema()
    if !description.isEmpty {
      schema["description"] = description
    }
    return schema
  }

  var isRequired: Bool {
    // If `nil` can be cast to `Value`, then `Value` must be an Optional type.
    let isOptional = Optional<Any>.none as? Value != nil
    return !isOptional && !hasDefaultValue
  }

  /// Decodes the parameter value from the decoder.
  ///
  /// This initializer is required to conform to the `Decodable` protocol. Since the `Tool` protocol
  /// inherits from `Decodable`, any property wrapper used within a `Tool` must also be `Decodable`.
  ///
  /// When `ToolManager` executes a tool, it uses `JSONDecoder` to decode the JSON arguments
  /// provided by the model into a `Tool` instance. This initializer is called automatically during
  /// that process to populate the `ToolParam`'s internal storage with the value provided by the
  /// model.
  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()

    if container.decodeNil() {
      if let nilValue = Optional<Any>.none as? Value {
        self.storage = nilValue
      } else {
        throw DecodingError.valueNotFound(
          Value.self,
          .init(
            codingPath: decoder.codingPath,
            debugDescription: "Received null for non-optional parameter"))
      }
    } else {
      self.storage = try container.decode(Value.self)
    }

    self.description = ""
    self.hasDefaultValue = true
  }
}

/// Example of how to define tools:
/// - Adopt the `Tool` protocol to define a struct/class as a tool.
/// - Use `@ToolParam` to define the parameters of the tool.
/// - The allowed parameter types are: String, Int, Bool, Float, Double, and Array of them.
/// - The return type of `run()` is `Any` and will be converted to JSON/String back to the model.
/// - Use the Swift Optional type (e.g., `String?`) to indicate that a parameter is optional.
///
/// ```swift
/// struct GetCurrentWeatherTool: Tool {
///   static let name = "get_current_weather"
///   static let description = "Get the current weather"
///
///   @ToolParam(description: "The city and state, e.g. San Francisco, CA")
///   var location: String
///
///   @ToolParam(description: "The temperature unit to use")
///   var unit: String = "celsius"
///
///   func run() async throws -> Any {
///     // In a real application, you would call a weather API here. This is just an example.
///     let temperature = getTemperature(for: location, in: unit)
///
///     return [
///       "temperature": temperature,
///       "unit": unit,
///     ]
///   }
/// }
/// ```
///  A protocol a struct/class must conform to be used as a tool by the LiteRT-LM model.
///
///  - SeeAlso: `ToolParam` for defining parameters of the tool.
public protocol Tool: Decodable {
  /// The unique name of the tool.
  static var name: String { get }
  /// A description of what the tool does.
  static var description: String { get }

  /// A required zero-argument initializer.
  init()
  /// The logic to execute when the tool is called.
  func run() async throws -> Any
}

extension Tool {
  /// Generates an OpenAPI-compliant JSON schema for the tool.
  public func getSchema() -> [String: Any] {
    var properties: [String: Any] = [:]
    var requiredFields: [String] = []

    let mirror = Mirror(reflecting: self)
    let useSnakeCase = ExperimentalFlags.convertCamelToSnakeCaseInToolDescription

    for child in mirror.children {
      guard let label = child.label else { continue }

      // Clean the name (remove the _ prefix and handle snake_case)
      let cleanName = label.cleanPropertyLabel(useSnakeCase: useSnakeCase)

      // Check if this property is a ToolParam
      if let param = child.value as? ToolParamProtocol {
        properties[cleanName] = param.getFullSchema()
        if param.isRequired {
          requiredFields.append(cleanName)
        }
      }
    }

    let toolName = useSnakeCase ? Self.name.camelToSnakeCase() : Self.name

    var parameters: [String: Any] = [
      "type": "object",
      "properties": properties,
    ]

    if !requiredFields.isEmpty {
      parameters["required"] = requiredFields
    }

    var schema: [String: Any] = [
      "type": "function",
      "function": [
        "name": toolName,
        "description": Self.description,
      ],
    ]

    if !properties.isEmpty {
      var functionBody = schema["function"] as! [String: Any]
      functionBody["parameters"] = parameters
      schema["function"] = functionBody
    }

    return schema
  }
}

extension String {
  /// Converts a camelCase string to snake_case.
  func camelToSnakeCase() -> String {
    let regex = try! NSRegularExpression(pattern: "([a-z0-9])([A-Z])", options: [])
    let range = NSRange(location: 0, length: self.utf16.count)
    return regex.stringByReplacingMatches(
      in: self, options: [], range: range, withTemplate: "$1_$2"
    ).lowercased()
  }

  /// Cleans a property label (removes '_' prefix) and optionally converts to snake_case.
  func cleanPropertyLabel(useSnakeCase: Bool) -> String {
    let cleaned = self.hasPrefix("_") ? String(self.dropFirst()) : self
    return useSnakeCase ? cleaned.camelToSnakeCase() : cleaned
  }
}
