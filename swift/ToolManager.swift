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

/// Manages the discovery, schema generation, and execution of tools.
public class ToolManager {

  /// A map of tool names to their corresponding Swift Types.
  /// We store the `Type` so we can instantiate new copies via Codable.
  private let toolRegistry: [String: Tool.Type]

  /// The JSON schema string representing all registered tools.
  /// This is computed once during initialization.
  public let toolsJsonDescription: String

  /// Initializes the ToolManager with a list of tool instances.
  ///
  /// - Parameter tools: A list of instantiated tools (e.g. `[GetWeatherTool()]`).
  ///   These instances are used *only* for schema generation. Execution uses new instances.
  public init(tools: [Tool]) {
    var registry: [String: Tool.Type] = [:]
    var schemaList: [[String: Any]] = []
    let useSnakeCase = ExperimentalFlags.convertCamelToSnakeCaseInToolDescription

    for tool in tools {
      let name = type(of: tool).name
      let toolNameInModel = useSnakeCase ? name.camelToSnakeCase() : name
      registry[toolNameInModel] = type(of: tool)

      // Use the provided instance to generate the schema via Mirror
      schemaList.append(tool.getSchema())
    }

    self.toolRegistry = registry

    // Serialize the schema list to a JSON string
    if let data = try? JSONSerialization.data(withJSONObject: schemaList, options: []),
      let jsonString = String(data: data, encoding: .utf8)
    {
      self.toolsJsonDescription = jsonString
    } else {
      self.toolsJsonDescription = "[]"
    }
  }

  /// Executes a tool based on the name and JSON arguments provided by the model.
  ///
  /// - Parameters:
  ///   - name: The name of the tool to execute.
  ///   - arguments: The arguments as a dictionary.
  /// - Returns: The result of the tool execution. This will be a JSON-compatible type (String, Number, Bool, Array, Dictionary).
  public func execute(name: String, arguments: [String: Any]) async throws -> Any {
    guard let ToolType = toolRegistry[name] else {
      throw LiteRTLMError.tool(.notFound(name: name))
    }

    // 1. Create a template instance to access default values.
    // This allows us to handle optional parameters that the model might omit from its output.
    // Swift Codable requires all keys to be present.
    // By manually merging defaults, we ensure decoding succeeds.
    let defaultTool = ToolType.init()
    var defaults: [String: Any] = [:]

    // 2. Reflect to extract default values.
    let mirror = Mirror(reflecting: defaultTool)
    let useSnakeCase = ExperimentalFlags.convertCamelToSnakeCaseInToolDescription

    for child in mirror.children {
      guard let label = child.label else { continue }

      // Clean the property wrapper name (remove _ prefix and handle snake_case)
      let paramName = label.cleanPropertyLabel(useSnakeCase: useSnakeCase)

      if let toolParam = child.value as? ToolParamProtocol {
        if let defaultValue = toolParam.wrappedValueAny {
          defaults[paramName] = defaultValue
        } else if !toolParam.isRequired {
          // If it's not required (i.e. optional or has default),
          // but currently nil, we should still provide a null value
          // so the decoder doesn't fail on missing key.
          defaults[paramName] = NSNull()
        }
      }
    }

    // 3. Merge provided arguments over the defaults.
    // The `arguments` take precedence.
    let mergedArgs = defaults.merging(arguments) { (_, new) in new }

    let argsData = try JSONSerialization.data(withJSONObject: mergedArgs)

    let decoder = JSONDecoder()
    if useSnakeCase {
      decoder.keyDecodingStrategy = .convertFromSnakeCase
    }

    // Create a new tool instance from the JSON arguments.
    // This populates the @ToolParam properties automatically.
    let toolInstance = try decoder.decode(ToolType, from: argsData)

    // Run the tool
    let result = try await toolInstance.run()

    // Normalize the result to ensure it is strictly JSON-compatible (mimicking Android logic)
    return normalizeResult(result)
  }

  /// Helper to recursively convert any result into a JSON-Serialization friendly format.
  /// - Collections: Recursively normalized.
  /// - Primitives: Passed through.
  /// - Custom Objects: Converted to String.
  private func normalizeResult(_ value: Any) -> Any {
    // Handle Dictionaries
    if let dict = value as? [String: Any] {
      var normalizedDict: [String: Any] = [:]
      for (key, val) in dict {
        normalizedDict[key] = normalizeResult(val)
      }
      return normalizedDict
    }

    // Handle Arrays
    if let array = value as? [Any] {
      return array.map { normalizeResult($0) }
    }

    // Handle Primitives (String, Number, Bool)
    if value is String || value is Int || value is Double || value is Float || value is Bool {
      return value
    }

    // Handle Void
    if value is Void {
      return ""
    }

    // Fallback for Custom Objects
    return String(describing: value)
  }
}
