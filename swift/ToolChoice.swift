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

/// Whether the model may, must, or must not call the conversation's tools for a message, and which
/// of them it may call.
///
/// It is only used if the conversation has tools and was created with
/// `ExperimentalFlags.enableConversationConstrainedDecoding` set to true. If no tool choice is
/// given, the model's default tool call constraint is used.
///
/// The choices are created with functions, e.g. `.none()`, so that they can't be confused with
/// `Optional.none` where an optional `ToolChoice` is expected.
public struct ToolChoice: Equatable, Sendable {
  private enum Choice: Equatable, Sendable {
    case auto
    case none
    case required
    case forcedFunction(name: String)
    case allowedTools(names: [String], required: Bool)
  }

  private let choice: Choice

  private init(_ choice: Choice) {
    self.choice = choice
  }

  /// The model decides whether to answer in text, call tools, or both ("auto").
  public static func auto() -> ToolChoice {
    return ToolChoice(.auto)
  }

  /// The model must answer in text only, with no tool calls ("none").
  public static func none() -> ToolChoice {
    return ToolChoice(.none)
  }

  /// The model must call at least one tool ("required").
  public static func required() -> ToolChoice {
    return ToolChoice(.required)
  }

  /// The model must call the function `name`, which must be one of the conversation's tools.
  ///
  /// - Parameter name: The name of the function to call.
  public static func forcedFunction(name: String) -> ToolChoice {
    return ToolChoice(.forcedFunction(name: name))
  }

  /// The model may only call the functions in `names`, which must be among the conversation's
  /// tools.
  ///
  /// - Parameters:
  ///   - names: The names of the functions that the model may call.
  ///   - required: If true, the model must call at least one of them. Otherwise, it may also answer
  ///     in text.
  public static func allowedTools(names: [String], required: Bool = false) -> ToolChoice {
    return ToolChoice(.allowedTools(names: names, required: required))
  }

  /// Returns the tool choice as the JSON that
  /// `litert_lm_conversation_optional_args_set_tool_choice` takes.
  ///
  /// - Throws: A `LiteRTLMError` if the tool choice can't be serialized.
  func jsonString() throws -> String {
    let json: Any
    switch choice {
    case .auto:
      json = "auto"
    case .none:
      json = "none"
    case .required:
      json = "required"
    case .forcedFunction(let name):
      json = ToolChoice.functionReference(name)
    case .allowedTools(let names, let required):
      let allowedTools: [String: Any] = [
        "mode": required ? "required" : "auto",
        "tools": names.map(ToolChoice.functionReference),
      ]
      json = ["type": "allowed_tools", "allowed_tools": allowedTools] as [String: Any]
    }
    let data = try JSONSerialization.data(withJSONObject: json, options: [.fragmentsAllowed])
    guard let jsonString = String(data: data, encoding: .utf8) else {
      throw LiteRTLMError.conversation(.invalidJson("Failed to serialize the tool choice."))
    }
    return jsonString
  }

  private static func functionReference(_ name: String) -> [String: Any] {
    return ["type": "function", "function": ["name": name]]
  }
}
