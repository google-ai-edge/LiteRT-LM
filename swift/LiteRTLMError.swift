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

/// Errors thrown by the LiteRT-LM Swift API.
public enum LiteRTLMError: Error, LocalizedError, Equatable {
  case engine(EngineError)
  case conversation(ConversationError)
  case config(ConfigError)
  case tool(ToolError)
  case message(MessageError)

  public var errorDescription: String? {
    switch self {
    case .engine(let error): return error.errorDescription
    case .conversation(let error): return error.errorDescription
    case .config(let error): return error.errorDescription
    case .tool(let error): return error.errorDescription
    case .message(let error): return error.errorDescription
    }
  }

  /// Specific errors related to the `Engine`.
  public enum EngineError: Error, LocalizedError, Equatable {
    case alreadyInitialized
    case failedToCreateSettings
    case failedToCreateEngine
    case notInitialized
    case failedToCreateSessionConfig
    case failedToCreateConversationConfig
    case failedToCreateConversation

    public var errorDescription: String? {
      switch self {
      case .alreadyInitialized:
        return "Engine is already initialized."
      case .failedToCreateSettings:
        return "Failed to create engine settings."
      case .failedToCreateEngine:
        return "Failed to create engine."
      case .notInitialized:
        return "Engine is not initialized."
      case .failedToCreateSessionConfig:
        return "Failed to create session config."
      case .failedToCreateConversationConfig:
        return "Failed to create conversation config."
      case .failedToCreateConversation:
        return "Failed to create conversation."
      }
    }
  }

  /// Specific errors related to `Conversation`.
  public enum ConversationError: Error, LocalizedError, Equatable {
    case notAlive
    case failedToSerializeMessage
    case invalidResponse(String)
    case recurringToolCallLimitExceeded(limit: Int)
    case failedToStartStream(status: Int)
    case invalidJson(String)
    case toolExecutionError(name: String, error: String)
    case benchmarkNotEnabled
    case benchmarkInfoUnavailable

    public var errorDescription: String? {
      switch self {
      case .notAlive:
        return "Conversation is not alive."
      case .failedToSerializeMessage:
        return "Failed to serialize message to JSON string."
      case .invalidResponse(let details):
        return "Invalid response from native layer: \(details)"
      case .recurringToolCallLimitExceeded(let limit):
        return "Exceeded recurring tool call limit of \(limit)"
      case .failedToStartStream(let status):
        return "Failed to start stream. Status: \(status)"
      case .invalidJson(let details):
        return "Invalid JSON: \(details)"
      case .toolExecutionError(let name, let error):
        return "Error processing tool call \(name): \(error)"
      case .benchmarkNotEnabled:
        return """
          Benchmark flag is not enabled. Please enable the flag by setting setting \
          ExperimentalFlags.enableBenchmark to true before initializing the Engine.
          """
      case .benchmarkInfoUnavailable:
        return "Failed to get benchmark info."
      }
    }
  }

  public enum ConfigError: Error, LocalizedError, Equatable {
    case invalidMaxNumTokens
    case invalidMaxNumImages(count: Int)
    case invalidTopK
    case invalidTopP
    case invalidTemperature
    case multipleSystemMessages

    public var errorDescription: String? {
      switch self {
      case .invalidMaxNumTokens:
        return "maxNumTokens must be positive or nil (use the default from model or engine)."
      case .invalidMaxNumImages:
        return "maxNumImages must be non-negative or nil (use the default from model or engine)."
      case .invalidTopK:
        return "topK should be positive."
      case .invalidTopP:
        return "topP not between 0 and 1"
      case .invalidTemperature:
        return "temperature should be non-negative"
      case .multipleSystemMessages:
        return "Cannot set both systemMessage and have system messages in initialMessages."
      }
    }
  }

  /// Specific errors related to tools.
  public enum ToolError: Error, LocalizedError, Equatable {
    case notFound(name: String)

    public var errorDescription: String? {
      switch self {
      case .notFound(let name):
        return "Tool '\(name)' not found."
      }
    }
  }

  /// Specific errors related to messages.
  public enum MessageError: Error, LocalizedError, Equatable {
    case failedToConvertToJson
    case invalidContent

    public var errorDescription: String? {
      switch self {
      case .failedToConvertToJson:
        return "Failed to convert Message to JSON string."
      case .invalidContent:
        return "No content found in JSON string. Cannot create Message."
      }
    }
  }
}
