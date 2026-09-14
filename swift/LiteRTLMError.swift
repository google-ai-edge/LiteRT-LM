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
import CLiteRTLM

/// Errors thrown by the LiteRT-LM Swift API.
public enum LiteRTLMError: Error, LocalizedError, Equatable {
  case engine(EngineError)
  case embeddingEngine(EmbeddingEngineError)
  case conversation(ConversationError)
  case config(ConfigError)
  case tool(ToolError)
  case message(MessageError)
  case modelInfo(ModelInfoError)

  public var errorDescription: String? {
    switch self {
    case .engine(let error): return error.errorDescription
    case .embeddingEngine(let error): return error.errorDescription
    case .conversation(let error): return error.errorDescription
    case .config(let error): return error.errorDescription
    case .tool(let error): return error.errorDescription
    case .message(let error): return error.errorDescription
    case .modelInfo(let error): return error.errorDescription
    }
  }

  // MARK: - Native Error Management APIs

  /// Returns the last error message recorded by the native LiteRT-LM C API on the calling thread,
  /// or `nil` if no error has occurred or the error state has been cleared.
  ///
  /// - Note: The native C API uses thread-local storage. When calling `async` Swift methods,
  ///   the operation may suspend and resume on a different thread. To inspect errors across
  ///   `async` operations, prefer inspecting the thrown `LiteRTLMError`, which captures the native
  ///   error message immediately on the invoking thread.
  public static func getLastErrorMessage() -> String? {
    guard let cString = litert_lm_get_last_error_message() else {
      return nil
    }
    return String(cString: cString)
  }

  /// Returns the last integer error code recorded by the native LiteRT-LM C API on the calling
  /// thread, or 0 (kOk) if no error has occurred or the error state has been cleared.
  ///
  /// - Note: The native C API uses thread-local storage. For `async` Swift methods, prefer
  ///   inspecting the thrown `LiteRTLMError`.
  public static func getLastErrorCode() -> Int {
    return Int(litert_lm_get_last_error_code())
  }

  /// Clears the last error recorded by the native LiteRT-LM C API on the calling thread,
  /// resetting the error message to `nil` and the error code to 0.
  public static func clearLastError() {
    litert_lm_clear_last_error()
  }

  /// Retrieves the last error message recorded on the calling thread and clears the thread-local
  /// error state. Returns `nil` if no error has occurred or if the recorded message is empty.
  internal static func consumeLastError() -> String? {
    defer { litert_lm_clear_last_error() }
    guard let cString = litert_lm_get_last_error_message() else {
      return nil
    }
    let message = String(cString: cString)
    return message.isEmpty ? nil : message
  }

  /// Specific errors related to the `Engine`.
  public enum EngineError: Error, LocalizedError, Equatable {
    case alreadyInitialized
    case failedToCreateSettings(String)
    case failedToCreateEngine(String)
    case notInitialized
    case failedToCreateSessionConfig(String)
    case failedToCreateConversationConfig(String)
    case failedToCreateConversation(String)
    case failedToSetLoraPath(String)
    case failedToSetAudioLoraPath(String)
    case failedToSetSupportedLoraRanks(String)
    case failedToSetSupportedAudioLoraRanks(String)
    case notOptedIntoExperimentalAPIs
    case failedToUpdateGPUEnableMetalResidencySet(String)

    public static let failedToCreateSettings = EngineError.failedToCreateSettings("")
    public static let failedToCreateEngine = EngineError.failedToCreateEngine("")
    public static let failedToCreateSessionConfig = EngineError.failedToCreateSessionConfig("")
    public static let failedToCreateConversationConfig =
      EngineError.failedToCreateConversationConfig("")
    public static let failedToCreateConversation = EngineError.failedToCreateConversation("")
    public static let failedToSetLoraPath = EngineError.failedToSetLoraPath("")
    public static let failedToSetAudioLoraPath = EngineError.failedToSetAudioLoraPath("")
    public static let failedToSetSupportedLoraRanks = EngineError.failedToSetSupportedLoraRanks("")
    public static let failedToSetSupportedAudioLoraRanks =
      EngineError.failedToSetSupportedAudioLoraRanks("")
    public static let failedToUpdateGPUEnableMetalResidencySet =
      EngineError.failedToUpdateGPUEnableMetalResidencySet("")

    public var errorDescription: String? {
      switch self {
      case .alreadyInitialized:
        return "Engine is already initialized."
      case .failedToCreateSettings(let details):
        return details.isEmpty
          ? "Failed to create engine settings."
          : "Failed to create engine settings: \(details)"
      case .failedToCreateEngine(let details):
        return details.isEmpty
          ? "Failed to create engine."
          : "Failed to create engine: \(details)"
      case .notInitialized:
        return "Engine is not initialized."
      case .failedToCreateSessionConfig(let details):
        return details.isEmpty
          ? "Failed to create session config."
          : "Failed to create session config: \(details)"
      case .failedToCreateConversationConfig(let details):
        return details.isEmpty
          ? "Failed to create conversation config."
          : "Failed to create conversation config: \(details)"
      case .failedToCreateConversation(let details):
        return details.isEmpty
          ? "Failed to create conversation."
          : "Failed to create conversation: \(details)"
      case .failedToSetLoraPath(let details):
        return details.isEmpty
          ? "Failed to set LoRA path."
          : "Failed to set LoRA path: \(details)"
      case .failedToSetAudioLoraPath(let details):
        return details.isEmpty
          ? "Failed to set Audio LoRA path."
          : "Failed to set Audio LoRA path: \(details)"
      case .failedToSetSupportedLoraRanks(let details):
        return details.isEmpty
          ? "Failed to set supported LoRA ranks."
          : "Failed to set supported LoRA ranks: \(details)"
      case .failedToSetSupportedAudioLoraRanks(let details):
        return details.isEmpty
          ? "Failed to set supported Audio LoRA ranks."
          : "Failed to set supported Audio LoRA ranks: \(details)"
      case .notOptedIntoExperimentalAPIs:
        return """
          Must opt into experimental APIs by calling `ExperimentalFlags.optIntoExperimentalAPIs()` \
          before calling this method.
          """
      case .failedToUpdateGPUEnableMetalResidencySet(let details):
        return details.isEmpty
          ? "Failed to update GPU enable Metal residency set."
          : "Failed to update GPU enable Metal residency set: \(details)"
      }
    }

    /// Equality comparison for `EngineError`.
    ///
    /// - Note: For backwards compatibility with legacy static error constants (which have empty
    ///   detail strings), two errors of the same case compare equal if their messages match or if
    ///   either message is empty.
    public static func == (lhs: EngineError, rhs: EngineError) -> Bool {
      switch (lhs, rhs) {
      case (.alreadyInitialized, .alreadyInitialized),
        (.notInitialized, .notInitialized),
        (.notOptedIntoExperimentalAPIs, .notOptedIntoExperimentalAPIs):
        return true
      case (.failedToCreateSettings(let a), .failedToCreateSettings(let b)),
        (.failedToCreateEngine(let a), .failedToCreateEngine(let b)),
        (.failedToCreateSessionConfig(let a), .failedToCreateSessionConfig(let b)),
        (.failedToCreateConversationConfig(let a), .failedToCreateConversationConfig(let b)),
        (.failedToCreateConversation(let a), .failedToCreateConversation(let b)),
        (.failedToSetLoraPath(let a), .failedToSetLoraPath(let b)),
        (.failedToSetAudioLoraPath(let a), .failedToSetAudioLoraPath(let b)),
        (.failedToSetSupportedLoraRanks(let a), .failedToSetSupportedLoraRanks(let b)),
        (.failedToSetSupportedAudioLoraRanks(let a), .failedToSetSupportedAudioLoraRanks(let b)),
        (
          .failedToUpdateGPUEnableMetalResidencySet(let a),
          .failedToUpdateGPUEnableMetalResidencySet(let b)
        ):
        return a == b || a.isEmpty || b.isEmpty
      default:
        return false
      }
    }
  }

  /// Specific errors related to the `EmbeddingEngine`.
  public enum EmbeddingEngineError: Error, LocalizedError, Equatable {
    case alreadyInitialized
    case failedToCreateSettings(String)
    case failedToCreateEngine(String)
    case notInitialized
    case failedToCreateInputData(String)
    case failedToComputeEmbedding(String)
    case failedToComputeEmbeddingBatch(String)

    public static let failedToCreateSettings = EmbeddingEngineError.failedToCreateSettings("")
    public static let failedToCreateEngine = EmbeddingEngineError.failedToCreateEngine("")
    public static let failedToCreateInputData = EmbeddingEngineError.failedToCreateInputData("")
    public static let failedToComputeEmbedding = EmbeddingEngineError.failedToComputeEmbedding("")
    public static let failedToComputeEmbeddingBatch =
      EmbeddingEngineError.failedToComputeEmbeddingBatch("")

    public var errorDescription: String? {
      switch self {
      case .alreadyInitialized:
        return "EmbeddingEngine is already initialized."
      case .failedToCreateSettings(let details):
        return details.isEmpty
          ? "Failed to create embedding engine settings."
          : "Failed to create embedding engine settings: \(details)"
      case .failedToCreateEngine(let details):
        return details.isEmpty
          ? "Failed to create embedding engine."
          : "Failed to create embedding engine: \(details)"
      case .notInitialized:
        return "EmbeddingEngine is not initialized."
      case .failedToCreateInputData(let details):
        return details.isEmpty
          ? "Failed to create input data for embedding."
          : "Failed to create input data for embedding: \(details)"
      case .failedToComputeEmbedding(let details):
        return details.isEmpty
          ? "Failed to compute embedding."
          : "Failed to compute embedding: \(details)"
      case .failedToComputeEmbeddingBatch(let details):
        return details.isEmpty
          ? "Failed to compute embedding batch."
          : "Failed to compute embedding batch: \(details)"
      }
    }

    /// Equality comparison for `EmbeddingEngineError`.
    ///
    /// - Note: For backwards compatibility with legacy static error constants (which have empty
    ///   detail strings), two errors of the same case compare equal if their messages match or if
    ///   either message is empty.
    public static func == (lhs: EmbeddingEngineError, rhs: EmbeddingEngineError) -> Bool {
      switch (lhs, rhs) {
      case (.alreadyInitialized, .alreadyInitialized),
        (.notInitialized, .notInitialized):
        return true
      case (.failedToCreateSettings(let a), .failedToCreateSettings(let b)),
        (.failedToCreateEngine(let a), .failedToCreateEngine(let b)),
        (.failedToCreateInputData(let a), .failedToCreateInputData(let b)),
        (.failedToComputeEmbedding(let a), .failedToComputeEmbedding(let b)),
        (.failedToComputeEmbeddingBatch(let a), .failedToComputeEmbeddingBatch(let b)):
        return a == b || a.isEmpty || b.isEmpty
      default:
        return false
      }
    }
  }

  /// Specific errors related to `Conversation`.
  public enum ConversationError: Error, LocalizedError, Equatable {
    case notAlive
    case failedToSerializeMessage
    case invalidResponse(String)
    case recurringToolCallLimitExceeded(limit: Int)
    case failedToStartStream(status: Int, message: String)
    case invalidJson(String)
    case toolExecutionError(name: String, error: String)
    case benchmarkNotEnabled
    case benchmarkInfoUnavailable
    case responseFormatNotEnabled

    public static func failedToStartStream(status: Int) -> ConversationError {
      .failedToStartStream(status: status, message: "")
    }

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
      case .failedToStartStream(let status, let details):
        return details.isEmpty
          ? "Failed to start stream. Status: \(status)"
          : "Failed to start stream (status \(status)): \(details)"
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
      case .responseFormatNotEnabled:
        return
          "responseFormat cannot be used unless enableResponseFormat=true was passed to ConversationConfig."
      }
    }

    /// Equality comparison for `ConversationError`.
    ///
    /// - Note: For backwards compatibility with legacy `failedToStartStream(status:)` without a
    ///   message, errors match if statuses are equal and messages match or either is empty.
    public static func == (lhs: ConversationError, rhs: ConversationError) -> Bool {
      switch (lhs, rhs) {
      case (.notAlive, .notAlive),
        (.failedToSerializeMessage, .failedToSerializeMessage),
        (.benchmarkNotEnabled, .benchmarkNotEnabled),
        (.benchmarkInfoUnavailable, .benchmarkInfoUnavailable),
        (.responseFormatNotEnabled, .responseFormatNotEnabled):
        return true
      case (.invalidResponse(let a), .invalidResponse(let b)),
        (.invalidJson(let a), .invalidJson(let b)):
        return a == b
      case (.recurringToolCallLimitExceeded(let a), .recurringToolCallLimitExceeded(let b)):
        return a == b
      case (.failedToStartStream(let s1, let m1), .failedToStartStream(let s2, let m2)):
        return s1 == s2 && (m1 == m2 || m1.isEmpty || m2.isEmpty)
      case (.toolExecutionError(let n1, let e1), .toolExecutionError(let n2, let e2)):
        return n1 == n2 && e1 == e2
      default:
        return false
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
    case invalidJsonSchema(String)

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
      case .invalidJsonSchema(let schema):
        return "Invalid JSON schema: \(schema)"
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

  /// Specific errors related to `ModelInfo`.
  public enum ModelInfoError: Error, LocalizedError, Equatable {
    case failedToLoadModel(String)

    public static let failedToLoadModel = ModelInfoError.failedToLoadModel("")
    public static let failedToCreateModelInfo = ModelInfoError.failedToLoadModel("")
    public static func failedToCreateModelInfo(_ details: String) -> ModelInfoError {
      .failedToLoadModel(details)
    }

    public var errorDescription: String? {
      switch self {
      case .failedToLoadModel(let details):
        return details.isEmpty
          ? "Failed to load model file."
          : "Failed to load model file: \(details)"
      }
    }

    /// Equality comparison for `ModelInfoError`.
    public static func == (lhs: ModelInfoError, rhs: ModelInfoError) -> Bool {
      switch (lhs, rhs) {
      case (.failedToLoadModel(let a), .failedToLoadModel(let b)):
        return a == b || a.isEmpty || b.isEmpty
      }
    }
  }
}
