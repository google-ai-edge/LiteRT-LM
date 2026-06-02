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

/// The backend to use for the LiteRT-LM engine.
///
/// Swift version of the C++'s `litert::lm::Backend`.
public enum Backend: Equatable {
  /// CPU LiteRT backend.
  case cpu(threadCount: Int? = nil)
  /// GPU LiteRT backend.
  case gpu

  public var rawValue: String {
    switch self {
    case .cpu: return "cpu"
    case .gpu: return "gpu"
    }
  }

  public init?(rawValue: String) {
    switch rawValue {
    case "cpu": self = .cpu()
    case "gpu": self = .gpu
    default: return nil
    }
  }
}

/// Configuration for the LiteRT-LM engine.
public struct EngineConfig {
  /// The file path to the LiteRT-LM model.
  public let modelPath: String
  /// The backend to use for the engine.
  public let backend: Backend
  /// The backend to use for the vision executor. If `nil`, vision executor will
  /// not be initialized.
  public let visionBackend: Backend?
  /// The backend to use for the audio executor. If `nil`, audio executor will
  /// not be initialized.
  public let audioBackend: Backend?
  /// The maximum number of the sum of input and output tokens. It is equivalent
  /// to the size of the kv-cache. When `nil`, use the default value from the model or the engine.
  public let maxNumTokens: Int?
  /// The directory for placing cache files. It should be a directory where the
  /// application has write access. If `nil`, it uses the directory of the `modelPath`.
  public let cacheDir: String?

  /// - Parameters:
  ///   - modelPath: The file path to the LiteRT-LM model.
  ///   - backend: The backend to use for the engine.
  ///   - visionBackend: The backend to use for the vision executor. If `nil`, vision executor
  ///     will not be initialized.
  ///   - audioBackend: The backend to use for the audio executor. If `nil`, audio executor will
  ///     not be initialized.
  ///   - maxNumTokens: The maximum number of the sum of input and output tokens. It is
  ///     equivalent to the size of the kv-cache. When `nil`, use the default value from the
  ///     model or the engine.
  ///   - cacheDir: The directory for placing cache files. It should be a directory where the
  ///     application has write access. If `nil`, it uses the directory of the `modelPath`.
  /// - Throws: `LiteRTLMError` if `maxNumTokens` is less than or equal to 0.
  public init(
    modelPath: String, backend: Backend = .cpu(), visionBackend: Backend? = nil,
    audioBackend: Backend? = nil,
    maxNumTokens: Int? = nil,
    cacheDir: String? = nil
  ) throws {
    if let maxNumTokens, maxNumTokens <= 0 {
      throw LiteRTLMError.config(.invalidMaxNumTokens)
    }
    self.modelPath = modelPath
    self.backend = backend
    self.visionBackend = visionBackend
    self.audioBackend = audioBackend
    self.maxNumTokens = maxNumTokens
    self.cacheDir = cacheDir
  }
}

/// Configuration for the sampling process.
public struct SamplerConfig {

  /// The number of most likely tokens (top logits) to consider at each step of sampling.
  public let topK: Int

  /// The cumulative probability threshold for nucleus sampling.
  public let topP: Float

  /// The temperature to use for sampling.
  public let temperature: Float

  /// The seed to use for randomization. Default to 0 (same default as engine code).
  public let seed: Int

  /// - Parameters:
  ///   - topK: The number of most likely tokens (top logits) to consider at each step of sampling.
  ///   - topP: The cumulative probability threshold for nucleus sampling.
  ///   - temperature: The temperature to use for sampling.
  ///   - seed: The seed to use for randomization. Default to 0 (same default as engine code).
  /// - Throws: `LiteRTLMError` if `topK` is less than or equal to 0, `topP` is not in [0, 1], or
  ///   `temperature` is less than 0.
  public init(
    topK: Int,
    topP: Float,
    temperature: Float,
    seed: Int = 0
  ) throws {
    if topK <= 0 {
      throw LiteRTLMError.config(.invalidTopK)
    }
    if topP < 0 || topP > 1 {
      throw LiteRTLMError.config(.invalidTopP)
    }
    if temperature < 0 {
      throw LiteRTLMError.config(.invalidTemperature)
    }

    self.topK = topK
    self.topP = topP
    self.temperature = temperature
    self.seed = seed
  }
}

/// Configuration fo the LiteRT-LM `Conversation`.
public struct ConversationConfig {
  // The system message to be used in the conversation.
  public let systemMessage: Message?

  // The initial messages to populate the conversation history.
  public let initialMessages: [Message]

  // The list of tool instances to be used in the conversation.
  public let tools: [Tool]

  // Configuration for the sampling process.
  // If `nil`, then uses the engine's default values.
  public let samplerConfig: SamplerConfig?

  /// - Parameters:
  ///   - systemMessage: The system message to be used in the conversation.
  ///   - initialMessages: The initial messages to populate the conversation history.
  ///   - tools: The list of tool instances to be used in the conversation.
  ///   - samplerConfig: Configuration for the sampling process. If `nil`, then uses the engine's
  ///     default values.
  public init(
    systemMessage: Message? = nil,
    initialMessages: [Message] = [],
    tools: [Tool] = [],
    samplerConfig: SamplerConfig? = nil
  ) {
    self.systemMessage = systemMessage.map { msg in
      msg.role == .system
        ? msg : Message(contents: msg.contents, role: .system, channels: msg.channels)
    }
    self.initialMessages = initialMessages
    self.tools = tools
    self.samplerConfig = samplerConfig
  }
}
