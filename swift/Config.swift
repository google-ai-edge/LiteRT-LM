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
public enum Backend: Equatable, Hashable, Sendable {
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
public struct EngineConfig: Hashable, Sendable {
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
  /// The rank of the text LoRA weights. If 0 or nil, LoRA is disabled.
  public let loraRank: Int?
  /// The rank of the audio LoRA weights. If 0 or nil, audio LoRA is disabled.
  public let audioLoraRank: Int?

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
  ///   - loraRank: The rank of the text LoRA weights.
  ///   - audioLoraRank: The rank of the audio LoRA weights.
  /// - Throws: `LiteRTLMError` if `maxNumTokens` is less than or equal to 0.
  public init(
    modelPath: String, backend: Backend = .cpu(), visionBackend: Backend? = nil,
    audioBackend: Backend? = nil,
    maxNumTokens: Int? = nil,
    cacheDir: String? = nil,
    loraRank: Int? = nil,
    audioLoraRank: Int? = nil
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
    self.loraRank = loraRank
    self.audioLoraRank = audioLoraRank
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

/// Configuration for thinking/reasoning generation.
public struct ThinkingConfig: Equatable {
  /// Whether thinking/reasoning generation is enabled.
  public let enableThinking: Bool
  /// The token budget for thinking/reasoning generation. Defaults to -1 (infinite budget).
  public let thinkingTokenBudget: Int

  /// - Parameters:
  ///   - enableThinking: Whether thinking/reasoning generation is enabled.
  ///   - thinkingTokenBudget: The token budget for thinking/reasoning generation. Defaults to -1
  ///     (infinite budget).
  public init(enableThinking: Bool = true, thinkingTokenBudget: Int = -1) {
    self.enableThinking = enableThinking
    self.thinkingTokenBudget = thinkingTokenBudget
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

  // The file path to the Text LoRA weights file.
  public let loraPath: String?

  // The file path to the Audio LoRA weights file.
  public let audioLoraPath: String?
  public let enableToolCallStreaming: Bool
  public let thinkingConfig: ThinkingConfig?
  public let automaticToolCalling: Bool
  public let enableResponseFormat: Bool
  public let visualTokenBudget: Int32?

  /// - Parameters:
  ///   - systemMessage: The system message to be used in the conversation.
  ///   - initialMessages: The initial messages to populate the conversation history.
  ///   - tools: The list of tool instances to be used in the conversation.
  ///   - samplerConfig: Configuration for the sampling process. If `nil`, then uses the engine's
  ///     default values.
  ///   - loraPath: The file path to the Text LoRA weights file.
  ///   - audioLoraPath: The file path to the Audio LoRA weights file.
  ///   - enableToolCallStreaming: Whether to enable conversation tool call streaming.
  ///   - thinkingConfig: Optional configuration for thinking/reasoning generation.
  ///   - automaticToolCalling: Whether to enable automatic tool calling. Default is true.
  ///   - enableResponseFormat: Whether to enable response format (constrained decoding). Default
  ///     is false.
  public init(
    systemMessage: Message? = nil,
    initialMessages: [Message] = [],
    tools: [Tool] = [],
    samplerConfig: SamplerConfig? = nil,
    loraPath: String? = nil,
    audioLoraPath: String? = nil,
    enableToolCallStreaming: Bool = false,
    thinkingConfig: ThinkingConfig? = nil,
    automaticToolCalling: Bool = true,
    enableResponseFormat: Bool = false,
    visualTokenBudget: Int32? = nil
  ) {
    self.systemMessage = systemMessage.flatMap { msg in
      if msg.toString.isEmpty {
        return nil
      }
      return msg.role == .system
        ? msg : Message(contents: msg.contents, role: .system, channels: msg.channels)
    }
    self.initialMessages = initialMessages
    self.tools = tools
    self.samplerConfig = samplerConfig
    self.loraPath = loraPath
    self.audioLoraPath = audioLoraPath
    self.enableToolCallStreaming = enableToolCallStreaming
    self.thinkingConfig = thinkingConfig
    self.automaticToolCalling = automaticToolCalling
    self.enableResponseFormat = enableResponseFormat
    self.visualTokenBudget = visualTokenBudget
  }
}

/// Configuration for penalizing repetitive tokens during generation.
public struct RepetitionPenaltyConfig {

  /// A multiplicative penalty applied to a token's logit if that token has appeared at least once
  /// inside the generated window history (e.g., 1.0 = no penalty, 1.2 = moderate penalty).
  ///
  /// Positive logits are divided by this parameter, and negative logits are multiplied
  /// (HuggingFace style). The parameter must be >= 1.0. Values less than 1.0 are automatically
  /// clamped to 1.0 during execution.
  public let repetitionPenalty: Float?

  /// A scalar subtracted from a logit if that token has appeared at least once inside the
  /// generated window history.
  ///
  /// Positive values discourage repetition, while negative values reward repeating tokens
  /// (OpenAI style). Defaults to 0.0.
  public let presencePenalty: Float?

  /// A scalar subtracted from a logit, scaled linearly by the number of times that token has
  /// previously appeared inside the generated window history.
  ///
  /// Positive values discourage repetition, while negative values reward repeating tokens
  /// (OpenAI style). Defaults to 0.0.
  public let frequencyPenalty: Float?

  /// The maximum number of recent tokens in generation history to consider when computing
  /// penalization.
  ///
  /// Tokens generated prior to this window are forgotten. A value of 0 means tracking all
  /// infinite generation history. Must be >= 0. Negative values are clamped to 0 during execution.
  public let windowSize: Int?

  /// - Parameters:
  ///   - repetitionPenalty: A multiplicative penalty applied to a token's logit if that token has
  ///     appeared at least once inside the generated window history.
  ///   - presencePenalty: A scalar subtracted from a logit if that token has appeared at least once
  ///     inside the generated window history.
  ///   - frequencyPenalty: A scalar subtracted from a logit, scaled linearly by the number of times
  ///     that token has previously appeared inside the generated window history.
  ///   - windowSize: The maximum number of recent tokens in generation history to consider.
  public init(
    repetitionPenalty: Float? = nil,
    presencePenalty: Float? = nil,
    frequencyPenalty: Float? = nil,
    windowSize: Int? = nil
  ) {
    self.repetitionPenalty = repetitionPenalty
    self.presencePenalty = presencePenalty
    self.frequencyPenalty = frequencyPenalty
    self.windowSize = windowSize
  }
}

/// Configuration for banning repetitive ngrams during generation.
public struct NoRepeatNgramConfig {

  /// The size of the ngram to ban (e.g. 5 means 5-grams). If set to an integer greater than 0,
  /// all ngrams of that size can only occur once. The logits of the banned tokens will be set to
  /// -inf. The value is clamped to [0, inf) during execution.
  public let noRepeatNgramSize: Int?

  /// The maximum number of recent tokens to consider for banning. Tokens older than this are
  /// forgotten. A value of 0 means track all infinite history. The value is clamped to [0, inf)
  /// during execution. If set less than the ngram size, the window size will be set to the ngram
  /// size to ensure that the ngram can be tracked.
  public let windowSize: Int?

  /// - Parameters:
  ///   - noRepeatNgramSize: The size of the ngram to ban.
  ///   - windowSize: The maximum number of recent tokens in generation history to consider.
  public init(
    noRepeatNgramSize: Int? = nil,
    windowSize: Int? = nil
  ) {
    self.noRepeatNgramSize = noRepeatNgramSize
    self.windowSize = windowSize
  }
}

/// Configuration for suppressing tokens during generation.
public struct SuppressTokensConfig {

  /// The list of token IDs to suppress.
  public let suppressTokens: [Int]

  /// - Parameters:
  ///   - suppressTokens: The list of token IDs to suppress. Can be any sequence of Ints (e.g.
  ///     Array or Set).
  public init<S: Sequence>(suppressTokens: S) where S.Element == Int {
    self.suppressTokens = Array(suppressTokens)
  }
}
