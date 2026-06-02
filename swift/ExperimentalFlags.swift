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
import OSLog

/// A struct to manage flags for APIs that are not yet considered mature.
///
/// These flags guard experimental APIs that may still undergo significant changes.
/// To use any experimental flags, first call `ExperimentalFlags.optIntoExperimentalAPIs()`.
public struct ExperimentalFlags {
  private static var optedIn = false

  private static let logger = Logger(
    subsystem: "com.google.odml.litertlm.swift",
    category: "ExperimentalFlags"
  )

  public static func optIntoExperimentalAPIs() {
    guard !optedIn else { return }
    logger.info("EXPERIMENTAL: LiteRTLM: Opting into experimental APIs....")
    optedIn = true
  }

  private static var _enableBenchmark: Bool = false

  public static var enableBenchmark: Bool {
    get { return _enableBenchmark }
    set {
      guard optedIn else {
        logger.error("LiteRTLM: Must opt into experimental APIs before setting this flag.")
        return
      }
      _enableBenchmark = newValue
    }
  }

  private static var _convertCamelToSnakeCaseInToolDescription: Bool = true

  /// Whether to convert the function and parameter names in to snake case for
  /// tool calling.
  ///
  /// Swift idiomatic style uses camelCase for names. However, many large
  /// language models are predominantly trained on datasets where snake_case is
  /// common.
  ///
  /// By default, this API converts Swift camelCase names to snake_case to
  /// potentially improve tool calling performance with models more familiar
  /// with snake_case.
  ///
  /// Set this flag to `false` if your model is specifically trained with
  /// camelCase tool descriptions to skip the conversion. Otherwise, the default
  /// of `true` (using snake_case) is recommended.
  ///
  /// Note: This flag is read only when a new [Conversation] is created.
  /// Changing this value will not affect any existing [Conversation] instances.
  public static var convertCamelToSnakeCaseInToolDescription: Bool {
    get { return _convertCamelToSnakeCaseInToolDescription }
    set {
      guard optedIn else {
        logger.error("LiteRTLM: Must opt into experimental APIs before setting this flag.")
        return
      }
      _convertCamelToSnakeCaseInToolDescription = newValue
    }
  }

  private static var _enableConversationConstrainedDecoding: Bool = false

  /// Whether to enable conversation constrained decoding. This is primarily
  /// used for function calling.
  ///
  /// Note: This flag is read only when a new [Conversation] is created.
  /// Changing this value will not affect any existing [Conversation] instances.
  public static var enableConversationConstrainedDecoding: Bool {
    get { return _enableConversationConstrainedDecoding }
    set {
      guard optedIn else {
        logger.error("LiteRTLM: Must opt into experimental APIs before setting this flag.")
        return
      }
      _enableConversationConstrainedDecoding = newValue
    }
  }

  private static var _enableSpeculativeDecoding: Bool? = nil

  /// Whether to enable speculative decoding.
  ///
  /// If true, enable speculative decoding; an error will be thrown if the model does not support it.
  /// If false, disable speculative decoding.
  ///
  /// Note: This flag is read only when a new [Engine] is created. Changing this value will not
  /// affect any existing [Engine] or [Conversation] instances.
  public static var enableSpeculativeDecoding: Bool? {
    get { return _enableSpeculativeDecoding }
    set {
      guard optedIn else {
        logger.error("LiteRTLM: Must opt into experimental APIs before setting this flag.")
        return
      }
      _enableSpeculativeDecoding = newValue
    }
  }

  private static var _visualTokenBudget: Int32? = nil

  /// The visual token budget.
  ///
  /// The number of visual tokens that the model can generate for a single image. If null, there is
  /// no budget limit and the engine use as much as needed.
  ///
  /// Currently, this is only supported by Gemma4. If this flag is set for a non-Gemma4 model, it
  /// will result in a no-ops. The Gemma4 budget options are 70, 140, 280, 560, or 1120 tokens. See
  /// https://ai.google.dev/gemma/docs/capabilities/vision#variable-resolution for more details.
  ///
  /// Note: This flag takes effect immediately and change alter the behaivor of created
  /// [Conversation].
  public static var visualTokenBudget: Int32? {
    get { return _visualTokenBudget }
    set {
      guard optedIn else {
        logger.error("LiteRTLM: Must opt into experimental APIs before setting this flag.")
        return
      }
      _visualTokenBudget = newValue
    }
  }

  // Prevent initializing the struct
  private init() {}
}
