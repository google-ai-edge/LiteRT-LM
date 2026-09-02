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

/// Supported input/output modalities.
public struct SupportedModalities: Equatable {
  public let text: Bool
  public let vision: Bool
  public let audio: Bool
  public let video: Bool
}

/// Hardware backends.
public enum BackendType: Hashable, Sendable {
  case cpu
  case gpu
  case npu
}

/// NPU brand options.
public enum NPUBrand: Hashable, Sendable {
  case unknown
  case qualcomm
  case googleTensor
  case mediaTek
}

/// Modality options.
public enum Modality: Int, Hashable, Sendable {
  case text = 0
  case vision = 1
  case audio = 2
  case video = 3

  internal var cValue: LiteRtLmModality {
    switch self {
    case .text: return kLiteRtLmModalityText
    case .vision: return kLiteRtLmModalityVision
    case .audio: return kLiteRtLmModalityAudio
    case .video: return kLiteRtLmModalityVideo
    }
  }
}

/// Default sampler parameters.
/// TODO: b/554164915 - Reuse SamplerConfig instead of SamplerParameters (matching Python),
/// once SamplerConfig can support optional fields to represent unset/0 values from models.
public struct SamplerParameters: Equatable {
  public let type: LiteRtLmSamplerType
  public let temperature: Float
  public let topK: Int
  public let topP: Float
}

/// Provides information about capabilities and features supported by a LiteRT-LM file.
///
/// ### Example Usage:
/// ```swift
/// // 1. Load the model metadata
/// guard let capabilities = Capabilities(modelPath: "/path/to/model.litertlm") else {
///   print("Failed to load model file capabilities.")
///   return
/// }
/// // 2. Query basic capability flags
/// let supportsThinking = capabilities.supportsThinking()
/// let supportsFunctionCall = capabilities.supportsFunctionCalling()
/// let hasSpeculativeDecoding = capabilities.hasSpeculativeDecodingSupport()
///
/// // 3. Check supported input modalities
/// if capabilities.inputModalities.vision {
///   print("Vision input is supported!")
/// }
///
/// // 4. Check vision token budget
/// let visionBudget = capabilities.maxVisionTokenBudget()
/// ```
public class Capabilities {
  private let handle: OpaquePointer

  /// Loads a LiteRT-LM file from the given path.
  /// Returns nil if the file cannot be opened.
  public init?(modelPath: String) {
    guard let handle = litert_lm_loaded_file_create(modelPath) else {
      return nil
    }
    self.handle = handle
  }

  /// Checks if the loaded LiteRT-LM file supports speculative decoding.
  public func hasSpeculativeDecodingSupport() -> Bool {
    return litert_lm_loaded_file_has_speculative_decoding_support(handle)
  }

  /// Checks if the loaded LiteRT-LM file supports thinking/reasoning.
  public func supportsThinking() -> Bool {
    return litert_lm_loaded_file_supports_thinking(handle)
  }

  /// Checks if the loaded LiteRT-LM file supports function calling/tool use.
  public func supportsFunctionCalling() -> Bool {
    return litert_lm_loaded_file_supports_function_calling(handle)
  }

  /// Returns the supported input modalities.
  public var inputModalities: SupportedModalities {
    return SupportedModalities(
      text: litert_lm_loaded_file_supports_input_modality(handle, kLiteRtLmModalityText),
      vision: litert_lm_loaded_file_supports_input_modality(handle, kLiteRtLmModalityVision),
      audio: litert_lm_loaded_file_supports_input_modality(handle, kLiteRtLmModalityAudio),
      video: litert_lm_loaded_file_supports_input_modality(handle, kLiteRtLmModalityVideo)
    )
  }

  /// Returns the default sampler parameters for the model.
  public var defaultSamplerParams: SamplerParameters {
    return SamplerParameters(
      type: litert_lm_loaded_file_sampler_type(handle),
      temperature: litert_lm_loaded_file_sampler_temperature(handle),
      topK: Int(litert_lm_loaded_file_sampler_top_k(handle)),
      topP: litert_lm_loaded_file_sampler_top_p(handle)
    )
  }

  /// Returns the maximum vision token budget for the model.
  /// Returns -1 if the model does not support vision or if the budget is not defined.
  public func maxVisionTokenBudget() -> Int {
    return Int(litert_lm_loaded_file_max_vision_token_budget(handle))
  }

  /// Returns the maximum supported context tokens for the loaded LiteRT-LM file.
  ///
  /// - If the model is static (`isDynamicContext()` is false), this is the fixed context size.
  /// - If the model is dynamic (`isDynamicContext()` is true), this is the largest context size that can be set.
  public func maxContextTokens() -> Int {
    return Int(litert_lm_loaded_file_max_context_tokens(handle))
  }

  /// Returns whether the loaded LiteRT-LM file has dynamic context.
  ///
  /// Dynamic context means the context size can be configured by the caller up to the maximum
  /// limit.
  public func isDynamicContext() -> Bool {
    return litert_lm_loaded_file_is_dynamic_context(handle)
  }

  /// Returns the list of vision signature selection choices, or nil if vision is not supported.
  public func visionSignatureSelection() -> [Int]? {
    let count = litert_lm_loaded_file_vision_signature_selection(handle, nil, 0)
    if count == -1 {
      return nil
    }
    if count == 0 {
      return []
    }
    var lengths = [Int32](repeating: 0, count: Int(count))
    let written = litert_lm_loaded_file_vision_signature_selection(handle, &lengths, count)
    guard written > 0 else {
      return []
    }
    return lengths[0..<Int(written)].map { Int($0) }
  }

  /// Returns the minimum LiteRT-LM runtime version required to run this model.
  /// Returns nil if not defined.
  public var minRuntimeVersion: String? {
    guard let versionChars = litert_lm_loaded_file_min_runtime_version(handle) else {
      return nil
    }
    return String(cString: versionChars)
  }

  /// Returns the set of supported backends for a given modality.
  public func supportedBackends(for modality: Modality) -> Set<BackendType> {
    let mask = litert_lm_loaded_file_modality_supported_backends(
      handle, modality.cValue
    )
    var backends = Set<BackendType>()
    if (mask & Int32(kLiteRtLmBackendCpu.rawValue)) != 0 { backends.insert(.cpu) }
    if (mask & Int32(kLiteRtLmBackendGpu.rawValue)) != 0 { backends.insert(.gpu) }
    if (mask & Int32(kLiteRtLmBackendNpu.rawValue)) != 0 { backends.insert(.npu) }
    return backends
  }

  /// Returns the detected NPU brand of the model for a given modality, or .unknown if not NPU-compiled.
  public func npuBrand(for modality: Modality) -> NPUBrand {
    let brand = litert_lm_loaded_file_modality_npu_brand(
      handle, modality.cValue
    )
    switch brand {
    case kLiteRtLmNpuBrandQualcomm: return .qualcomm
    case kLiteRtLmNpuBrandGoogleTensor: return .googleTensor
    case kLiteRtLmNpuBrandMediaTek: return .mediaTek
    default: return .unknown
    }
  }

  deinit {
    litert_lm_loaded_file_delete(handle)
  }
}
