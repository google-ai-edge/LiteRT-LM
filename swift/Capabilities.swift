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
  case intel
  case samsung
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
///
/// // 2. Query basic capability flags
/// let supportsThinking = capabilities.supportsThinking()
/// let supportsFunctionCall = capabilities.supportsFunctionCalling()
/// let hasSpeculativeDecoding = capabilities.hasSpeculativeDecodingSupport()
///
/// // 3. Inspect context limits and runtime version requirements
/// let maxContext = capabilities.maxContextTokens()
/// let isDynamic = capabilities.isDynamicContext()
/// if let minVersion = capabilities.minRuntimeVersion {
///   print("Minimum required LiteRT-LM runtime version: \(minVersion)")
/// }
///
/// // 4. Check supported input modalities and vision signatures
/// if capabilities.inputModalities.vision {
///   let visionBudget = capabilities.maxVisionTokenBudget()
///   if let signatures = capabilities.visionSignatureSelection() {
///     print("Supported vision token capacities: \(signatures)")
///   }
/// }
///
/// // 5. Inspect hardware backends (ordered by priority), NPU brand, etc.
/// let textBackends = capabilities.supportedBackends(for: .text)  // e.g. [.cpu, .gpu, .npu]
/// if let defaultBackend = textBackends.first {
///   print("Default backend for text: \(defaultBackend)")  // e.g. .cpu or .npu
/// }
///
/// if textBackends.contains(.npu) {
///   let brand = capabilities.npuBrand(for: .text)  // e.g. .qualcomm, .googleTensor, .mediaTek
///   if let socName = capabilities.socName(for: .text) {
///     print("Target NPU SoC: \(socName) (\(brand))")  // e.g. "SM8750 (qualcomm)"
///   }
/// }
///
/// // 6. Retrieve default sampler parameters
/// let sampler = capabilities.defaultSamplerParams
/// print("Temperature: \(sampler.temperature), TopK: \(sampler.topK), TopP: \(sampler.topP)")
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

  /// Returns the list of supported backends for a given modality, ordered by
  /// priority (first is default).
  public func supportedBackends(for modality: Modality) -> [BackendType] {
    let count = litert_lm_loaded_file_modality_supported_backends(
      handle, modality.cValue, nil, 0
    )
    guard count > 0 else { return [] }
    var cBackends = [LiteRtLmBackendType](
      repeating: LiteRtLmBackendType(0), count: Int(count)
    )
    let written = litert_lm_loaded_file_modality_supported_backends(
      handle, modality.cValue, &cBackends, count
    )
    guard written > 0 else { return [] }
    let validCount = min(Int(written), cBackends.count)
    return cBackends[0..<validCount].compactMap { cType in
      switch cType {
      case kLiteRtLmBackendTypeCpu: return .cpu
      case kLiteRtLmBackendTypeGpu: return .gpu
      case kLiteRtLmBackendTypeNpu: return .npu
      default: return nil
      }
    }
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
    case kLiteRtLmNpuBrandIntel: return .intel
    case kLiteRtLmNpuBrandSamsung: return .samsung
    default: return .unknown
    }
  }

  /// Returns the NPU SoC name string for a given modality, or nil if not set.
  public func socName(for modality: Modality) -> String? {
    guard let chars = litert_lm_loaded_file_modality_soc_name(handle, modality.cValue) else {
      return nil
    }
    return String(cString: chars)
  }

  deinit {
    litert_lm_loaded_file_delete(handle)
  }
}
