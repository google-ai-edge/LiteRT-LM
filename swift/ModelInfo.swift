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

/// Model type options.
public enum ModelType: Hashable, Sendable {
  case unknown
  case llm
  case embedding
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

/// Calls a native model info accessor that returns a status code and writes its result to an
/// out-parameter, and returns the result.
///
/// The model info accessors are exposed as non-throwing Swift APIs, so a failure (or a value the
/// model does not define, reported as `kLiteRtLmStatusNotFound`) yields `fallback`, which is the
/// value these APIs reported before the native API returned status codes. Scalar out-parameters
/// are not written on failure, and pointer out-parameters are set to NULL.
private func modelInfoValue<T>(
  _ fallback: T, _ body: (UnsafeMutablePointer<T>) -> LiteRtLmStatusCode
) -> T {
  var value = fallback
  if body(&value) != kLiteRtLmStatusOk {
    return fallback
  }
  return value
}

/// Helper to fetch an array of Int32 values from a two-pass C API function.
///
/// Returns nil if the values are not defined by the model (or the call fails).
private func fetchIntArray(
  handle: OpaquePointer,
  cFunction: (
    OpaquePointer?, UnsafeMutablePointer<Int32>?, Int32, UnsafeMutablePointer<Int32>?
  ) -> LiteRtLmStatusCode
) -> [Int]? {
  var count: Int32 = 0
  guard cFunction(handle, nil, 0, &count) == kLiteRtLmStatusOk else {
    return nil
  }
  if count == 0 {
    return []
  }
  var values = [Int32](repeating: 0, count: Int(count))
  var written: Int32 = 0
  guard cFunction(handle, &values, count, &written) == kLiteRtLmStatusOk, written > 0 else {
    return []
  }
  return values[0..<min(Int(written), values.count)].map { Int($0) }
}

/// Capabilities specific to Large Language Models (LLM).
public class LLMCapability {
  private let handle: OpaquePointer
  private let owner: AnyObject

  internal init(handle: OpaquePointer, owner: AnyObject) {
    self.handle = handle
    self.owner = owner
  }

  /// Checks if the loaded LiteRT-LM file supports speculative decoding.
  public func hasSpeculativeDecodingSupport() -> Bool {
    return modelInfoValue(false) {
      litert_lm_loaded_file_has_speculative_decoding_support(handle, $0)
    }
  }

  /// Checks if the loaded LiteRT-LM file supports thinking/reasoning.
  public func supportsThinking() -> Bool {
    return modelInfoValue(false) { litert_lm_loaded_file_supports_thinking(handle, $0) }
  }

  /// Checks if the loaded LiteRT-LM file supports function calling/tool use.
  public func supportsFunctionCalling() -> Bool {
    return modelInfoValue(false) {
      litert_lm_loaded_file_supports_function_calling(handle, $0)
    }
  }

  /// Returns the default sampler parameters for the model.
  public var defaultSamplerParams: SamplerParameters {
    return SamplerParameters(
      type: modelInfoValue(kLiteRtLmSamplerTypeUnspecified) {
        litert_lm_loaded_file_sampler_type(handle, $0)
      },
      temperature: modelInfoValue(Float(0)) {
        litert_lm_loaded_file_sampler_temperature(handle, $0)
      },
      topK: Int(modelInfoValue(Int32(0)) { litert_lm_loaded_file_sampler_top_k(handle, $0) }),
      topP: modelInfoValue(Float(0)) { litert_lm_loaded_file_sampler_top_p(handle, $0) }
    )
  }

  /// Returns whether the loaded LiteRT-LM file has dynamic context.
  ///
  /// Dynamic context means the context size can be configured by the caller
  /// up to the maximum limit.
  public func isDynamicContext() -> Bool {
    return modelInfoValue(false) { litert_lm_loaded_file_is_dynamic_context(handle, $0) }
  }

  /// Returns the maximum vision token budget for the model.
  /// Returns -1 if the model does not support vision or if not defined.
  public func maxVisionTokenBudget() -> Int {
    return Int(
      modelInfoValue(Int32(-1)) { litert_lm_loaded_file_max_vision_token_budget(handle, $0) })
  }

  /// Returns the list of vision signature selection choices, or nil if
  /// vision is not supported.
  public func visionSignatureSelection() -> [Int]? {
    return fetchIntArray(
      handle: handle,
      cFunction: litert_lm_loaded_file_vision_signature_selection
    )
  }
}

public typealias LlmCapability = LLMCapability

/// Capabilities specific to Embedding models.
public class EmbeddingCapability {
  private let handle: OpaquePointer
  private let owner: AnyObject

  internal init(handle: OpaquePointer, owner: AnyObject) {
    self.handle = handle
    self.owner = owner
  }

  /// Returns the output embedding dimension for the model.
  /// Returns nil if not defined.
  public func dimension() -> Int? {
    let dim = modelInfoValue(Int32(-1)) {
      litert_lm_loaded_file_embedding_dimension(handle, $0)
    }
    return dim > 0 ? Int(dim) : nil
  }

  /// Returns the list of supported embedding signature lengths, or nil if
  /// not defined.
  public func signatureSelection() -> [Int]? {
    return fetchIntArray(
      handle: handle,
      cFunction: litert_lm_loaded_file_embedding_signature_selection
    )
  }

  /// Returns the maximum vision token budget for the model.
  /// Returns -1 if the model does not support vision or if not defined.
  public func maxVisionTokenBudget() -> Int {
    return Int(
      modelInfoValue(Int32(-1)) { litert_lm_loaded_file_max_vision_token_budget(handle, $0) })
  }

  /// Returns the list of vision signature selection choices, or nil if
  /// vision is not supported.
  public func visionSignatureSelection() -> [Int]? {
    return fetchIntArray(
      handle: handle,
      cFunction: litert_lm_loaded_file_vision_signature_selection
    )
  }
}

/// Provides information about capabilities and metadata of a LiteRT-LM file.
///
/// ### Example Usage:
/// ```swift
/// // 1. Load the model metadata
/// guard let modelInfo = ModelInfo(modelPath: "/path/to/model.litertlm") else {
///   print("Failed to load model file info.")
///   return
/// }
///
/// // 2. Access LLM or Embedding capabilities
/// if modelInfo.isEmbeddingModel(), let embedding = modelInfo.embedding {
///   let dim = embedding.dimension() // e.g. 768
///   let signatures = embedding.signatureSelection() // e.g. [128, 256, 512]
/// } else if modelInfo.isLlmModel(), let llm = modelInfo.llm {
///   let supportsThinking = llm.supportsThinking()
///   let supportsFunctionCall = llm.supportsFunctionCalling()
///   let hasSpeculativeDecoding = llm.hasSpeculativeDecodingSupport()
///
///   // Retrieve default sampler parameters
///   let sampler = llm.defaultSamplerParams
///   print("Temp: \(sampler.temperature), TopK: \(sampler.topK), TopP: \(sampler.topP)")
/// }
///
/// // 3. Inspect context limits and runtime version requirements
/// let maxContext = modelInfo.maxContextTokens()
/// let isDynamic = modelInfo.isDynamicContext()
/// if let minVersion = modelInfo.minRuntimeVersion {
///   print("Minimum required LiteRT-LM runtime version: \(minVersion)")
/// }
///
/// // 4. Check supported input modalities and vision signatures
/// if modelInfo.inputModalities.vision {
///   let visionBudget = modelInfo.maxVisionTokenBudget()
///   if let signatures = modelInfo.visionSignatureSelection() {
///     print("Supported vision token capacities: \(signatures)")
///   }
/// }
///
/// // 5. Inspect hardware backends (ordered by priority), NPU brand, etc.
/// let textBackends = modelInfo.supportedBackends(for: .text)
/// if let defaultBackend = textBackends.first {
///   print("Default backend for text: \(defaultBackend)")
/// }
///
/// if textBackends.contains(.npu) {
///   let brand = modelInfo.npuBrand(for: .text)
///   if let socName = modelInfo.socName(for: .text) {
///     print("Target NPU SoC: \(socName) (\(brand))")
///   }
/// }
/// ```
public class ModelInfo {
  private static let logger = Logger(
    subsystem: "com.google.odml.litertlm.swift",
    category: "ModelInfo"
  )

  private let handle: OpaquePointer

  /// Loads a LiteRT-LM file from the given path.
  ///
  /// - Parameter modelPath: The path to the LiteRT-LM model file.
  /// - Throws: A `LiteRTLMError` if the model file cannot be loaded.
  public init(throwingModelPath modelPath: String) throws {
    var handle: OpaquePointer?
    let status = litert_lm_loaded_file_create(modelPath, &handle)
    guard status == kLiteRtLmStatusOk, let handle else {
      let errorMsg = LiteRTLMError.consumeLastError() ?? ""
      if !errorMsg.isEmpty {
        Self.logger.error("Failed to load model file at '\(modelPath)': \(errorMsg)")
      } else {
        Self.logger.error("Failed to load model file at '\(modelPath)'")
      }
      throw LiteRTLMError.modelInfo(.failedToLoadModel(errorMsg))
    }
    self.handle = handle
  }

  /// Loads a LiteRT-LM file from the given path.
  /// Returns nil if the file cannot be opened.
  public init?(modelPath: String) {
    var handle: OpaquePointer?
    let status = litert_lm_loaded_file_create(modelPath, &handle)
    guard status == kLiteRtLmStatusOk, let handle else {
      if let errorMsg = LiteRTLMError.getLastErrorMessage(), !errorMsg.isEmpty {
        Self.logger.error("Failed to load model file at '\(modelPath)': \(errorMsg)")
      } else {
        Self.logger.error("Failed to load model file at '\(modelPath)'")
      }
      return nil
    }
    self.handle = handle
  }

  /// LLM-specific capabilities, or nil if the model is not an LLM.
  public var llm: LLMCapability? {
    return isLlmModel() ? LLMCapability(handle: handle, owner: self) : nil
  }

  /// Embedding-specific capabilities, or nil if the model is not an embedding model.
  public var embedding: EmbeddingCapability? {
    return isEmbeddingModel() ? EmbeddingCapability(handle: handle, owner: self) : nil
  }

  /// Returns the supported input modalities.
  public var inputModalities: SupportedModalities {
    let supports = { (modality: LiteRtLmModality) -> Bool in
      modelInfoValue(false) {
        litert_lm_loaded_file_supports_input_modality(self.handle, modality, $0)
      }
    }
    return SupportedModalities(
      text: supports(kLiteRtLmModalityText),
      vision: supports(kLiteRtLmModalityVision),
      audio: supports(kLiteRtLmModalityAudio),
      video: supports(kLiteRtLmModalityVideo)
    )
  }

  /// Returns the maximum vision token budget for the model.
  /// Returns -1 if the model does not support vision or if not defined.
  public func maxVisionTokenBudget() -> Int {
    return Int(
      modelInfoValue(Int32(-1)) { litert_lm_loaded_file_max_vision_token_budget(handle, $0) })
  }

  /// Returns the maximum supported context tokens for the loaded LiteRT-LM file.
  ///
  /// - If the model is static (`isDynamicContext()` is false), this is the
  ///   fixed context size.
  /// - If the model is dynamic (`isDynamicContext()` is true), this is the
  ///   largest context size that can be set.
  public func maxContextTokens() -> Int {
    return Int(
      modelInfoValue(UInt32(0)) { litert_lm_loaded_file_max_context_tokens(handle, $0) })
  }

  /// Returns whether the loaded LiteRT-LM file has dynamic context.
  ///
  /// Dynamic context means the context size can be configured by the caller
  /// up to the maximum limit.
  public func isDynamicContext() -> Bool {
    return modelInfoValue(false) { litert_lm_loaded_file_is_dynamic_context(handle, $0) }
  }

  /// Returns the list of vision signature selection choices, or nil if
  /// vision is not supported.
  public func visionSignatureSelection() -> [Int]? {
    return fetchIntArray(
      handle: handle,
      cFunction: litert_lm_loaded_file_vision_signature_selection
    )
  }

  /// Returns the type of the loaded LiteRT-LM model.
  public func modelType() -> ModelType {
    let modelType = modelInfoValue(kLiteRtLmModelTypeUnknown) {
      litert_lm_loaded_file_model_type(handle, $0)
    }
    switch modelType {
    case kLiteRtLmModelTypeLlm: return .llm
    case kLiteRtLmModelTypeEmbedding: return .embedding
    default: return .unknown
    }
  }

  /// Returns whether the loaded LiteRT-LM file is an embedding model.
  public func isEmbeddingModel() -> Bool {
    return modelType() == .embedding
  }

  /// Returns whether the loaded LiteRT-LM file is an LLM (generative) model.
  public func isLlmModel() -> Bool {
    return modelType() == .llm
  }

  /// Returns the minimum LiteRT-LM runtime version required to run this model.
  /// Returns nil if not defined.
  public var minRuntimeVersion: String? {
    guard
      let versionChars = modelInfoValue(
        nil as UnsafePointer<CChar>?,
        {
          litert_lm_loaded_file_min_runtime_version(handle, $0)
        })
    else {
      return nil
    }
    return String(cString: versionChars)
  }

  /// Returns the list of supported backends for a given modality, ordered by
  /// priority (first is default).
  public func supportedBackends(for modality: Modality) -> [BackendType] {
    let count = modelInfoValue(Int32(0)) {
      litert_lm_loaded_file_modality_supported_backends(handle, modality.cValue, nil, 0, $0)
    }
    guard count > 0 else { return [] }
    var cBackends = [LiteRtLmBackendType](
      repeating: LiteRtLmBackendType(0), count: Int(count)
    )
    let written = modelInfoValue(Int32(0)) {
      litert_lm_loaded_file_modality_supported_backends(
        handle, modality.cValue, &cBackends, count, $0)
    }
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

  /// Returns the detected NPU brand of the model for a given modality, or
  /// .unknown if not NPU-compiled.
  public func npuBrand(for modality: Modality) -> NPUBrand {
    let brand = modelInfoValue(kLiteRtLmNpuBrandUnknown) {
      litert_lm_loaded_file_modality_npu_brand(handle, modality.cValue, $0)
    }
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
    guard
      let chars = modelInfoValue(
        nil as UnsafePointer<CChar>?,
        {
          litert_lm_loaded_file_modality_soc_name(handle, modality.cValue, $0)
        })
    else {
      return nil
    }
    return String(cString: chars)
  }

  deinit {
    litert_lm_loaded_file_delete(handle)
  }
}
