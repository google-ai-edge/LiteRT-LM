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

/// Manages the lifecycle of a LiteRT-LM Embedding Engine, providing an interface for interacting
/// with native embedding models.
///
/// Example usage:
/// ```swift
/// let config = EmbeddingEngineConfig(modelPath: "...")
/// let engine = EmbeddingEngine(config: config)
/// try await engine.initialize()
/// let response = try await engine.computeEmbedding(contents: [.text("Hello world")])
/// await engine.close()
/// ```
public actor EmbeddingEngine {
  private let logger = Logger(
    subsystem: "com.google.odml.litertlm.swift",
    category: "EmbeddingEngine"
  )

  /// The configuration for the embedding engine.
  public let config: EmbeddingEngineConfig

  /// The native handle to the LiteRT-LM embedding engine. A non-nil value indicates an initialized engine.
  private var handle: OpaquePointer? = nil

  /// - Parameter config: The configuration for the embedding engine.
  public init(config: EmbeddingEngineConfig) {
    self.config = config
  }

  deinit {
    if let handle {
      self.handle = nil
      litert_lm_embedding_engine_delete(handle)
    }
  }

  /// Returns `true` if the engine is initialized and ready for use; `false` otherwise.
  public func isInitialized() -> Bool {
    return handle != nil
  }

  /// Initializes the native LiteRT-LM embedding engine.
  ///
  /// - Throws: A `LiteRTLMError` if the engine fails to initialize.
  public func initialize() throws {
    if isInitialized() {
      throw LiteRTLMError.embeddingEngine(.alreadyInitialized)
    }

    let backendStr = config.backend.rawValue
    let visionBackendStr = config.visionBackend?.rawValue
    let audioBackendStr = config.audioBackend?.rawValue

    guard
      let settings = litert_lm_embedding_engine_settings_create(
        config.modelPath, backendStr, visionBackendStr, audioBackendStr)
    else {
      throw LiteRTLMError.embeddingEngine(.failedToCreateSettings)
    }

    defer { litert_lm_embedding_engine_settings_delete(settings) }

    if case .cpu(let threadCount) = config.audioBackend, let threadCount, threadCount > 0 {
      litert_lm_embedding_engine_settings_set_audio_num_threads(settings, Int32(threadCount))
    }

    if let cacheDir = config.cacheDir {
      litert_lm_embedding_engine_settings_set_cache_dir(settings, cacheDir)
    }

    if let maxInputLength = config.maxInputLength {
      litert_lm_embedding_engine_settings_set_max_input_length(
        settings, Int32(maxInputLength))
    }

    if let visionTokensPerImage = config.visionTokensPerImage {
      litert_lm_embedding_engine_settings_set_vision_tokens_per_image(
        settings, Int32(visionTokensPerImage))
    }

    guard let engineHandle = litert_lm_embedding_engine_create(settings) else {
      throw LiteRTLMError.embeddingEngine(.failedToCreateEngine)
    }

    self.handle = engineHandle
  }

  /// Computes an embedding for multimodal input contents.
  ///
  /// - Parameters:
  ///   - contents: An array of `Content` items to embed.
  ///   - options: Options for embedding generation.
  /// - Returns: An `EmbeddingResponse` containing the embedding vector.
  /// - Throws: `LiteRTLMError` if not initialized or if computation fails.
  public func computeEmbedding(
    contents: [Content],
    options: EmbeddingOptions = EmbeddingOptions()
  ) throws -> EmbeddingResponse {
    guard let handle else {
      throw LiteRTLMError.embeddingEngine(.notInitialized)
    }

    var inputPointers: [OpaquePointer?] = []
    defer {
      for ptr in inputPointers {
        if let ptr {
          litert_lm_input_data_delete(ptr)
        }
      }
    }

    for item in contents {
      guard let ptr = createInputData(item) else {
        throw LiteRTLMError.embeddingEngine(.failedToCreateInputData)
      }
      inputPointers.append(ptr)
    }

    guard let optionsHandle = litert_lm_embedding_options_create() else {
      throw LiteRTLMError.embeddingEngine(.failedToCreateSettings)
    }
    defer { litert_lm_embedding_options_delete(optionsHandle) }
    if let normalize = options.normalize {
      litert_lm_embedding_options_set_normalize(optionsHandle, normalize)
    }
    if let insertSpecialTokens = options.insertSpecialTokens {
      litert_lm_embedding_options_set_insert_special_tokens(
        optionsHandle, insertSpecialTokens
      )
    }
    if let outputSize = options.outputSize {
      litert_lm_embedding_options_set_output_size(
        optionsHandle, Int32(outputSize)
      )
    }
    if let visionTokensPerImage = options.visionTokensPerImage {
      litert_lm_embedding_options_set_vision_tokens_per_image(
        optionsHandle, Int32(visionTokensPerImage)
      )
    }

    let responseHandle = inputPointers.withUnsafeBufferPointer { buffer in
      litert_lm_embedding_engine_compute_embedding(
        handle,
        buffer.baseAddress,
        buffer.count,
        optionsHandle
      )
    }

    guard let responseHandle else {
      throw LiteRTLMError.embeddingEngine(.failedToComputeEmbedding)
    }
    defer { litert_lm_embedding_response_delete(responseHandle) }

    let size = litert_lm_embedding_response_get_size(responseHandle)
    guard let valuesPtr = litert_lm_embedding_response_get_values(responseHandle) else {
      if size == 0 {
        return EmbeddingResponse(embedding: [])
      }
      throw LiteRTLMError.embeddingEngine(.failedToComputeEmbedding)
    }

    let valuesBuffer = UnsafeBufferPointer(start: valuesPtr, count: size)
    return EmbeddingResponse(embedding: Array(valuesBuffer))
  }

  /// Computes an embedding for multimodal input contents provided as variadic arguments.
  public func computeEmbedding(
    contents: Content...,
    options: EmbeddingOptions = EmbeddingOptions()
  ) throws -> EmbeddingResponse {
    try computeEmbedding(contents: contents, options: options)
  }

  /// Computes embeddings for a batch of multimodal input requests.
  ///
  /// - Parameters:
  ///   - contentsBatch: An array of requests, where each request is an array of `Content`.
  ///   - options: Options for embedding generation.
  /// - Returns: An array of `EmbeddingResponse` objects.
  /// - Throws: `LiteRTLMError` if not initialized or if computation fails.
  public func computeEmbeddingBatch(
    contentsBatch: [[Content]],
    options: EmbeddingOptions = EmbeddingOptions()
  ) throws -> [EmbeddingResponse] {
    guard let handle else {
      throw LiteRTLMError.embeddingEngine(.notInitialized)
    }

    var allInputPointers: [OpaquePointer?] = []
    var numInputsPerBatch: [Int] = []

    defer {
      for ptr in allInputPointers {
        if let ptr {
          litert_lm_input_data_delete(ptr)
        }
      }
    }

    for req in contentsBatch {
      let startIndex = allInputPointers.count
      for item in req {
        guard let ptr = createInputData(item) else {
          throw LiteRTLMError.embeddingEngine(.failedToCreateInputData)
        }
        allInputPointers.append(ptr)
      }
      numInputsPerBatch.append(allInputPointers.count - startIndex)
    }

    guard let optionsHandle = litert_lm_embedding_options_create() else {
      throw LiteRTLMError.embeddingEngine(.failedToCreateSettings)
    }
    defer { litert_lm_embedding_options_delete(optionsHandle) }
    if let normalize = options.normalize {
      litert_lm_embedding_options_set_normalize(optionsHandle, normalize)
    }
    if let insertSpecialTokens = options.insertSpecialTokens {
      litert_lm_embedding_options_set_insert_special_tokens(
        optionsHandle, insertSpecialTokens
      )
    }
    if let outputSize = options.outputSize {
      litert_lm_embedding_options_set_output_size(
        optionsHandle, Int32(outputSize)
      )
    }
    if let visionTokensPerImage = options.visionTokensPerImage {
      litert_lm_embedding_options_set_vision_tokens_per_image(
        optionsHandle, Int32(visionTokensPerImage)
      )
    }

    // Prepare arrays of pointers for the C API batch call using pinned flat buffer
    let responsesHandle = allInputPointers.withUnsafeBufferPointer { flatBuf in
      var reqBasePointers: [UnsafePointer<OpaquePointer?>?] = []
      var currentOffset = 0
      for count in numInputsPerBatch {
        if count == 0 {
          reqBasePointers.append(nil)
        } else {
          reqBasePointers.append(flatBuf.baseAddress.map { $0 + currentOffset })
          currentOffset += count
        }
      }

      return reqBasePointers.withUnsafeBufferPointer { batchBuf in
        numInputsPerBatch.withUnsafeBufferPointer { numInputsBuf in
          litert_lm_embedding_engine_compute_embedding_batch(
            handle,
            batchBuf.baseAddress,
            numInputsBuf.baseAddress,
            contentsBatch.count,
            optionsHandle
          )
        }
      }
    }

    guard let responsesHandle else {
      throw LiteRTLMError.embeddingEngine(.failedToComputeEmbeddingBatch)
    }
    defer { litert_lm_embedding_responses_delete(responsesHandle) }

    let count = litert_lm_embedding_responses_get_size(responsesHandle)
    var results: [EmbeddingResponse] = []
    results.reserveCapacity(count)

    for i in 0..<count {
      guard let respPtr = litert_lm_embedding_responses_get_at(responsesHandle, i) else {
        throw LiteRTLMError.embeddingEngine(.failedToComputeEmbeddingBatch)
      }
      let size = litert_lm_embedding_response_get_size(respPtr)
      guard let valuesPtr = litert_lm_embedding_response_get_values(respPtr) else {
        if size == 0 {
          results.append(EmbeddingResponse(embedding: []))
          continue
        }
        throw LiteRTLMError.embeddingEngine(.failedToComputeEmbeddingBatch)
      }
      let valuesBuffer = UnsafeBufferPointer(start: valuesPtr, count: size)
      results.append(EmbeddingResponse(embedding: Array(valuesBuffer)))
    }

    return results
  }

  /// Closes the engine and releases native resources.
  public func close() {
    if let handle {
      self.handle = nil
      litert_lm_embedding_engine_delete(handle)
    }
  }

  private func createInputData(_ item: Content) -> OpaquePointer? {
    switch item {
    case .text(let text):
      return text.withCString({ cStr in
        litert_lm_input_data_create(kLiteRtLmInputDataTypeText, cStr, strlen(cStr))
      })
    case .imageData(let data):
      return data.withUnsafeBytes({ rawBuffer in
        litert_lm_input_data_create(
          kLiteRtLmInputDataTypeImage, rawBuffer.baseAddress, data.count)
      })
    case .imageFile(let path):
      guard
        let data = try? Data(contentsOf: URL(fileURLWithPath: path))
      else {
        return nil
      }
      return data.withUnsafeBytes({ rawBuffer in
        litert_lm_input_data_create(
          kLiteRtLmInputDataTypeImage, rawBuffer.baseAddress, data.count)
      })
    case .audioData(let data):
      return data.withUnsafeBytes({ rawBuffer in
        litert_lm_input_data_create(
          kLiteRtLmInputDataTypeAudio, rawBuffer.baseAddress, data.count)
      })
    case .audioFile(let path):
      guard
        let data = try? Data(contentsOf: URL(fileURLWithPath: path))
      else {
        return nil
      }
      return data.withUnsafeBytes({ rawBuffer in
        litert_lm_input_data_create(
          kLiteRtLmInputDataTypeAudio, rawBuffer.baseAddress, data.count)
      })
    case .toolResponse:
      return nil
    }
  }
}
