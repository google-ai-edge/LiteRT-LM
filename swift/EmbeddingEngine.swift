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

    let settingsError: (String) -> LiteRTLMError = {
      .embeddingEngine(.failedToCreateSettings($0))
    }

    let settings = try LiteRTLMError.create(
      "litert_lm_embedding_engine_settings_create", settingsError
    ) { out in
      litert_lm_embedding_engine_settings_create(
        config.modelPath, backendStr, visionBackendStr, audioBackendStr, out)
    }

    defer { litert_lm_embedding_engine_settings_delete(settings) }

    if case .cpu(let threadCount) = config.backend, let threadCount, threadCount > 0 {
      try LiteRTLMError.check(
        litert_lm_embedding_engine_settings_set_num_threads(settings, Int32(threadCount)),
        "litert_lm_embedding_engine_settings_set_num_threads", settingsError)
    }

    if case .cpu(let threadCount) = config.audioBackend, let threadCount, threadCount > 0 {
      try LiteRTLMError.check(
        litert_lm_embedding_engine_settings_set_audio_num_threads(settings, Int32(threadCount)),
        "litert_lm_embedding_engine_settings_set_audio_num_threads", settingsError)
    }

    if let cacheDir = config.cacheDir {
      try LiteRTLMError.check(
        litert_lm_embedding_engine_settings_set_cache_dir(settings, cacheDir),
        "litert_lm_embedding_engine_settings_set_cache_dir", settingsError)
    }

    if let maxInputLength = config.maxInputLength {
      try LiteRTLMError.check(
        litert_lm_embedding_engine_settings_set_max_input_length(
          settings, Int32(maxInputLength)),
        "litert_lm_embedding_engine_settings_set_max_input_length", settingsError)
    }

    if let visionTokensPerImage = config.visionTokensPerImage {
      try LiteRTLMError.check(
        litert_lm_embedding_engine_settings_set_vision_tokens_per_image(
          settings, Int32(visionTokensPerImage)),
        "litert_lm_embedding_engine_settings_set_vision_tokens_per_image", settingsError)
    }

    let engineHandle = try LiteRTLMError.create(
      "litert_lm_embedding_engine_create", { .embeddingEngine(.failedToCreateEngine($0)) }
    ) { out in
      litert_lm_embedding_engine_create(settings, out)
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
      let ptr = try createInputData(item)
      inputPointers.append(ptr)
    }

    let optionsHandle = try createOptions(options)
    defer { litert_lm_embedding_options_delete(optionsHandle) }

    let computeError: (String) -> LiteRTLMError = {
      .embeddingEngine(.failedToComputeEmbedding($0))
    }
    let responseHandle = try LiteRTLMError.create(
      "litert_lm_embedding_engine_compute_embedding", computeError
    ) { out in
      inputPointers.withUnsafeBufferPointer { buffer in
        litert_lm_embedding_engine_compute_embedding(
          handle,
          buffer.baseAddress,
          buffer.count,
          optionsHandle,
          out
        )
      }
    }
    defer { litert_lm_embedding_response_delete(responseHandle) }

    return try readResponse(responseHandle, computeError)
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
        let ptr = try createInputData(item)
        allInputPointers.append(ptr)
      }
      numInputsPerBatch.append(allInputPointers.count - startIndex)
    }

    let optionsHandle = try createOptions(options)
    defer { litert_lm_embedding_options_delete(optionsHandle) }

    let batchError: (String) -> LiteRTLMError = {
      .embeddingEngine(.failedToComputeEmbeddingBatch($0))
    }

    // Prepare arrays of pointers for the C API batch call using pinned flat buffer
    let responsesHandle = try LiteRTLMError.create(
      "litert_lm_embedding_engine_compute_embedding_batch", batchError
    ) { out in
      allInputPointers.withUnsafeBufferPointer { flatBuf in
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
              optionsHandle,
              out
            )
          }
        }
      }
    }
    defer { litert_lm_embedding_responses_delete(responsesHandle) }

    let count = try LiteRTLMError.get(
      0, "litert_lm_embedding_responses_get_size", batchError
    ) { out in
      litert_lm_embedding_responses_get_size(responsesHandle, out)
    }
    var results: [EmbeddingResponse] = []
    results.reserveCapacity(count)

    for i in 0..<count {
      // The response is borrowed from `responsesHandle` and must not be deleted.
      let respPtr = try LiteRTLMError.create(
        "litert_lm_embedding_responses_get_at", batchError
      ) { out in
        litert_lm_embedding_responses_get_at(responsesHandle, i, out)
      }
      results.append(try readResponse(respPtr, batchError))
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

  /// Creates native embedding options from `options`. The caller owns the returned handle and
  /// must release it with `litert_lm_embedding_options_delete`.
  private func createOptions(_ options: EmbeddingOptions) throws -> OpaquePointer {
    let optionsError: (String) -> LiteRTLMError = {
      .embeddingEngine(.failedToCreateSettings($0))
    }
    let optionsHandle = try LiteRTLMError.create(
      "litert_lm_embedding_options_create", optionsError
    ) { out in
      litert_lm_embedding_options_create(out)
    }
    do {
      if let normalize = options.normalize {
        try LiteRTLMError.check(
          litert_lm_embedding_options_set_normalize(optionsHandle, normalize),
          "litert_lm_embedding_options_set_normalize", optionsError)
      }
      if let insertSpecialTokens = options.insertSpecialTokens {
        try LiteRTLMError.check(
          litert_lm_embedding_options_set_insert_special_tokens(
            optionsHandle, insertSpecialTokens
          ),
          "litert_lm_embedding_options_set_insert_special_tokens", optionsError)
      }
      if let outputSize = options.outputSize {
        try LiteRTLMError.check(
          litert_lm_embedding_options_set_output_size(
            optionsHandle, Int32(outputSize)
          ),
          "litert_lm_embedding_options_set_output_size", optionsError)
      }
      if let visionTokensPerImage = options.visionTokensPerImage {
        try LiteRTLMError.check(
          litert_lm_embedding_options_set_vision_tokens_per_image(
            optionsHandle, Int32(visionTokensPerImage)
          ),
          "litert_lm_embedding_options_set_vision_tokens_per_image", optionsError)
      }
    } catch {
      litert_lm_embedding_options_delete(optionsHandle)
      throw error
    }
    return optionsHandle
  }

  /// Copies the embedding vector out of the native response `response`, which is not consumed.
  private func readResponse(
    _ response: OpaquePointer, _ makeError: (String) -> LiteRTLMError
  ) throws -> EmbeddingResponse {
    let size = try LiteRTLMError.get(
      0, "litert_lm_embedding_response_get_size", makeError
    ) { out in
      litert_lm_embedding_response_get_size(response, out)
    }
    let values = try LiteRTLMError.get(
      nil as UnsafePointer<Float>?, "litert_lm_embedding_response_get_values", makeError
    ) { out in
      litert_lm_embedding_response_get_values(response, out)
    }
    guard let values else {
      if size == 0 {
        return EmbeddingResponse(embedding: [])
      }
      throw makeError("litert_lm_embedding_response_get_values returned null values")
    }
    return EmbeddingResponse(embedding: Array(UnsafeBufferPointer(start: values, count: size)))
  }

  private func createInputData(_ item: Content) throws -> OpaquePointer {
    let inputDataError: (String) -> LiteRTLMError = {
      .embeddingEngine(.failedToCreateInputData($0))
    }
    switch item {
    case .text(let text):
      return try LiteRTLMError.create("litert_lm_input_data_create", inputDataError) { out in
        text.withCString { cStr in
          litert_lm_input_data_create(kLiteRtLmInputDataTypeText, cStr, text.utf8.count, out)
        }
      }
    case .imageData(let data):
      return try LiteRTLMError.create("litert_lm_input_data_create", inputDataError) { out in
        data.withUnsafeBytes { rawBuffer in
          litert_lm_input_data_create(
            kLiteRtLmInputDataTypeImage, rawBuffer.baseAddress, data.count, out)
        }
      }
    case .imageFile(let path):
      let data: Data
      do {
        data = try Data(contentsOf: URL(fileURLWithPath: path))
      } catch {
        throw LiteRTLMError.embeddingEngine(
          .failedToCreateInputData(
            "Failed to read image file at '\(path)': \(error.localizedDescription)"))
      }
      return try LiteRTLMError.create("litert_lm_input_data_create", inputDataError) { out in
        data.withUnsafeBytes { rawBuffer in
          litert_lm_input_data_create(
            kLiteRtLmInputDataTypeImage, rawBuffer.baseAddress, data.count, out)
        }
      }
    case .audioData(let data):
      return try LiteRTLMError.create("litert_lm_input_data_create", inputDataError) { out in
        data.withUnsafeBytes { rawBuffer in
          litert_lm_input_data_create(
            kLiteRtLmInputDataTypeAudio, rawBuffer.baseAddress, data.count, out)
        }
      }
    case .audioFile(let path):
      let data: Data
      do {
        data = try Data(contentsOf: URL(fileURLWithPath: path))
      } catch {
        throw LiteRTLMError.embeddingEngine(
          .failedToCreateInputData(
            "Failed to read audio file at '\(path)': \(error.localizedDescription)"))
      }
      return try LiteRTLMError.create("litert_lm_input_data_create", inputDataError) { out in
        data.withUnsafeBytes { rawBuffer in
          litert_lm_input_data_create(
            kLiteRtLmInputDataTypeAudio, rawBuffer.baseAddress, data.count, out)
        }
      }
    case .toolResponse:
      throw LiteRTLMError.embeddingEngine(
        .failedToCreateInputData("Tool responses are not supported for embeddings"))
    }
  }
}
