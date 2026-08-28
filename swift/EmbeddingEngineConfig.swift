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

/// Configuration for the LiteRT-LM embedding engine.
public struct EmbeddingEngineConfig: Hashable, Sendable {
  /// The file path to the LiteRT-LM embedding model.
  public let modelPath: String
  /// The backend to use for the main embedding encoder.
  public let backend: Backend
  /// The backend to use for the vision encoder. If `nil`, vision encoder will not be initialized.
  public let visionBackend: Backend?
  /// The backend to use for the audio encoder. If `nil`, audio encoder will not be initialized.
  public let audioBackend: Backend?
  /// The directory for placing cache files. If `nil`, it uses the directory of the `modelPath`.
  public let cacheDir: String?
  /// Maximum sequence length (in tokens) for text encoder signatures.
  public let maxInputLength: Int?
  /// Desired number of vision tokens generated per image.
  public let visionTokensPerImage: Int?

  /// - Parameters:
  ///   - modelPath: The file path to the LiteRT-LM embedding model.
  ///   - backend: The backend to use for the embedding engine. Defaults to `.cpu()`.
  ///   - visionBackend: The backend to use for the vision encoder.
  ///   - audioBackend: The backend to use for the audio encoder.
  ///   - cacheDir: The directory for placing cache files.
  ///   - maxInputLength: Maximum sequence length (in tokens) for text encoder signatures.
  ///   - visionTokensPerImage: Desired number of vision tokens generated per image.
  public init(
    modelPath: String,
    backend: Backend = .cpu(),
    visionBackend: Backend? = nil,
    audioBackend: Backend? = nil,
    cacheDir: String? = nil,
    maxInputLength: Int? = nil,
    visionTokensPerImage: Int? = nil
  ) {
    self.modelPath = modelPath
    self.backend = backend
    self.visionBackend = visionBackend
    self.audioBackend = audioBackend
    self.cacheDir = cacheDir
    self.maxInputLength = maxInputLength
    self.visionTokensPerImage = visionTokensPerImage
  }
}

/// Configuration options for an embedding computation.
public struct EmbeddingOptions: Hashable, Sendable {
  /// Whether to L2-normalize the output embedding vector. If `nil`, uses the C++ engine default.
  public let normalize: Bool?

  /// Whether to automatically insert special tokens (BOS, EOS, start/end of image, start/end of audio). If `nil`, uses the C++ engine default.
  public let insertSpecialTokens: Bool?

  /// The output embedding size to truncate the embedding to. If `nil`, uses the C++ engine default.
  public let outputSize: Int?

  /// The number of vision soft tokens to generate per image.
  ///
  /// If `nil`, uses the C++ engine default.
  public let visionTokensPerImage: Int?

  /// - Parameters:
  ///   - normalize: Whether to L2-normalize the output embedding vector. If `nil`, uses the C++ engine default.
  ///   - insertSpecialTokens: Whether to automatically insert special tokens. If `nil`, uses the C++ engine default.
  ///   - outputSize: The output embedding size to truncate to. If `nil`, uses the C++ engine default.
  ///   - visionTokensPerImage: The number of vision soft tokens to generate per image. If `nil`,
  ///     uses the C++ engine default.
  public init(
    normalize: Bool? = nil,
    insertSpecialTokens: Bool? = nil,
    outputSize: Int? = nil,
    visionTokensPerImage: Int? = nil
  ) {
    self.normalize = normalize
    self.insertSpecialTokens = insertSpecialTokens
    self.outputSize = outputSize
    self.visionTokensPerImage = visionTokensPerImage
  }
}

/// Represents the embedding result for an input item.
public struct EmbeddingResponse: Hashable, Sendable {
  /// Dense float vector output representation.
  public let embedding: [Float]

  /// - Parameter embedding: Dense float vector output representation.
  public init(embedding: [Float]) {
    self.embedding = embedding
  }
}
