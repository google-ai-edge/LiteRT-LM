// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import LiteRTLM
import XCTest

/// Returns the full path to a test data resource.
private func testDataPath(forResource resource: String) -> String {
  guard let testSrcdir = ProcessInfo.processInfo.environment["TEST_SRCDIR"] else {
    fatalError("TEST_SRCDIR not set.")
  }
  return "\(testSrcdir)/\(resource)"
}

class EmbeddingEngineTests: XCTestCase {

  private var modelPath: String {
    let modelResource =
      + "runtime/testdata/test_embedding.litertlm"
    return testDataPath(forResource: modelResource)
  }

  func testEmbeddingOptions_DefaultsAreNil() {
    let options = EmbeddingOptions()
    XCTAssertNil(options.normalize)
    XCTAssertNil(options.insertSpecialTokens)

    let customOptions = EmbeddingOptions(normalize: false, insertSpecialTokens: true)
    XCTAssertEqual(customOptions.normalize, false)
    XCTAssertEqual(customOptions.insertSpecialTokens, true)
  }

  func testEmbeddingEngineConfig_IsCorrectlySet() async throws {
    let config = EmbeddingEngineConfig(
      modelPath: modelPath,
      backend: .cpu(),
      cacheDir: NSTemporaryDirectory()
    )

    let engine = EmbeddingEngine(config: config)
    let engineConfig = await engine.config

    XCTAssertEqual(engineConfig.modelPath, modelPath)
    XCTAssertEqual(engineConfig.cacheDir, NSTemporaryDirectory())
    let isInit = await engine.isInitialized()
    XCTAssertFalse(isInit)
  }

  func testComputeEmbedding_Success() async throws {
    let config = EmbeddingEngineConfig(
      modelPath: modelPath,
      backend: .cpu()
    )
    let engine = EmbeddingEngine(config: config)
    try await engine.initialize()

    // Test with default options (nil options respect engine defaults)
    let responseDefault = try await engine.computeEmbedding(
      contents: [.text("'s")]
    )
    XCTAssertFalse(responseDefault.embedding.isEmpty)

    // Test with explicit normalized options
    let response = try await engine.computeEmbedding(
      contents: [.text("'s")],
      options: EmbeddingOptions(normalize: true)
    )

    XCTAssertFalse(response.embedding.isEmpty)

    // Verify L2 normalization
    var sumSquares: Float = 0.0
    for val in response.embedding {
      sumSquares += val * val
    }
    let norm = sqrt(sumSquares)
    XCTAssertEqual(norm, 1.0, accuracy: 1e-4)

    await engine.close()
  }

  func testComputeEmbeddingBatch_Success() async throws {
    let config = EmbeddingEngineConfig(
      modelPath: modelPath,
      backend: .cpu()
    )
    let engine = EmbeddingEngine(config: config)
    try await engine.initialize()

    let responses = try await engine.computeEmbeddingBatch(
      contentsBatch: [[.text("'s")], [.text("'s")]],
      options: EmbeddingOptions(normalize: true)
    )

    XCTAssertEqual(responses.count, 2)
    XCTAssertFalse(responses[0].embedding.isEmpty)
    XCTAssertFalse(responses[1].embedding.isEmpty)
    XCTAssertEqual(responses[0].embedding.count, responses[1].embedding.count)

    await engine.close()
  }

  func testComputeEmbedding_UninitializedThrows() async throws {
    let config = EmbeddingEngineConfig(
      modelPath: modelPath,
      backend: .cpu()
    )
    let engine = EmbeddingEngine(config: config)

    do {
      _ = try await engine.computeEmbedding(contents: [.text("'s")])
      XCTFail("Expected uninitialized error")
    } catch let error as LiteRTLMError {
      XCTAssertEqual(error, .embeddingEngine(.notInitialized))
    }
  }

  func testInitialize_InvalidPathThrows() async throws {
    let config = EmbeddingEngineConfig(
      modelPath: "/invalid/path/nonexistent.litertlm",
      backend: .cpu()
    )
    let engine = EmbeddingEngine(config: config)

    do {
      try await engine.initialize()
      XCTFail("Expected initialization failure")
    } catch {
      // Expected error
    }
  }
}
