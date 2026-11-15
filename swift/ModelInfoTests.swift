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

import LiteRTLM
import XCTest

/// Returns the full path to a test data resource.
private func testDataPath(forResource resource: String) -> String {
  guard let testSrcdir = ProcessInfo.processInfo.environment["TEST_SRCDIR"] else {
    fatalError("TEST_SRCDIR not set.")
  }
  return "\(testSrcdir)/\(resource)"
}

class ModelInfoTests: XCTestCase {

  func testInit_SuccessfulWithValidModel() {
    // swift-format-ignore
    let modelResource =
      "runtime/testdata/test_lm.litertlm"
    let modelPath = testDataPath(forResource: modelResource)

    let modelInfo = ModelInfo(modelPath: modelPath)
    XCTAssertNotNil(modelInfo)
  }

  func testInit_ReturnsNilWithInvalidModelPath() {
    LiteRTLMError.clearLastError()
    let modelInfo = ModelInfo(modelPath: "/non/existent/path.litertlm")
    XCTAssertNil(modelInfo)
    XCTAssertNotNil(LiteRTLMError.getLastErrorMessage())
  }

  func testThrowingInit_SuccessfulWithValidModel() throws {
    // swift-format-ignore
    let modelResource =
      "runtime/testdata/test_lm.litertlm"
    let modelPath = testDataPath(forResource: modelResource)

    let modelInfo = try ModelInfo(throwingModelPath: modelPath)
    XCTAssertNotNil(modelInfo)
  }

  func testThrowingInit_ThrowsWithInvalidModelPath() {
    XCTAssertThrowsError(try ModelInfo(throwingModelPath: "/non/existent/path.litertlm")) { error in
      guard let litertError = error as? LiteRTLMError else {
        XCTFail("Expected LiteRTLMError, got \(error)")
        return
      }
      XCTAssertEqual(litertError, .modelInfo(.failedToLoadModel))
      guard case .modelInfo(.failedToLoadModel(let message)) = litertError else {
        XCTFail("Expected failedToLoadModel error, got \(litertError)")
        return
      }
      XCTAssertFalse(message.isEmpty)
      XCTAssertTrue(litertError.localizedDescription.contains(message))
      XCTAssertNil(LiteRTLMError.getLastErrorMessage())
    }
  }

  func testHasSpeculativeDecodingSupport() {
    // swift-format-ignore
    let modelResource =
      "runtime/testdata/test_lm.litertlm"
    let modelPath = testDataPath(forResource: modelResource)

    guard let modelInfo = ModelInfo(modelPath: modelPath) else {
      XCTFail("Failed to load model info")
      return
    }

    guard let llm = modelInfo.llm else {
      XCTFail("modelInfo.llm is nil for LLM model")
      return
    }

    XCTAssertNil(modelInfo.embedding)

    // Verify that calling hasSpeculativeDecodingSupport doesn't crash.
    let supportsSpeculativeDecoding = llm.hasSpeculativeDecodingSupport()
    XCTAssertFalse(supportsSpeculativeDecoding)

    // Verify thinking and function calling (false for legacy test model)
    XCTAssertFalse(llm.supportsThinking())
    XCTAssertFalse(llm.supportsFunctionCalling())
    XCTAssertEqual(modelInfo.modelType(), .llm)
    XCTAssertFalse(modelInfo.isEmbeddingModel())
    XCTAssertTrue(modelInfo.isLlmModel())
    XCTAssertEqual(modelInfo.maxVisionTokenBudget(), -1)
    XCTAssertEqual(llm.maxVisionTokenBudget(), -1)
    XCTAssertNil(modelInfo.visionSignatureSelection())
    XCTAssertNil(llm.visionSignatureSelection())

    // Verify modalities
    XCTAssertTrue(modelInfo.inputModalities.text)
    XCTAssertFalse(modelInfo.inputModalities.vision)
    XCTAssertFalse(modelInfo.inputModalities.audio)
    XCTAssertFalse(modelInfo.inputModalities.video)

    // Verify default sampler parameters (from model config)
    let defaultSamplerParams = llm.defaultSamplerParams
    XCTAssertEqual(defaultSamplerParams.type.rawValue, 2)
    XCTAssertEqual(defaultSamplerParams.temperature, 0.0)
    XCTAssertEqual(defaultSamplerParams.topK, 1)
    XCTAssertEqual(defaultSamplerParams.topP, 0.7)

    // Verify context capabilities (128 from TFLite graph and static context)
    XCTAssertEqual(modelInfo.maxContextTokens(), 128)
    XCTAssertFalse(modelInfo.isDynamicContext())
    XCTAssertFalse(llm.isDynamicContext())

    // Verify minRuntimeVersion is nil for legacy model
    XCTAssertNil(modelInfo.minRuntimeVersion)

    // Verify modality-specific backends for text (defaults to CPU and GPU)
    XCTAssertEqual(modelInfo.supportedBackends(for: .text), [.cpu, .gpu])
    // Verify modality-specific backends for vision (not present -> empty)
    XCTAssertEqual(modelInfo.supportedBackends(for: .vision), [])

    XCTAssertEqual(modelInfo.npuBrand(for: .text), .unknown)
    XCTAssertNil(modelInfo.socName(for: .text))
  }

  func testEmbeddingModel_returnsExpectedCapabilities() {
    // swift-format-ignore
    let modelResource =
      "runtime/testdata/test_embedding.litertlm"
    let modelPath = testDataPath(forResource: modelResource)

    guard let modelInfo = ModelInfo(modelPath: modelPath) else {
      XCTFail("Failed to load model info")
      return
    }

    guard let embedding = modelInfo.embedding else {
      XCTFail("modelInfo.embedding is nil for embedding model")
      return
    }

    XCTAssertNil(modelInfo.llm)
    XCTAssertEqual(modelInfo.modelType(), .embedding)
    XCTAssertTrue(modelInfo.isEmbeddingModel())
    XCTAssertFalse(modelInfo.isLlmModel())
    XCTAssertEqual(embedding.dimension(), 768)
    XCTAssertEqual(modelInfo.maxContextTokens(), 128)
    XCTAssertFalse(modelInfo.isDynamicContext())
    guard let lengths = embedding.signatureSelection() else {
      XCTFail("signatureSelection returned nil")
      return
    }
    XCTAssertEqual(lengths, [128])
  }

  func testVisionSignatureSelection_returnsLengthsForMultimodal() {
    // swift-format-ignore
    let modelResource =
      "runtime/testdata/dummy_vision_with_adapter.litertlm"
    let modelPath = testDataPath(forResource: modelResource)

    guard let modelInfo = ModelInfo(modelPath: modelPath) else {
      XCTFail("Failed to load model info")
      return
    }

    XCTAssertTrue(modelInfo.inputModalities.vision)
    guard let lengths = modelInfo.visionSignatureSelection() else {
      XCTFail("visionSignatureSelection returned nil")
      return
    }
    XCTAssertEqual(lengths, [5])
  }

  func testChainedCapabilityAccess_RetainsOwnerDuringCall() {
    // swift-format-ignore
    let modelResource =
      "runtime/testdata/test_lm.litertlm"
    let modelPath = testDataPath(forResource: modelResource)

    XCTAssertEqual(ModelInfo(modelPath: modelPath)?.llm?.hasSpeculativeDecodingSupport(), false)
    XCTAssertEqual(ModelInfo(modelPath: modelPath)?.llm?.supportsThinking(), false)
  }
}
