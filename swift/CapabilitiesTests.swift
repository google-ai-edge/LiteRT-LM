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

class CapabilitiesTests: XCTestCase {

  func testInit_SuccessfulWithValidModel() {
    // swift-format-ignore
    let modelResource =
      "runtime/testdata/test_lm.litertlm"
    let modelPath = testDataPath(forResource: modelResource)

    let capabilities = Capabilities(modelPath: modelPath)
    XCTAssertNotNil(capabilities)
  }

  func testInit_ReturnsNilWithInvalidModelPath() {
    let capabilities = Capabilities(modelPath: "/non/existent/path.litertlm")
    XCTAssertNil(capabilities)
  }

  func testHasSpeculativeDecodingSupport() {
    // swift-format-ignore
    let modelResource =
      "runtime/testdata/test_lm.litertlm"
    let modelPath = testDataPath(forResource: modelResource)

    guard let capabilities = Capabilities(modelPath: modelPath) else {
      XCTFail("Failed to load model capabilities")
      return
    }

    // Verify that calling hasSpeculativeDecodingSupport doesn't crash and returns a boolean.
    let supportsSpeculativeDecoding = capabilities.hasSpeculativeDecodingSupport()
    XCTAssertFalse(supportsSpeculativeDecoding)

    // Verify thinking and function calling (false for legacy test model)
    XCTAssertFalse(capabilities.supportsThinking())
    XCTAssertFalse(capabilities.supportsFunctionCalling())
    XCTAssertEqual(capabilities.maxVisionTokenBudget(), -1)

    // Verify modalities
    XCTAssertTrue(capabilities.inputModalities.text)
    XCTAssertFalse(capabilities.inputModalities.vision)
    XCTAssertFalse(capabilities.inputModalities.audio)
    XCTAssertFalse(capabilities.inputModalities.video)

    // Verify default sampler parameters (from model config)
    let defaultSamplerParams = capabilities.defaultSamplerParams
    XCTAssertEqual(defaultSamplerParams.type.rawValue, 2)
    XCTAssertEqual(defaultSamplerParams.temperature, 0.0)
    XCTAssertEqual(defaultSamplerParams.topK, 1)
    XCTAssertEqual(defaultSamplerParams.topP, 0.7)
  }
}
