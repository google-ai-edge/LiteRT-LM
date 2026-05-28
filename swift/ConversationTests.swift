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

import XCTest

@testable import LiteRTLM

/// Returns the full path to a test data resource.
func testDataPath(forResource resource: String) -> String {
  guard let testSrcdir = ProcessInfo.processInfo.environment["TEST_SRCDIR"] else {
    fatalError("TEST_SRCDIR not set.")
  }
  return "\(testSrcdir)/\(resource)"
}

class ConversationTests: XCTestCase {

  var engine: Engine!

  override func setUp() async throws {
    try await super.setUp()
    let modelResource =
      "runtime/testdata/test_lm.litertlm"
    let modelPath = testDataPath(forResource: modelResource)
    ExperimentalFlags.optIntoExperimentalAPIs()
    ExperimentalFlags.enableBenchmark = true
    ExperimentalFlags.visualTokenBudget = 10
    let engineConfig = try EngineConfig(
      modelPath: modelPath, maxNumTokens: 10, cacheDir: NSTemporaryDirectory())
    engine = Engine(engineConfig: engineConfig)
    try await engine.initialize()
  }

  func testCreateConversationWithSystemMessageWithoutSamplerConfig() async throws {
    let config = ConversationConfig(
      systemMessage: Message("You are a helpful assistant."))

    let conversation = try await self.engine.createConversation(with: config)
    XCTAssertTrue(conversation.isAlive)
  }

  func testConversationConfigSystemMessageSerialization() throws {
    let config = ConversationConfig(
      systemMessage: Message("Talk like a pirate", role: .system))

    guard let systemMessage = config.systemMessage else {
      XCTFail("systemMessage should not be nil")
      return
    }

    let contentsJsonStr = try systemMessage.contents.jsonString

    // Validate it results in the expected "flat" content JSON
    guard let data = contentsJsonStr.data(using: String.Encoding.utf8),
      let arr = try JSONSerialization.jsonObject(with: data) as? [[String: Any]]
    else {
      XCTFail("Should be a flat JSON array")
      return
    }

    XCTAssertEqual(arr.count, 1)
    XCTAssertEqual(arr[0]["type"] as? String, "text")
    XCTAssertEqual(arr[0]["text"] as? String, "Talk like a pirate")
    XCTAssertNil(arr[0]["role"])
  }

  func testCreateConversationWithoutSystemMessageAndSamplerConfig() async throws {
    let conversation = try await self.engine.createConversation()
    XCTAssertTrue(conversation.isAlive)
  }

  func testCreateConversationWithSamplerConfigAndSystemMessage() async throws {
    let config = ConversationConfig(
      systemMessage: Message("You are a helpful assistant."),
      samplerConfig: try SamplerConfig(topK: 10, topP: 0.5, temperature: 0.5, seed: 123))
    let conversation = try await self.engine.createConversation(with: config)
    XCTAssertTrue(conversation.isAlive)
  }

  func testCreateConversationWithSamplerConfigWithoutSystemMessage() async throws {
    let config = ConversationConfig(
      samplerConfig: try SamplerConfig(topK: 10, topP: 0.5, temperature: 0.5, seed: 123))
    let conversation = try await self.engine.createConversation(with: config)
    XCTAssertTrue(conversation.isAlive)
  }

  func testSendMessageReturnsMessage() async throws {
    let conversation = try await self.engine.createConversation(with: ConversationConfig())
    XCTAssertTrue(conversation.isAlive)

    let response = try await conversation.sendMessage(Message("Hello"))
    XCTAssertEqual(response.contents.count, 1)
    switch response.contents[0] {
    case .text(let text):
      XCTAssertFalse(text.isEmpty)
    default:
      XCTFail("Response should be text but received \(response.contents[0])")
    }
  }

  func testSendMessageStream() async throws {
    let conversation = try await self.engine.createConversation(with: ConversationConfig())
    XCTAssertTrue(conversation.isAlive)

    let message = Message("How are you")
    var accumulatedText = ""
    var chunkCount = 0

    for try await chunk in conversation.sendMessageStream(message) {
      chunkCount += 1
      accumulatedText += chunk.toString
    }
    XCTAssertEqual(chunkCount, 6)
    XCTAssertEqual(accumulatedText, "Ꮝgdockdict इक इक इकद्दा")

  }

  func testSendMessageStreamAndCancel() async throws {
    let conversation = try await self.engine.createConversation(with: ConversationConfig())
    XCTAssertTrue(conversation.isAlive)

    let message = Message("Hello")
    var accumulatedText = ""
    var chunkCount = 0
    var didReceiveError: Error?

    let stream = conversation.sendMessageStream(message)

    try conversation.cancel()

    do {
      for try await chunk in stream {
        chunkCount += 1
        accumulatedText += chunk.toString
      }
      XCTFail("Stream should have thrown an error.")
    } catch {
      didReceiveError = error
    }

    XCTAssertNotNil(didReceiveError, "Stream should throw an error when cancelled.")
    if let error = didReceiveError as? LiteRTLMError,
      case .conversation(let conversationError) = error,
      case .invalidResponse(let details) = conversationError
    {
      XCTAssertTrue(
        details.contains("CANCELLED"),
        "Error details should contain 'CANCELLED', but was: \(details)")
    } else {
      XCTFail(
        "Expected LiteRTLMError.conversation(.invalidResponse(...)) containing 'CANCELLED', but got: \(String(describing: didReceiveError))"
      )
    }
  }

  func testBenchmarkInfoOnEmptyConversation() async throws {
    let conversation = try await self.engine.createConversation(with: ConversationConfig())
    XCTAssertTrue(conversation.isAlive)

    // No messages are sent in the conversation, so all the benchmark values should be 0.
    let benchmarkInfo = try conversation.getBenchmarkInfo()
    XCTAssertEqual(benchmarkInfo.lastPrefillTokenCount, 0)
    XCTAssertEqual(benchmarkInfo.lastDecodeTokenCount, 0)
  }

  func testBenchmarkInfoOnNonEmptyConversation() async throws {
    let conversation = try await self.engine.createConversation(with: ConversationConfig())
    XCTAssertTrue(conversation.isAlive)
    let _ = try await conversation.sendMessage(Message("Hello World"))
    let benchmarkInfo = try conversation.getBenchmarkInfo()
    XCTAssertGreaterThan(benchmarkInfo.lastPrefillTokenCount, 0)
    XCTAssertGreaterThan(benchmarkInfo.lastDecodeTokenCount, 0)
  }

  func testSendMessageWithExtraContextReturnsMessage() async throws {
    let conversation = try await self.engine.createConversation(with: ConversationConfig())
    XCTAssertTrue(conversation.isAlive)

    let extraContext: [String: Any] = ["key": "value"]
    let response = try await conversation.sendMessage(
      Message("How are you"), extraContext: extraContext)

    XCTAssertEqual(response.contents.count, 1)
    switch response.contents[0] {
    case .text(let text):
      XCTAssertFalse(text.isEmpty)
    default:
      XCTFail("Response should be text but received \(response.contents[0])")
    }
  }

  func testSendMessageStreamWithExtraContext() async throws {
    let conversation = try await self.engine.createConversation(with: ConversationConfig())
    XCTAssertTrue(conversation.isAlive)

    let message = Message("How are you")
    let extraContext: [String: Any] = ["key": "value"]
    var accumulatedText = ""
    var chunkCount = 0

    for try await chunk in conversation.sendMessageStream(message, extraContext: extraContext) {
      chunkCount += 1
      accumulatedText += chunk.toString
    }

    XCTAssertGreaterThan(chunkCount, 0)
    XCTAssertFalse(accumulatedText.isEmpty)
  }

  func testCreateConversationWithInitialMessages() async throws {
    let config = ConversationConfig(
      initialMessages: [
        Message("Hello", role: .user),
        Message("Hi there!", role: .model),
      ]
    )

    let conversation = try await self.engine.createConversation(with: config)
    XCTAssertTrue(conversation.isAlive)

    let response = try await conversation.sendMessage(Message("How are you?"))
    XCTAssertFalse(response.contents.isEmpty)
  }

  func testCreateConversationWithBothSystemMessageAndInitialSystemMessageFails() async throws {
    let config = ConversationConfig(
      systemMessage: Message("You are a helpful assistant."),
      initialMessages: [
        Message("Hello", role: .system),
        Message("Hi there!", role: .model),
      ]
    )

    do {
      let _ = try await self.engine.createConversation(with: config)
      XCTFail("Should have thrown error for multiple system messages")
    } catch let error as LiteRTLMError {
      XCTAssertEqual(error, LiteRTLMError.config(.multipleSystemMessages))
    } catch {
      XCTFail("Unexpected error: \(error)")
    }
  }

  func testCreateConversationWithMultipleInitialSystemMessagesFails() async throws {
    let config = ConversationConfig(
      initialMessages: [
        Message("Hello", role: .system),
        Message("Hi there!", role: .system),
      ]
    )

    do {
      let _ = try await self.engine.createConversation(with: config)
      XCTFail("Should have thrown error for multiple system messages")
    } catch let error as LiteRTLMError {
      XCTAssertEqual(error, LiteRTLMError.config(.multipleSystemMessages))
    } catch {
      XCTFail("Unexpected error: \(error)")
    }
  }

  func testCreateConversationWithSingleInitialSystemMessageSucceeds() async throws {
    let config = ConversationConfig(
      initialMessages: [
        Message("You are a helpful assistant.", role: .system),
        Message("Hello", role: .user),
      ]
    )

    let conversation = try await self.engine.createConversation(with: config)
    XCTAssertTrue(conversation.isAlive)
  }

  func testRenderMessageIntoString() async throws {
    let conversation = try await self.engine.createConversation(with: ConversationConfig())
    XCTAssertTrue(conversation.isAlive)

    let rendered = try conversation.renderMessageIntoString(Message("Hello world", role: .user))
    XCTAssertFalse(rendered.isEmpty)
    XCTAssertTrue(rendered.contains("Hello world"))
  }
}
