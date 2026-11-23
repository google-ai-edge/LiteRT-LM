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

class MessageTests: XCTestCase {
  func testJsonStringVsContentsJsonString() throws {
    let message = Message("test", role: .system)

    let jsonStr = try message.jsonString
    let contentsJsonStr = try message.contents.jsonString

    // Verify jsonString returns a JSON object with "role": "system"
    guard let jsonData = jsonStr.data(using: String.Encoding.utf8),
      let jsonObj = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any]
    else {
      XCTFail("jsonString should be a valid JSON object")
      return
    }
    XCTAssertEqual(jsonObj["role"] as? String, "system")

    // Verify contents.jsonString returns a JSON array of content objects
    guard let contentsData = contentsJsonStr.data(using: String.Encoding.utf8),
      let contentsArr = try JSONSerialization.jsonObject(with: contentsData) as? [[String: Any]]
    else {
      XCTFail("contents.jsonString should be a valid JSON array")
      return
    }
    XCTAssertEqual(contentsArr.count, 1)
    XCTAssertEqual(contentsArr[0]["type"] as? String, "text")
    XCTAssertEqual(contentsArr[0]["text"] as? String, "test")
  }

  func testContentsCreationAndCollectionBehavior() {
    // Test empty()
    let emptyContents = Contents.empty()
    XCTAssertTrue(emptyContents.isEmpty)
    XCTAssertEqual(emptyContents.count, 0)

    // Test of(String)
    let stringContents = Contents.of("hello")
    XCTAssertEqual(stringContents.count, 1)
    if case .text(let str) = stringContents[0] {
      XCTAssertEqual(str, "hello")
    } else {
      XCTFail("Expected text content")
    }

    // Test of(Content...)
    let variadicContents = Contents.of(.text("a"), .text("b"))
    XCTAssertEqual(variadicContents.count, 2)
    XCTAssertEqual(variadicContents.startIndex, 0)
    XCTAssertEqual(variadicContents.endIndex, 2)

    // Test of([Content])
    let arrayContents = Contents.of([.text("x"), .text("y")])
    XCTAssertEqual(arrayContents.count, 2)

    // Test RandomAccessCollection iteration
    var count = 0
    for _ in variadicContents {
      count += 1
    }
    XCTAssertEqual(count, 2)
  }

  func testContentsToString() {
    let contents = Contents.of(.text("Hello"), .text("World"))
    XCTAssertEqual(contents.toString, "Hello World")
  }

  func testMessageWithContentsInit() {
    let contents = Contents.of(.text("Hello Message"))
    let message = Message(contents: contents, role: .model)
    XCTAssertEqual(message.role, .model)
    XCTAssertEqual(message.toString, "Hello Message")
  }

  func testUserBuiltMessageHasNoEndOfTurnFields() {
    let message = Message("Hello", role: .model)

    XCTAssertNil(message.finishReason)
    XCTAssertNil(message.executorStats)
  }

  func testEndOfTurnMessageHasNoContent() throws {
    let message = Message(finishReason: "stop", executorStats: ExecutorStats(moduleName: "LLM"))

    XCTAssertEqual(message.role, .model)
    XCTAssertTrue(message.contents.isEmpty)
    XCTAssertTrue(message.channels.isEmpty)
    XCTAssertTrue(message.toolCalls.isEmpty)
    XCTAssertEqual(message.toString, "")
    XCTAssertEqual(message.finishReason, "stop")
    XCTAssertEqual(message.executorStats, ExecutorStats(moduleName: "LLM"))
  }

  func testJsonToMessageParsesEndOfTurnFields() throws {
    let json = """
      {
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi"}],
        "finish_reason": "stop",
        "executor_stats": {
          "module_name": "LLM",
          "metrics": {"plain_steps": 3, "acceptance_rate": 0.5},
          "latencies_us": {"plain_step_time": 51234},
          "substats": [
            {"module_name": "Drafter", "metrics": {"draft_steps": 2}, "latencies_us": {},
             "substats": []}
          ]
        }
      }
      """

    let message = try Conversation.jsonToMessage(json)

    XCTAssertEqual(message.toString, "Hi")
    XCTAssertEqual(message.finishReason, "stop")
    XCTAssertEqual(
      message.executorStats,
      ExecutorStats(
        moduleName: "LLM",
        metrics: ["plain_steps": 3, "acceptance_rate": 0.5],
        latenciesInMicroseconds: ["plain_step_time": 51234],
        substats: [ExecutorStats(moduleName: "Drafter", metrics: ["draft_steps": 2])]
      ))
  }

  func testJsonToMessageParsesTheSharedTestdata() throws {
    // The tests of every binding parse this file, and the C++ end_of_turn_test checks that it
    // matches the writer. Its unknown `future_field` is ignored.
    let testSrcdir = try XCTUnwrap(ProcessInfo.processInfo.environment["TEST_SRCDIR"])
    // swift-format-ignore
    let resource =
      "runtime/testdata/end_of_turn.json"
    let json = try String(contentsOfFile: "\(testSrcdir)/\(resource)", encoding: .utf8)

    let message = try Conversation.jsonToMessage(json)

    XCTAssertEqual(message.role, .model)
    XCTAssertTrue(message.contents.isEmpty)
    XCTAssertEqual(message.finishReason, "max_tokens")
    XCTAssertEqual(
      message.executorStats,
      ExecutorStats(
        moduleName: "LLM",
        metrics: ["plain_steps": 3, "acceptance_rate": 0.5],
        latenciesInMicroseconds: ["plain_step_time": 51234],
        substats: [ExecutorStats(moduleName: "Drafter", metrics: ["draft_steps": 2])]
      ))
  }

  func testJsonToMessageParsesNullExecutorStatsAsNil() throws {
    let json = #"{"role": "assistant", "finish_reason": "max_tokens", "executor_stats": null}"#

    let message = try Conversation.jsonToMessage(json)

    XCTAssertEqual(message.finishReason, "max_tokens")
    XCTAssertNil(message.executorStats)
  }

  func testJsonToMessageParsesMissingStatsFieldsAsEmpty() throws {
    let message = try Conversation.jsonToMessage(
      #"{"finish_reason": "stop", "executor_stats": {}}"#)

    XCTAssertEqual(message.executorStats, ExecutorStats())
  }

  func testJsonToMessageKeepsUnknownFinishReason() throws {
    let message = try Conversation.jsonToMessage(#"{"finish_reason": "new_reason"}"#)

    XCTAssertEqual(message.finishReason, "new_reason")
    XCTAssertNil(message.executorStats)
  }

  func testJsonToMessageSkipsNonNumericStats() throws {
    let json = """
      {"finish_reason": "stop", "executor_stats": {"module_name": 1,
        "metrics": {"name": "text", "flag": true, "steps": 2},
        "latencies_us": {
          "large": 9007199254740993,
          "fractional": 12.7,
          "overflow": 9223372036854775808,
          "underflow": -9223372036854775809,
          "time": [1],
          "flag": false
        }, "substats": "x"}}
      """

    let message = try Conversation.jsonToMessage(json)

    XCTAssertEqual(
      message.executorStats,
      ExecutorStats(
        metrics: ["steps": 2],
        latenciesInMicroseconds: ["large": 9_007_199_254_740_993]
      ))
  }

  func testJsonToMessageSkipsMalformedSubstats() throws {
    let json = """
      {"finish_reason": "stop", "executor_stats": {"substats": [1, {"module_name": "Sub"}]}}
      """

    let message = try Conversation.jsonToMessage(json)

    XCTAssertEqual(
      message.executorStats, ExecutorStats(substats: [ExecutorStats(moduleName: "Sub")]))
  }

  func testJsonToMessageWithoutFinishReasonStringHasNoEndOfTurnFields() throws {
    let message = try Conversation.jsonToMessage(
      #"""
      {"content": [{"type": "text", "text": "Hi"}], "finish_reason": null,
       "executor_stats": {"module_name": "LLM"}}
      """#)

    XCTAssertNil(message.finishReason)
    XCTAssertNil(message.executorStats)
  }

  func testJsonToMessageWithoutContentOrFinishReasonThrows() {
    XCTAssertThrowsError(
      try Conversation.jsonToMessage(#"{"executor_stats": {"module_name": "LLM"}}"#)
    ) { error in
      XCTAssertEqual(error as? LiteRTLMError, LiteRTLMError.message(.invalidContent))
    }
  }

  func testEndOfTurnFieldsAreNotSerialized() throws {
    let message = try Conversation.jsonToMessage(
      #"""
      {"role": "assistant", "content": [{"type": "text", "text": "Hi"}],
       "finish_reason": "stop", "executor_stats": {"module_name": "LLM"}}
      """#)

    XCTAssertEqual(Set(message.toJson.keys), ["role", "content"])
    XCTAssertEqual(
      Set(Message(finishReason: "stop", executorStats: ExecutorStats()).toJson.keys), ["role"])
  }

  func testExecutorStatsFromInvalidJsonThrows() {
    XCTAssertThrowsError(try ExecutorStats(jsonString: "not json"))
    XCTAssertThrowsError(try ExecutorStats(jsonString: "[]"))
  }
}
