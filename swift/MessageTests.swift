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
}
