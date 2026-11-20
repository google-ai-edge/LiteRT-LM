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

  func testVideoFramesToJsonWithAllFrameKinds() throws {
    let jpegData = Data([1, 2, 3])
    let rgbData = Data(repeating: 7, count: 4 * 2 * 3)
    let content = Content.videoFrames(
      [
        .imageData(jpegData, timestamp: 0.0),
        .imageFile("/tmp/f1.png", timestamp: 0.5),
        .pixels(rgbData, width: 4, height: 2, timestamp: 1.0),
      ],
      fps: 2.0
    )

    let json = content.toJson
    XCTAssertEqual(json["type"] as? String, "video")
    XCTAssertEqual(json["fps"] as? Double, 2.0)

    let frames = try XCTUnwrap(json["frames"] as? [[String: Any]])
    XCTAssertEqual(frames.count, 3)

    XCTAssertEqual(frames[0]["blob"] as? String, jpegData.base64EncodedString())
    XCTAssertEqual(frames[0]["timestamp"] as? Double, 0.0)

    XCTAssertEqual(frames[1]["path"] as? String, "/tmp/f1.png")
    XCTAssertEqual(frames[1]["timestamp"] as? Double, 0.5)

    XCTAssertEqual(frames[2]["pixels"] as? String, rgbData.base64EncodedString())
    XCTAssertEqual(frames[2]["width"] as? Int, 4)
    XCTAssertEqual(frames[2]["height"] as? Int, 2)
    XCTAssertEqual(frames[2]["timestamp"] as? Double, 1.0)
  }

  func testVideoFramesToJsonOmitsOptionalFields() throws {
    let content = Content.videoFrames([.imageFile("/tmp/f0.png")])

    let json = content.toJson
    XCTAssertEqual(json["type"] as? String, "video")
    XCTAssertNil(json["fps"])

    let frames = try XCTUnwrap(json["frames"] as? [[String: Any]])
    XCTAssertEqual(frames.count, 1)
    XCTAssertEqual(frames[0]["path"] as? String, "/tmp/f0.png")
    XCTAssertNil(frames[0]["timestamp"])
  }

  func testVideoFramesRoundTripsThroughConversationJsonToMessage() throws {
    let jpegData = Data([1, 2, 3])
    let rgbData = Data(repeating: 5, count: 2 * 1 * 3)
    let expectedFrames: [VideoFrame] = [
      .imageFile("/tmp/f0.png", timestamp: 0.0),
      .imageData(jpegData, timestamp: 0.5),
      .pixels(rgbData, width: 2, height: 1, timestamp: 1.0),
    ]
    let message = Message(
      contents: [.videoFrames(expectedFrames, fps: 2.0), .text("What happens?")],
      role: .user
    )

    let parsed = try Conversation.jsonToMessage(try message.jsonString)
    XCTAssertEqual(parsed.role, .user)
    XCTAssertEqual(parsed.contents.count, 2)

    if case .videoFrames(let frames, let fps) = parsed.contents[0] {
      XCTAssertEqual(frames, expectedFrames)
      XCTAssertEqual(fps, 2.0)
    } else {
      XCTFail("Expected .videoFrames content")
    }
    if case .text(let text) = parsed.contents[1] {
      XCTAssertEqual(text, "What happens?")
    } else {
      XCTFail("Expected .text content")
    }
  }

  func testConversationJsonToMessageRejectsOverflowingVideoFrameDimensions() throws {
    let json = """
      {
        "role": "user",
        "content": [
          {
            "type": "video",
            "frames": [
              {"pixels": "AAAA", "width": \(Int.max), "height": 2},
              {"path": "/tmp/valid.png"}
            ]
          }
        ]
      }
      """

    let parsed = try Conversation.jsonToMessage(json)
    XCTAssertEqual(parsed.contents.count, 1)
    if case .videoFrames(let frames, let fps) = parsed.contents[0] {
      XCTAssertEqual(frames, [.imageFile("/tmp/valid.png")])
      XCTAssertNil(fps)
    } else {
      XCTFail("Expected .videoFrames content")
    }
  }

  func testConversationJsonToMessageRejectsMismatchedPixelsAndNegativeTimestamps() throws {
    // 2x2 RGB requires 12 bytes, while "AAAA" decodes to 3 bytes.
    let json = """
      {
        "role": "user",
        "content": [
          {
            "type": "video",
            "frames": [
              {"pixels": "AAAA", "width": 2, "height": 2},
              {"path": "/tmp/neg.png", "timestamp": -0.5},
              {"path": "/tmp/neg_zero.png", "timestamp": -0.0},
              {"path": "/tmp/valid.png", "timestamp": 0.0}
            ]
          }
        ]
      }
      """

    let parsed = try Conversation.jsonToMessage(json)
    XCTAssertEqual(parsed.contents.count, 1)
    if case .videoFrames(let frames, let fps) = parsed.contents[0] {
      XCTAssertEqual(frames, [.imageFile("/tmp/valid.png", timestamp: 0.0)])
      XCTAssertNil(fps)
    } else {
      XCTFail("Expected .videoFrames content")
    }
  }

  func testConversationJsonToMessageRejectsInvalidFps() throws {
    for invalidFps in ["0", "-2.0"] {
      let json = """
        {
          "role": "user",
          "content": [
            {
              "type": "video",
              "fps": \(invalidFps),
              "frames": [{"path": "/tmp/f0.png"}]
            },
            {"type": "text", "text": "fallback"}
          ]
        }
        """

      let parsed = try Conversation.jsonToMessage(json)
      XCTAssertEqual(parsed.contents.count, 1)
      if case .text(let text) = parsed.contents[0] {
        XCTAssertEqual(text, "fallback")
      } else {
        XCTFail("Expected invalid-fps video part to be skipped")
      }
    }
  }

  func testVideoFrameHashableAndExpectedRgbByteCount() {
    let frame1 = VideoFrame.imageData(Data([1, 2, 3]), timestamp: 0.5)
    let frame2 = VideoFrame.imageData(Data([1, 2, 3]), timestamp: 0.5)
    let frame3 = VideoFrame.imageFile("/tmp/f.png", timestamp: 0.5)
    let set: Set<VideoFrame> = [frame1, frame2, frame3]
    XCTAssertEqual(set.count, 2)

    XCTAssertEqual(VideoFrame.expectedRgbByteCount(width: 4, height: 2), 24)
    XCTAssertNil(VideoFrame.expectedRgbByteCount(width: 0, height: 2))
    XCTAssertNil(VideoFrame.expectedRgbByteCount(width: 4, height: -1))
    XCTAssertNil(VideoFrame.expectedRgbByteCount(width: Int.max, height: 2))
  }
}
