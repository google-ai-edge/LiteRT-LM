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

/// The role of the message in a conversation.
public enum Role: String {
  case system
  case user
  case model = "assistant"
  case tool
}

/// One pre-decoded frame of `Content.videoFrames`.
///
/// LiteRT-LM does not decode video containers; decode and sample frames in the application and pass
/// them here. The runtime handles resizing and patchification.
public struct VideoFrame: Hashable, Sendable {
  /// The image payload of a video frame.
  public enum Source: Hashable, Sendable {
    /// Encoded image bytes (e.g. JPEG or PNG).
    case imageData(Data)
    /// Absolute path to an encoded image file.
    case imageFile(String)
    /// Raw RGB888 HWC uint8 pixels (`width * height * 3` bytes), sent without re-encoding.
    case pixels(data: Data, width: Int, height: Int)
  }

  /// The frame's image payload.
  public let source: Source

  /// Presentation time of the frame in the source video, in seconds (non-negative). If `nil`, the
  /// runtime uses `index / fps`.
  public let timestamp: Double?

  /// Creates a `VideoFrame` from an encoded or raw image source.
  public init(source: Source, timestamp: Double? = nil) {
    if let timestamp {
      precondition(
        timestamp.isFinite && timestamp.sign == .plus,
        "Video frame timestamp must be a non-negative finite number.")
    }
    if case .pixels(let data, let width, let height) = source {
      guard let expectedBytes = Self.expectedRgbByteCount(width: width, height: height) else {
        preconditionFailure("Video frame width and height must be positive and not overflow.")
      }
      precondition(
        data.count == expectedBytes,
        "Video frame pixels must have width * height * 3 bytes.")
    }
    self.source = source
    self.timestamp = timestamp
  }

  /// Returns `width * height * 3` when `width > 0`, `height > 0`, and the product does not overflow
  /// `Int`; otherwise returns `nil`.
  static func expectedRgbByteCount(width: Int, height: Int) -> Int? {
    guard width > 0 && height > 0 else { return nil }
    let (pixelCount, pixelOverflow) = width.multipliedReportingOverflow(by: height)
    guard !pixelOverflow else { return nil }
    let (byteCount, byteOverflow) = pixelCount.multipliedReportingOverflow(by: 3)
    return byteOverflow ? nil : byteCount
  }

  /// Creates a `VideoFrame` from encoded image bytes (e.g. JPEG or PNG).
  public static func imageData(_ data: Data, timestamp: Double? = nil) -> VideoFrame {
    VideoFrame(source: .imageData(data), timestamp: timestamp)
  }

  /// Creates a `VideoFrame` from an absolute path to an encoded image file.
  public static func imageFile(_ path: String, timestamp: Double? = nil) -> VideoFrame {
    VideoFrame(source: .imageFile(path), timestamp: timestamp)
  }

  /// Creates a `VideoFrame` from raw RGB888 HWC uint8 pixels (`width * height * 3` bytes).
  public static func pixels(
    _ data: Data, width: Int, height: Int, timestamp: Double? = nil
  ) -> VideoFrame {
    VideoFrame(source: .pixels(data: data, width: width, height: height), timestamp: timestamp)
  }

  /// Convert to JSON format. Used internally.
  var toJson: [String: Any] {
    var dict: [String: Any]
    switch source {
    case .imageData(let bytes):
      dict = ["blob": bytes.base64EncodedString()]
    case .imageFile(let absPath):
      dict = ["path": absPath]
    case .pixels(let data, let width, let height):
      dict = [
        "pixels": data.base64EncodedString(),
        "width": width,
        "height": height,
      ]
    }
    if let timestamp {
      dict["timestamp"] = timestamp
    }
    return dict
  }
}

/// Represents a content in the `Message` of the conversation.
///
/// Example Usage:
/// ```swift
/// let textContent = Content.text("What's the capital of France?")
/// let imageContent = Content.imageData(imageData)
/// let audioContent = Content.audioFile(path: "/path/to/audio.wav")
/// ```
public enum Content {
  /// Text.
  case text(String)
  /// Image provided as raw bytes.
  case imageData(Data)
  /// Image provided by an absolute file path.
  case imageFile(String)
  /// Audio provided as raw bytes.
  case audioData(Data)
  /// Audio provided by an absolute file path.
  case audioFile(String)
  /// Tool response.
  case toolResponse(name: String, response: Any, id: String = "")
  /// Video provided as pre-decoded frames sampled by the caller.
  ///
  /// Requires a model whose metadata reports the Video input modality. The runtime handles
  /// resizing, patchification, and temporal frame grouping.
  case videoFrames([VideoFrame], fps: Double? = nil)

  /// Convert to JSON format. Used internally.
  var toJson: [String: Any] {
    switch self {
    case .text(let text):
      return ["type": "text", "text": text]
    case .imageData(let bytes):
      return ["type": "image", "blob": bytes.base64EncodedString()]
    case .imageFile(let absPath):
      return ["type": "image", "path": absPath]
    case .audioData(let bytes):
      return ["type": "audio", "blob": bytes.base64EncodedString()]
    case .audioFile(let absPath):
      return ["type": "audio", "path": absPath]
    case .toolResponse(let name, let response, let id):
      var dict: [String: Any] = [
        "type": "tool_response",
        "name": name,
        "response": response,
      ]
      if !id.isEmpty {
        dict["id"] = id
      }
      return dict
    case .videoFrames(let frames, let fps):
      precondition(!frames.isEmpty, "A video must have at least one frame.")
      if let fps {
        precondition(fps.isFinite && fps > 0, "Video fps must be a positive finite number.")
      }
      var dict: [String: Any] = [
        "type": "video",
        "frames": frames.map { $0.toJson },
      ]
      if let fps {
        dict["fps"] = fps
      }
      return dict
    }
  }
}

/// Represents a message in the conversation. A message can contain multiple `Content`s.
///
/// Example Usage:
/// ```swift
/// let textMessage = Message("What's the capital of France?")
/// let modelMessage = Message("The capital of France is Paris.", role: .model)
/// let audioContentMessage = Message(of: someAudioContent)
/// let multiContentMessage = Message(of: someText, someImageContent)
/// let multiContentMessageFromArray = Message(contents: [someText, someImageContent])
/// ```
public struct Message {

  private let logger = Logger(
    subsystem: "com.google.odml.litertlm.swift",
    category: "Message"
  )

  /// The role of the message.
  public let role: Role

  /// The contents of the message.
  public let contents: Contents

  /// The channels of the message.
  public let channels: [String: String]

  /// The tool calls in the message.
  public let toolCalls: [ToolCall]

  /// Why the turn ended, set only on the message that ends a turn: the response of
  /// `Conversation.sendMessage`, and the last element of `Conversation.sendMessageStream`, which
  /// has no content. A non-nil value is how to detect the end of a turn.
  ///
  /// - "stop": the model finished its response, or the response reached `maxOutputTokens`.
  /// - "max_tokens": the conversation filled the engine's context window
  ///   (`EngineConfig.maxNumTokens`), which cut the response off.
  ///
  /// More values may come; treat unknown values as the end of the turn too. It's never sent back to
  /// the model.
  public let finishReason: String?

  /// The executor stats of the turn, set only with `finishReason`, or `nil` if the turn has none.
  /// It's never sent back to the model.
  public let executorStats: ExecutorStats?

  // TODO: (b/459796564): Update constructor to `Message.role("messageTextHere")`
  /// Creates a `Message` from a text string.
  /// - Parameter text: The text content of the message.
  public init(_ text: String, role: Role = .user, channels: [String: String] = [:]) {
    self.init(
      contents: Contents(contents: [.text(text)]), role: role, channels: channels, toolCalls: [])
  }

  /// Creates a `Message` from one or more `Content`s.
  /// - Parameter contents: The list of contents for this message.
  public init(of contents: Content..., role: Role = .user) {
    precondition(!contents.isEmpty, "Contents should not be empty.")
    self.init(
      contents: Contents(contents: contents), role: role, channels: [:], toolCalls: [],
      finishReason: nil, executorStats: nil)
  }

  /// Creates a `Message` from a list of `Content`s.
  /// - Parameter contents: The list of contents for this message.
  public init(
    contents: [Content], role: Role = .user, channels: [String: String] = [:],
    toolCalls: [ToolCall] = []
  ) {
    self.init(
      contents: Contents(contents: contents), role: role, channels: channels, toolCalls: toolCalls)
  }

  /// Creates a `Message` from a `Contents` object.
  /// - Parameter contents: The contents object for this message.
  public init(
    contents: Contents, role: Role = .user, channels: [String: String] = [:],
    toolCalls: [ToolCall] = []
  ) {
    precondition(
      !contents.isEmpty || !channels.isEmpty || !toolCalls.isEmpty,
      "Contents, channels and toolCalls should not all be empty.")
    self.init(
      contents: contents, role: role, channels: channels, toolCalls: toolCalls,
      finishReason: nil, executorStats: nil)
  }

  /// Creates the content-less model message that ends a streamed turn.
  init(finishReason: String, executorStats: ExecutorStats?) {
    self.init(
      contents: .empty(), role: .model, channels: [:], toolCalls: [],
      finishReason: finishReason, executorStats: executorStats)
  }

  /// Creates a `Message` without checking its content, as the runtime's messages that end a turn
  /// may have none.
  init(
    contents: Contents, role: Role, channels: [String: String], toolCalls: [ToolCall],
    finishReason: String?, executorStats: ExecutorStats?
  ) {
    self.contents = contents
    self.role = role
    self.channels = channels
    self.toolCalls = toolCalls
    self.finishReason = finishReason
    self.executorStats = executorStats
  }

  /// Convert to JSON format. Used internally.
  var toJson: [String: Any] {
    var dict: [String: Any] = ["role": role.rawValue]
    if !contents.isEmpty {
      dict["content"] = contents.toJson
    }
    if !channels.isEmpty {
      dict["channels"] = channels
    }
    if !toolCalls.isEmpty {
      dict["tool_calls"] = toolCalls.map { toolCall in
        let tcDict: [String: Any] = [
          "id": toolCall.id,
          "type": "function",
          "function": [
            "name": toolCall.name,
            "arguments": toolCall.arguments,
          ],
        ]
        return tcDict
      }
    }
    return dict
  }

  /// Convenience property to get the text content of the message.
  /// Returns a string by concatenating all text content items, separated by a space.
  /// Returns an empty string if no text content exists.
  public var toString: String {
    return contents.toString
  }

  /// A computed property that returns the JSON string representation
  /// of the Message, or nil if conversion fails.
  var jsonString: String {
    get throws {
      let jsonData = try JSONSerialization.data(withJSONObject: self.toJson, options: [])
      guard let resultString = String(data: jsonData, encoding: .utf8) else {
        throw LiteRTLMError.message(.failedToConvertToJson)
      }
      return resultString
    }
  }
}

/// Represents a collection of `Content` items in the `Message` of the conversation.
///
/// Example Usage:
/// ```swift
/// let contents = Contents(contents: [.text("Hello")])
/// ```
public struct Contents: RandomAccessCollection {
  public typealias Element = Content
  public typealias Index = Int

  /// The list of underlying content items.
  public var contents: [Content]

  public var startIndex: Int { contents.startIndex }
  public var endIndex: Int { contents.endIndex }

  public subscript(position: Int) -> Content {
    return contents[position]
  }

  /// Creates a `Contents` from a list of `Content`s.
  public init(contents: [Content]) {
    self.contents = contents
  }

  /// Creates an empty `Contents`.
  public static func empty() -> Contents {
    return Contents(contents: [])
  }

  /// Creates a `Contents` from a text string.
  public static func of(_ text: String) -> Contents {
    return Contents(contents: [.text(text)])
  }

  /// Creates a `Contents` from one or more `Content`s.
  public static func of(_ contents: Content...) -> Contents {
    return Contents(contents: contents)
  }

  /// Creates a `Contents` from a list of `Content`s.
  public static func of(_ contents: [Content]) -> Contents {
    return Contents(contents: contents)
  }

  /// Convert to JSON format. Used internally.
  var toJson: [[String: Any]] {
    return contents.map { $0.toJson }
  }

  /// A computed property that returns the JSON string representation
  /// of the Contents array.
  var jsonString: String {
    get throws {
      let jsonData = try JSONSerialization.data(withJSONObject: self.toJson, options: [])
      guard let resultString = String(data: jsonData, encoding: .utf8) else {
        throw LiteRTLMError.message(.failedToConvertToJson)
      }
      return resultString
    }
  }

  /// Returns a string by concatenating all text content items, separated by a space.
  public var toString: String {
    contents.compactMap { content in
      if case .text(let str) = content { return str } else { return nil }
    }.joined(separator: " ")
  }
}

/// Represents a tool call from the model.
public struct ToolCall {
  public let name: String
  public let id: String
  public let arguments: [String: Any]

  public init(name: String, id: String, arguments: [String: Any]) {
    self.name = name
    self.id = id
    self.arguments = arguments
  }
}
