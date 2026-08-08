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
//
// Acknowledgements:
// This implementation was originally authored by @john-rocky and ported
// from the open-source repository: https://github.com/john-rocky/swift-litert-lm/tree/main
//
// Audio and video through the Foundation Models API.
//
// Apple's FM transcript has built-in text and *image* segments, but no audio or
// video. The custom-segment hook (`Transcript.CustomSegment`) lets a backend
// carry arbitrary modalities through the FM API. These segments feed audio and
// sampled video frames into LiteRT's encoders.
//
//   let answer = try await session.respond {
//     LiteRTAudioSegment(data: wavBytes)
//     "Transcribe the spoken words."
//   }

#if canImport(FoundationModels) && compiler(>=6.4)

  import Foundation
  import FoundationModels

  /// A Foundation Models prompt segment carrying audio for a LiteRT backend.
  ///
  /// `Transcript.CustomSegment` supplies `promptRepresentation`, `description`, and
  /// equality for free; we only provide `id` + `content`. Include it in a prompt
  /// via the `@PromptBuilder` overloads of `respond` / `streamResponse`.
  @available(iOS 27.0, macOS 27.0, *)
  public struct LiteRTAudioSegment: Transcript.CustomSegment {
    public struct Content: Codable, Equatable, Sendable {
      public var data: Data
      public init(data: Data) { self.data = data }
    }

    public let id: String
    public var content: Content

    /// - Parameters:
    ///   - data: Raw audio bytes (WAV / supported container).
    ///   - id: Stable identifier for the segment.
    public init(data: Data, id: String = UUID().uuidString) {
      self.id = id
      self.content = Content(data: data)
    }

    /// Convenience for audio already on disk.
    public init(fileURL: URL, id: String = UUID().uuidString) throws {
      self.id = id
      self.content = Content(data: try Data(contentsOf: fileURL))
    }
  }

  /// A Foundation Models prompt segment carrying sampled video frames (image bytes,
  /// in temporal order). The executor feeds them to the model as a sequence of
  /// images.
  @available(iOS 27.0, macOS 27.0, *)
  public struct LiteRTVideoSegment: Transcript.CustomSegment {
    public struct Content: Codable, Equatable, Sendable {
      public var frames: [Data]
      public init(frames: [Data]) { self.frames = frames }
    }

    public let id: String
    public var content: Content

    public init(frames: [Data], id: String = UUID().uuidString) {
      self.id = id
      self.content = Content(frames: frames)
    }
  }

#endif
