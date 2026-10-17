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
// Audio and video through the Foundation Models API — currently unavailable.
//
// This file held `LiteRTAudioSegment` and `LiteRTVideoSegment`, two
// `Transcript.CustomSegment` conformances that carried audio bytes and video
// frames into a prompt, since FM's transcript has built-in text and image
// segments but nothing for either.
//
// The iOS 27 beta-5 SDK removed the hook. `Transcript.Segment` is now
// text / structure / attachment only, `Transcript.Attachment` is a closed enum
// whose sole case is `.image`, and `Transcript.CustomSegment` no longer exists —
// so a third-party segment type has no representation at all, and the executor's
// matching `.custom` branch went with it.
//
// The engine side is untouched: `Content.audioData` and multi-image prompts
// still work through the LiteRTLM API directly. What is gone is the route in
// through a `LanguageModelSession` prompt. If a later SDK reopens custom
// segments, restore both halves from the history of this file.

#if canImport(FoundationModels) && compiler(>=6.4)

  import Foundation
  import FoundationModels

#endif
