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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_QWEN3_TTS_QWEN3_TTS_STOP_CRITERIA_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_QWEN3_TTS_QWEN3_TTS_STOP_CRITERIA_H_

namespace litert::omni::tts {

// Ends a Qwen3-TTS utterance that keeps generating after its text was spoken.
//
// The Talker only stops when it samples the codec EOS token. Near the end of
// the text EOS competes with the next codebook-0 token, and if sampling misses
// it the Talker can lock into repeating one (silence or noise) codebook-0
// token until `max_frames`, appending tens of seconds of noise to the speech.
// The criteria stop such utterances by:
// - stopping once the same codebook-0 token was generated
//   `kMaxRepeatedFrames` times in a row after all the text was fed (normal
//   speech repeats a token at most ~4 times), and
// - capping the utterance at `kMaxFramesPerTextToken` frames per text token
//   plus `kExtraFrames` (normal speech takes ~2-4.5 frames per token).
class Qwen3TtsStopCriteria {
 public:
  // Maximum number of consecutive frames with the same codebook-0 token after
  // all the text was fed.
  static constexpr int kMaxRepeatedFrames = 8;
  // Frame budget per text token, about twice the slowest normal speech rate.
  static constexpr int kMaxFramesPerTextToken = 8;
  // Frames added to the per-token budget, for very short texts.
  static constexpr int kExtraFrames = 40;

  // Creates the stop criteria of one utterance.
  //
  // args
  // - text_tokens: Number of text tokens (including the TTS EOS token) fed to
  //   the Talker one per frame while generating.
  // - max_frames: Configured maximum number of frames of the utterance.
  Qwen3TtsStopCriteria(int text_tokens, int max_frames);

  // Returns the maximum number of frames of the utterance.
  //
  // returns
  // - `max_frames`, reduced to the text-length budget if that is smaller.
  int FrameLimit() const;

  // Records the codebook-0 token generated at `frame` and returns whether the
  // utterance should end after it.
  //
  // args
  // - frame: Zero-based index of the generated frame, increasing by one per
  //   call.
  // - cb0: Codebook-0 token generated at `frame` (not the codec EOS token).
  //
  // returns
  // - True if the utterance ended its text and keeps repeating `cb0`, or
  //   reached `FrameLimit()`.
  bool ShouldStop(int frame, int cb0);

 private:
  int text_tokens_;
  int frame_limit_;
  int last_cb0_ = -1;
  int repeated_frames_ = 0;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_QWEN3_TTS_QWEN3_TTS_STOP_CRITERIA_H_
