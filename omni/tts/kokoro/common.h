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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_COMMON_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_COMMON_H_

#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::omni::tts::kokoro {

// =============================================================================
// Model & Pipeline Architecture Constants
// =============================================================================

// Maximum phoneme sequence length supported across bucketing configurations.
constexpr int kMaxTokens = 256;

// Default target sequence length for bucketing.
constexpr int kDefaultBucketSize = 128;

// Capacity divisor threshold when searching backwards for chunk split points.
constexpr int kSplitCapacityDivisor = 2;

// =============================================================================
// Voice Style Embedding Constants
// =============================================================================

// Number of token embeddings in the voice pack style table (510 token
// embeddings).
constexpr int kVoicePackNumTokens = 510;

// Maximum token index in the 510x256 voice pack embedding table (0..509).
constexpr int kMaxVoiceTokens = 509;

// Total embedding dimension per voice style token (256 floats).
constexpr int kVoiceEmbeddingDim = 256;

// Dimension of style vector slice passed to the prosody predictor (first 128
// floats).
constexpr int kStyleSliceDim = 128;

// Total float count of a pre-extracted voice pack embedding file (510 * 256 =
// 130,560).
constexpr int kVoicePackSize = kVoicePackNumTokens * kVoiceEmbeddingDim;

// =============================================================================
// Token IDs in Kokoro Phoneme Vocabulary
// =============================================================================

// BOS (Beginning of Sequence) and EOS (End of Sequence) token ID in Kokoro
// vocabulary.
constexpr int kBosTokenId = 0;
constexpr int kEosTokenId = 0;

// Whitespace delimiter token ID in Kokoro vocabulary (' ').
constexpr int kSpaceTokenId = 16;

// Range of sentence and clause punctuation token IDs in Kokoro vocabulary
// (1..6: ';', ':', ',', '.', '!', '?').
constexpr int kMinPunctuationTokenId = 1;
constexpr int kMaxPunctuationTokenId = 6;

// =============================================================================
// Tensor & Feature Dimensions
// =============================================================================

// Expanded phoneme text embeddings feature dimension (formerly t_en: 640
// floats).
constexpr int kExpandedTextEmbeddingDim = 640;

// Acoustic feature representation tensor dimension (formerly asr: 512 floats).
constexpr int kAcousticFeatureDim = 512;

// Combined pitch (F0) and energy (N) auxiliary contour dimension (f0_n: 1024
// floats).
constexpr int kF0NDim = 1024;

// Maximum number of output speech acoustic frames supported (512 frames).
constexpr int kMaxSpeechFrames = 512;

// TODO(b/538727793): refactor to configurable parameters.
// =============================================================================
// Audio DSP & Synthesis Constants
// =============================================================================

// Output audio sampling rate in Hz (24kHz).
constexpr int kSampleRate = 24000;

// Audio hop size multiplier from speech frames to time-domain PCM samples
// (24000 / 40 = 600).
constexpr int kAudioHop = 600;

// Number of frequency bins in inverse short-time Fourier transform (iSTFT).
constexpr int kIstftBins = 11;

// Window frame length in samples for iSTFT synthesis.
constexpr int kIstftFrameLen = 20;

// Hop size in samples between consecutive iSTFT synthesis frames.
constexpr int kIstftHop = 5;

// Overlap-add sum normalization factor for Hann synthesis window with hop=5.
constexpr float kHannOverlapAddScale = 1.5f;

// Loads a 510x256 voice style embedding vector from binary file.
//
// args
// - model_dir: Base model directory containing voice files.
// - voice_identifier: Voice name (e.g. "af_heart") or binary filename.
//
// returns
// - Float vector containing 130,560 elements on success, or error status.
absl::StatusOr<std::vector<float>> LoadVoiceEmbedding(
    absl::string_view model_dir,
    absl::string_view voice_identifier = "af_heart");

}  // namespace litert::omni::tts::kokoro

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_KOKORO_COMMON_H_
