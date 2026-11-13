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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_LFM2_AUDIO_LFM2_AUDIO_MODEL_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_LFM2_AUDIO_LFM2_AUDIO_MODEL_CONFIG_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/executor/executor_settings_base.h"

namespace litert::omni::any2any {
namespace lfm2_audio {

// LFM2-1.2B hybrid short-conv + GQA LLM backbone dimensions.
inline constexpr int kLlmHiddenSize = 2048;
inline constexpr int kLlmVocabSize = 65536;
inline constexpr int kDefaultMaxKvLen = 2048;

// FastConformer audio encoder + NeMo FilterbankFeatures frontend constants.
inline constexpr int kAudioMelBins = 128;
inline constexpr int kAudioNfft = 512;
inline constexpr int kAudioWinLength = 400;
inline constexpr int kAudioHopLength = 160;
inline constexpr float kAudioPreemph = 0.97f;
inline constexpr float kAudioLogZeroGuard = 5.960464477539063e-08f;  // 2^-24
inline constexpr int kAudioSubsamplingFactor = 8;

// 8-codebook RQ-Transformer Depthformer constants.
inline constexpr int kTtsHiddenSize = 1024;
inline constexpr int kTtsVocabSize = 2049;
inline constexpr int kTtsNumLayers = 6;
inline constexpr int kTtsNumCodebooks = 8;
inline constexpr int kTtsMaxCacheLen = 8;

// LFM2AudioDetokenizer (LFM2 + ISTFT) constants.
inline constexpr int kDetokenizerUpsampleSamples = 1920;  // 6 * 320
inline constexpr int kDefaultVocoderChunkFrames = 64;
// Exact causal receptive field of `DetokLfm2Model` (5 conv layers * 2 + 3
// sliding-window-30 attention layers * 29 = 97 upsampled steps = ceil(97/6) =
// 17 code frames) plus 1 frame for ISTFT window overlap = 18 left frames and 1
// right frame.
inline constexpr int kDetokenizerLeftContextFrames = 18;
inline constexpr int kDetokenizerRightContextFrames = 1;

inline constexpr int kAudioInputSampleRateHz = 16000;
inline constexpr int kAudioOutputSampleRateHz = 24000;

// LiquidAI/LFM2.5-Audio-1.5B special token strings and IDs.
inline constexpr absl::string_view kPadToken = "<|pad|>";
inline constexpr absl::string_view kBosToken = "<|startoftext|>";
inline constexpr absl::string_view kImStartToken = "<|im_start|>";
inline constexpr absl::string_view kImEndToken = "<|im_end|>";
inline constexpr absl::string_view kAudioStartToken = "<|audio_start|>";
inline constexpr absl::string_view kTextEndToken = "<|text_end|>";

inline constexpr int kPadTokenId = 0;
inline constexpr int kBosTokenId = 1;
inline constexpr int kImStartTokenId = 6;
inline constexpr int kImEndTokenId = 7;
inline constexpr int kAudioStartTokenId = 128;
inline constexpr int kTextEndTokenId = 130;
inline constexpr int kAudioEmbedPlaceholderTokenId = -100;
inline constexpr int kInterleavedNText = 6;
inline constexpr int kInterleavedNAudio = 12;

inline constexpr absl::string_view kDefaultInterleavedSystemPrompt =
    "Respond with interleaved text and audio.";
inline constexpr absl::string_view kDefaultAsrSystemPrompt = "Perform ASR.";
inline constexpr absl::string_view kDefaultSequentialTextSystemPrompt =
    "Respond with text.";
inline constexpr absl::string_view kDefaultTtsSystemPrompt = "Perform TTS.";

}  // namespace lfm2_audio

// Generation scheduling mode for LiquidAI LFM2.5-Audio-1.5B.
enum class Lfm2AudioGenerationMode {
  // Automatically selects `kInterleaved` or `kSequential` from `generate_audio`
  // and `system_prompt`.
  kAuto,
  // Interleaves 6 text tokens and 12 audio frames until `<|text_end|>` and
  // `EOAudio`.
  kInterleaved,
  // Sequential generation (`generate_sequential` in `liquid-audio`): used for
  // ASR (`"Perform ASR."`), text-only responses (`"Respond with text."`), and
  // pure TTS (`"Perform TTS."`).
  kSequential,
};

// Configuration for LiquidAI LFM2.5-Audio-1.5B Any-to-Any inference.
struct Lfm2AudioModelConfig {
  int llm_hidden_size = lfm2_audio::kLlmHiddenSize;
  int llm_vocab_size = lfm2_audio::kLlmVocabSize;
  int max_kv_len = lfm2_audio::kDefaultMaxKvLen;

  int audio_mel_bins = lfm2_audio::kAudioMelBins;
  int audio_n_fft = lfm2_audio::kAudioNfft;
  int audio_win_length = lfm2_audio::kAudioWinLength;
  int audio_hop_length = lfm2_audio::kAudioHopLength;
  float audio_preemph = lfm2_audio::kAudioPreemph;
  float audio_log_zero_guard = lfm2_audio::kAudioLogZeroGuard;
  int audio_subsampling_factor = lfm2_audio::kAudioSubsamplingFactor;

  int tts_hidden_size = lfm2_audio::kTtsHiddenSize;
  int tts_vocab_size = lfm2_audio::kTtsVocabSize;
  int tts_num_layers = lfm2_audio::kTtsNumLayers;
  int tts_num_codebooks = lfm2_audio::kTtsNumCodebooks;
  int tts_max_cache_len = lfm2_audio::kTtsMaxCacheLen;
  int vocoder_chunk_frames = lfm2_audio::kDefaultVocoderChunkFrames;
  int detokenizer_upsample_samples = lfm2_audio::kDetokenizerUpsampleSamples;

  int audio_input_sample_rate_hz = lfm2_audio::kAudioInputSampleRateHz;
  int audio_output_sample_rate_hz = lfm2_audio::kAudioOutputSampleRateHz;

  int pad_token_id = lfm2_audio::kPadTokenId;
  int bos_token_id = lfm2_audio::kBosTokenId;
  int im_start_token_id = lfm2_audio::kImStartTokenId;
  int im_end_token_id = lfm2_audio::kImEndTokenId;
  int audio_start_token_id = lfm2_audio::kAudioStartTokenId;
  int text_end_token_id = lfm2_audio::kTextEndTokenId;
  int audio_embed_placeholder_token_id =
      lfm2_audio::kAudioEmbedPlaceholderTokenId;
  int interleaved_n_text = lfm2_audio::kInterleavedNText;
  int interleaved_n_audio = lfm2_audio::kInterleavedNAudio;

  Lfm2AudioGenerationMode generation_mode = Lfm2AudioGenerationMode::kAuto;
  bool generate_audio = true;
  int max_new_tokens = 512;
  int max_tts_tokens = 1024;
  float temperature = 0.0f;
  float top_p = 0.8f;
  int top_k = 50;
  bool do_sample = false;
  uint64_t seed = 42;

  float tts_temperature = 0.8f;
  float tts_top_p = 1.0f;
  int tts_top_k = 4;

  std::string system_prompt =
      std::string(lfm2_audio::kDefaultInterleavedSystemPrompt);
  std::string voice_name = "default";
  std::string language = "en-US";

  std::optional<lm::Backend> audio_encoder_backend = std::nullopt;
  std::optional<lm::Backend> llm_backend = std::nullopt;
  std::optional<lm::Backend> tts_backend = std::nullopt;
  std::optional<lm::Backend> vocoder_backend = std::nullopt;
  bool force_fp32_on_gpu = false;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_LFM2_AUDIO_LFM2_AUDIO_MODEL_CONFIG_H_
