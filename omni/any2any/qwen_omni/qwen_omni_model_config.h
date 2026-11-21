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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_MODEL_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_MODEL_CONFIG_H_

#include <cstdint>
#include <optional>
#include <string>

#include "runtime/executor/executor_settings_base.h"

namespace litert::omni::any2any {
namespace qwen_omni {

// Thinker LLM defaults (Qwen2.5-Omni-3B = 2048):
inline constexpr int kLlmHiddenSize = 2048;
inline constexpr int kLlmVocabSize = 151936;

// Talker speech-codec decoder defaults (Qwen2.5-Omni-3B Talker = 896,
// 24 layers, 2 KV heads, 64 head dim, 8448 codec vocab):
inline constexpr int kTtsHiddenSize = 896;
inline constexpr int kTtsVocabSize = 8448;
inline constexpr int kVocoderCodebookSize = 8193;
inline constexpr int kTtsNumLayers = 24;
inline constexpr int kTtsNumKvHeads = 2;
inline constexpr int kTtsHeadDim = 64;
inline constexpr int kTtsMaxCacheLen = 1024;
inline constexpr int kTtsTextChunkSize = 10;
inline constexpr int kTtsNumMelBins = 80;
inline constexpr int kTtsNumCodebooks = 1;

// Audio & Vision encoder defaults:
inline constexpr int kAudioInputSampleRateHz = 16000;
inline constexpr int kAudioOutputSampleRateHz = 24000;
inline constexpr int kVisionImageSize = 280;
inline constexpr int kVisionPatchSize = 14;
inline constexpr int kVisionSpatialMergeSize = 2;
inline constexpr int kDefaultVisionTokens = 100;  // (280 / 14 / 2)^2 = 100
inline constexpr int kAudioMelBins = 128;
inline constexpr int kAudioChunkFrames = 200;  // 2 * n_window (2.0s at 100fps)
inline constexpr int kAudioPoolStep = 2;       // Conv1d stride-2 + AvgPool = 4x

// ChatML and multimodal special token IDs in Qwen2.5-Omni
// (`configuration_qwen2_5_omni.py`):
inline constexpr int kEosTokenId = 151643;        // <|endoftext|>
inline constexpr int kImStartTokenId = 151644;    // <|im_start|>
inline constexpr int kImEndTokenId = 151645;      // <|im_end|>
inline constexpr int kAudioTokenId = 151646;      // <|AUDIO|>
inline constexpr int kAudioBosTokenId = 151647;   // <|audio_bos|>
inline constexpr int kAudioEosTokenId = 151648;   // <|audio_eos|>
inline constexpr int kVisionBosTokenId = 151652;  // <|vision_bos|>
inline constexpr int kVisionEosTokenId = 151653;  // <|vision_eos|>
inline constexpr int kImageTokenId = 151655;      // <|IMAGE|>
inline constexpr int kVideoTokenId = 151656;      // <|VIDEO|>

// Talker codec & text special token IDs (`configuration_qwen2_5_omni.py` and
// `spk_dict.pt`):
inline constexpr int kTtsCodecPadTokenId = 8292;     // tts_codec_pad_token_id
inline constexpr int kTtsCodecStartTokenId = 8293;   // tts_codec_start_token_id
inline constexpr int kTtsCodecEndTokenId = 8294;     // tts_codec_end_token_id
inline constexpr int kTtsCodecMaskTokenId = 8296;    // tts_codec_mask_token_id
inline constexpr int kTtsTextPadTokenId = 151859;    // tts_text_pad_token_id
inline constexpr int kTtsTextStartTokenId = 151860;  // tts_text_start_token_id
inline constexpr int kTtsTextEndTokenId = 151861;    // tts_text_end_token_id
// `spk_dict["Ethan"]["bos_token"]` and `spk_dict["Chelsie"]["bos_token"]`.
inline constexpr int kEthanTextBosTokenId = 151870;
inline constexpr int kChelsieTextBosTokenId = 151872;

// Aliases used by `QwenOmniTtsStage` and `QwenOmniVocoderStage`:
inline constexpr int kTtsTextEosTokenId = kTtsTextEndTokenId;
inline constexpr int kTtsAudioBosTokenId = kTtsCodecStartTokenId;
inline constexpr int kTtsEosTokenId = kTtsCodecEndTokenId;
inline constexpr int kTtsSilenceTokenId = kTtsCodecPadTokenId;

// 24 kHz Token2Wav block-aligned vocoder window parameters (`block_size = 24`
// mel frames = `12` codec tokens at `repeats = 2`; `12 + 24 + 12 = 48` codec
// tokens = 4 full DiT blocks; each codec token produces `480` PCM samples):
inline constexpr int kVocoderHistoryTokens = 12;
inline constexpr int kVocoderChunkTokens = 24;
inline constexpr int kVocoderLookaheadTokens = 12;
inline constexpr int kVocoderSamplesPerToken = 480;

}  // namespace qwen_omni

// Model-specific configuration settings for Qwen2.5-Omni-3B (Thinker-Talker +
// Token2Wav) Any-to-Any multimodal models.
struct QwenOmniModelConfig {
  // Thinker LLM backbone dimensions:
  int llm_hidden_size = qwen_omni::kLlmHiddenSize;
  int llm_vocab_size = qwen_omni::kLlmVocabSize;

  // Talker speech-codec decoder dimensions:
  int tts_hidden_size = qwen_omni::kTtsHiddenSize;
  int tts_vocab_size = qwen_omni::kTtsVocabSize;
  int tts_num_layers = qwen_omni::kTtsNumLayers;
  int tts_num_kv_heads = qwen_omni::kTtsNumKvHeads;
  int tts_head_dim = qwen_omni::kTtsHeadDim;
  int tts_max_cache_len = qwen_omni::kTtsMaxCacheLen;
  int tts_text_chunk_size = qwen_omni::kTtsTextChunkSize;
  int tts_num_mel_bins = qwen_omni::kTtsNumMelBins;
  int tts_num_codebooks = qwen_omni::kTtsNumCodebooks;
  int tts_text_bos_token_id = qwen_omni::kChelsieTextBosTokenId;
  int tts_text_eos_token_id = qwen_omni::kTtsTextEosTokenId;
  int tts_text_pad_token_id = qwen_omni::kTtsTextPadTokenId;
  int tts_audio_bos_token_id = qwen_omni::kTtsAudioBosTokenId;
  int tts_eos_token_id = qwen_omni::kTtsEosTokenId;
  int tts_silence_token_id = qwen_omni::kTtsSilenceTokenId;

  // Token2Wav vocoder parameters:
  int vocoder_history_tokens = qwen_omni::kVocoderHistoryTokens;
  int vocoder_chunk_tokens = qwen_omni::kVocoderChunkTokens;
  int vocoder_lookahead_tokens = qwen_omni::kVocoderLookaheadTokens;
  int vocoder_samples_per_token = qwen_omni::kVocoderSamplesPerToken;

  // Audio and vision encoder parameters:
  int audio_input_sample_rate_hz = qwen_omni::kAudioInputSampleRateHz;
  int audio_output_sample_rate_hz = qwen_omni::kAudioOutputSampleRateHz;
  int vision_image_size = qwen_omni::kVisionImageSize;
  int vision_patch_size = qwen_omni::kVisionPatchSize;
  int vision_spatial_merge_size = qwen_omni::kVisionSpatialMergeSize;
  int image_token_id = qwen_omni::kImageTokenId;
  int video_token_id = qwen_omni::kVideoTokenId;
  int vision_bos_token_id = qwen_omni::kVisionBosTokenId;
  int vision_eos_token_id = qwen_omni::kVisionEosTokenId;
  int audio_num_mel_bins = qwen_omni::kAudioMelBins;
  int audio_chunk_frames = qwen_omni::kAudioChunkFrames;
  int audio_pool_step = qwen_omni::kAudioPoolStep;
  int audio_token_id = qwen_omni::kAudioTokenId;
  int audio_bos_token_id = qwen_omni::kAudioBosTokenId;
  int audio_eos_token_id = qwen_omni::kAudioEosTokenId;

  // Generation and sampling settings (matching
  // `Qwen2_5OmniForConditionalGeneration.generate` defaults):
  bool generate_audio = false;
  bool is_moe = false;
  int max_new_tokens = 512;
  int max_tts_tokens = 1024;
  bool do_sample = false;
  float temperature = 0.7f;
  float top_p = 0.8f;
  int top_k = 20;
  bool tts_do_sample = true;
  float tts_temperature = 0.9f;
  float tts_top_p = 0.8f;
  int tts_top_k = 40;
  float tts_repetition_penalty = 1.05f;
  std::optional<uint64_t> seed = 42;

  std::string system_prompt =
      "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
      "Group, capable of perceiving auditory and visual inputs, as well as "
      "generating text and speech.";
  std::string voice_name = "Chelsie";
  std::string language = "en-US";

  // Optional per-component backend overrides:
  std::optional<lm::Backend> vision_backend;
  std::optional<lm::Backend> audio_encoder_backend;
  std::optional<lm::Backend> llm_backend;
  std::optional<lm::Backend> tts_backend;
  std::optional<lm::Backend> vocoder_backend;
  bool force_fp32_on_gpu = false;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_QWEN_OMNI_QWEN_OMNI_MODEL_CONFIG_H_
