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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_MODEL_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_MODEL_CONFIG_H_

#include <cstdint>
#include <optional>
#include <string>

#include "runtime/executor/executor_settings_base.h"

namespace litert::omni::any2any {
namespace minicpm_o {

inline constexpr int kLlmHiddenSize = 4096;
inline constexpr int kLlmVocabSize = 151748;
inline constexpr int kTtsHiddenSize = 768;
inline constexpr int kTtsVocabSize = 6562;
inline constexpr int kTtsNumLayers = 20;
inline constexpr int kTtsMaxCacheLen = 256;
inline constexpr int kTtsTextChunkSize = 10;
inline constexpr int kTtsNumMelBins = 100;
inline constexpr int kAudioInputSampleRateHz = 16000;
inline constexpr int kAudioOutputSampleRateHz = 24000;
inline constexpr int kVisionScaleResolution = 448;
inline constexpr int kVisionPatchSize = 14;
inline constexpr int kVisionMaxSliceNums = 9;
inline constexpr int kNumImageQueries = 64;
inline constexpr int kAudioMelBins = 80;
inline constexpr int kAudioChunkFrames = 100;
inline constexpr int kAudioPoolStep = 5;
inline constexpr int kAudioTokensPerChunk = 10;
inline constexpr int kImStartTokenId = 151644;
inline constexpr int kImEndTokenId = 151645;
inline constexpr int kEosTokenId = 151643;
inline constexpr int kAudioStartTokenId = 151665;
inline constexpr int kAudioEndTokenId = 151666;
// Sentinel token ID placed in `MiniCpmOEmbeddingSegment::token_ids` and
// `MiniCpmOFrontendOutput::token_ids` for continuous audio encoder embeddings;
// never passed to the token embedder.
inline constexpr int kAudioEmbedPlaceholderTokenId = -2;
inline constexpr int kThinkStartTokenId = 151667;
inline constexpr int kThinkEndTokenId = 151668;
// Qwen3 token ID for "\n\n" emitted immediately after `</think>`.
inline constexpr int kQwenDoubleNewlineTokenId = 271;
inline constexpr int kImageStartTokenId = 151669;
inline constexpr int kImageEndTokenId = 151670;
inline constexpr int kSliceStartTokenId = 151671;
inline constexpr int kSliceEndTokenId = 151672;
inline constexpr int kImageIdStartTokenId = 151673;
inline constexpr int kImageIdEndTokenId = 151674;
// Sentinel token ID placed in `MiniCpmOEmbeddingSegment::token_ids` and
// `MiniCpmOFrontendOutput::token_ids` for continuous vision encoder embeddings;
// never passed to the token embedder.
inline constexpr int kImageEmbedPlaceholderTokenId = -1;
// Sentinel token ID placed in `MiniCpmOEmbeddingSegment::token_ids` between
// rows of a high-resolution image slice grid; replaced by the tokenized
// newline sequence in `MiniCpmOFrontendStage::FrameImageSegment`.
inline constexpr int kImageSliceRowBreakTokenId = -3;
inline constexpr int kTtsTextEosTokenId = 151672;
inline constexpr int kTtsAudioBosTokenId = 151676;
inline constexpr int kTtsBosTokenId = 151703;
inline constexpr int kTtsEosTokenId = 6561;
inline constexpr int kTtsSilenceTokenId = 4218;
inline constexpr int kVocoderHistoryTokens = 4;
inline constexpr int kVocoderChunkTokens = 25;
inline constexpr int kVocoderLookaheadTokens = 3;
inline constexpr int kVocoderInputTokens =
    kVocoderHistoryTokens + kVocoderChunkTokens + kVocoderLookaheadTokens;
inline constexpr int kVocoderSamplesPerToken = 960;
inline constexpr int kVocoderOutputSamples =
    (kVocoderHistoryTokens + kVocoderChunkTokens) * kVocoderSamplesPerToken;

}  // namespace minicpm_o

// Model-specific configuration settings for MiniCPM-o 4.5 Any2Any.
struct MiniCpmOModelConfig {
  // Hidden state dimension of the Qwen3-8B LLM backbone.
  int llm_hidden_size = minicpm_o::kLlmHiddenSize;
  // Vocabulary size of the Qwen3-8B LLM backbone.
  int llm_vocab_size = minicpm_o::kLlmVocabSize;
  // Conditioning hidden dimension of the TTS projector and acoustic decoder.
  int tts_hidden_size = minicpm_o::kTtsHiddenSize;
  // Discrete audio token vocabulary size of the TTS acoustic decoder.
  int tts_vocab_size = minicpm_o::kTtsVocabSize;
  // Number of LLaMA transformer layers in the CosyVoice2 TTS decoder.
  int tts_num_layers = minicpm_o::kTtsNumLayers;
  // Maximum KV-cache sequence length for the CosyVoice2 TTS decoder.
  int tts_max_cache_len = minicpm_o::kTtsMaxCacheLen;
  // Number of LLM-generated text tokens grouped per TTS streaming chunk.
  int tts_text_chunk_size = minicpm_o::kTtsTextChunkSize;
  // Number of mel-spectrogram frequency bins of the internal Token2Wav CFM
  // flow network, retained for proto metadata compatibility.
  int tts_num_mel_bins = minicpm_o::kTtsNumMelBins;
  // Number of historical speech tokens prepended per Token2Wav vocoder chunk.
  int vocoder_history_tokens = minicpm_o::kVocoderHistoryTokens;
  // Number of active speech tokens synthesized per Token2Wav vocoder chunk.
  int vocoder_chunk_tokens = minicpm_o::kVocoderChunkTokens;
  // Number of lookahead speech tokens appended per Token2Wav vocoder chunk.
  int vocoder_lookahead_tokens = minicpm_o::kVocoderLookaheadTokens;
  // Number of 24 kHz PCM waveform samples emitted per 25 Hz speech token.
  int vocoder_samples_per_token = minicpm_o::kVocoderSamplesPerToken;
  // Expected sample rate in Hz for input audio waveforms.
  int audio_input_sample_rate_hz = minicpm_o::kAudioInputSampleRateHz;
  // Output sample rate in Hz for vocoder-synthesized speech waveforms.
  int audio_output_sample_rate_hz = minicpm_o::kAudioOutputSampleRateHz;
  // Target resolution in pixels for each vision slice/tile.
  int vision_scale_resolution = minicpm_o::kVisionScaleResolution;
  // Spatial patch size in pixels for the SigLIP2 vision encoder.
  int vision_patch_size = minicpm_o::kVisionPatchSize;
  // Maximum number of high-resolution image slices in dynamic high-res slicing.
  int vision_max_slice_nums = minicpm_o::kVisionMaxSliceNums;
  // Number of resampler query tokens emitted per image slice.
  int num_image_queries = minicpm_o::kNumImageQueries;
  // Temporal pooling factor applied to Whisper audio encoder frames.
  int audio_pool_step = minicpm_o::kAudioPoolStep;

  // Whether to synthesize speech audio output in addition to text responses.
  bool generate_audio = false;
  // Maximum number of text tokens generated by the LLM per turn.
  int max_new_tokens = 512;
  // Maximum number of acoustic tokens/frames generated by the TTS decoder
  // (values >= 1024 are scaled up to at least 20 acoustic tokens per spoken
  // text token for long responses).
  int max_tts_tokens = 1024;
  // Whether to use stochastic sampling (top-k/top-p/temperature) instead of
  // greedy argmax decoding.
  bool do_sample = false;
  // Sampling temperature for LLM text token generation.
  float temperature = 0.7f;
  // Nucleus (top-p) cumulative probability threshold for LLM text sampling.
  float top_p = 0.8f;
  // Top-k candidate count for LLM text sampling.
  int top_k = 20;
  // Sampling temperature for TTS acoustic token generation.
  float tts_temperature = 0.8f;
  // Nucleus (top-p) cumulative probability threshold for TTS sampling.
  float tts_top_p = 0.85f;
  // Top-k candidate count for TTS acoustic token sampling.
  int tts_top_k = 25;
  // Repetition penalty applied over recent TTS audio codes (`> 1.0f`).
  float tts_repetition_penalty = 1.05f;
  // Sliding window size (in recent audio codes) for TTS repetition penalty.
  int tts_repetition_window = 16;
  // Optional random seed for reproducible stochastic sampling.
  std::optional<uint64_t> seed = 42;

  // Default system prompt prepended on the first conversation turn.
  std::string system_prompt = "You are a helpful multimodal assistant.";
  // Speaker voice identifier for TTS synthesis.
  std::string voice_name = "default";
  // Target language locale tag for prompt formatting and speech synthesis.
  std::string language = "en-US";

  // Filename of the SigLIP2 vision encoder + resampler TFLite model.
  std::string vision_encoder_file = "vision_encoder.tflite";
  // Filename of the Whisper-medium audio encoder + projector TFLite model.
  std::string audio_encoder_file = "audio_encoder.tflite";
  // Filename of the token embedding lookup TFLite model.
  std::string text_embedder_file = "text_embedder.tflite";
  // Filename of the Qwen3-8B multimodal LLM backbone TFLite model.
  std::string llm_backbone_file = "llm_backbone.tflite";
  // Filename of the LLM-to-TTS hidden state and text token projector TFLite
  // model (`(float[1, 1, llm_hidden_size], int32[1, 1]) ->
  // float[1, 1, tts_hidden_size]`).
  std::string tts_projector_file = "tts_projector.tflite";
  // Filename of the KV-cached CosyVoice2 autoregressive TTS decoder TFLite
  // model (`(cond_emb, prev_audio_code, step_idx, kv_cache) ->
  // (code_logits, new_kv_cache)`).
  std::string tts_decoder_file = "tts_decoder.tflite";
  // Filename of the 24 kHz Token2Wav (CosyVoice2 CausalMaskedDiffWithXvec
  // Conformer + DiT CFM + HiFTGenerator neural source-filter iSTFT) vocoder
  // TFLite model (`int32[1, 32] -> float[1, 27840]`).
  std::string audio_vocoder_file = "audio_vocoder.tflite";
  // Filename of the HuggingFace tokenizer JSON file.
  std::string tokenizer_file = "tokenizer.json";

  // Optional backend override for the vision encoder model.
  std::optional<lm::Backend> vision_backend;
  // Optional backend override for the audio encoder model.
  std::optional<lm::Backend> audio_encoder_backend;
  // Optional backend override for the LLM backbone and text embedder models.
  std::optional<lm::Backend> llm_backend;
  // Optional backend override for the TTS projector and decoder models.
  std::optional<lm::Backend> tts_backend;
  // Optional backend override for the audio vocoder model.
  std::optional<lm::Backend> vocoder_backend;
};

}  // namespace litert::omni::any2any

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_ANY2ANY_MINICPM_O_MINICPM_O_MODEL_CONFIG_H_
