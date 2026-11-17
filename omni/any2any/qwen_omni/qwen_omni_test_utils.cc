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

#include "omni/any2any/qwen_omni/qwen_omni_test_utils.h"

#include "omni/any2any/qwen_omni/qwen_omni_model_config.h"

namespace litert::omni::any2any {

QwenOmniModelConfig MakeSmallQwenOmniTestConfig(bool generate_audio) {
  QwenOmniModelConfig config;
  config.generate_audio = generate_audio;
  config.llm_hidden_size = 4;
  config.llm_vocab_size = generate_audio ? 28 : 32;
  config.max_new_tokens = 1;
  config.do_sample = false;
  config.tts_hidden_size = 4;
  config.tts_vocab_size = 8;
  config.tts_num_layers = 1;
  config.tts_max_cache_len = 64;
  config.max_tts_tokens = 4;
  config.tts_do_sample = false;
  config.tts_temperature = 0.0f;
  config.tts_silence_token_id = 5;
  config.tts_audio_bos_token_id = 6;
  config.tts_eos_token_id = 7;
  config.tts_text_pad_token_id = 14;
  config.tts_text_bos_token_id = 15;
  config.tts_text_eos_token_id = 16;
  config.audio_num_mel_bins = 128;
  config.audio_chunk_frames = 10;
  config.audio_pool_step = 2;
  config.audio_bos_token_id = 6;
  config.audio_eos_token_id = 7;
  config.audio_token_id = 10;
  config.vision_image_size = 28;
  config.image_token_id = 11;
  config.video_token_id = 12;
  config.vision_bos_token_id = 8;
  config.vision_eos_token_id = 9;
  config.vocoder_history_tokens = 1;
  config.vocoder_chunk_tokens = 2;
  config.vocoder_lookahead_tokens = 1;
  config.vocoder_samples_per_token = 4;
  return config;
}

}  // namespace litert::omni::any2any
