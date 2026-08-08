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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_STAGE_OPTIONS_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_STAGE_OPTIONS_H_

#include <cstdint>
#include <optional>
#include <string>

namespace litert::omni::tts {

struct Qwen3StageOptions {
  std::string model_dir;
  std::string cache_dir;
  // TODO b/538727793 the following fields are fixed for now, will introduce
  // model type specific settings later.
  std::string talker_file = "talker_int4.tflite";
  std::string mtp_file = "mtp_fp32.tflite";
  std::string codec_file = "codec_decoder_fp32.tflite";
  std::string text_embedding_file = "text_embedding.tflite";
  std::string text_projection_file = "text_projection.tflite";
  std::string codec_embedding_file = "codec_embedding.tflite";
  std::string mtp_embedding_file = "mtp_embedding.tflite";
  std::string tokenizer_file = "tokenizer.json";
  std::string speaker_file = "voices/demo_speaker.bin";
  int num_threads = 4;
  int mtp_threads = 2;
  std::string language = "english";
  bool do_sample = true;
  int top_k = 50;
  float temperature = 0.9f;
  float repetition_penalty = 1.05f;
  int max_frames = 512;
  std::optional<uint64_t> seed = 1;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_STAGE_OPTIONS_H_
