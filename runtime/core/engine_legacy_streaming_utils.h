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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_ENGINE_LEGACY_STREAMING_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_ENGINE_LEGACY_STREAMING_UTILS_H_

#include "runtime/engine/engine_settings.h"
#include "runtime/proto/llm_metadata.pb.h"

// Helpers for creating the legacy engine from a model stream.

namespace litert::lm {

// Returns whether a streamed vision or audio model section (encoder, adapter,
// or end-of-modality model) should be cached for creating the vision or audio
// executor. Sections of disabled modalities are skipped instead.
//
// args:
//   model_type: Type of the streamed TFLiteModel section.
//   engine_settings: Engine settings indicating which modalities are enabled.
// returns:
//   True if `model_type` is a vision or audio model whose modality is enabled
//   in `engine_settings`; false otherwise (including for model types that are
//   not vision or audio models).
bool ShouldCacheStreamedModel(proto::LlmMetadata::TfLiteModelType model_type,
                              const EngineSettings& engine_settings);

// Returns the number of media embedding cache slots that an artisan executor
// should reserve for embeddings from a LiteRT audio encoder.
//
// The artisan executor's media embedding cache is allocated when it's created,
// which may be before the audio encoder section (or `LlmMetadata` setting the
// max number of tokens) is streamed, so this only depends on the model's
// header.
//
// args:
//   has_tflite_audio_encoder: Whether the model has a LiteRT audio encoder
//     section (`TF_LITE_AUDIO_ENCODER_HW`).
//   artisan_has_audio_encoder: Whether an audio encoder is part of the artisan
//     model's weights.
//   engine_settings: Current engine settings (whose max number of tokens may
//     still be 0 if `LlmMetadata` has not been applied yet).
//   default_max_num_audio_tokens: Default number of audio tokens to reserve.
// returns:
//   `default_max_num_audio_tokens` (capped at `max_num_tokens` when positive)
//   if audio is enabled and the model has a LiteRT audio encoder rather than
//   one in the artisan model's weights; 0 otherwise.
int GetNumExternalAudioTokensToReserve(bool has_tflite_audio_encoder,
                                       bool artisan_has_audio_encoder,
                                       const EngineSettings& engine_settings,
                                       int default_max_num_audio_tokens);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_ENGINE_LEGACY_STREAMING_UTILS_H_
