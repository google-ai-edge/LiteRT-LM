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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_COMMON_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_COMMON_H_

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::omni::tts {
namespace qwen3_tts {

constexpr int kTtsBos = 151672;
constexpr int kTtsEos = 151673;
constexpr int kTtsPad = 151671;

constexpr int kCodecVocab = 3072;
constexpr int kCodecPad = 2148;
constexpr int kCodecBos = 2149;
constexpr int kCodecEos = 2150;
constexpr int kCodecThink = 2154;
constexpr int kCodecNoThink = 2155;
constexpr int kCodecThinkBos = 2156;
constexpr int kCodecThinkEos = 2157;

constexpr int kHiddenDim = 1024;
constexpr int kNumCodeGroups = 16;
constexpr int kFrameChunkSize = 75;
constexpr int kVocoderOverlapContext = 25;
constexpr int kAudioSampleRateHz = 24000;
constexpr float kNegInf = -1e9f;
constexpr absl::string_view kPromptTemplate =
    "<|im_start|>assistant\n%s<|im_end|>\n<|im_start|>assistant\n";
}  // namespace qwen3_tts

// Resolves a language name to its corresponding codec language ID.
//
// args
// - language: Language name string view (e.g. "english", "chinese").
//
// returns
// - Integer language ID on success, or absl::InvalidArgumentError if
// unsupported.
absl::StatusOr<int> GetLanguageId(absl::string_view language);

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_COMMON_H_
