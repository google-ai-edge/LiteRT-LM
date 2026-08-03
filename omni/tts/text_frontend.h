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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TEXT_FRONTEND_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TEXT_FRONTEND_H_

#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "omni/base/stage.h"

namespace litert::omni::tts {

// Generic frontend output payload for text frontend.
// TODO: b/538727793. Refactor the outputs to a more generic structure.
// Currently these are tailored for qwen3-tts model.
struct FrontendOutput {
  std::vector<int> token_ids;
  std::vector<float> prompt_embeddings;
  int prompt_len = 0;
  std::vector<float> trailing_embeddings;
  int trailing_len = 0;
  std::vector<float> tts_pad_embedding;
};

// Abstract interface for Stage 2: Text preprocessing, tokenization, and
// embedding generation.
class TextFrontend : public SingleThreadedStageWithDeque<FrontendOutput> {
 public:
  using FrontendOutput = ::litert::omni::tts::FrontendOutput;

  explicit TextFrontend(Stage<std::string>* absl_nonnull text_source)
      : text_source_(*text_source) {}

  ~TextFrontend() override = default;

  // Resets internal state for a new TTS stream.
  virtual void Reset() = 0;

 protected:
  bool NeedScheduleInternal() const override {
    return text_source_.HasOutput();
  }

  Stage<std::string>& text_source_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TEXT_FRONTEND_H_
