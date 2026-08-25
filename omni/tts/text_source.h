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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TEXT_SOURCE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TEXT_SOURCE_H_

#include <string>

#include "omni/base/stage.h"

namespace litert::omni::tts {

// Abstract interface for providing raw text strings or chunks for speech
// synthesis.
//
// Subclasses (such as FileTextSource or StreamTextSource) populate the output
// deque with text strings when scheduled.
class TextSource : public SingleThreadedStageWithDeque<std::string> {
 public:
  ~TextSource() override = default;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_TEXT_SOURCE_H_
