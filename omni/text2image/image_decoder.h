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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_IMAGE_DECODER_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_IMAGE_DECODER_H_

#include "omni/base/io_types.h"
#include "omni/base/stage.h"

namespace litert::omni::text2image {

// Base stage that decodes denoised latent representations into `Output`
// (`ImageOutput` with interleaved RGB888 pixel data).
class ImageDecoder : public SingleThreadedStageWithDeque<Output> {
 public:
  ~ImageDecoder() override = default;
};

}  // namespace litert::omni::text2image

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_IMAGE_DECODER_H_
