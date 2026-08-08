// Copyright (C) 2026 Samsung Electronics Co. LTD.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_LFM2_DATA_PROCESSOR_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_LFM2_DATA_PROCESSOR_CONFIG_H_

#include <string>
#include <vector>

namespace litert::lm {

// Config for LFM2DataProcessor.
struct Lfm2DataProcessorConfig {
  // The string for beginning of image token.
  std::string boi_token = "<|image_start|>";
  // The string for end of image token.
  std::string eoi_token = "<|image_end|>";

  // The patch width that the image preprocessor should patchify the image to.
  int patch_width = 16;
  // The patch height that the image preprocessor should patchify the image to.
  int patch_height = 16;
  // The maximum number of patches that the image preprocessor should patchify
  // the image to.
  int max_num_patches = 1024;
  // The pooling kernel size that the image preprocessor should use for
  // patchifying the image.
  int pooling_kernel_size = 2;

  // The dimensions the image preprocessor should resize the image to.
  int image_height = 512;
  int image_width = 512;

  // Per-channel mean/std for image normalization. The image is normalized as
  // (pixel * rescale_factor - mean) / std.
  std::vector<float> normalization_mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> normalization_std = {0.5f, 0.5f, 0.5f};
  float normalization_rescale_factor = 1.0f / 255.0f;
};

struct Lfm2DataProcessorArguments {};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_MODEL_DATA_PROCESSOR_LFM2_DATA_PROCESSOR_CONFIG_H_
