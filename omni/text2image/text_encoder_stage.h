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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_TEXT_ENCODER_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_TEXT_ENCODER_STAGE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/text2image/prompt_source.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::text2image {

// Conditioning embeddings produced by the text encoder stage, preserving the
// input generation metadata (`ImageGenInputMetadata`).
struct TextEncoderOutput {
  ImageGenInputMetadata metadata;
  std::vector<float> prompt_embeds;
};

// Configuration for `TextEncoderStage`.
// TODO: b/568027544 - These default values are specific to Bonsai-FLUX.2 /
// FLUX.2-klein (Qwen3 text encoder). Move these defaults to metadata later.
// Default token framing encodes the Qwen3 chat template:
// `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
struct TextEncoderConfig {
  // Input buffer indices for `TextEncoderStage`.
  struct InputIndices {
    size_t input_ids = 0;
    size_t attention_mask = 1;
  };

  // Padded sequence length expected by the text encoder model. User prompt
  // tokens exceeding `seq_len - (prefix_token_ids.size() +
  // suffix_token_ids.size())` are truncated while preserving the prefix and
  // suffix framing tokens.
  int seq_len = 256;
  // Token ID used to right-pad `input_ids` to `seq_len`
  // (151643 = `<|endoftext|>`).
  int32_t pad_token_id = 151643;
  // Optional text prefix prepended to the user prompt before tokenization.
  std::string prompt_prefix = "user\n";
  // Token IDs prepended before the tokenized prompt (151644 = `<|im_start|>`).
  std::vector<int32_t> prefix_token_ids = {151644};
  // Token IDs appended after the tokenized prompt before padding:
  // 151645 = `<|im_end|>`, 198 = `\n`, 151644 = `<|im_start|>`,
  // 77091 = `assistant`, 151667 = `<think>`, 271 = `\n\n`,
  // 151668 = `</think>`.
  std::vector<int32_t> suffix_token_ids = {151645, 198, 151644, 77091, 198,
                                           151667, 271, 151668, 271};
  // Input buffer indices for the runner.
  InputIndices input_indices;
};

// Text frontend and prompt encoder stage for text-to-image pipelines.
// Tokenizes the input prompt, applies configured prefix/suffix token framing
// and padding, and runs the text encoder `LiteRtRunner` to produce conditioning
// embeddings.
class TextEncoderStage
    : public SingleThreadedStageWithDeque<TextEncoderOutput> {
 public:
  using Config = TextEncoderConfig;
  using InputIndices = TextEncoderConfig::InputIndices;

  static absl::StatusOr<std::unique_ptr<TextEncoderStage>> Create(
      Stage<Text2ImagePrompt>* absl_nonnull prompt_source,
      std::unique_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::unique_ptr<LiteRtRunner> absl_nonnull runner,
      const Config& config = {});

  ~TextEncoderStage() override = default;

 protected:
  bool NeedScheduleInternal() const override;

  absl::Status ScheduleInternal() override;

 private:
  TextEncoderStage(Stage<Text2ImagePrompt>* absl_nonnull prompt_source,
                   std::unique_ptr<support::Tokenizer> absl_nonnull tokenizer,
                   std::unique_ptr<LiteRtRunner> absl_nonnull runner,
                   std::vector<TensorBuffer> input_buffers,
                   std::vector<TensorBuffer> output_buffers, Config config);

  Stage<Text2ImagePrompt>& prompt_source_;
  const std::unique_ptr<support::Tokenizer> absl_nonnull tokenizer_;
  const std::unique_ptr<LiteRtRunner> absl_nonnull runner_;
  std::vector<TensorBuffer> input_buffers_;
  std::vector<TensorBuffer> output_buffers_;
  const Config config_;
};

}  // namespace litert::omni::text2image

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_TEXT_ENCODER_STAGE_H_
