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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_KLEIN_TEXT_ENCODER_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_KLEIN_TEXT_ENCODER_STAGE_H_

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/text2image/prompt_source.h"
#include "omni/text2image/text_encoder_stage.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::text2image {

// Configuration for `KleinTextEncoderStage` (sharded 3-part Qwen3 text encoder
// with host-side FP16 token embedding table lookup).
struct KleinTextEncoderConfig {
  struct ShardInputIndices {
    size_t hidden = 0;          // args_0: [1, seq_len, hidden_dim]
    size_t attention_mask = 1;  // args_1: [1, num_heads, seq_len, seq_len]
    size_t cos = 2;             // args_2: [1, seq_len, head_dim]
    size_t sin = 3;             // args_3: [1, seq_len, head_dim]
  };

  // Token framing and padding configuration.
  TextEncoderConfig framing;
  // Concatenated 3-tap output embedding dimension (`3 * hidden_dim = 7680`).
  int prompt_dim = 7680;
  // RoPE base frequency theta for Qwen3 decoder layers.
  float rope_theta = 1000000.0f;
  // Input buffer indices for each of the 3 text encoder shards.
  std::array<ShardInputIndices, 3> shard_input_indices = {};
};

// Stage 1 for sharded FLUX.2-klein models:
//   1. Tokenizes and frames the input prompt to `seq_len` tokens.
//   2. Looks up initial token embeddings `[1, seq_len, hidden_dim]` from a
//      host-side FP16 embedding table (`qwen_embed_fp16.bin`).
//   3. Builds the 4D causal + padding attention mask
//      `[1, num_heads, seq_len, seq_len]` and Qwen3 rotary embeddings
//      `(cos, sin)` `[1, seq_len, head_dim]`.
//   4. Runs the 3 Qwen3 decoder shards sequentially to collect the 3 tap
//      tensors and interleaves them into `[1, seq_len, 3 * hidden_dim]`.
class KleinTextEncoderStage
    : public SingleThreadedStageWithDeque<TextEncoderOutput> {
 public:
  using Config = KleinTextEncoderConfig;
  using ShardInputIndices = KleinTextEncoderConfig::ShardInputIndices;

  // Note: `embed_table_fp16` is stored as a non-owning `absl::string_view`. The
  // caller is responsible for keeping the underlying data alive until this
  // stage is destroyed.
  static absl::StatusOr<std::unique_ptr<KleinTextEncoderStage>> Create(
      Stage<Text2ImagePrompt>* absl_nonnull prompt_source,
      std::unique_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::array<std::unique_ptr<LiteRtRunner>, 3> shard_runners,
      absl::string_view embed_table_fp16, const Config& config = {});

  ~KleinTextEncoderStage() override = default;

 protected:
  bool NeedScheduleInternal() const override;

  absl::Status ScheduleInternal() override;

 private:
  KleinTextEncoderStage(
      Stage<Text2ImagePrompt>* absl_nonnull prompt_source,
      std::unique_ptr<support::Tokenizer> absl_nonnull tokenizer,
      std::array<std::unique_ptr<LiteRtRunner>, 3> shard_runners,
      std::array<std::vector<TensorBuffer>, 3> shard_input_buffers,
      std::array<std::vector<TensorBuffer>, 3> shard_output_buffers,
      absl::string_view embed_table_fp16, Config config, int hidden_dim,
      int num_heads);

  Stage<Text2ImagePrompt>& prompt_source_;
  const std::unique_ptr<support::Tokenizer> absl_nonnull tokenizer_;
  std::array<std::unique_ptr<LiteRtRunner>, 3> shard_runners_;
  std::array<std::vector<TensorBuffer>, 3> shard_input_buffers_;
  std::array<std::vector<TensorBuffer>, 3> shard_output_buffers_;
  const absl::string_view embed_table_fp16_;
  const Config config_;
  const int hidden_dim_;
  const int num_heads_;
};

}  // namespace litert::omni::text2image

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_KLEIN_TEXT_ENCODER_STAGE_H_
