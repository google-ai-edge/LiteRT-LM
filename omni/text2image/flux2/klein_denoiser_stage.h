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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_KLEIN_DENOISER_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_KLEIN_DENOISER_STAGE_H_

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/text2image/flux2/flux2_denoiser_stage.h"
#include "omni/text2image/flux2/flux2_model_config.h"
#include "omni/text2image/text_encoder_stage.h"

namespace litert::omni::text2image {

// Input buffer indices for the 8 sharded FLUX.2-klein DiT subgraphs.
struct KleinDenoiserInputIndices {
  struct Initial {
    size_t hidden = 0;  // args_0: [1, tokens, packed_ch]
    size_t enc = 1;     // args_1: [1, seq_len, prompt_dim]
    size_t t = 2;       // args_2: [1]
  };
  struct DoubleBlock {
    size_t image = 0;    // args_0: [1, tokens, dit_hidden_dim]
    size_t text = 1;     // args_1: [1, seq_len, dit_hidden_dim]
    size_t cos = 2;      // args_2: [1, seq_len + tokens, 1, head_dim]
    size_t sin = 3;      // args_3: [1, seq_len + tokens, 1, head_dim]
    size_t mod_img = 4;  // args_4: [1, 6 * dit_hidden_dim]
    size_t mod_txt = 5;  // args_5: [1, 6 * dit_hidden_dim]
  };
  struct SingleBlock {
    size_t joint = 0;       // args_0: [1, seq_len + tokens, dit_hidden_dim]
    size_t cos = 1;         // args_1: [1, seq_len + tokens, 1, head_dim]
    size_t sin = 2;         // args_2: [1, seq_len + tokens, 1, head_dim]
    size_t mod_single = 3;  // args_3: [1, 3 * dit_hidden_dim]
  };
  struct Final {
    size_t joint = 0;  // args_0: [1, seq_len + tokens, dit_hidden_dim]
    size_t temb = 1;   // args_1: [1, dit_hidden_dim] (or [1] timestep)
  };

  Initial initial;
  std::array<DoubleBlock, 2> double_blocks = {};
  std::array<SingleBlock, 4> single_blocks = {};
  Final final_stage;
};

// Stage 2 for sharded FLUX.2-klein models:
// Orchestrates the 8 DiT subgraphs (`dit_initial`, `dit_double_block_0..1`,
// `dit_single_block_0..3`, `dit_final`) across the FlowMatch Euler integration
// loop to produce denoised packed latents `[1, tokens, packed_ch]`.
class KleinDenoiserStage
    : public SingleThreadedStageWithDeque<Flux2DenoiserOutput> {
 public:
  using InputIndices = KleinDenoiserInputIndices;

  struct Runners {
    std::unique_ptr<LiteRtRunner> initial;
    std::array<std::unique_ptr<LiteRtRunner>, 2> double_blocks;
    std::array<std::unique_ptr<LiteRtRunner>, 4> single_blocks;
    std::unique_ptr<LiteRtRunner> final_stage;
  };

  static absl::StatusOr<std::unique_ptr<KleinDenoiserStage>> Create(
      Stage<TextEncoderOutput>* absl_nonnull text_encoder,
      const Flux2ModelConfig& config, Runners runners,
      InputIndices input_indices = {});

  ~KleinDenoiserStage() override = default;

 protected:
  bool NeedScheduleInternal() const override;

  absl::Status ScheduleInternal() override;

 private:
  struct Buffers {
    std::vector<TensorBuffer> initial_in;
    std::vector<TensorBuffer> initial_out;
    std::array<std::vector<TensorBuffer>, 2> double_in;
    std::array<std::vector<TensorBuffer>, 2> double_out;
    std::array<std::vector<TensorBuffer>, 4> single_in;
    std::array<std::vector<TensorBuffer>, 4> single_out;
    std::vector<TensorBuffer> final_in;
    std::vector<TensorBuffer> final_out;
  };

  KleinDenoiserStage(Stage<TextEncoderOutput>* absl_nonnull text_encoder,
                     Flux2ModelConfig config, Runners runners,
                     InputIndices input_indices, Buffers buffers,
                     int dit_hidden_dim, bool initial_outputs_temb);

  Stage<TextEncoderOutput>& text_encoder_;
  const Flux2ModelConfig config_;
  Runners runners_;
  const InputIndices input_indices_;
  Buffers buffers_;
  const int dit_hidden_dim_;
  const bool initial_outputs_temb_;
};

}  // namespace litert::omni::text2image

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2IMAGE_FLUX2_KLEIN_DENOISER_STAGE_H_
