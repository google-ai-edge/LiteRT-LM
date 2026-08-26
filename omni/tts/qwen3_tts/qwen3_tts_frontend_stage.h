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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_FRONTEND_STAGE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_FRONTEND_STAGE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/tts/qwen3_tts/qwen3_tts_io_types.h"
#include "omni/tts/qwen3_tts/qwen3_tts_model_config.h"
#include "support/tokenizer/huggingface_tokenizer.h"

namespace litert::omni::tts {

// Stage 1: Text Frontend & Prompt Framing for Qwen3-TTS
class Qwen3TtsFrontendStage
    : public SingleThreadedStageWithDeque<Qwen3TtsFrontendOutput> {
 public:
  // Creates a Qwen3TtsFrontendStage instance and initializes its resources.
  //
  // args
  // - text_source: Stage that provides the input text.
  // - model_dir: Path to directory containing tokenizer and speaker files.
  // - config: Model-specific configuration for Qwen3-TTS.
  // - resources: Shared ModelResources container with compiled models.
  //
  // returns
  // - Unique pointer to the created Qwen3TtsFrontendStage on success, or an
  // error
  //   status on failure.
  static absl::StatusOr<std::unique_ptr<Qwen3TtsFrontendStage>> Create(
      Stage<std::string>* absl_nonnull text_source,
      const std::string& model_dir,
      const Qwen3TtsModelConfig& config,
      std::shared_ptr<ModelResources> absl_nonnull resources);

  ~Qwen3TtsFrontendStage() override = default;

  // Resets the stage state back to idle and clears any pending outputs.
  void Reset() override;

 protected:
  bool NeedScheduleInternal() const override {
    return text_source_.HasOutput();
  }

  // Executes one step of frontend stage processing asynchronously.
  //
  // returns
  // - absl::OkStatus() on success, or error status on failure.
  absl::Status ScheduleInternal() override;

 private:
  Qwen3TtsFrontendStage(Stage<std::string>* absl_nonnull text_source,
                        Qwen3TtsModelConfig config,
                        std::shared_ptr<ModelResources> absl_nonnull resources)
      : text_source_(*text_source),
        config_(std::move(config)),
        resources_(std::move(resources)) {}

  Stage<std::string>& text_source_;

  // Embeds a single audio codec token ID using the codec embedding model.
  //
  // args
  // - code_id: Codec token ID to embed.
  //
  // returns
  // - Embedding float vector of size `kHiddenDim` on success, or error status.
  absl::StatusOr<std::vector<float>> EmbedCodecToken(int code_id);

  // Embeds a sequence of text token IDs using text embedding and projection.
  //
  // args
  // - ids: Vector of text token IDs to embed.
  //
  // returns
  // - Projected embedding float vector of size `ids.size() * kHiddenDim` on
  //   success, or error status.
  absl::StatusOr<std::vector<float>> EmbedText(const std::vector<int>& ids);

  // Projects text embedding rows to hidden dimension space using the text
  // projection model.
  //
  // args
  // - rows: Unprojected embedding float data rows.
  // - num_rows: Number of embedding rows to project.
  //
  // returns
  // - Projected embedding vector of size `num_rows * kHiddenDim` on success,
  //   or error status.
  absl::StatusOr<std::vector<float>> ProjectText(const std::vector<float>& rows,
                                                 int num_rows);

  // Tokenizes input text and constructs prompt and trailing embeddings.
  //
  // args
  // - input_text: Raw input text string to frame into prompt embeddings.
  //
  // returns
  // - Qwen3TtsFrontendOutput containing prompt and trailing embeddings on
  // success,
  //   or error status.
  absl::StatusOr<Qwen3TtsFrontendOutput> BuildPrompt(
      const std::string& input_text);

  // Appends the elementwise sum of two float spans to an output vector.
  //
  // args
  // - a: First float span operand.
  // - b: Second float span operand.
  // - out: Vector to append elementwise sums onto.
  void AppendSum(absl::Span<const float> a, absl::Span<const float> b,
                 std::vector<float>& out);

  // Embeds a codec token ID and appends the elementwise sum of base embedding
  // and codec embedding to an output vector.
  //
  // args
  // - base: Base embedding span operand.
  // - code_id: Codec token ID to embed and sum with base.
  // - out: Vector to append elementwise sums onto.
  //
  // returns
  // - absl::OkStatus() on success, or error status on codec embedding lookup
  //   failure.
  absl::Status AppendCodecSum(absl::Span<const float> base, int code_id,
                              std::vector<float>& out);

  Qwen3TtsModelConfig config_;
  std::shared_ptr<ModelResources> resources_;
  std::unique_ptr<support::HuggingFaceTokenizer> tokenizer_;

  // Required compiled models for text frontend stage
  std::shared_ptr<CompiledModel> text_embedding_model_;
  std::shared_ptr<CompiledModel> text_projection_model_;
  std::shared_ptr<CompiledModel> codec_embedding_model_;

  // Pre-allocated TensorBuffers initialized once to avoid runtime allocation
  std::vector<TensorBuffer> text_emb_input_buffers_;
  std::vector<TensorBuffer> text_emb_output_buffers_;
  std::vector<TensorBuffer> text_proj_input_buffers_;
  std::vector<TensorBuffer> text_proj_output_buffers_;
  std::vector<TensorBuffer> codec_emb_input_buffers_;
  std::vector<TensorBuffer> codec_emb_output_buffers_;

  // Precomputed system embeddings for fixed tokens (kTtsBos, kTtsEos, kTtsPad)
  std::vector<float> tts_bos_;
  std::vector<float> tts_eos_;
  std::vector<float> tts_pad_;

  std::vector<float> speaker_emb_;
};

}  // namespace litert::omni::tts

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TTS_QWEN3_TTS_QWEN3_TTS_FRONTEND_STAGE_H_
