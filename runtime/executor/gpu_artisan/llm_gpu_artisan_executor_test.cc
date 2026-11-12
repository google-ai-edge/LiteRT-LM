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

#include "runtime/executor/gpu_artisan/llm_gpu_artisan_executor.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/common/tensor.h"  // from @ml_drift
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/bitmap.h"
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/gpu_artisan/ml_drift/llm_gpu_runner.h"
#include "runtime/executor/gpu_artisan/ml_drift/speculative_utils.h"
#include "runtime/executor/gpu_artisan/ml_drift/tensor_loaders/llm_tensor_loader.h"
#include "runtime/executor/gpu_artisan/ml_drift/transformer.h"
#include "runtime/executor/gpu_artisan/proto/sampler_params.pb.h"
#include "runtime/executor/gpu_artisan/utils/bitmap.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "support/util/convert_tensor_buffer.h"

namespace odml::infra {
namespace {

using ::litert::lm::Backend;
using ::litert::lm::ConstrainedDecoder;
using ::litert::lm::ExecutorAudioData;
using ::litert::lm::ExecutorDecodeParams;
using ::litert::lm::ExecutorInputs;
using ::litert::lm::ExecutorPrefillParams;
using ::litert::lm::ExecutorTextData;
using ::litert::lm::ExecutorVisionData;
using ::litert::lm::LlmExecutorSettings;
using ::litert::lm::ModelAssets;
using ::litert::support::CopyToTensorBuffer;
using ::ml_drift::speculative_utils::TokenAtTimeStep;
using ::testing::ElementsAre;

constexpr int kVocabSize = 16;

// A fake LlmGpuRunner that emulates the 1-token-lookahead pipelining of
// LlmWebGpu / LlmMetal (the runners used by Chrome on Windows/Linux/macOS):
//
//  * InitInputTokens() (prefill) runs post-processing right away and samples
//    the *first output token* under whatever constraint mask is currently
//    uploaded (ready_tokens_ == 1 in the real runners).
//  * PeekNextToken() exposes that already-sampled, not-yet-returned token.
//  * GetNextTokens() returns the pending token and immediately samples the
//    following one under the mask uploaded at that time.
//
// "Sampling" is greedy and fully deterministic: the fake model always prefers
// kModelPreferredToken (think of it as the `{"` token). If the mask disallows
// it, the lowest allowed token id wins.
class FakeLookaheadGpuRunner : public gpu::LlmGpuRunner {
 public:
  static constexpr int kModelPreferredToken = 7;

  gpu::LlmGpuRunnerType GetLlmGpuRunnerType() override {
    return gpu::LlmGpuRunnerType::kWebGpu;
  }

  absl::Status InitInputTokens(const std::vector<int>& input_tokens,
                               int current_step, bool wait_for_completion,
                               const std::atomic_bool* cancel) override {
    ++num_prefill_calls_;
    prefilled_tokens_.push_back(input_tokens);
    prefill_wait_for_completion_.push_back(wait_for_completion);
    time_step_ = current_step + input_tokens.size();
    pending_token_ = Sample();
    return absl::OkStatus();
  }

  // Emulates the media embedding cache of the real runners, which holds at
  // most `max_num_images_` images.
  absl::Status FillVisionEmbeddings(
      const ml_drift::TensorFloat32& vision_embedding_tensor,
      const std::vector<ml_drift::TensorFloat32>& vision_ple_tensors,
      int image_index) override {
    if (image_index < 0 || image_index >= max_num_images_) {
      return absl::InvalidArgumentError(
          absl::StrCat("Image index ", image_index, " must be less than ",
                       max_num_images_, "."));
    }
    filled_image_indices_.push_back(image_index);
    filled_image_first_values_.push_back(vision_embedding_tensor.data[0]);
    return absl::OkStatus();
  }

  absl::Status FillAudioEmbeddingsWithTensorBuffer(
      const ::litert::TensorBuffer& audio_tensor_buffer, int index) override {
    filled_audio_offsets_.push_back(index);
    auto tensor_type = audio_tensor_buffer.TensorType();
    if (!tensor_type.HasValue()) return absl::InternalError("TensorType");
    const auto dims = tensor_type->Layout().Dimensions();
    auto values =
        ::litert::lm::ReferTensorBufferAsSpan<float>(audio_tensor_buffer);
    if (!values.HasValue()) return absl::InternalError("Refer");
    filled_audio_num_tokens_.push_back(values->size() / dims.back());
    filled_audio_first_values_.push_back((*values)[0]);
    return absl::OkStatus();
  }

  absl::StatusOr<TokenAtTimeStep> PeekNextToken() override {
    return TokenAtTimeStep(time_step_, pending_token_);
  }

  absl::Status GetNextTokens(std::vector<std::vector<int>>* output_tokens,
                             UpdateConstraintFn update_constraint_fn) override {
    if (output_tokens->size() != 1 || (*output_tokens)[0].size() != 1) {
      return absl::UnimplementedError("Fake only supports 1x1 outputs.");
    }
    (*output_tokens)[0][0] = pending_token_;
    ++time_step_;
    pending_token_ = Sample();
    return absl::OkStatus();
  }

  absl::Status UpdateConstraintMask(
      const llm_utils::Bitmap& constraint_bitmap) override {
    ++num_mask_updates_;
    for (int i = 0; i < kVocabSize; ++i) {
      allowed_[i] = constraint_bitmap.Get(i);
    }
    return absl::OkStatus();
  }

  // Unused pure virtuals.
  absl::Status CleanCache(int time_step,
                          bool skip_ringbuffer_delta_check) override {
    return absl::OkStatus();
  }
  absl::Status LoadLoRA(
      uint32_t lora_id, int lora_rank,
      std::unique_ptr<gpu::LlmTensorLoader> tensor_loader) override {
    return absl::UnimplementedError("LoadLoRA");
  }
  absl::Status UnloadLoRA(uint32_t lora_id) override {
    return absl::UnimplementedError("UnloadLoRA");
  }
  absl::Status UseLoRA(std::optional<uint32_t> lora_id) override {
    return absl::OkStatus();
  }
  absl::Status UseLayers(const absl::flat_hash_set<int>& layers,
                         int final_ln) override {
    return absl::OkStatus();
  }
  absl::Status UpdateSamplerParams(
      const proto::SamplerParameters& new_params,
      std::shared_ptr<std::default_random_engine> rand_gen) override {
    return absl::OkStatus();
  }
  int GetMaxTopK() override { return 1; }

  int num_prefill_calls_ = 0;
  int num_mask_updates_ = 0;
  int max_num_images_ = 0;
  std::vector<std::vector<int>> prefilled_tokens_;
  std::vector<bool> prefill_wait_for_completion_;
  std::vector<int> filled_image_indices_;
  std::vector<float> filled_image_first_values_;
  std::vector<int> filled_audio_offsets_;
  std::vector<int> filled_audio_num_tokens_;
  std::vector<float> filled_audio_first_values_;

 private:
  int Sample() const {
    if (allowed_[kModelPreferredToken]) return kModelPreferredToken;
    for (int i = 0; i < kVocabSize; ++i) {
      if (allowed_[i]) return i;
    }
    return -1;
  }

  std::vector<bool> allowed_ = std::vector<bool>(kVocabSize, true);
  int pending_token_ = -1;
  int time_step_ = 0;
};

// Forces a fixed token sequence. Unlike FakeConstraint, ComputeNext() rejects
// a token that is not allowed in the current state, exactly like llguidance's
// llg_commit_token() does ("token ... doesn't satisfy the grammar").
class StrictSequenceConstraint : public ::litert::lm::Constraint {
 public:
  class SeqState : public State {
   public:
    explicit SeqState(int index) : index_(index) {}
    const int index_;
  };

  explicit StrictSequenceConstraint(std::vector<int> token_ids)
      : token_ids_(std::move(token_ids)) {}

  std::unique_ptr<State> Start() const override {
    return std::make_unique<SeqState>(0);
  }
  bool IsEnded(const State& state) const override {
    return Index(state) >= token_ids_.size();
  }
  int GetVocabularySize() const override { return kVocabSize; }

  absl::StatusOr<std::unique_ptr<State>> ComputeNext(const State& state,
                                                     int token) const override {
    const int index = Index(state);
    if (index >= token_ids_.size()) {
      return absl::InvalidArgumentError("Constraint already ended.");
    }
    if (token_ids_[index] != token) {
      return absl::InternalError(absl::StrCat(
          "token ", token, " doesn't satisfy the grammar; expected ",
          token_ids_[index], " at step ", index));
    }
    return std::make_unique<SeqState>(index + 1);
  }

  absl::StatusOr<std::unique_ptr<::litert::lm::Bitmap>> ComputeBitmap(
      const State& state) const override {
    const int index = Index(state);
    if (index >= token_ids_.size()) {
      return std::make_unique<::litert::lm::AllAllowedBitmap>();
    }
    return std::make_unique<::litert::lm::SingleAllowedTokenBitmap>(
        token_ids_[index]);
  }

 private:
  static int Index(const State& state) {
    return static_cast<const SeqState&>(state).index_;
  }
  const std::vector<int> token_ids_;
};

class LlmGpuArtisanExecutorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto runner = std::make_unique<FakeLookaheadGpuRunner>();
    runner_ = runner.get();
    ASSERT_OK_AND_ASSIGN(auto model_assets,
                         ModelAssets::Create("/path/to/fake/model"));
    ASSERT_OK_AND_ASSIGN(
        auto settings,
        LlmExecutorSettings::CreateDefault(model_assets, Backend::GPU_ARTISAN));
    ml_drift::TransformerConfig text_config{};
    text_config.vocabulary_size = kVocabSize;
    executor_ = std::make_unique<LlmGpuArtisanExecutor>(
        settings, std::move(runner), text_config);
  }

  // Prefills with no constraint attached.
  absl::Status Prefill(const std::vector<int>& ids) {
    auto tensor = CopyToTensorBuffer<int32_t>(
        absl::MakeConstSpan(ids), {1, static_cast<int>(ids.size())});
    if (!tensor.HasValue()) return absl::InternalError("CopyToTensorBuffer");
    ExecutorInputs inputs(ExecutorTextData(std::move(*tensor)), std::nullopt,
                          std::nullopt);
    ExecutorPrefillParams params;
    params.SetWaitForCompletion(true);
    return executor_->Prefill(inputs, params);
  }

  // Decodes with a ConstrainedDecoder attached.
  absl::StatusOr<std::vector<int>> ConstrainedDecode(
      const std::vector<int>& forced) {
    StrictSequenceConstraint constraint(forced);
    ABSL_ASSIGN_OR_RETURN(auto decoder,
                          ConstrainedDecoder::CreateForHost(&constraint));
    ExecutorDecodeParams params;
    params.SetConstrainedDecoder(decoder.get());
    std::vector<int> out;
    for (int i = 0; i < forced.size(); ++i) {
      ABSL_ASSIGN_OR_RETURN(auto tokens, executor_->Decode(params));
      out.push_back(tokens[0][0]);
    }
    return out;
  }

  // Runs Turn 0: an unconstrained prefill followed by a constrained decode.
  // On the first constrained turn, `Decode()` sees a null `constraint_state`,
  // uploads the initial constraint mask, and re-prefills the last input token
  // so that the lookahead token is re-sampled under the constraint.
  void RunFirstConstrainedTurn() {
    ASSERT_OK(Prefill({10, 11, 12}));
    EXPECT_EQ(runner_->num_prefill_calls_, 1);

    ASSERT_OK_AND_ASSIGN(auto tokens, ConstrainedDecode({2, 3, 1}));
    ASSERT_THAT(tokens, ElementsAre(2, 3, 1));
    // The last input token was re-prefilled to resample the lookahead token.
    EXPECT_EQ(runner_->num_prefill_calls_, 2);
  }

  FakeLookaheadGpuRunner* runner_ = nullptr;
  std::unique_ptr<LlmGpuArtisanExecutor> executor_;
};

// After a finished constrained turn, an unconstrained prefill followed by a
// new constrained decode must constrain the *first* token of the new turn.
TEST_F(LlmGpuArtisanExecutorTest,
       SecondConstrainedTurnAfterUnconstrainedPrefillIsConstrained) {
  RunFirstConstrainedTurn();

  // Turn 1: unconstrained prefill samples the lookahead token `7` with an
  // all-allowed mask.
  ASSERT_OK(Prefill({13, 14}));
  const int prefills_before_decode = runner_->num_prefill_calls_;

  // Now Decode must re-sample it under the new constraint.
  absl::StatusOr<std::vector<int>> turn1 = ConstrainedDecode({2, 3, 1});
  EXPECT_OK(turn1);
  if (turn1.ok()) {
    EXPECT_THAT(*turn1, ElementsAre(2, 3, 1));
  }
  EXPECT_EQ(runner_->num_prefill_calls_, prefills_before_decode + 1)
      << "The lookahead token was not re-sampled under the new constraint.";
}

// Same as above but with a different constraint on the second turn.
TEST_F(LlmGpuArtisanExecutorTest, SecondTurnUsesItsOwnConstraint) {
  RunFirstConstrainedTurn();

  ASSERT_OK(Prefill({13, 14}));
  absl::StatusOr<std::vector<int>> turn1 = ConstrainedDecode({4, 5, 1});
  EXPECT_OK(turn1);
  if (turn1.ok()) {
    EXPECT_THAT(*turn1, ElementsAre(4, 5, 1));
  }
}

// A turn that is cut short (e.g. max_output_tokens / cancellation) must not
// leak its mid-grammar state into the next turn.
TEST_F(LlmGpuArtisanExecutorTest, TruncatedTurnDoesNotLeakStateIntoNextTurn) {
  ASSERT_OK(Prefill({10, 11, 12}));
  {
    StrictSequenceConstraint constraint({2, 3, 1});
    ASSERT_OK_AND_ASSIGN(auto decoder,
                         ConstrainedDecoder::CreateForHost(&constraint));
    ExecutorDecodeParams params;
    params.SetConstrainedDecoder(decoder.get());
    // Only one of the three steps: the constraint state is left at index 1.
    ASSERT_OK_AND_ASSIGN(auto tokens, executor_->Decode(params));
    ASSERT_THAT(tokens[0], ElementsAre(2));
  }

  ASSERT_OK(Prefill({13, 14}));
  absl::StatusOr<std::vector<int>> turn1 = ConstrainedDecode({2, 3, 1});
  EXPECT_OK(turn1);
  if (turn1.ok()) {
    EXPECT_THAT(*turn1, ElementsAre(2, 3, 1));
  }
}

constexpr int kVisionTokensPerImage = 3;
constexpr int kModelDimension = 2;
constexpr int kVision = ExecutorVisionData::kSpecialToken;
constexpr int kVisionEnd = ExecutorVisionData::kEndToken;
constexpr int kAudio = ExecutorAudioData::kSpecialToken;
constexpr float kEndOfVisionValue = -100.0f;

class LlmGpuArtisanExecutorVisionTest : public ::testing::Test {
 protected:
  void CreateExecutor(int max_num_images) {
    auto runner = std::make_unique<FakeLookaheadGpuRunner>();
    runner->max_num_images_ = max_num_images;
    runner_ = runner.get();
    ASSERT_OK_AND_ASSIGN(auto model_assets,
                         ModelAssets::Create("/path/to/fake/model"));
    ASSERT_OK_AND_ASSIGN(
        auto settings,
        LlmExecutorSettings::CreateDefault(model_assets, Backend::GPU_ARTISAN));
    ml_drift::TransformerConfig text_config{};
    text_config.vocabulary_size = kVocabSize;
    text_config.vision_tokens_num = kVisionTokensPerImage;
    text_config.model_dimension = kModelDimension;
    text_config.max_num_images = max_num_images;
    executor_ = std::make_unique<LlmGpuArtisanExecutor>(
        settings, std::move(runner), text_config,
        std::vector<float>(kModelDimension, kEndOfVisionValue));
  }

  // Prefills `ids` with vision embeddings for every kVision token. The
  // embedding of the i-th vision token is filled with the value `i`.
  absl::Status PrefillWithVision(const std::vector<int>& ids,
                                 bool wait_for_completion = false) {
    return PrefillWithMedia(ids, wait_for_completion);
  }

  // Prefills `ids` with vision embeddings for every kVision token (if any) and
  // projected audio embeddings for every kAudio token (if any). The embedding
  // of the i-th vision token and of the i-th audio token is filled with the
  // value `i`.
  absl::Status PrefillWithMedia(const std::vector<int>& ids,
                                bool wait_for_completion = false) {
    const int num_vision_tokens = std::count(ids.begin(), ids.end(), kVision);
    std::vector<float> embeddings;
    for (int i = 0; i < num_vision_tokens; ++i) {
      embeddings.insert(embeddings.end(), kModelDimension, i);
    }
    auto ids_tensor = CopyToTensorBuffer<int32_t>(
        absl::MakeConstSpan(ids), {1, static_cast<int>(ids.size())});
    if (!ids_tensor.HasValue()) return absl::InternalError("ids tensor");
    auto embeddings_tensor =
        CopyToTensorBuffer<float>(absl::MakeConstSpan(embeddings),
                                  {1, num_vision_tokens, kModelDimension});
    if (!embeddings_tensor.HasValue()) {
      return absl::InternalError("embeddings tensor");
    }
    std::optional<ExecutorVisionData> vision_data;
    if (num_vision_tokens > 0) {
      vision_data =
          ExecutorVisionData(std::move(*embeddings_tensor), std::nullopt);
    }
    std::optional<ExecutorAudioData> audio_data;
    const int num_audio_tokens = std::count(ids.begin(), ids.end(), kAudio);
    if (num_audio_tokens > 0) {
      std::vector<float> audio_embeddings;
      for (int i = 0; i < num_audio_tokens; ++i) {
        audio_embeddings.insert(audio_embeddings.end(), kModelDimension, i);
      }
      auto audio_tensor =
          CopyToTensorBuffer<float>(absl::MakeConstSpan(audio_embeddings),
                                    {1, num_audio_tokens, kModelDimension});
      if (!audio_tensor.HasValue()) return absl::InternalError("audio tensor");
      audio_data = ExecutorAudioData(std::move(*audio_tensor), std::nullopt);
    }
    ExecutorInputs inputs(ExecutorTextData(std::move(*ids_tensor)),
                          std::move(vision_data), std::move(audio_data));
    ExecutorPrefillParams params;
    params.SetWaitForCompletion(wait_for_completion);
    return executor_->Prefill(inputs, params);
  }

  FakeLookaheadGpuRunner* runner_ = nullptr;
  std::unique_ptr<LlmGpuArtisanExecutor> executor_;
};

TEST_F(LlmGpuArtisanExecutorVisionTest, ImagesWithinLimitArePrefilledAtOnce) {
  CreateExecutor(/*max_num_images=*/2);

  ASSERT_OK(PrefillWithVision(
      {10, kVision, kVision, kVisionEnd, 11, kVision, kVisionEnd, 12}));

  EXPECT_THAT(runner_->filled_image_indices_, ElementsAre(0, 1));
  // The first vision token of each image.
  EXPECT_THAT(runner_->filled_image_first_values_, ElementsAre(0.0f, 2.0f));
  EXPECT_THAT(runner_->prefilled_tokens_,
              ElementsAre(ElementsAre(10, -1, -2, -3, 11, -4, -5, 12)));
  EXPECT_THAT(runner_->prefill_wait_for_completion_, ElementsAre(false));
}

TEST_F(LlmGpuArtisanExecutorVisionTest,
       ImagesExceedingLimitAreSplitIntoMultiplePrefills) {
  CreateExecutor(/*max_num_images=*/1);

  ASSERT_OK(PrefillWithVision(
      {10, kVision, kVision, kVisionEnd, 11, kVision, kVisionEnd, 12}));

  // Each image is stored in slot 0 of the media embedding cache.
  EXPECT_THAT(runner_->filled_image_indices_, ElementsAre(0, 0));
  EXPECT_THAT(runner_->filled_image_first_values_, ElementsAre(0.0f, 2.0f));
  EXPECT_THAT(
      runner_->prefilled_tokens_,
      ElementsAre(ElementsAre(10, -1, -2, -3, 11), ElementsAre(-1, -2, 12)));
  // Every chunk but the last waits for completion before the media embedding
  // cache is overwritten.
  EXPECT_THAT(runner_->prefill_wait_for_completion_, ElementsAre(true, false));

  // The next prefill continues after all the tokens of both chunks.
  ASSERT_OK(PrefillWithVision({13, kVision, 14}));
  EXPECT_THAT(runner_->filled_image_indices_, ElementsAre(0, 0, 0));
  EXPECT_THAT(runner_->prefilled_tokens_.back(), ElementsAre(13, -1, 14));
}

TEST_F(LlmGpuArtisanExecutorVisionTest,
       ImagesInLaterTurnsReuseMediaEmbeddingCache) {
  CreateExecutor(/*max_num_images=*/1);

  ASSERT_OK(PrefillWithVision({10, kVision, kVisionEnd}));
  ASSERT_OK(PrefillWithVision({11, kVision, kVision, kVisionEnd}));

  EXPECT_THAT(runner_->filled_image_indices_, ElementsAre(0, 0));
  EXPECT_THAT(
      runner_->prefilled_tokens_,
      ElementsAre(ElementsAre(10, -1, -2), ElementsAre(11, -1, -2, -3)));
}

TEST_F(LlmGpuArtisanExecutorVisionTest, AudioOnlyIsStoredFromSlotZero) {
  CreateExecutor(/*max_num_images=*/1);

  ASSERT_OK(PrefillWithMedia({10, kAudio, kAudio, 11}));

  EXPECT_THAT(runner_->filled_image_indices_, ::testing::IsEmpty());
  EXPECT_THAT(runner_->filled_audio_offsets_, ElementsAre(0));
  EXPECT_THAT(runner_->prefilled_tokens_,
              ElementsAre(ElementsAre(10, -1, -2, 11)));
}

TEST_F(LlmGpuArtisanExecutorVisionTest,
       AudioIsStoredAfterImageSlotsAndGatheredFromThere) {
  CreateExecutor(/*max_num_images=*/2);

  // The image uses 2 of the 3 tokens of its slot, so the audio is stored at
  // offset 3 and must be gathered with ids -4, -5 rather than -3, -4.
  ASSERT_OK(PrefillWithMedia({10, kVision, kVision, 11, kAudio, kAudio, 12}));

  EXPECT_THAT(runner_->filled_image_indices_, ElementsAre(0));
  EXPECT_THAT(runner_->filled_audio_offsets_,
              ElementsAre(kVisionTokensPerImage));
  EXPECT_THAT(runner_->prefilled_tokens_,
              ElementsAre(ElementsAre(10, -1, -2, 11, -4, -5, 12)));
}

TEST_F(LlmGpuArtisanExecutorVisionTest,
       ImageAndAudioInLaterTurnsReuseMediaEmbeddingCache) {
  CreateExecutor(/*max_num_images=*/2);

  ASSERT_OK(PrefillWithMedia({10, kVision, kVisionEnd}));
  ASSERT_OK(PrefillWithMedia({11, kAudio, 12}));
  ASSERT_OK(PrefillWithMedia({13, kVision, kVisionEnd, kAudio, kAudio, 14}));

  EXPECT_THAT(runner_->filled_image_indices_, ElementsAre(0, 0));
  EXPECT_THAT(runner_->filled_audio_offsets_,
              ElementsAre(0, kVisionTokensPerImage));
  EXPECT_THAT(runner_->prefilled_tokens_,
              ElementsAre(ElementsAre(10, -1, -2), ElementsAre(11, -1, 12),
                          ElementsAre(13, -1, -2, -4, -5, 14)));
}

TEST_F(LlmGpuArtisanExecutorVisionTest,
       AudioIsPrefilledWithoutImagesIfTheyDoNotFitTogether) {
  CreateExecutor(/*max_num_images=*/1);

  // The media embedding cache has room for 1 image (3 tokens), so the image
  // and the 2 audio tokens are prefilled separately.
  ASSERT_OK(PrefillWithMedia({13, kVision, kVisionEnd, kAudio, kAudio, 14}));

  EXPECT_THAT(runner_->filled_image_indices_, ElementsAre(0));
  EXPECT_THAT(runner_->filled_audio_offsets_, ElementsAre(0));
  EXPECT_THAT(runner_->prefilled_tokens_,
              ElementsAre(ElementsAre(13, -1, -2), ElementsAre(-1, -2, 14)));
  EXPECT_THAT(runner_->prefill_wait_for_completion_, ElementsAre(true, false));
}

TEST_F(LlmGpuArtisanExecutorVisionTest,
       AudioWithImagesExceedingLimitIsSplitWithoutSplittingAudio) {
  CreateExecutor(/*max_num_images=*/2);

  // With 1 audio token, the chunk with the audio has room for 1 image.
  ASSERT_OK(PrefillWithMedia({10, kVision, kVisionEnd, 11, kVision, kVisionEnd,
                              12, kVision, kVisionEnd, kAudio, 13}));

  EXPECT_THAT(runner_->filled_image_indices_, ElementsAre(0, 1, 0));
  // Each image's first vision token value.
  EXPECT_THAT(runner_->filled_image_first_values_,
              ElementsAre(0.0f, 1.0f, 2.0f));
  EXPECT_THAT(runner_->filled_audio_offsets_,
              ElementsAre(kVisionTokensPerImage));
  EXPECT_THAT(runner_->prefilled_tokens_,
              ElementsAre(ElementsAre(10, -1, -2, 11, -4, -5, 12),
                          ElementsAre(-1, -2, -4, 13)));
  EXPECT_THAT(runner_->prefill_wait_for_completion_, ElementsAre(true, false));
}

TEST_F(LlmGpuArtisanExecutorVisionTest,
       MultipleAudioClipsArePrefilledOneClipPerChunk) {
  CreateExecutor(/*max_num_images=*/1);

  ASSERT_OK(PrefillWithMedia({10, kAudio, kAudio, 11, kAudio, 12}));

  EXPECT_THAT(runner_->filled_audio_offsets_, ElementsAre(0, 0));
  EXPECT_THAT(runner_->filled_audio_num_tokens_, ElementsAre(2, 1));
  // The first audio token of each clip.
  EXPECT_THAT(runner_->filled_audio_first_values_, ElementsAre(0.0f, 2.0f));
  EXPECT_THAT(runner_->prefilled_tokens_,
              ElementsAre(ElementsAre(10, -1, -2, 11), ElementsAre(-1, 12)));
  EXPECT_THAT(runner_->prefill_wait_for_completion_, ElementsAre(true, false));
}

TEST_F(LlmGpuArtisanExecutorVisionTest, ImagesBetweenAudioClipsAreSplit) {
  CreateExecutor(/*max_num_images=*/1);

  // With room for 1 image, an audio clip leaves no room for images, so every
  // image and every audio clip is prefilled separately.
  ASSERT_OK(PrefillWithMedia(
      {kAudio, 10, kVision, kVisionEnd, 11, kVision, kVisionEnd, kAudio, 12}));

  EXPECT_THAT(runner_->filled_image_indices_, ElementsAre(0, 0));
  EXPECT_THAT(runner_->filled_image_first_values_, ElementsAre(0.0f, 1.0f));
  EXPECT_THAT(runner_->filled_audio_offsets_, ElementsAre(0, 0));
  EXPECT_THAT(runner_->filled_audio_num_tokens_, ElementsAre(1, 1));
  EXPECT_THAT(runner_->filled_audio_first_values_, ElementsAre(0.0f, 1.0f));
  EXPECT_THAT(runner_->prefilled_tokens_,
              ElementsAre(ElementsAre(-1, 10), ElementsAre(-1, -2, 11),
                          ElementsAre(-1, -2), ElementsAre(-1, 12)));
}

}  // namespace
}  // namespace odml::infra
