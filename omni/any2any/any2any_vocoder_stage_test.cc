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

#include "omni/any2any/any2any_vocoder_stage.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "omni/any2any/any2any_vocoder_test_utils.h"
#include "omni/any2any/io_types.h"
#include "omni/base/io_types.h"
#include "omni/base/mock_litert_runner.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::any2any {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::FloatEq;
using ::testing::FloatNear;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;

Any2AnyVocoderStage::Options MakeBaseOptions() {
  Any2AnyVocoderStage::Options options;
  options.history_tokens = 1;
  options.chunk_tokens = 2;
  options.lookahead_tokens = 1;
  options.samples_per_token = 2;
  options.audio_output_sample_rate_hz = 24000;
  options.generate_audio = true;
  return options;
}

TEST(Any2AnyVocoderStageTest, CreateValidatesOptionsAndRequiredRunner) {
  FakeTtsStage tts_stage;
  Any2AnyVocoderStage::Options base = MakeBaseOptions();

  {
    Any2AnyVocoderStage::Options bad = base;
    bad.history_tokens = -1;
    EXPECT_THAT(Any2AnyVocoderStage::Create(&tts_stage, bad).status(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    Any2AnyVocoderStage::Options bad = base;
    bad.chunk_tokens = 0;
    EXPECT_THAT(Any2AnyVocoderStage::Create(&tts_stage, bad).status(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    Any2AnyVocoderStage::Options bad = base;
    bad.lookahead_tokens = -1;
    EXPECT_THAT(Any2AnyVocoderStage::Create(&tts_stage, bad).status(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    Any2AnyVocoderStage::Options bad = base;
    bad.samples_per_token = 0;
    EXPECT_THAT(Any2AnyVocoderStage::Create(&tts_stage, bad).status(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    Any2AnyVocoderStage::Options bad = base;
    bad.audio_output_sample_rate_hz = 0;
    EXPECT_THAT(Any2AnyVocoderStage::Create(&tts_stage, bad).status(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Missing runner rejected when `generate_audio` is true.
  EXPECT_THAT(Any2AnyVocoderStage::Create(&tts_stage, base, nullptr).status(),
              StatusIs(absl::StatusCode::kNotFound));

  // Empty input/output buffers rejected with `kInternal`.
  auto empty_runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*empty_runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) { return std::vector<TensorBuffer>{}; });
  ON_CALL(*empty_runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) { return std::vector<TensorBuffer>{}; });
  EXPECT_THAT(
      Any2AnyVocoderStage::Create(&tts_stage, base, std::move(empty_runner))
          .status(),
      StatusIs(absl::StatusCode::kInternal));
}

TEST(Any2AnyVocoderStageTest,
     CreateRejectsMismatchedInputAndOutputElementTypes) {
  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  FakeTtsStage tts_stage;
  Any2AnyVocoderStage::Options options = MakeBaseOptions();

  // Int64[1, 2] has the same packed byte size (16 bytes) as Int32[1, 4],
  // verifying that element type is validated independently of byte size.
  {
    auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
    ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
        .WillByDefault([&env](absl::string_view)
                           -> absl::StatusOr<std::vector<TensorBuffer>> {
          RankedTensorType type(ElementType::Int64, Layout(Dimensions({1, 2})));
          LITERT_ASSIGN_OR_RETURN(
              TensorBuffer buf,
              TensorBuffer::CreateManaged(*env, TensorBufferType::kHostMemory,
                                          type, 2 * sizeof(int64_t)));
          std::vector<TensorBuffer> bufs;
          bufs.push_back(std::move(buf));
          return bufs;
        });
    ON_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
        .WillByDefault([&env](absl::string_view)
                           -> absl::StatusOr<std::vector<TensorBuffer>> {
          RankedTensorType type(ElementType::Float32,
                                Layout(Dimensions({1, 6})));
          LITERT_ASSIGN_OR_RETURN(
              TensorBuffer buf,
              TensorBuffer::CreateManaged(*env, TensorBufferType::kHostMemory,
                                          type, 6 * sizeof(float)));
          std::vector<TensorBuffer> bufs;
          bufs.push_back(std::move(buf));
          return bufs;
        });
    EXPECT_THAT(
        Any2AnyVocoderStage::Create(&tts_stage, options, std::move(runner))
            .status(),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Int32")));
  }

  // Float16[1, 12] has the same packed byte size (24 bytes) as Float32[1, 6].
  {
    auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
    ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
        .WillByDefault([&env](absl::string_view)
                           -> absl::StatusOr<std::vector<TensorBuffer>> {
          RankedTensorType type(ElementType::Int32, Layout(Dimensions({1, 4})));
          LITERT_ASSIGN_OR_RETURN(
              TensorBuffer buf,
              TensorBuffer::CreateManaged(*env, TensorBufferType::kHostMemory,
                                          type, 4 * sizeof(int32_t)));
          std::vector<TensorBuffer> bufs;
          bufs.push_back(std::move(buf));
          return bufs;
        });
    ON_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
        .WillByDefault([&env](absl::string_view)
                           -> absl::StatusOr<std::vector<TensorBuffer>> {
          RankedTensorType type(ElementType::Float16,
                                Layout(Dimensions({1, 12})));
          LITERT_ASSIGN_OR_RETURN(
              TensorBuffer buf,
              TensorBuffer::CreateManaged(*env, TensorBufferType::kHostMemory,
                                          type, 12 * sizeof(uint16_t)));
          std::vector<TensorBuffer> bufs;
          bufs.push_back(std::move(buf));
          return bufs;
        });
    EXPECT_THAT(
        Any2AnyVocoderStage::Create(&tts_stage, options, std::move(runner))
            .status(),
        StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Float32")));
  }
}

TEST(Any2AnyVocoderStageTest, CreateAndSynthesizeUsePackedSizeOnPaddedBuffers) {
  auto env = ::litert::Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  FakeTtsStage tts_stage;
  Any2AnyVocoderStage::Options options = MakeBaseOptions();
  options.padding_mode = Any2AnyVocoderStage::PaddingMode::kEdgeClamp;

  // Allocate 64-byte hardware buffers for packed 16-byte input and 24-byte
  // output tensors so Size() > PackedSize().
  constexpr size_t kPaddedAllocationBytes = 64;
  auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault([&env](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        RankedTensorType type(ElementType::Int32, Layout(Dimensions({1, 4})));
        LITERT_ASSIGN_OR_RETURN(
            TensorBuffer buf,
            TensorBuffer::CreateManaged(*env, TensorBufferType::kHostMemory,
                                        type, kPaddedAllocationBytes));
        LITERT_ASSIGN_OR_RETURN(size_t alloc_size, buf.Size());
        LITERT_ASSIGN_OR_RETURN(size_t packed_size, buf.PackedSize());
        EXPECT_GT(alloc_size, packed_size);
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(buf));
        return bufs;
      });
  ON_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault([&env](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        RankedTensorType type(ElementType::Float32, Layout(Dimensions({1, 6})));
        LITERT_ASSIGN_OR_RETURN(
            TensorBuffer buf,
            TensorBuffer::CreateManaged(*env, TensorBufferType::kHostMemory,
                                        type, kPaddedAllocationBytes));
        LITERT_ASSIGN_OR_RETURN(size_t alloc_size, buf.Size());
        LITERT_ASSIGN_OR_RETURN(size_t packed_size, buf.PackedSize());
        EXPECT_GT(alloc_size, packed_size);
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(buf));
        return bufs;
      });
  EXPECT_CALL(*runner, Run(absl::string_view(""), _, _))
      .WillOnce([](absl::string_view, absl::Span<const TensorBuffer> inputs,
                   absl::Span<const TensorBuffer> outputs) -> absl::Status {
        std::vector<int32_t> codes(4, 0);
        LITERT_RETURN_IF_ERROR(
            const_cast<TensorBuffer&>(inputs[0]).Read<int32_t>(
                absl::MakeSpan(codes)));
        EXPECT_THAT(codes, ElementsAre(3, 3, 4, 4));
        std::vector<float> pcm = {30.0f, 31.0f, 30.0f, 31.0f, 40.0f, 41.0f};
        LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(outputs[0])
                                   .Write<float>(absl::MakeConstSpan(pcm)));
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Any2AnyVocoderStage> stage,
      Any2AnyVocoderStage::Create(&tts_stage, options, std::move(runner)));
  tts_stage.Push(
      Any2AnyTtsOutput{.is_text_final = true, .audio_codes = {3, 4}});
  ASSERT_OK(stage->Schedule());

  ASSERT_OK_AND_ASSIGN(Output audio_out, stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(audio_out));
  EXPECT_THAT(std::get<AudioOutput>(audio_out).pcm_samples,
              ElementsAre(FloatEq(30.0f), FloatEq(31.0f), FloatEq(40.0f),
                          FloatEq(41.0f)));
}

TEST(Any2AnyVocoderStageTest, StrictModeVsInferredLayoutBufferValidation) {
  FakeTtsStage tts_stage;
  Any2AnyVocoderStage::Options strict_opts =
      MakeBaseOptions();  // window=4, trimmed=3, spt=2 -> 6.
  strict_opts.allow_inferred_layout = false;

  EXPECT_THAT(Any2AnyVocoderStage::Create(&tts_stage, strict_opts,
                                          MakeMockVocoderRunner(4, 6))
                  .status(),
              IsOk());
  // Full window output (8 samples = 32 bytes vs expected 24 bytes) is rejected
  // in strict mode with actual vs. expected byte sizes in the message.
  EXPECT_THAT(Any2AnyVocoderStage::Create(&tts_stage, strict_opts,
                                          MakeMockVocoderRunner(4, 8))
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("32"), HasSubstr("24"))));
  // Wrong input tokens (3 tokens = 12 bytes vs expected 16 bytes) rejected in
  // strict mode.
  EXPECT_THAT(Any2AnyVocoderStage::Create(&tts_stage, strict_opts,
                                          MakeMockVocoderRunner(3, 6))
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("12"), HasSubstr("16"))));

  Any2AnyVocoderStage::Options inferred_opts = MakeBaseOptions();
  inferred_opts.samples_per_token = 4;  // Deliberately mismatched.
  inferred_opts.allow_inferred_layout = true;
  inferred_opts.padding_mode = Any2AnyVocoderStage::PaddingMode::kEdgeClamp;

  // Branch 1: matching input window (4), output divisible by window (20/4=5).
  {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Any2AnyVocoderStage> stage,
        Any2AnyVocoderStage::Create(
            &tts_stage, inferred_opts,
            MakeMockVocoderRunner(4, 20, /*recorded_windows=*/nullptr,
                                  /*samples_per_token=*/5)));
    tts_stage.Push(
        Any2AnyTtsOutput{.is_text_final = false, .audio_codes = {1, 2}});
    ASSERT_OK(stage->Schedule());
    ASSERT_OK_AND_ASSIGN(Output out, stage->GetOutput());
    ASSERT_TRUE(std::holds_alternative<AudioOutput>(out));
    EXPECT_THAT(std::get<AudioOutput>(out).pcm_samples,
                ElementsAre(FloatEq(10.0f), FloatEq(11.0f), FloatEq(12.0f),
                            FloatEq(13.0f), FloatEq(14.0f), FloatEq(20.0f),
                            FloatEq(21.0f), FloatEq(22.0f), FloatEq(23.0f),
                            FloatEq(24.0f)));
  }

  // Branch 2: matching input window (4), output divisible by trimmed (9/3=3).
  {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Any2AnyVocoderStage> stage,
        Any2AnyVocoderStage::Create(
            &tts_stage, inferred_opts,
            MakeMockVocoderRunner(4, 9, /*recorded_windows=*/nullptr,
                                  /*samples_per_token=*/3)));
    tts_stage.Push(
        Any2AnyTtsOutput{.is_text_final = false, .audio_codes = {1, 2}});
    ASSERT_OK(stage->Schedule());
    ASSERT_OK_AND_ASSIGN(Output out, stage->GetOutput());
    ASSERT_TRUE(std::holds_alternative<AudioOutput>(out));
    EXPECT_THAT(std::get<AudioOutput>(out).pcm_samples,
                ElementsAre(FloatEq(10.0f), FloatEq(11.0f), FloatEq(12.0f),
                            FloatEq(20.0f), FloatEq(21.0f), FloatEq(22.0f)));
  }

  // Branch 3: non-matching input tokens (3), output divisible by in_tokens
  // (6/3=2) -> history=0, chunk=3, lookahead=0.
  {
    std::vector<std::vector<int32_t>> recorded;
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Any2AnyVocoderStage> stage,
        Any2AnyVocoderStage::Create(&tts_stage, inferred_opts,
                                    MakeMockVocoderRunner(3, 6, &recorded)));
    tts_stage.Push(
        Any2AnyTtsOutput{.is_text_final = false, .audio_codes = {1, 2}});
    ASSERT_OK(stage->Schedule());
    ASSERT_EQ(recorded.size(), 1);
    EXPECT_THAT(recorded[0], ElementsAre(1, 2, 2));
    ASSERT_OK_AND_ASSIGN(Output out, stage->GetOutput());
    ASSERT_TRUE(std::holds_alternative<AudioOutput>(out));
    EXPECT_THAT(std::get<AudioOutput>(out).pcm_samples,
                ElementsAre(FloatEq(10.0f), FloatEq(11.0f), FloatEq(20.0f),
                            FloatEq(21.0f)));
  }

  // Incompatible output size (5 floats = 20 bytes vs expected 48 or 64 bytes)
  // still fails with actual vs. expected byte sizes.
  EXPECT_THAT(
      Any2AnyVocoderStage::Create(&tts_stage, inferred_opts,
                                  MakeMockVocoderRunner(4, 5))
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("20"), HasSubstr("48"), HasSubstr("64"))));
}

TEST(Any2AnyVocoderStageTest,
     ConstantPaddingAndSeamCrossfadeBlendHistoryOverlap) {
  FakeTtsStage tts_stage;
  Any2AnyVocoderStage::Options options = MakeBaseOptions();
  options.padding_mode = Any2AnyVocoderStage::PaddingMode::kConstant;
  options.pad_token_id = 999;
  options.max_seam_crossfade_samples = 48;

  std::unique_ptr<NiceMock<MockLiteRtRunner>> runner =
      MakeMockVocoderRunner(/*input_tokens=*/4, /*output_samples=*/6);

  std::vector<std::vector<int32_t>> captured_windows;
  int call_idx = 0;
  EXPECT_CALL(*runner, Run(absl::string_view(""), _, _))
      .Times(2)
      .WillRepeatedly(
          [&](absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            std::vector<int32_t> window_in(4, 0);
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[0]).Read<int32_t>(
                    absl::MakeSpan(window_in)));
            captured_windows.push_back(window_in);
            const float bias = (call_idx == 1) ? 3.0f : 0.0f;
            ++call_idx;
            std::vector<float> pcm = {
                static_cast<float>(window_in[0]) + bias,
                static_cast<float>(window_in[0] * 10) + bias,
                static_cast<float>(window_in[1]) + bias,
                static_cast<float>(window_in[1] * 10) + bias,
                static_cast<float>(window_in[2]) + bias,
                static_cast<float>(window_in[2] * 10) + bias,
            };
            LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(outputs[0])
                                       .Write<float>(absl::MakeConstSpan(pcm)));
            return absl::OkStatus();
          });

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Any2AnyVocoderStage> stage,
      Any2AnyVocoderStage::Create(&tts_stage, options, std::move(runner)));

  tts_stage.Push(Any2AnyTtsOutput{
      .text = "Hello",
      .has_text = true,
      .is_text_final = true,
      .audio_codes = {1, 2, 3},
  });
  ASSERT_OK(stage->Schedule());

  ASSERT_EQ(captured_windows.size(), 2);
  EXPECT_THAT(captured_windows[0], ElementsAre(999, 1, 2, 3));
  EXPECT_THAT(captured_windows[1], ElementsAre(2, 3, 999, 999));

  ASSERT_OK_AND_ASSIGN(Output text_out, stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(text_out));
  EXPECT_EQ(std::get<TextOutput>(text_out).confirmed_text, "Hello");

  ASSERT_OK_AND_ASSIGN(Output audio_out, stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(audio_out));
  EXPECT_THAT(std::get<AudioOutput>(audio_out).pcm_samples,
              ElementsAre(FloatNear(1.0f, 1e-5f), FloatNear(10.0f, 1e-5f),
                          FloatNear(3.0f, 1e-5f), FloatNear(22.0f, 1e-5f),
                          FloatNear(6.0f, 1e-5f), FloatNear(33.0f, 1e-5f)));

  ASSERT_OK_AND_ASSIGN(Output eot_out, stage->GetOutput());
  EXPECT_TRUE(std::holds_alternative<EndOfOutputTurn>(eot_out));
}

TEST(Any2AnyVocoderStageTest, CapsSeamCrossfadeToMaxSeamCrossfadeSamples) {
  FakeTtsStage tts_stage;
  Any2AnyVocoderStage::Options options;
  options.history_tokens = 1;
  options.chunk_tokens = 1;
  options.lookahead_tokens = 0;
  options.samples_per_token = 64;
  options.audio_output_sample_rate_hz = 24000;
  options.generate_audio = true;
  options.max_seam_crossfade_samples = 48;

  std::unique_ptr<NiceMock<MockLiteRtRunner>> runner =
      MakeMockVocoderRunner(/*input_tokens=*/2, /*output_samples=*/128);
  int call_idx = 0;
  EXPECT_CALL(*runner, Run(absl::string_view(""), _, _))
      .Times(2)
      .WillRepeatedly(
          [&](absl::string_view, absl::Span<const TensorBuffer> /*inputs*/,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            const float val = (call_idx++ == 0) ? 1.0f : 2.0f;
            std::vector<float> pcm(128, val);
            LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(outputs[0])
                                       .Write<float>(absl::MakeConstSpan(pcm)));
            return absl::OkStatus();
          });

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Any2AnyVocoderStage> stage,
      Any2AnyVocoderStage::Create(&tts_stage, options, std::move(runner)));
  tts_stage.Push(
      Any2AnyTtsOutput{.is_text_final = true, .audio_codes = {1, 2}});
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(Output audio_out, stage->GetOutput());
  const std::vector<float>& pcm = std::get<AudioOutput>(audio_out).pcm_samples;
  ASSERT_EQ(pcm.size(), 128);
  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(pcm[i], 1.0f);
  }
  EXPECT_GT(pcm[16], 1.0f);
  EXPECT_LT(pcm[63], 2.0f);
  EXPECT_FLOAT_EQ(pcm[64], 2.0f);
}

TEST(Any2AnyVocoderStageTest, EdgeClampPaddingAndMaxValidCodeValidation) {
  FakeTtsStage tts_stage;
  Any2AnyVocoderStage::Options options = MakeBaseOptions();
  options.padding_mode = Any2AnyVocoderStage::PaddingMode::kEdgeClamp;
  options.max_seam_crossfade_samples = 0;
  options.max_valid_code = 8;
  options.allow_inferred_layout = true;

  std::vector<std::vector<int32_t>> recorded;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Any2AnyVocoderStage> stage,
      Any2AnyVocoderStage::Create(&tts_stage, options,
                                  MakeMockVocoderRunner(4, 8, &recorded)));

  // Rejects negative and >= max_valid_code tokens.
  tts_stage.Push(Any2AnyTtsOutput{.audio_codes = {-1}});
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));
  tts_stage.Push(Any2AnyTtsOutput{.audio_codes = {0, 8}});
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // Valid codes synthesize with edge-clamp padding and no seam crossfade.
  tts_stage.Push(Any2AnyTtsOutput{
      .is_text_final = true,
      .audio_codes = {1, 2, 3, 4, 5},
  });
  ASSERT_OK(stage->Schedule());
  ASSERT_EQ(recorded.size(), 3);
  EXPECT_THAT(recorded[0], ElementsAre(1, 1, 2, 3));
  EXPECT_THAT(recorded[1], ElementsAre(2, 3, 4, 5));
  EXPECT_THAT(recorded[2], ElementsAre(4, 5, 5, 5));

  ASSERT_OK_AND_ASSIGN(Output audio_out, stage->GetOutput());
  EXPECT_THAT(std::get<AudioOutput>(audio_out).pcm_samples,
              ElementsAre(FloatEq(10.0f), FloatEq(11.0f), FloatEq(20.0f),
                          FloatEq(21.0f), FloatEq(30.0f), FloatEq(31.0f),
                          FloatEq(40.0f), FloatEq(41.0f), FloatEq(50.0f),
                          FloatEq(51.0f)));
}

TEST(Any2AnyVocoderStageTest,
     GenerateAudioFalseIgnoresNonNullRunnerAndSkipsSynthesis) {
  FakeTtsStage tts_stage;
  Any2AnyVocoderStage::Options options = MakeBaseOptions();
  options.generate_audio = false;

  auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  EXPECT_CALL(*runner, CreateInputBuffers(_)).Times(0);
  EXPECT_CALL(*runner, CreateOutputBuffers(_)).Times(0);
  EXPECT_CALL(*runner, Run(_, _, _)).Times(0);

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Any2AnyVocoderStage> stage,
      Any2AnyVocoderStage::Create(&tts_stage, options, std::move(runner)));

  tts_stage.Push(Any2AnyTtsOutput{
      .text = "TextOnly",
      .has_text = true,
      .is_text_final = true,
      .audio_codes = {1, 2, 3},
  });
  ASSERT_OK(stage->Schedule());

  ASSERT_OK_AND_ASSIGN(Output text_out, stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(text_out));
  EXPECT_EQ(std::get<TextOutput>(text_out).confirmed_text, "TextOnly");

  ASSERT_OK_AND_ASSIGN(Output eot_out, stage->GetOutput());
  EXPECT_TRUE(std::holds_alternative<EndOfOutputTurn>(eot_out));
  EXPECT_FALSE(stage->HasOutput());
}

TEST(Any2AnyVocoderStageTest,
     SupportsNullRunnerFlushResetAndRunnerErrorPropagation) {
  FakeTtsStage tts_stage;
  Any2AnyVocoderStage::Options disabled_opts = MakeBaseOptions();
  disabled_opts.generate_audio = false;

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Any2AnyVocoderStage> stage,
                       Any2AnyVocoderStage::Create(&tts_stage, disabled_opts));
  EXPECT_FALSE(stage->NeedSchedule());
  EXPECT_OK(stage->Schedule());

  tts_stage.Push(Any2AnyTtsOutput{
      .text = "Part1",
      .has_text = true,
      .is_text_final = false,
      .audio_codes = {1, 2},
  });
  tts_stage.Push(Any2AnyTtsOutput{
      .text = "Part2",
      .has_text = true,
      .is_text_final = true,
  });
  EXPECT_TRUE(stage->NeedSchedule());
  ASSERT_OK(stage->Flush());

  ASSERT_OK_AND_ASSIGN(Output out1, stage->GetOutput());
  EXPECT_EQ(std::get<TextOutput>(out1).confirmed_text, "Part1");
  // Reset discards remaining queued outputs.
  stage->Reset();
  EXPECT_FALSE(stage->HasOutput());

  // Runner execution error propagates cleanly while preserving already-emitted
  // `TextOutput` from the same `Any2AnyTtsOutput` item.
  Any2AnyVocoderStage::Options err_opts = MakeBaseOptions();
  err_opts.history_tokens = 0;
  err_opts.chunk_tokens = 1;
  err_opts.lookahead_tokens = 0;
  err_opts.samples_per_token = 2;
  std::unique_ptr<NiceMock<MockLiteRtRunner>> err_runner =
      MakeMockVocoderRunner(1, 2);
  EXPECT_CALL(*err_runner, Run(absl::string_view(""), _, _))
      .WillOnce(Return(absl::InternalError("vocoder failed")));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Any2AnyVocoderStage> err_stage,
      Any2AnyVocoderStage::Create(&tts_stage, err_opts, std::move(err_runner)));
  tts_stage.Push(Any2AnyTtsOutput{
      .text = "Partial",
      .has_text = true,
      .is_text_final = true,
      .audio_codes = {1},
  });
  EXPECT_THAT(err_stage->Schedule(), StatusIs(absl::StatusCode::kInternal));
  ASSERT_OK_AND_ASSIGN(Output partial_text, err_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(partial_text));
  EXPECT_EQ(std::get<TextOutput>(partial_text).confirmed_text, "Partial");
  EXPECT_FALSE(err_stage->HasOutput());
}

}  // namespace
}  // namespace litert::omni::any2any
