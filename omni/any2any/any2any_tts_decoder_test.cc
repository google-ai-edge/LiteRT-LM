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

#include "omni/any2any/any2any_tts_decoder.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "omni/base/mock_litert_runner.h"
#include "support/util/convert_tensor_buffer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::any2any {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::NiceMock;
using ::testing::Return;

Any2AnyTtsDecoder::Options MakeBaseOptions() {
  Any2AnyTtsDecoder::Options options;
  options.llm_hidden_size = 4;
  options.llm_vocab_size = 100;
  options.tts_hidden_size = 4;
  options.tts_vocab_size = 8;
  options.tts_num_layers = 1;
  options.tts_max_cache_len = 8;
  options.generate_audio = true;
  return options;
}

std::unique_ptr<NiceMock<MockLiteRtRunner>> MakeProjectorRunner(
    int llm_hidden_size, int tts_hidden_size) {
  auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault([llm_hidden_size](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(
            auto hidden_buf,
            support::CreateTensorBuffer<float>({1, 1, llm_hidden_size}));
        LITERT_ASSIGN_OR_RETURN(auto token_buf,
                                support::CreateTensorBuffer<int32_t>({1, 1}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(hidden_buf));
        bufs.push_back(std::move(token_buf));
        return bufs;
      });
  ON_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault([tts_hidden_size](absl::string_view)
                         -> absl::StatusOr<std::vector<TensorBuffer>> {
        LITERT_ASSIGN_OR_RETURN(
            auto out_buf,
            support::CreateTensorBuffer<float>({1, 1, tts_hidden_size}));
        std::vector<TensorBuffer> bufs;
        bufs.push_back(std::move(out_buf));
        return bufs;
      });
  return runner;
}

std::unique_ptr<NiceMock<MockLiteRtRunner>> MakeDecoderRunner(
    int tts_hidden_size, int tts_vocab_size, int tts_num_layers,
    int tts_max_cache_len, int kv_dim) {
  auto runner = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*runner, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [tts_hidden_size, tts_num_layers, tts_max_cache_len, kv_dim](
              absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto cond_buf,
                support::CreateTensorBuffer<float>({1, 1, tts_hidden_size}));
            LITERT_ASSIGN_OR_RETURN(
                auto code_buf, support::CreateTensorBuffer<int32_t>({1, 1}));
            LITERT_ASSIGN_OR_RETURN(auto step_buf,
                                    support::CreateTensorBuffer<int32_t>({1}));
            LITERT_ASSIGN_OR_RETURN(
                auto kv_buf,
                support::CreateTensorBuffer<float>(
                    {1, 2 * tts_num_layers, tts_max_cache_len, kv_dim}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(cond_buf));
            bufs.push_back(std::move(code_buf));
            bufs.push_back(std::move(step_buf));
            bufs.push_back(std::move(kv_buf));
            return bufs;
          });
  ON_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [tts_vocab_size, tts_num_layers, tts_max_cache_len, kv_dim](
              absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            LITERT_ASSIGN_OR_RETURN(
                auto logits_buf,
                support::CreateTensorBuffer<float>({1, 1, tts_vocab_size}));
            LITERT_ASSIGN_OR_RETURN(
                auto kv_buf,
                support::CreateTensorBuffer<float>(
                    {1, 2 * tts_num_layers, tts_max_cache_len, kv_dim}));
            std::vector<TensorBuffer> bufs;
            bufs.push_back(std::move(logits_buf));
            bufs.push_back(std::move(kv_buf));
            return bufs;
          });
  return runner;
}

TEST(Any2AnyTtsDecoderTest, CreateValidatesDimensionsAndRequiredRunners) {
  auto base = MakeBaseOptions();

  {
    auto bad = base;
    bad.llm_hidden_size = 0;
    EXPECT_THAT(Any2AnyTtsDecoder::Create(bad).status(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    auto bad = base;
    bad.tts_hidden_size = -1;
    EXPECT_THAT(Any2AnyTtsDecoder::Create(bad).status(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    auto bad = base;
    bad.tts_vocab_size = 0;
    EXPECT_THAT(Any2AnyTtsDecoder::Create(bad).status(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    auto bad = base;
    bad.tts_num_layers = 0;
    EXPECT_THAT(Any2AnyTtsDecoder::Create(bad).status(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    auto bad = base;
    bad.tts_max_cache_len = 0;
    EXPECT_THAT(Any2AnyTtsDecoder::Create(bad).status(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Null projector / decoder rejected when `generate_audio` is true.
  EXPECT_THAT(Any2AnyTtsDecoder::Create(base, /*tts_projector_runner=*/nullptr,
                                        MakeDecoderRunner(4, 8, 1, 8, 4))
                  .status(),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(Any2AnyTtsDecoder::Create(base, MakeProjectorRunner(4, 4),
                                        /*tts_decoder_runner=*/nullptr)
                  .status(),
              StatusIs(absl::StatusCode::kNotFound));

  // Null runners succeed when `generate_audio` is false, and stepping APIs
  // return `kFailedPrecondition`.
  auto no_audio = base;
  no_audio.generate_audio = false;
  ASSERT_OK_AND_ASSIGN(auto decoder, Any2AnyTtsDecoder::Create(no_audio));
  std::vector<float> hidden(4, 0.0f);
  std::vector<float> cond(4, 0.0f);
  std::vector<float> logits(8, 0.0f);
  EXPECT_THAT(
      decoder->ProjectTokenConditioning(hidden, 1, absl::MakeSpan(cond)),
      StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(decoder->ClearKvCache(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(decoder->RunDecoderStep(cond, -1, 0, absl::MakeSpan(logits)),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(Any2AnyTtsDecoderTest, CreateValidatesProjectorAndDecoderBuffers) {
  auto options = MakeBaseOptions();

  // Projector buffer count mismatch.
  auto bad_count_proj = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*bad_count_proj, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            return std::vector<TensorBuffer>{};
          });
  ON_CALL(*bad_count_proj, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            return std::vector<TensorBuffer>{};
          });
  EXPECT_THAT(Any2AnyTtsDecoder::Create(options, std::move(bad_count_proj),
                                        MakeDecoderRunner(4, 8, 1, 8, 4))
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Projector byte size mismatch (input hidden_size 8 != 4, output 2 != 4).
  EXPECT_THAT(Any2AnyTtsDecoder::Create(options, MakeProjectorRunner(8, 4),
                                        MakeDecoderRunner(4, 8, 1, 8, 4))
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(Any2AnyTtsDecoder::Create(options, MakeProjectorRunner(4, 2),
                                        MakeDecoderRunner(4, 8, 1, 8, 4))
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Decoder buffer count mismatch.
  auto bad_count_dec = std::make_unique<NiceMock<MockLiteRtRunner>>();
  ON_CALL(*bad_count_dec, CreateInputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            return std::vector<TensorBuffer>{};
          });
  ON_CALL(*bad_count_dec, CreateOutputBuffers(absl::string_view("")))
      .WillByDefault(
          [](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            return std::vector<TensorBuffer>{};
          });
  EXPECT_THAT(Any2AnyTtsDecoder::Create(options, MakeProjectorRunner(4, 4),
                                        std::move(bad_count_dec))
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Decoder logits vocab size mismatch.
  EXPECT_THAT(Any2AnyTtsDecoder::Create(options, MakeProjectorRunner(4, 4),
                                        MakeDecoderRunner(4, 9, 1, 8, 4))
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Strict mode rejects mismatched KV cache length.
  EXPECT_THAT(Any2AnyTtsDecoder::Create(options, MakeProjectorRunner(4, 4),
                                        MakeDecoderRunner(4, 8, 1, 16, 4))
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Any2AnyTtsDecoderTest, SupportsGqaMhaAndInferred4DKvCacheShapes) {
  auto options = MakeBaseOptions();
  options.tts_num_kv_heads = 1;
  options.tts_head_dim = 2;  // GQA kv_dim = 2 vs MHA hidden_size = 4.

  // GQA KV cache accepted.
  EXPECT_THAT(Any2AnyTtsDecoder::Create(options, MakeProjectorRunner(4, 4),
                                        MakeDecoderRunner(4, 8, 1, 8, 2))
                  .status(),
              IsOk());

  // MHA KV cache also accepted.
  EXPECT_THAT(Any2AnyTtsDecoder::Create(options, MakeProjectorRunner(4, 4),
                                        MakeDecoderRunner(4, 8, 1, 8, 4))
                  .status(),
              IsOk());

  // With `allow_inferred_kv_shape = true`, 4D `[1, 2*L, max_cache_len, D]`
  // updates `tts_num_layers()` and `tts_max_cache_len()`.
  options.allow_inferred_kv_shape = true;
  ASSERT_OK_AND_ASSIGN(auto inferred, Any2AnyTtsDecoder::Create(
                                          options, MakeProjectorRunner(4, 4),
                                          MakeDecoderRunner(4, 8, 3, 12, 2)));
  EXPECT_EQ(inferred->tts_num_layers(), 3);
  EXPECT_EQ(inferred->tts_max_cache_len(), 12);
  std::vector<float> inferred_cond(4, 0.0f);
  std::vector<float> inferred_logits(8, 0.0f);
  EXPECT_THAT(inferred->RunDecoderStep(inferred_cond, -1, 11,
                                       absl::MakeSpan(inferred_logits)),
              IsOk());
  EXPECT_THAT(inferred->RunDecoderStep(inferred_cond, -1, 12,
                                       absl::MakeSpan(inferred_logits)),
              StatusIs(absl::StatusCode::kInvalidArgument));

  auto make_custom_kv_decoder = [](auto make_kv_in, auto make_kv_out) {
    auto dec = std::make_unique<NiceMock<MockLiteRtRunner>>();
    ON_CALL(*dec, CreateInputBuffers(absl::string_view("")))
        .WillByDefault([make_kv_in](absl::string_view)
                           -> absl::StatusOr<std::vector<TensorBuffer>> {
          LITERT_ASSIGN_OR_RETURN(
              auto cond_buf, support::CreateTensorBuffer<float>({1, 1, 4}));
          LITERT_ASSIGN_OR_RETURN(auto code_buf,
                                  support::CreateTensorBuffer<int32_t>({1, 1}));
          LITERT_ASSIGN_OR_RETURN(auto step_buf,
                                  support::CreateTensorBuffer<int32_t>({1}));
          LITERT_ASSIGN_OR_RETURN(auto kv_buf, make_kv_in());
          std::vector<TensorBuffer> bufs;
          bufs.push_back(std::move(cond_buf));
          bufs.push_back(std::move(code_buf));
          bufs.push_back(std::move(step_buf));
          bufs.push_back(std::move(kv_buf));
          return bufs;
        });
    ON_CALL(*dec, CreateOutputBuffers(absl::string_view("")))
        .WillByDefault([make_kv_out](absl::string_view)
                           -> absl::StatusOr<std::vector<TensorBuffer>> {
          LITERT_ASSIGN_OR_RETURN(
              auto logits_buf, support::CreateTensorBuffer<float>({1, 1, 8}));
          LITERT_ASSIGN_OR_RETURN(auto kv_buf, make_kv_out());
          std::vector<TensorBuffer> bufs;
          bufs.push_back(std::move(logits_buf));
          bufs.push_back(std::move(kv_buf));
          return bufs;
        });
    return dec;
  };

  // Odd layer dimension in 4D KV tensor is rejected even when inference is on.
  EXPECT_THAT(
      Any2AnyTtsDecoder::Create(
          options, MakeProjectorRunner(4, 4),
          make_custom_kv_decoder(
              []() {
                return support::CreateTensorBuffer<float>({1, 3, 12, 2});
              },
              []() {
                return support::CreateTensorBuffer<float>({1, 3, 12, 2});
              }))
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Mismatched `kv_in != kv_out` byte sizes rejected even with inference on.
  EXPECT_THAT(
      Any2AnyTtsDecoder::Create(
          options, MakeProjectorRunner(4, 4),
          make_custom_kv_decoder(
              []() {
                return support::CreateTensorBuffer<float>({1, 6, 12, 2});
              },
              []() {
                return support::CreateTensorBuffer<float>({1, 6, 10, 2});
              }))
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Non-4D (3D) KV tensor rejected even when inference is on.
  EXPECT_THAT(
      Any2AnyTtsDecoder::Create(
          options, MakeProjectorRunner(4, 4),
          make_custom_kv_decoder(
              []() { return support::CreateTensorBuffer<float>({6, 12, 2}); },
              []() { return support::CreateTensorBuffer<float>({6, 12, 2}); }))
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Same total byte size (576 bytes) but mismatched 4D shapes between `kv_in`
  // and `kv_out` is rejected in both configured and inferred modes.
  auto strict_opts = MakeBaseOptions();
  strict_opts.allow_inferred_kv_shape = false;
  EXPECT_THAT(
      Any2AnyTtsDecoder::Create(
          strict_opts, MakeProjectorRunner(4, 4),
          make_custom_kv_decoder(
              []() { return support::CreateTensorBuffer<float>({1, 2, 8, 4}); },
              []() {
                return support::CreateTensorBuffer<float>({1, 4, 4, 4});
              }))
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      Any2AnyTtsDecoder::Create(
          options, MakeProjectorRunner(4, 4),
          make_custom_kv_decoder(
              []() {
                return support::CreateTensorBuffer<float>({1, 6, 12, 2});
              },
              []() {
                return support::CreateTensorBuffer<float>({1, 4, 18, 2});
              }))
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Non-Float32 (`int32_t`) KV tensors with matching byte sizes are rejected
  // in both configured and inferred modes.
  EXPECT_THAT(
      Any2AnyTtsDecoder::Create(
          strict_opts, MakeProjectorRunner(4, 4),
          make_custom_kv_decoder(
              []() {
                return support::CreateTensorBuffer<int32_t>({1, 2, 8, 4});
              },
              []() {
                return support::CreateTensorBuffer<int32_t>({1, 2, 8, 4});
              }))
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      Any2AnyTtsDecoder::Create(
          options, MakeProjectorRunner(4, 4),
          make_custom_kv_decoder(
              []() {
                return support::CreateTensorBuffer<int32_t>({1, 6, 12, 2});
              },
              []() {
                return support::CreateTensorBuffer<int32_t>({1, 6, 12, 2});
              }))
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Overflowing KV-cache dimensions are rejected without integer wrap-around.
  {
    auto overflow_gqa = MakeBaseOptions();
    overflow_gqa.tts_num_kv_heads = std::numeric_limits<int>::max();
    overflow_gqa.tts_head_dim = std::numeric_limits<int>::max();
    EXPECT_THAT(
        Any2AnyTtsDecoder::Create(overflow_gqa, MakeProjectorRunner(4, 4),
                                  MakeDecoderRunner(4, 8, 1, 8, 4))
            .status(),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    // `65536 * 65536` would overflow 32-bit signed `int` to 0 if unguarded.
    auto int_overflow_gqa = MakeBaseOptions();
    int_overflow_gqa.tts_num_kv_heads = 65536;
    int_overflow_gqa.tts_head_dim = 65536;
    EXPECT_THAT(
        Any2AnyTtsDecoder::Create(int_overflow_gqa, MakeProjectorRunner(4, 4),
                                  MakeDecoderRunner(4, 8, 1, 8, 2))
            .status(),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }
  {
    auto overflow_steps = MakeBaseOptions();
    overflow_steps.tts_num_layers = std::numeric_limits<int>::max();
    overflow_steps.tts_max_cache_len = std::numeric_limits<int>::max();
    EXPECT_THAT(
        Any2AnyTtsDecoder::Create(overflow_steps, MakeProjectorRunner(4, 4),
                                  MakeDecoderRunner(4, 8, 1, 8, 4))
            .status(),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(Any2AnyTtsDecoderTest,
     ProjectAndDecodeValidateVocabBoundsAndSwapKvBuffers) {
  auto options = MakeBaseOptions();
  options.llm_vocab_size = 50;
  options.tts_vocab_size = 8;

  auto proj = MakeProjectorRunner(4, 4);
  std::vector<int32_t> captured_proj_ids;
  EXPECT_CALL(*proj, Run(absl::string_view(""), _, _))
      .Times(2)
      .WillRepeatedly(
          [&](absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            std::vector<float> hidden(4, 0.0f);
            int32_t tok = 0;
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[0]).Read<float>(
                    absl::MakeSpan(hidden)));
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[1]).Read<int32_t>(
                    absl::MakeSpan(&tok, 1)));
            captured_proj_ids.push_back(tok);
            std::vector<float> out = {hidden[0] + static_cast<float>(tok),
                                      hidden[1], hidden[2], hidden[3]};
            LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(outputs[0])
                                       .Write<float>(absl::MakeConstSpan(out)));
            return absl::OkStatus();
          });

  auto dec = MakeDecoderRunner(4, 8, 1, 8, 4);
  std::vector<float> kv_slot0_seen;
  EXPECT_CALL(*dec, Run(absl::string_view(""), _, _))
      .Times(3)
      .WillRepeatedly(
          [&](absl::string_view, absl::Span<const TensorBuffer> inputs,
              absl::Span<const TensorBuffer> outputs) -> absl::Status {
            int32_t step_idx = 0;
            std::vector<float> kv(2 * 8 * 4, 0.0f);
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[2]).Read<int32_t>(
                    absl::MakeSpan(&step_idx, 1)));
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(inputs[3]).Read<float>(
                    absl::MakeSpan(kv)));
            kv_slot0_seen.push_back(kv[0]);
            kv[0] = static_cast<float>(step_idx + 10);
            std::vector<float> logits(8, 0.0f);
            logits[step_idx % 8] = 5.0f;
            LITERT_RETURN_IF_ERROR(
                const_cast<TensorBuffer&>(outputs[0])
                    .Write<float>(absl::MakeConstSpan(logits)));
            LITERT_RETURN_IF_ERROR(const_cast<TensorBuffer&>(outputs[1])
                                       .Write<float>(absl::MakeConstSpan(kv)));
            return absl::OkStatus();
          });

  ASSERT_OK_AND_ASSIGN(
      auto decoder,
      Any2AnyTtsDecoder::Create(options, std::move(proj), std::move(dec)));

  std::vector<float> hidden = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> cond(4, 0.0f);
  std::vector<float> short_hidden(3, 0.0f);
  std::vector<float> long_hidden(5, 0.0f);
  std::vector<float> short_cond(3, 0.0f);
  std::vector<float> long_cond(5, 0.0f);
  // Out-of-range text token IDs (including `< -1`) and mismatched spans
  // rejected without invoking the projector runner.
  EXPECT_THAT(
      decoder->ProjectTokenConditioning(hidden, 50, absl::MakeSpan(cond)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      decoder->ProjectTokenConditioning(hidden, -2, absl::MakeSpan(cond)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      decoder->ProjectTokenConditioning(
          hidden, std::numeric_limits<int32_t>::min(), absl::MakeSpan(cond)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      decoder->ProjectTokenConditioning(short_hidden, 1, absl::MakeSpan(cond)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      decoder->ProjectTokenConditioning(long_hidden, 1, absl::MakeSpan(cond)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      decoder->ProjectTokenConditioning(hidden, 1, absl::MakeSpan(short_cond)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      decoder->ProjectTokenConditioning(hidden, 1, absl::MakeSpan(long_cond)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  // `-1` placeholder and valid ID accepted.
  ASSERT_OK(
      decoder->ProjectTokenConditioning(hidden, -1, absl::MakeSpan(cond)));
  EXPECT_THAT(cond, ElementsAre(0.0f, 2.0f, 3.0f, 4.0f));
  ASSERT_OK(decoder->ProjectTokenConditioning(hidden, 7, absl::MakeSpan(cond)));
  EXPECT_THAT(cond, ElementsAre(8.0f, 2.0f, 3.0f, 4.0f));
  EXPECT_THAT(captured_proj_ids, ElementsAre(-1, 7));

  std::vector<float> logits(8, 0.0f);
  std::vector<float> short_logits(7, 0.0f);
  std::vector<float> long_logits(9, 0.0f);
  // Out-of-range `prev_audio_code`, `step_idx`, and mismatched spans rejected
  // without invoking the decoder runner.
  EXPECT_THAT(decoder->RunDecoderStep(cond, 8, 0, absl::MakeSpan(logits)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(decoder->RunDecoderStep(cond, -2, 0, absl::MakeSpan(logits)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(decoder->RunDecoderStep(cond, std::numeric_limits<int32_t>::min(),
                                      0, absl::MakeSpan(logits)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(decoder->RunDecoderStep(cond, -1, -1, absl::MakeSpan(logits)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(decoder->RunDecoderStep(cond, -1, 8, absl::MakeSpan(logits)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      decoder->RunDecoderStep(short_cond, -1, 0, absl::MakeSpan(logits)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(decoder->RunDecoderStep(long_cond, -1, 0, absl::MakeSpan(logits)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      decoder->RunDecoderStep(cond, -1, 0, absl::MakeSpan(short_logits)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(decoder->RunDecoderStep(cond, -1, 0, absl::MakeSpan(long_logits)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(decoder->RunDecoderStep(cond, -1, 0, absl::Span<float>{}),
              StatusIs(absl::StatusCode::kInvalidArgument));

  ASSERT_OK(decoder->ClearKvCache());
  // Passing `std::nullopt` skips logits readback while still swapping KV.
  logits.assign(8, -1.0f);
  ASSERT_OK(decoder->RunDecoderStep(cond, -1, 0, /*out_logits=*/std::nullopt));
  EXPECT_FLOAT_EQ(logits[0], -1.0f);
  // Step 1 must observe the KV value written at step 0 via buffer swap and
  // populate `logits` when a span is provided.
  ASSERT_OK(decoder->RunDecoderStep(cond, 0, 1, absl::MakeSpan(logits)));
  EXPECT_FLOAT_EQ(logits[1], 5.0f);
  // Clearing KV cache resets slot 0 back to 0.0f.
  ASSERT_OK(decoder->ClearKvCache());
  ASSERT_OK(decoder->RunDecoderStep(cond, -1, 0, absl::MakeSpan(logits)));
  EXPECT_FLOAT_EQ(logits[0], 5.0f);
  EXPECT_THAT(kv_slot0_seen, ElementsAre(0.0f, 10.0f, 0.0f));

  // Unbounded `llm_vocab_size = 0` still rejects `token_id < -1`, allows
  // arbitrary non-negative token IDs, and propagates runner errors.
  auto unbounded_opts = MakeBaseOptions();
  unbounded_opts.llm_vocab_size = 0;
  auto err_proj = MakeProjectorRunner(4, 4);
  EXPECT_CALL(*err_proj, Run(absl::string_view(""), _, _))
      .WillOnce(Return(absl::InternalError("proj fail")));
  auto err_dec = MakeDecoderRunner(4, 8, 1, 8, 4);
  EXPECT_CALL(*err_dec, Run(absl::string_view(""), _, _))
      .WillOnce(Return(absl::InternalError("dec fail")));
  ASSERT_OK_AND_ASSIGN(
      auto err_decoder,
      Any2AnyTtsDecoder::Create(unbounded_opts, std::move(err_proj),
                                std::move(err_dec)));
  EXPECT_THAT(
      err_decoder->ProjectTokenConditioning(hidden, -2, absl::MakeSpan(cond)),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(err_decoder->ProjectTokenConditioning(hidden, 999999,
                                                    absl::MakeSpan(cond)),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(err_decoder->RunDecoderStep(cond, -1, 0, absl::MakeSpan(logits)),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(Any2AnyTtsDecoderTest, ApplyRepetitionPenaltySupportsBothModes) {
  // Frequency-window mode: positive logits divided by penalty^freq, negative
  // logits multiplied by penalty^freq, respecting `window_size` and ignoring
  // out-of-range codes.
  std::vector<float> freq_logits = {8.0f, -2.0f, 4.0f};
  std::vector<int> history = {2, 0, 0, 1, -1, 99};
  // Window of last 5 elements: {0, 0, 1, -1, 99} -> code 2 outside window,
  // code 0 repeated twice, code 1 once.
  Any2AnyTtsDecoder::ApplyRepetitionPenalty(
      absl::MakeSpan(freq_logits), history, /*penalty=*/2.0f,
      Any2AnyTtsDecoder::RepetitionPenaltyMode::kFrequencyWindow,
      /*window_size=*/5);
  EXPECT_THAT(freq_logits,
              ElementsAre(FloatNear(2.0f, 1e-5f), FloatNear(-4.0f, 1e-5f),
                          FloatNear(4.0f, 1e-5f)));

  // Disabled, non-finite (`NaN`/`+Inf`), or non-positive-window penalty is a
  // no-op in both modes.
  constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();
  std::vector<float> untouched = {8.0f, -2.0f};
  Any2AnyTtsDecoder::ApplyRepetitionPenalty(
      absl::MakeSpan(untouched), {0, 1}, /*penalty=*/1.0f,
      Any2AnyTtsDecoder::RepetitionPenaltyMode::kFrequencyWindow,
      /*window_size=*/4);
  Any2AnyTtsDecoder::ApplyRepetitionPenalty(
      absl::MakeSpan(untouched), {0, 1}, /*penalty=*/kNan,
      Any2AnyTtsDecoder::RepetitionPenaltyMode::kFrequencyWindow,
      /*window_size=*/4);
  Any2AnyTtsDecoder::ApplyRepetitionPenalty(
      absl::MakeSpan(untouched), {0, 1}, /*penalty=*/kInf,
      Any2AnyTtsDecoder::RepetitionPenaltyMode::kFrequencyWindow,
      /*window_size=*/4);
  Any2AnyTtsDecoder::ApplyRepetitionPenalty(
      absl::MakeSpan(untouched), {0, 1}, /*penalty=*/kNan,
      Any2AnyTtsDecoder::RepetitionPenaltyMode::kUniqueHistory);
  Any2AnyTtsDecoder::ApplyRepetitionPenalty(
      absl::MakeSpan(untouched), {0, 1}, /*penalty=*/kInf,
      Any2AnyTtsDecoder::RepetitionPenaltyMode::kUniqueHistory);
  Any2AnyTtsDecoder::ApplyRepetitionPenalty(
      absl::MakeSpan(untouched), {0, 1}, /*penalty=*/2.0f,
      Any2AnyTtsDecoder::RepetitionPenaltyMode::kFrequencyWindow,
      /*window_size=*/0);
  EXPECT_THAT(untouched, ElementsAre(8.0f, -2.0f));

  // Unique-history mode: applies penalty once per unique code across entire
  // history regardless of multiplicity or window_size.
  std::vector<float> unique_logits = {8.0f, -2.0f, 4.0f};
  Any2AnyTtsDecoder::ApplyRepetitionPenalty(
      absl::MakeSpan(unique_logits), {2, 0, 0, 1, 1, -1, 99}, /*penalty=*/2.0f,
      Any2AnyTtsDecoder::RepetitionPenaltyMode::kUniqueHistory);
  EXPECT_THAT(unique_logits,
              ElementsAre(FloatNear(4.0f, 1e-5f), FloatNear(-4.0f, 1e-5f),
                          FloatNear(2.0f, 1e-5f)));

  // Member overload forwards decoder options and resets its seen-code epoch
  // table cleanly across sequential calls in `kUniqueHistory` mode.
  auto opts = MakeBaseOptions();
  opts.generate_audio = false;
  opts.repetition_penalty = 2.0f;
  opts.repetition_penalty_mode =
      Any2AnyTtsDecoder::RepetitionPenaltyMode::kUniqueHistory;
  ASSERT_OK_AND_ASSIGN(auto decoder, Any2AnyTtsDecoder::Create(opts));
  std::vector<float> member_logits = {6.0f, -3.0f, 10.0f};
  decoder->ApplyRepetitionPenalty(absl::MakeSpan(member_logits),
                                  {0, 0, 1, -1, 99});
  EXPECT_THAT(member_logits,
              ElementsAre(FloatNear(3.0f, 1e-5f), FloatNear(-6.0f, 1e-5f),
                          FloatNear(10.0f, 1e-5f)));

  // Second call with a disjoint history must penalize only the new history
  // entries (index 2) without leaking prior epoch entries (indices 0 and 1).
  std::vector<float> second_logits = {6.0f, -3.0f, 10.0f};
  decoder->ApplyRepetitionPenalty(absl::MakeSpan(second_logits), {2, 2});
  EXPECT_THAT(second_logits,
              ElementsAre(FloatNear(6.0f, 1e-5f), FloatNear(-3.0f, 1e-5f),
                          FloatNear(5.0f, 1e-5f)));

  // Third call with an overlapping history re-penalizes index 0 once.
  std::vector<float> third_logits = {6.0f, -3.0f, 10.0f};
  decoder->ApplyRepetitionPenalty(absl::MakeSpan(third_logits), {0, 0});
  EXPECT_THAT(third_logits,
              ElementsAre(FloatNear(3.0f, 1e-5f), FloatNear(-3.0f, 1e-5f),
                          FloatNear(10.0f, 1e-5f)));
}

TEST(Any2AnyTtsDecoderTest,
     SampleAudioCodeHandlesGreedyNonFiniteAndDeterministicStochasticSampling) {
  constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();

  auto greedy_opts = MakeBaseOptions();
  greedy_opts.generate_audio = false;
  greedy_opts.do_sample = false;
  ASSERT_OK_AND_ASSIGN(auto greedy, Any2AnyTtsDecoder::Create(greedy_opts));

  EXPECT_EQ(greedy->SampleAudioCode({}), 0);
  EXPECT_EQ(greedy->SampleAudioCode({kNan, -kInf, -1.5f, -3.0f}), 2);
  EXPECT_EQ(greedy->SampleAudioCode({100.0f, kInf, kInf}), 1);
  EXPECT_EQ(greedy->SampleAudioCode({kNan, -kInf, kNan}), 0);

  // Stochastic sampling with non-finite handling, top-k, top-p, and Reset().
  auto sample_opts = MakeBaseOptions();
  sample_opts.generate_audio = false;
  sample_opts.do_sample = true;
  sample_opts.temperature = 1.0f;
  sample_opts.top_k = 2;
  sample_opts.top_p = 0.8f;
  sample_opts.seed = 777;
  ASSERT_OK_AND_ASSIGN(auto sampler, Any2AnyTtsDecoder::Create(sample_opts));

  EXPECT_EQ(sampler->SampleAudioCode({kNan, kInf, 5.0f}), 1);
  EXPECT_EQ(sampler->SampleAudioCode({kNan, -kInf}), 0);

  // Top-k=2 restricts draws to indices {1, 3} (highest two finite logits), and
  // `Reset()` replays the exact same non-constant sequence.
  std::vector<float> logits = {kNan, 2.0f, -10.0f, 2.0f, -kInf};
  std::vector<int> first_run;
  for (int i = 0; i < 16; ++i) {
    const int code = sampler->SampleAudioCode(logits);
    EXPECT_TRUE(code == 1 || code == 3);
    first_run.push_back(code);
  }
  EXPECT_THAT(first_run, Contains(1));
  EXPECT_THAT(first_run, Contains(3));
  sampler->Reset();
  std::vector<int> second_run;
  for (int i = 0; i < 16; ++i) {
    second_run.push_back(sampler->SampleAudioCode(logits));
  }
  EXPECT_EQ(first_run, second_run);

  // Top-p=0.5 genuinely drops low-probability tail candidates: with logits
  // {1.0, 0.0, -10.0}, index 0 has ~73% probability (> 0.5), so indices 1 and 2
  // are pruned before sampling.
  auto top_p_opts = MakeBaseOptions();
  top_p_opts.generate_audio = false;
  top_p_opts.do_sample = true;
  top_p_opts.temperature = 1.0f;
  top_p_opts.top_k = 0;
  top_p_opts.top_p = 0.5f;
  top_p_opts.seed = 123;
  ASSERT_OK_AND_ASSIGN(auto top_p_sampler,
                       Any2AnyTtsDecoder::Create(top_p_opts));
  std::vector<float> skewed_logits = {1.0f, 0.0f, -10.0f};
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(top_p_sampler->SampleAudioCode(skewed_logits), 0);
  }

  // `NaN` temperature with `do_sample = true` falls back safely to greedy
  // selection rather than producing `NaN` weights.
  auto nan_temp_opts = MakeBaseOptions();
  nan_temp_opts.generate_audio = false;
  nan_temp_opts.do_sample = true;
  nan_temp_opts.temperature = kNan;
  ASSERT_OK_AND_ASSIGN(auto nan_temp_sampler,
                       Any2AnyTtsDecoder::Create(nan_temp_opts));
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(nan_temp_sampler->SampleAudioCode({1.0f, 5.0f, 2.0f}), 1);
  }
  EXPECT_EQ(nan_temp_sampler->SampleAudioCode({kNan, -kInf, -2.0f, -5.0f}), 2);
  EXPECT_EQ(nan_temp_sampler->SampleAudioCode({10.0f, kInf, 20.0f}), 1);

  // `+Inf` temperature with extreme finite logits stays finite and valid.
  auto inf_temp_opts = MakeBaseOptions();
  inf_temp_opts.generate_audio = false;
  inf_temp_opts.do_sample = true;
  inf_temp_opts.temperature = kInf;
  ASSERT_OK_AND_ASSIGN(auto inf_temp_sampler,
                       Any2AnyTtsDecoder::Create(inf_temp_opts));
  constexpr float kMaxFloat = std::numeric_limits<float>::max();
  for (int i = 0; i < 8; ++i) {
    const int code = inf_temp_sampler->SampleAudioCode({-kMaxFloat, kMaxFloat});
    EXPECT_TRUE(code == 0 || code == 1);
  }
}

TEST(Any2AnyTtsDecoderTest, ComputeMaxTotalCodesAndValidateLlmAudioInputs) {
  EXPECT_EQ(Any2AnyTtsDecoder::ComputeMaxTotalCodes(10, 0), 1);
  EXPECT_EQ(Any2AnyTtsDecoder::ComputeMaxTotalCodes(10, 50), 50);
  EXPECT_EQ(Any2AnyTtsDecoder::ComputeMaxTotalCodes(10, 1024), 1024);
  EXPECT_EQ(Any2AnyTtsDecoder::ComputeMaxTotalCodes(100, 1024), 2000);
  EXPECT_EQ(Any2AnyTtsDecoder::ComputeMaxTotalCodes(-5, 1024), 1024);

  EXPECT_THAT(Any2AnyTtsDecoder::ValidateLlmAudioInputs(
                  /*num_tokens=*/2, /*llm_hidden_size=*/3,
                  /*generated_token_ids=*/{10, 20},
                  /*llm_hidden_states=*/{1, 2, 3, 4, 5, 6},
                  /*prompt_token_ids=*/{1},
                  /*prompt_hidden_states=*/{7, 8, 9}),
              IsOk());
  EXPECT_THAT(Any2AnyTtsDecoder::ValidateLlmAudioInputs(
                  /*num_tokens=*/-1, /*llm_hidden_size=*/3, {}, {}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(Any2AnyTtsDecoder::ValidateLlmAudioInputs(
                  /*num_tokens=*/1, /*llm_hidden_size=*/0, {}, {}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      Any2AnyTtsDecoder::ValidateLlmAudioInputs(
          /*num_tokens=*/2, /*llm_hidden_size=*/3, {10, 20}, {1, 2, 3, 4}),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      Any2AnyTtsDecoder::ValidateLlmAudioInputs(
          /*num_tokens=*/2, /*llm_hidden_size=*/3, {10}, {1, 2, 3, 4, 5, 6}),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(Any2AnyTtsDecoder::ValidateLlmAudioInputs(
                  /*num_tokens=*/1, /*llm_hidden_size=*/3, {10}, {1, 2, 3},
                  /*prompt_token_ids=*/{1, 2},
                  /*prompt_hidden_states=*/{7, 8, 9}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::omni::any2any
