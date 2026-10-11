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

#include "omni/asr/ctc_decoder.h"

#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::asr {
namespace {

TEST(CtcDecoderTest, CreateInvalidVocabSizeFails) {
  auto decoder_or = CtcDecoder::Create(/*vocab_size=*/0);
  EXPECT_FALSE(decoder_or.ok());
}

TEST(CtcDecoderTest, CreateValidVocabSizeSucceeds) {
  auto decoder_or = CtcDecoder::Create(/*vocab_size=*/4);
  ASSERT_OK(decoder_or.status());
  EXPECT_NE(*decoder_or, nullptr);
}

TEST(CtcDecoderTest, DecodeEmptyEncoderOutputsReturnsEmpty) {
  auto decoder_or = CtcDecoder::Create(/*vocab_size=*/4);
  ASSERT_OK(decoder_or.status());
  std::vector<::litert::TensorBuffer> empty_outputs;
  auto tokens_or = (*decoder_or)->Decode(empty_outputs);
  ASSERT_OK(tokens_or.status());
  EXPECT_TRUE(tokens_or->empty());
}

TEST(CtcDecoderTest, DecodeIncludesEndOfChunkToken) {
  ASSERT_OK_AND_ASSIGN(auto decoder, CtcDecoder::Create(/*vocab_size=*/2));
  alignas(64) float logits[] = {1.0f, 0.0f, 0.0f, 1.0f};
  ::litert::RankedTensorType tensor_type(
      ::litert::ElementType::Float32,
      ::litert::Layout(::litert::Dimensions{2, 2}));
  auto buffer_or = ::litert::TensorBuffer::CreateFromHostMemory(
      tensor_type, logits, sizeof(logits));
  ASSERT_TRUE(buffer_or.HasValue());
  std::vector<::litert::TensorBuffer> outputs;
  outputs.push_back(std::move(*buffer_or));

  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(outputs));
  ASSERT_GE(tokens.size(), 1);
  EXPECT_TRUE(tokens.back().IsEndOfChunk());
  EXPECT_EQ(tokens.back().timestamp_ms, 2);
}

TEST(CtcDecoderTest, SetDecodeRangeValidFractionBoundsDecoding) {
  constexpr int kNumFrames = 8;
  ASSERT_OK_AND_ASSIGN(auto decoder, CtcDecoder::Create(/*vocab_size=*/2));
  // Frame t emits token (t % 2), so every frame yields a new token.
  alignas(64) float logits[kNumFrames * 2];
  for (int t = 0; t < kNumFrames; ++t) {
    logits[t * 2] = (t % 2 == 0) ? 1.0f : 0.0f;
    logits[t * 2 + 1] = (t % 2 == 0) ? 0.0f : 1.0f;
  }
  ::litert::RankedTensorType tensor_type(
      ::litert::ElementType::Float32,
      ::litert::Layout(::litert::Dimensions{kNumFrames, 2}));
  auto buffer_or = ::litert::TensorBuffer::CreateFromHostMemory(
      tensor_type, logits, sizeof(logits));
  ASSERT_TRUE(buffer_or.HasValue());
  std::vector<::litert::TensorBuffer> outputs;
  outputs.push_back(std::move(*buffer_or));

  // 25% of the window is real audio: round(0.25 * 8) = 2 valid frames, plus
  // the margin, are decoded; the end-of-chunk token still marks the window end.
  decoder->SetDecodeRange({.valid_fraction = 0.25f});
  ASSERT_OK_AND_ASSIGN(auto tokens, decoder->Decode(outputs));
  const int expected_frames = 2 + CtcDecoder::kValidFrameMargin;
  ASSERT_EQ(tokens.size(), expected_frames + 1);
  EXPECT_EQ(tokens[expected_frames - 1].timestamp_ms, expected_frames - 1);
  EXPECT_TRUE(tokens.back().IsEndOfChunk());
  EXPECT_EQ(tokens.back().timestamp_ms, kNumFrames);

  // The bound only applies to the next Decode().
  ASSERT_OK_AND_ASSIGN(tokens, decoder->Decode(outputs));
  EXPECT_EQ(tokens.size(), kNumFrames + 1);

  // Fractions outside (0, 1) mean "decode everything".
  decoder->SetDecodeRange({.valid_fraction = 0.0f});
  ASSERT_OK_AND_ASSIGN(tokens, decoder->Decode(outputs));
  EXPECT_EQ(tokens.size(), kNumFrames + 1);
}

}  // namespace
}  // namespace litert::omni::asr
