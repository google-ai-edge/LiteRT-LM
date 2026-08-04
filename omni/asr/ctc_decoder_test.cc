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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "support/util/test_utils.h"  // from @litert  // IWYU pragma: keep for ASSERT_OK

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

}  // namespace
}  // namespace litert::omni::asr
