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
//
// C++ port of `src/decoders/sequence.rs` (tests) from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/decoders/sequence.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "support/tokenizer/tokenizers/decoders/ctc.h"
#include "support/tokenizer/tokenizers/decoders/decoders.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/metaspace.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace decoders {
namespace {

TEST(SequenceTest, SequenceBasic) {
  std::vector<DecoderWrapper> decoders = {
      DecoderWrapper(CTC()),
      DecoderWrapper(pre_tokenizers::Metaspace()),
  };
  Sequence decoder(decoders);
  std::vector<std::string> tokens = {"▁", "▁", "H", "H", "i",
                                     "i", "▁", "y", "o", "u"};
  ASSERT_OK_AND_ASSIGN(std::string out_tokens, decoder.Decode(tokens));
  EXPECT_EQ(out_tokens, "Hi you");
}

}  // namespace
}  // namespace decoders
}  // namespace tokenizers
