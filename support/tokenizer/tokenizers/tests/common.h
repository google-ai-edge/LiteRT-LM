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
// C++ port of `tests/common/mod.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TESTS_COMMON_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TESTS_COMMON_H_

#include <cstdint>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/decoders/wordpiece.h"
#include "support/tokenizer/tokenizers/models/bpe/model.h"
#include "support/tokenizer/tokenizers/models/wordpiece/wordpiece.h"
#include "support/tokenizer/tokenizers/normalizers/bert.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/bert.h"
#include "support/tokenizer/tokenizers/pre_tokenizers/byte_level.h"
#include "support/tokenizer/tokenizers/processors/bert.h"
#include "support/tokenizer/tokenizers/tokenizer/tokenizer.h"

namespace tokenizers {
namespace test_common {

inline constexpr char kTestdataDir[] =
    "support/tokenizer/tokenizers/testdata/";

inline std::string TestdataPath(absl::string_view filename) {
  return (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
          std::string(filename))
      .string();
}

inline Tokenizer GetEmpty() { return Tokenizer(models::bpe::BPE()); }

inline models::bpe::BPE GetByteLevelBpe() {
  return models::bpe::BPE::FromFile(TestdataPath("gpt2-vocab.json"),
                                    TestdataPath("gpt2-merges.txt"))
      .Build()
      .value();
}

inline Tokenizer GetByteLevel(bool add_prefix_space, bool trim_offsets) {
  Tokenizer tokenizer(GetByteLevelBpe());
  tokenizer
      .WithPreTokenizer(
          pre_tokenizers::ByteLevel().AddPrefixSpace(add_prefix_space))
      .WithDecoder(pre_tokenizers::ByteLevel())
      .WithPostProcessor(pre_tokenizers::ByteLevel().TrimOffsets(trim_offsets));

  return tokenizer;
}

inline models::wordpiece::WordPiece GetBertWordpiece() {
  return models::wordpiece::WordPiece::FromFile(
             TestdataPath("bert-base-uncased-vocab.txt"))
      .Build()
      .value();
}

inline Tokenizer GetBert() {
  Tokenizer tokenizer(GetBertWordpiece());
  uint32_t sep = tokenizer.GetModel().TokenToId("[SEP]").value();
  uint32_t cls = tokenizer.GetModel().TokenToId("[CLS]").value();
  (void)tokenizer.WithNormalizer(normalizers::BertNormalizer());
  tokenizer.WithPreTokenizer(pre_tokenizers::BertPreTokenizer())
      .WithDecoder(decoders::WordPiece())
      .WithPostProcessor(processors::BertProcessing(
          {std::string("[SEP]"), sep}, {std::string("[CLS]"), cls}));

  return tokenizer;
}

}  // namespace test_common
}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TESTS_COMMON_H_
