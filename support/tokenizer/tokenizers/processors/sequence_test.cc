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
// C++ port of the `#[cfg(test)]` module of `src/processors/sequence.rs` from
// the HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0, Copyright
// 2020 The HuggingFace Team).

#include "support/tokenizer/tokenizers/processors/sequence.h"

#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/processors/processors.h"
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace processors {
namespace {

TEST(SequenceTest, ProcessChain) {
  Encoding start(std::vector<uint32_t>(5, 0), std::vector<uint32_t>(5, 0),
                 {"Ġ", "ĠĠĠĠHelloĠĠ", "ĠĠHello", "HelloĠĠ", "ĠĠĠĠ"}, {},
                 {{0, 1}, {0, 11}, {11, 18}, {18, 25}, {25, 29}}, {}, {}, {},
                 {});

  ByteLevel bytelevel = ByteLevel().TrimOffsets(true);
  Sequence sequence({PostProcessorWrapper(bytelevel)});
  Encoding expected(std::vector<uint32_t>(5, 0), std::vector<uint32_t>(5, 0),
                    {"Ġ", "ĠĠĠĠHelloĠĠ", "ĠĠHello", "HelloĠĠ", "ĠĠĠĠ"}, {},
                    {{0, 0}, {4, 9}, {13, 18}, {18, 23}, {29, 29}}, {}, {}, {},
                    {{0, {0, 5}}});

  ASSERT_OK_AND_ASSIGN(Encoding bytelevel_out,
                       bytelevel.Process(start, std::nullopt, false));
  EXPECT_EQ(expected, bytelevel_out);
  ASSERT_OK_AND_ASSIGN(Encoding sequence_out,
                       sequence.Process(start, std::nullopt, false));
  EXPECT_EQ(expected, sequence_out);

  Encoding pair_expected(std::vector<uint32_t>(10, 0),
                         {0, 0, 0, 0, 0, 1, 1, 1, 1, 1},
                         {"Ġ", "ĠĠĠĠHelloĠĠ", "ĠĠHello", "HelloĠĠ", "ĠĠĠĠ", "Ġ",
                          "ĠĠĠĠHelloĠĠ", "ĠĠHello", "HelloĠĠ", "ĠĠĠĠ"},
                         {},
                         {{0, 0},
                          {4, 9},
                          {13, 18},
                          {18, 23},
                          {29, 29},
                          {0, 0},
                          {4, 9},
                          {13, 18},
                          {18, 23},
                          {29, 29}},
                         {}, {}, {}, {{0, {0, 5}}, {1, {5, 10}}});
  ASSERT_OK_AND_ASSIGN(Encoding bytelevel_pair_out,
                       bytelevel.Process(start, start, false));
  EXPECT_EQ(pair_expected, bytelevel_pair_out);
  ASSERT_OK_AND_ASSIGN(Encoding sequence_pair_out,
                       sequence.Process(start, start, false));
  EXPECT_EQ(pair_expected, sequence_pair_out);
}

}  // namespace
}  // namespace processors
}  // namespace tokenizers
