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
// C++ port of `src/models/unigram/model.rs` (`#[cfg(test)] mod tests`) from the
// HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/models/unigram/model.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "support/tokenizer/tokenizers/models/unigram/lattice.h"
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/util/status_macros.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers::models::unigram {
namespace {

TEST(UnigramModelTest, TestPopulateNodesUnk) {
  const Vocab pieces = {{"<unk>", 0.0}};
  ASSERT_OK_AND_ASSIGN(Unigram model, Unigram::From(pieces, 0, false));

  Lattice lattice = Lattice::From("abc", model.bos_id(), model.eos_id());
  model.PopulateNodes(lattice);

  EXPECT_EQ(lattice.begin_nodes()[0].size(), 1);
  EXPECT_EQ(lattice.begin_nodes()[1].size(), 1);
  EXPECT_EQ(lattice.begin_nodes()[2].size(), 1);
  EXPECT_EQ(lattice.begin_nodes()[0][0]->id, 0);
  EXPECT_EQ(lattice.begin_nodes()[1][0]->id, 0);
  EXPECT_EQ(lattice.begin_nodes()[2][0]->id, 0);
  EXPECT_EQ(lattice.begin_nodes()[0][0]->node_id, 2);
  EXPECT_EQ(lattice.begin_nodes()[1][0]->node_id, 3);
  EXPECT_EQ(lattice.begin_nodes()[2][0]->node_id, 4);
}

TEST(UnigramModelTest, TestPopulateNodes) {
  const Vocab pieces = {
      {"<unk>", 0.0}, {"a", 0.1}, {"b", 0.2}, {"ab", 0.3}, {"bc", 0.4},
  };
  ASSERT_OK_AND_ASSIGN(Unigram model, Unigram::From(pieces, 0, false));

  Lattice lattice = Lattice::From("abc", model.bos_id(), model.eos_id());
  model.PopulateNodes(lattice);

  EXPECT_EQ(lattice.begin_nodes()[0].size(), 2);  // a, ab
  EXPECT_EQ(lattice.begin_nodes()[1].size(), 2);  // b, bc
  EXPECT_EQ(lattice.begin_nodes()[2].size(), 1);  // c(unk)

  // Id is the vocabulary id from Unigram model
  // node_id is simply the rank of the given node in the lattice.
  EXPECT_EQ(lattice.begin_nodes()[0][0]->id, 1);
  EXPECT_EQ(lattice.begin_nodes()[0][1]->id, 3);
  EXPECT_EQ(lattice.begin_nodes()[1][0]->id, 2);
  EXPECT_EQ(lattice.begin_nodes()[1][1]->id, 4);
  EXPECT_EQ(lattice.begin_nodes()[2][0]->id, 0);
  EXPECT_EQ(lattice.begin_nodes()[0][0]->node_id, 2);
  EXPECT_EQ(lattice.begin_nodes()[0][1]->node_id, 3);
  EXPECT_EQ(lattice.begin_nodes()[1][0]->node_id, 4);
  EXPECT_EQ(lattice.begin_nodes()[1][1]->node_id, 5);
  EXPECT_EQ(lattice.begin_nodes()[2][0]->node_id, 6);
}

TEST(UnigramModelTest, TestEncode) {
  const Vocab sentencepieces = {
      {"<unk>", 0.0}, {"a", 0.0},  {"b", 0.0},   {"c", 0.0},     {"d", 0.0},
      {"cd", 1.0},    {"ab", 2.0}, {"abc", 5.0}, {"abcd", 10.0},
  };

  ASSERT_OK_AND_ASSIGN(Unigram model, Unigram::From(sentencepieces, 0, false));
  ASSERT_OK_AND_ASSIGN(std::vector<std::string> result, model.Encode("abcd"));
  EXPECT_EQ(result, (std::vector<std::string>{"abcd"}));
}

TEST(UnigramModelTest, TestEncode2) {
  const Vocab sentencepieces = {
      {"<unk>", 0.0},     {"ab", 0.0}, {"cd", -0.1}, {"abc", -0.2},
      {"a", -0.3},        {"b", -0.4}, {"c", -0.5},  {"ABC", -0.5},
      {"abcdabcd", 20.0},  // User defined just max the scores.
      {"q", 20.5},        {"r", 20.5}, {"qr", -0.5},
  };

  ASSERT_OK_AND_ASSIGN(Unigram model, Unigram::From(sentencepieces, 0, false));

  for (const bool is_optimized : {true, false}) {
    model.SetOptimized(is_optimized);
    ASSERT_OK_AND_ASSIGN(std::vector<std::string> res, model.Encode("abc"));
    EXPECT_EQ(res, (std::vector<std::string>{"abc"}));
    ASSERT_OK_AND_ASSIGN(res, model.Encode("AB"));
    EXPECT_EQ(res, (std::vector<std::string>{"AB"}));

    model.SetFuseUnk(false);
    ASSERT_OK_AND_ASSIGN(res, model.Encode("AB"));
    EXPECT_EQ(res, (std::vector<std::string>{"A", "B"}));
    model.SetFuseUnk(true);
    ASSERT_OK_AND_ASSIGN(res, model.Encode("AB"));
    EXPECT_EQ(res, (std::vector<std::string>{"AB"}));

    ASSERT_OK_AND_ASSIGN(res, model.Encode("abcd"));
    EXPECT_EQ(res, (std::vector<std::string>{"ab", "cd"}));
    ASSERT_OK_AND_ASSIGN(res, model.Encode("abcc"));
    EXPECT_EQ(res, (std::vector<std::string>{"abc", "c"}));
    ASSERT_OK_AND_ASSIGN(res, model.Encode("xabcabaabcdd"));
    EXPECT_EQ(res, (std::vector<std::string>{"x", "abc", "ab", "a", "ab", "cd",
                                             "d"}));
    model.SetFuseUnk(false);
    ASSERT_OK_AND_ASSIGN(res, model.Encode("xyz東京"));
    EXPECT_EQ(res, (std::vector<std::string>{"x", "y", "z", "東", "京"}));
    model.SetFuseUnk(true);
    ASSERT_OK_AND_ASSIGN(res, model.Encode("xyz東京"));
    EXPECT_EQ(res, (std::vector<std::string>{"xyz東京"}));

    // User encoded in original version
    ASSERT_OK_AND_ASSIGN(res, model.Encode("ABC"));
    EXPECT_EQ(res, (std::vector<std::string>{"ABC"}));
    ASSERT_OK_AND_ASSIGN(res, model.Encode("abABCcd"));
    EXPECT_EQ(res, (std::vector<std::string>{"ab", "ABC", "cd"}));
    ASSERT_OK_AND_ASSIGN(res, model.Encode("ababcdabcdcd"));
    EXPECT_EQ(res, (std::vector<std::string>{"ab", "abcdabcd", "cd"}));
    ASSERT_OK_AND_ASSIGN(res, model.Encode("abqrcd"));
    EXPECT_EQ(res, (std::vector<std::string>{"ab", "q", "r", "cd"}));
  }
}

TEST(UnigramModelTest, TestUnigramBytefallback) {
  // In [97]: processor.encode_as_pieces("⅐⅛⅑ ")
  // Out[97]: ['▁', '<0xE2>', '<0x85>', '<0x90>', '⅛', '<0xE2>', '<0x85>',
  // '<0x91>', '▁']
  const Vocab sentencepieces = {
      {"<unk>", 0.0},
      {"<0xC3>", -0.01},
      {"<0xA9>", -0.03},
  };
  ASSERT_OK_AND_ASSIGN(Unigram unigram, Unigram::From(sentencepieces, 0, true));
  ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, unigram.Tokenize("é"));
  EXPECT_EQ(tokens, (std::vector<Token>{
                        Token{1, "<0xC3>", {0, 2}},
                        Token{2, "<0xA9>", {0, 2}},
                    }));

  ASSERT_OK_AND_ASSIGN(tokens, unigram.Tokenize("?é"));
  EXPECT_EQ(tokens[0].id, 0);
}

}  // namespace
}  // namespace tokenizers::models::unigram
