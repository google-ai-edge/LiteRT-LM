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
// C++ port of `src/models/unigram/lattice.rs` (`#[cfg(test)] mod tests`) from
// the HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/models/unigram/lattice.h"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"  // from @com_google_absl

namespace tokenizers::models::unigram {
namespace {

TEST(LatticeTest, SetSentence) {
  Lattice lattice = Lattice::From("", 1, 2);

  EXPECT_EQ(lattice.Len(), 0);
  EXPECT_EQ(lattice.Sentence(), "");
  EXPECT_EQ(lattice.Surface(0), "");

  Lattice lattice2 = Lattice::From("test", 1, 2);
  EXPECT_EQ(lattice2.Len(), 4);
  EXPECT_EQ(lattice2.Sentence(), "test");
  EXPECT_EQ(lattice2.Surface(0), "test");
  EXPECT_EQ(lattice2.Surface(1), "est");
  EXPECT_EQ(lattice2.Surface(2), "st");
  EXPECT_EQ(lattice2.Surface(3), "t");

  NodeRef bos = lattice2.BosNode();
  NodeRef eos = lattice2.EosNode();

  EXPECT_EQ(bos->id, 1);
  EXPECT_EQ(eos->id, 2);
  EXPECT_EQ(lattice2.end_nodes()[0].front()->id, bos->id);
  EXPECT_EQ(lattice2.begin_nodes()[4].front()->id, eos->id);

  Lattice lattice3 = Lattice::From("テストab", 1, 2);
  EXPECT_EQ(lattice3.Len(), 11);
  EXPECT_EQ(lattice3.Sentence(), "テストab");
  EXPECT_EQ(lattice3.Surface(0), "テストab");
  EXPECT_EQ(lattice3.Surface(1), "ストab");
  EXPECT_EQ(lattice3.Surface(2), "トab");
  EXPECT_EQ(lattice3.Surface(3), "ab");
  EXPECT_EQ(lattice3.Surface(4), "b");
}

TEST(LatticeTest, InsertTest) {
  Lattice lattice = Lattice::From("ABあい", 1, 2);

  lattice.Insert(0, 1, 0.0, 3);
  lattice.Insert(1, 1, 0.0, 4);
  lattice.Insert(2, 3, 0.0, 5);
  lattice.Insert(5, 3, 0.0, 6);
  lattice.Insert(0, 2, 0.0, 7);
  lattice.Insert(1, 4, 0.0, 8);
  lattice.Insert(2, 6, 0.0, 9);
  // 0 & 1 are bos and eos
  const Node& node0 = *lattice.nodes()[2];
  const Node& node1 = *lattice.nodes()[3];
  const Node& node2 = *lattice.nodes()[4];
  const Node& node3 = *lattice.nodes()[5];
  const Node& node4 = *lattice.nodes()[6];
  const Node& node5 = *lattice.nodes()[7];
  const Node& node6 = *lattice.nodes()[8];

  EXPECT_EQ(lattice.Piece(node0), "A");
  EXPECT_EQ(lattice.Piece(node1), "B");
  EXPECT_EQ(lattice.Piece(node2), "あ");
  EXPECT_EQ(lattice.Piece(node3), "い");
  EXPECT_EQ(lattice.Piece(node4), "AB");
  EXPECT_EQ(lattice.Piece(node5), "Bあ");
  EXPECT_EQ(lattice.Piece(node6), "あい");

  EXPECT_EQ(node0.pos, 0);
  EXPECT_EQ(node1.pos, 1);
  EXPECT_EQ(node2.pos, 2);
  EXPECT_EQ(node3.pos, 5);
  EXPECT_EQ(node4.pos, 0);
  EXPECT_EQ(node5.pos, 1);
  EXPECT_EQ(node6.pos, 2);

  EXPECT_EQ(node0.length, 1);
  EXPECT_EQ(node1.length, 1);
  EXPECT_EQ(node2.length, 3);
  EXPECT_EQ(node3.length, 3);
  EXPECT_EQ(node4.length, 2);
  EXPECT_EQ(node5.length, 4);
  EXPECT_EQ(node6.length, 6);

  EXPECT_EQ(lattice.BosNode()->id, 1);
  EXPECT_EQ(lattice.EosNode()->id, 2);
  EXPECT_EQ(node0.id, 3);
  EXPECT_EQ(node1.id, 4);
  EXPECT_EQ(node2.id, 5);
  EXPECT_EQ(node3.id, 6);
  EXPECT_EQ(node4.id, 7);
  EXPECT_EQ(node5.id, 8);
  EXPECT_EQ(node6.id, 9);

  EXPECT_EQ(lattice.begin_nodes()[0].size(), 2);
  EXPECT_EQ(lattice.begin_nodes()[1].size(), 2);
  EXPECT_EQ(lattice.begin_nodes()[2].size(), 2);
  EXPECT_EQ(lattice.begin_nodes()[5].size(), 1);
  EXPECT_EQ(lattice.begin_nodes()[8].size(), 1);

  EXPECT_EQ(lattice.end_nodes()[0].size(), 1);
  EXPECT_EQ(lattice.end_nodes()[1].size(), 1);
  EXPECT_EQ(lattice.end_nodes()[2].size(), 2);
  EXPECT_EQ(lattice.end_nodes()[5].size(), 2);
  EXPECT_EQ(lattice.end_nodes()[8].size(), 2);

  EXPECT_EQ(lattice.begin_nodes()[0][0]->id, node0.id);
  EXPECT_EQ(lattice.begin_nodes()[0][1]->id, node4.id);
  EXPECT_EQ(lattice.begin_nodes()[1][0]->id, node1.id);
  EXPECT_EQ(lattice.begin_nodes()[1][1]->id, node5.id);
  EXPECT_EQ(lattice.begin_nodes()[2][0]->id, node2.id);
  EXPECT_EQ(lattice.begin_nodes()[2][1]->id, node6.id);
  EXPECT_EQ(lattice.begin_nodes()[5][0]->id, node3.id);
  EXPECT_EQ(lattice.EosNode()->id, lattice.begin_nodes()[8][0]->id);

  EXPECT_EQ(lattice.BosNode()->id, lattice.end_nodes()[0][0]->id);
  EXPECT_EQ(node0.id, lattice.end_nodes()[1][0]->id);
  EXPECT_EQ(node1.id, lattice.end_nodes()[2][0]->id);
  EXPECT_EQ(node4.id, lattice.end_nodes()[2][1]->id);
  EXPECT_EQ(node2.id, lattice.end_nodes()[5][0]->id);
  EXPECT_EQ(node5.id, lattice.end_nodes()[5][1]->id);
  EXPECT_EQ(node3.id, lattice.end_nodes()[8][0]->id);
  EXPECT_EQ(node6.id, lattice.end_nodes()[8][1]->id);
}

TEST(LatticeTest, TestViterbi) {
  Lattice lattice = Lattice::From("ABC", 1, 2);
  EXPECT_TRUE(lattice.Viterbi().empty());
  // Still incomplete
  lattice.Insert(0, 1, 0.0, 3);
  EXPECT_TRUE(lattice.Viterbi().empty());
  lattice.Insert(1, 1, 0.0, 4);
  lattice.Insert(2, 1, 0.0, 5);
  // XXX: In sentence piece this is not tested, still incomplete ?
  EXPECT_EQ(lattice.Viterbi().size(), 3);
}

TEST(LatticeTest, TestViterbi2) {
  Lattice lattice = Lattice::From("ABC", 1, 2);

  lattice.Insert(0, 1, 0.0, 3);
  lattice.Insert(1, 1, 0.0, 4);
  lattice.Insert(2, 1, 0.0, 5);

  EXPECT_EQ(lattice.Tokens(), (std::vector<std::string>{"A", "B", "C"}));

  lattice.Insert(0, 2, 2.0, 6);
  EXPECT_EQ(lattice.Tokens(), (std::vector<std::string>{"AB", "C"}));

  lattice.Insert(1, 2, 5.0, 7);
  EXPECT_EQ(lattice.Tokens(), (std::vector<std::string>{"A", "BC"}));

  lattice.Insert(0, 3, 10.0, 8);
  EXPECT_EQ(lattice.Tokens(), (std::vector<std::string>{"ABC"}));
}

TEST(LatticeTest, TestNbest) {
  Lattice lattice = Lattice::From("ABC", 1, 2);
  lattice.Insert(0, 1, 0.0, 3);
  lattice.Insert(1, 1, 0.0, 4);
  lattice.Insert(2, 1, 0.0, 5);
  lattice.Insert(0, 2, 2.0, 6);
  lattice.Insert(1, 2, 5.0, 7);
  lattice.Insert(0, 3, 10.0, 8);

  const std::vector<std::vector<std::string>> nbests = lattice.NbestTokens(10);
  EXPECT_EQ(nbests, (std::vector<std::vector<std::string>>{
                        {"ABC"},
                        {"A", "BC"},
                        {"AB", "C"},
                        {"A", "B", "C"},
                    }));

  EXPECT_TRUE(lattice.NbestTokens(0).empty());
  EXPECT_EQ(lattice.NbestTokens(1),
            (std::vector<std::vector<std::string>>{{"ABC"}}));
}

TEST(LatticeTest, TestLogSumExp) {
  double x = 0.0;

  const std::vector<double> v = {1.0, 2.0, 3.0};
  for (size_t i = 0; i < v.size(); ++i) {
    x = LogSumExp(x, v[i], i == 0);
  }
  double sum_exp = 0.0;
  for (const double n : v) {
    sum_exp += std::exp(n);
  }
  EXPECT_NEAR(x, std::log(sum_exp), 0.001);
}

TEST(LatticeTest, TestPopulate) {
  Lattice lattice = Lattice::From("ABC", 1, 2);
  lattice.Insert(0, 1, 1.0, 3);  // A
  lattice.Insert(1, 1, 1.2, 4);  // B
  lattice.Insert(2, 1, 2.5, 5);  // C
  lattice.Insert(0, 2, 3.0, 6);  // AB
  lattice.Insert(1, 2, 4.0, 7);  // BC
  lattice.Insert(0, 3, 2.0, 8);  // ABC

  std::vector<double> probs(9, 0.0);
  const double p1 = std::exp(1.0 + 1.2 + 2.5);
  const double p2 = std::exp(3.0 + 2.5);
  const double p3 = std::exp(1.0 + 4.0);
  const double p4 = std::exp(2.0);
  const double z = p1 + p2 + p3 + p4;

  const double log_z = lattice.PopulateMarginal(1.0, absl::MakeSpan(probs));

  EXPECT_NEAR(log_z, std::log(z), 0.001);
  EXPECT_NEAR(probs[0], 0.0, 0.001);
  EXPECT_NEAR(probs[1], 0.0, 0.001);
  EXPECT_NEAR(probs[2], 0.0, 0.001);
  EXPECT_NEAR(probs[3], (p1 + p3) / z, 0.001);
  EXPECT_NEAR(probs[4], p1 / z, 0.001);
  EXPECT_NEAR(probs[5], (p1 + p2) / z, 0.001);
  EXPECT_NEAR(probs[6], p2 / z, 0.001);
  EXPECT_NEAR(probs[7], p3 / z, 0.001);
  EXPECT_NEAR(probs[8], p4 / z, 0.001);
}

}  // namespace
}  // namespace tokenizers::models::unigram
