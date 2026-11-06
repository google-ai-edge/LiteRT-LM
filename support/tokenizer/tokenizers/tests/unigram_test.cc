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
// C++ port of `tests/unigram.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.
//
// PORT NOTE: `test_train_unigram_from_file` is not ported (`UnigramTrainer` /
// model training is out of scope).

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/unigram/lattice.h"
#include "support/tokenizer/tokenizers/models/unigram/model.h"
#include "support/tokenizer/tokenizers/tests/common.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace tokenizers {
namespace {

using ::tokenizers::models::unigram::Lattice;
using ::tokenizers::models::unigram::Unigram;
using ::tokenizers::test_common::TestdataPath;

TEST(UnigramIntegrationTest, UnigramFromFile) {
  ASSERT_OK_AND_ASSIGN(Unigram model,
                       Unigram::Load(TestdataPath("unigram.json")));
  const char* string = "吾輩《わがはい》は猫である。名前はまだ無い。";
  ASSERT_OK_AND_ASSIGN(std::vector<Token> tokens, model.Tokenize(string));
  std::vector<std::string> values;
  values.reserve(tokens.size());
  for (const Token& tok : tokens) {
    values.push_back(tok.value);
  }
  EXPECT_EQ(values, (std::vector<std::string>{
                        "吾輩",
                        "《",
                        "わが",
                        "はい",
                        "》",
                        "は",
                        "猫",
                        "である",
                        "。",
                        "名前",
                        "はまだ",
                        "無い",
                        "。",
                    }));
}

TEST(UnigramIntegrationTest, Sample) {
  Lattice lattice = Lattice::From("ABC", 0, 2);
  lattice.Insert(0, 1, 1.0, 3);  // A
  lattice.Insert(1, 1, 1.2, 4);  // B
  lattice.Insert(2, 1, 1.5, 5);  // C
  lattice.Insert(0, 2, 1.6, 6);  // AB
  lattice.Insert(1, 2, 1.7, 7);  // BC
  lattice.Insert(0, 3, 1.8, 8);  // ABC

  std::vector<double> thetas = {0.0, 0.01, 0.5, 0.7, 1.0};

  for (double theta : thetas) {
    absl::flat_hash_map<std::string, double> probs;
    probs.emplace("A B C", std::exp(theta * (1.0 + 1.2 + 1.5)));
    probs.emplace("AB C", std::exp(theta * (1.6 + 1.5)));
    probs.emplace("A BC", std::exp(theta * (1.0 + 1.7)));
    probs.emplace("ABC", std::exp(theta * 1.8));

    // Computes expected probabilities.
    double z = 0.0;
    for (const auto& [_, p] : probs) {
      z += p;
    }
    for (auto& [_, p] : probs) {
      p /= z;
    }

    const int n_trials = 10000;
    absl::flat_hash_map<std::string, uint32_t> freq;
    for (int i = 0; i < n_trials; ++i) {
      ASSERT_OK_AND_ASSIGN(std::vector<std::string> pieces,
                           lattice.SampleToken(theta));
      std::string s = absl::StrJoin(pieces, " ");
      freq[s] += 1;
    }

    EXPECT_EQ(freq.size(), probs.size());
    for (const auto& [s, p] : probs) {
      EXPECT_NEAR(
          1.0 * static_cast<double>(freq[s]) / static_cast<double>(n_trials), p,
          0.03);
    }
  }
}

}  // namespace
}  // namespace tokenizers
