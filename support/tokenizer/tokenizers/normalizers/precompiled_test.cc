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
// C++ port of the tests of `src/normalizers/precompiled.rs` from the
// HuggingFace `tokenizers` crate v0.23.2
// (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team).

#include "support/tokenizer/tokenizers/normalizers/precompiled.h"

#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"

namespace tokenizers::normalizers {
namespace {

TEST(PrecompiledTest, ExpansionFollowedByRemoval) {
  // Simulate transformations from "™\x1eg" to "TMg"
  std::vector<std::pair<char32_t, int64_t>> transformations;

  NormalizedString n = NormalizedString::From(
      "™\x1e"
      "g");
  precompiled::Replace(transformations, "™", "TM");
  precompiled::Replace(transformations, "\x1e", "");
  transformations.push_back({U'g', 0});

  n.Transform(transformations, 0);

  EXPECT_EQ(n.Get(), "TMg");
}

}  // namespace
}  // namespace tokenizers::normalizers
