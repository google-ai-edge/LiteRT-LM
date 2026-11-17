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

#include "runtime/components/random_state.h"

#include <cstdint>
#include <random>

#include <gtest/gtest.h>

namespace litert::lm {
namespace {

TEST(RandomStateTest, FromSeedSeedsTheSamplingEngine) {
  RandomState state = RandomState::FromSeed(42);

  std::default_random_engine expected(42);
  EXPECT_EQ(state.sampling, expected);
}

TEST(RandomStateTest, FromSeedSaltsTheLenienceStreams) {
  RandomState state = RandomState::FromSeed(42);

  EXPECT_EQ(state.drafter_lenience, std::mt19937_64(42 ^ 0x4c454e49));
  EXPECT_EQ(state.draft_verifier_lenience, std::mt19937_64(42 ^ 0x44524654));
}

TEST(RandomStateTest, FromSeedTakesNegativeSeedsAsUnsigned) {
  RandomState state = RandomState::FromSeed(-1);

  EXPECT_EQ(state.drafter_lenience,
            std::mt19937_64(uint64_t{0xffffffff} ^ 0x4c454e49));
}

TEST(RandomStateTest, CopyContinuesFromTheOriginal) {
  RandomState original = RandomState::FromSeed(42);
  original.sampling.discard(3);

  original.drafter_lenience.discard(2);

  RandomState copy = original;

  EXPECT_EQ(copy.sampling(), original.sampling());
  EXPECT_EQ(copy.drafter_lenience(), original.drafter_lenience());
  EXPECT_EQ(copy.draft_verifier_lenience(), original.draft_verifier_lenience());
}

TEST(RandomStateTest, CopiesAdvanceIndependently) {
  RandomState original = RandomState::FromSeed(42);
  RandomState copy = original;
  const RandomState untouched = original;

  original.sampling.discard(5);

  EXPECT_EQ(copy.sampling, untouched.sampling);
  EXPECT_NE(copy.sampling, original.sampling);
}

}  // namespace
}  // namespace litert::lm
