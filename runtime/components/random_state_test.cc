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

#include <random>

#include <gtest/gtest.h>

namespace litert::lm {
namespace {

TEST(RandomStateTest, FromSeedSeedsTheSamplingEngine) {
  RandomState state = RandomState::FromSeed(42);

  std::default_random_engine expected(42);
  EXPECT_EQ(state.sampling, expected);
}

TEST(RandomStateTest, CopyContinuesFromTheOriginal) {
  RandomState original = RandomState::FromSeed(42);
  original.sampling.discard(3);

  RandomState copy = original;

  EXPECT_EQ(copy.sampling(), original.sampling());
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
