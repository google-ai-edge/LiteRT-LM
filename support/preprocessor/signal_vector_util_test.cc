// Copyright 2025 The ODML Authors.
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

#include "support/preprocessor/signal_vector_util.h"

#include <cmath>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "support/util/test_utils.h"  // NOLINT

namespace litert::support {
namespace {

TEST(VectorTest, SmootherCoefficientFromScale) {
  float scale1, coefficient;
  // scale1 is the smoothing scale of one forward/backward pass.  The net
  // smoothing scale is scale1 * sqrt(2.0).  Using scale1 makes the
  // expected coefficient values a little simpler.
  scale1 = 2;
  coefficient = SmootherCoefficientFromScale(scale1 * M_SQRT2);
  // This is one coincidentally simple value that we know:
  EXPECT_NEAR(coefficient, 1 / scale1, 1e-6);
  // Other values we can bound:
  scale1 = 1;
  coefficient = SmootherCoefficientFromScale(scale1 * M_SQRT2);
  EXPECT_LT(coefficient, 1 / scale1);
  scale1 = 4;
  coefficient = SmootherCoefficientFromScale(scale1 * M_SQRT2);
  EXPECT_GT(coefficient, 1 / scale1);
  // Small-scale approximation.
  scale1 = 0.1;
  coefficient = SmootherCoefficientFromScale(scale1 * M_SQRT2);
  EXPECT_NEAR(coefficient, 1.0 - 0.5 * scale1 * scale1, 1e-4);
  // Large-scale approximation.
  scale1 = 1000;
  coefficient = SmootherCoefficientFromScale(scale1 * M_SQRT2);
  EXPECT_NEAR(coefficient, sqrt(2.0) / scale1, 1e-4);
}

TEST(VectorTest, ForwardSmoothVector) {
  std::vector<float> v1({0, 0, 1, 0, 0});  // Impulse input.
  const float initial_state = 1;
  float state = initial_state;
  ForwardSmoothVector(0.5, &state, &v1);
  EXPECT_EQ(v1[0], 0.5);          // initial_state * coefficient.
  EXPECT_EQ(v1[1], 0.25);         // Decay by factor 0.5.
  EXPECT_EQ(v1[2], 0.5 + 0.125);  // The impulse comes in here.
  EXPECT_EQ(v1[3], v1[2] / 2);
  EXPECT_EQ(v1[4], v1[3] / 2);
  EXPECT_EQ(state, v1[4]);  // Final state is last value stored.
}

TEST(VectorTest, BackwardSmoothVector) {
  std::vector<float> v1({0, 0, 1, 0, 0});  // Impulse input.
  const float initial_state = 1;
  float state = initial_state;
  BackwardSmoothVector(0.5, &state, &v1);
  // Just reversed indices from the ForwardSmoother test.
  EXPECT_EQ(v1[4], 0.5);
  EXPECT_EQ(v1[3], 0.25);
  EXPECT_EQ(v1[2], 0.5 + 0.125);
  EXPECT_EQ(v1[1], v1[2] / 2);
  EXPECT_EQ(v1[0], v1[1] / 2);
  EXPECT_EQ(state, v1[0]);
}

TEST(VectorTest, SmoothVector) {
  std::vector<float> v1({0, 0, 0, 1, 0, 0, 0});
  SmoothVector(0.5, &v1);
  std::vector<float> expected_half({0.07, 0.105, 0.15, 0.19});
  for (int i = 0; i < expected_half.size(); ++i) {
    // Expect the result to be pretty nearly symmetric.
    EXPECT_NEAR(expected_half[i], v1[i], 5e-3);
    EXPECT_NEAR(v1[i], v1[v1.size() - 1 - i], 3e-3);
  }
  EXPECT_NEAR(std::accumulate(v1.begin(), v1.end(), 0.0), 0.83, 5e-3);
  // Try more room to allow sum to be closer to 1.
  std::vector<float> v2({0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0});
  SmoothVector(0.5, &v2);
  EXPECT_NEAR(std::accumulate(v2.begin(), v2.end(), 0.0), 0.97, 5e-3);
}

}  // namespace
}  // namespace litert::support
