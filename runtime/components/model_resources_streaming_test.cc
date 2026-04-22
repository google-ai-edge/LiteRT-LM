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

#include "runtime/components/model_resources_streaming.h"

#include <optional>

#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"

namespace litert::lm {
namespace {

TEST(ModelResourcesStreamingTest, GetTFLiteModel) {
  ModelResourcesStreaming model_resources;
  EXPECT_EQ(model_resources.GetTFLiteModel(ModelType::kUnknown).status().code(),
            absl::StatusCode::kUnimplemented);
}

TEST(ModelResourcesStreamingTest, GetTFLiteModelBuffer) {
  ModelResourcesStreaming model_resources;
  EXPECT_EQ(
      model_resources.GetTFLiteModelBuffer(ModelType::kUnknown).status().code(),
      absl::StatusCode::kUnimplemented);
}

TEST(ModelResourcesStreamingTest, GetScopedFile) {
  ModelResourcesStreaming model_resources;
  EXPECT_EQ(model_resources.GetScopedFile().status().code(),
            absl::StatusCode::kUnimplemented);
}

TEST(ModelResourcesStreamingTest, GetWeightsSectionOffset) {
  ModelResourcesStreaming model_resources;
  EXPECT_EQ(
      model_resources.GetWeightsSectionOffset(ModelType::kUnknown)
          .status()
          .code(),
      absl::StatusCode::kUnimplemented);
}

TEST(ModelResourcesStreamingTest, GetTFLiteModelBackendConstraint) {
  ModelResourcesStreaming model_resources;
  EXPECT_EQ(
      model_resources.GetTFLiteModelBackendConstraint(ModelType::kUnknown),
      std::nullopt);
}

TEST(ModelResourcesStreamingTest, GetTokenizer) {
  ModelResourcesStreaming model_resources;
  EXPECT_EQ(model_resources.GetTokenizer().status().code(),
            absl::StatusCode::kUnimplemented);
}

TEST(ModelResourcesStreamingTest, GetLlmMetadata) {
  ModelResourcesStreaming model_resources;
  EXPECT_EQ(model_resources.GetLlmMetadata().status().code(),
            absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace litert::lm
