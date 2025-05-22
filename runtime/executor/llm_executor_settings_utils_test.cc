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

#include "runtime/executor/llm_executor_settings_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "third_party/odml/infra/genai/inference/proto/llm_inference_engine.pb.h"
#include "runtime/executor/llm_executor_settings.h"

namespace {

using ::litert::lm::ActivationDataType;
using ::litert::lm::Backend;
using ::litert::lm::ConvertActivationDataType;
using ::litert::lm::ConvertBackend;
using ::odml::infra::proto::SessionConfig;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

TEST(LlmExecutorUtilsTest, ConvertBackendSuccess) {
  EXPECT_THAT(ConvertBackend(SessionConfig::XNNPACK),
              IsOkAndHolds(Backend::CPU));
  EXPECT_THAT(ConvertBackend(SessionConfig::ML_DRIFT),
              IsOkAndHolds(Backend::GPU));
  EXPECT_THAT(ConvertBackend(SessionConfig::GOOGLE_TENSOR),
              IsOkAndHolds(Backend::GOOGLE_TENSOR_ARTISAN));
}

TEST(LlmExecutorUtilsTest, ConvertBackendFail) {
  EXPECT_THAT(ConvertBackend(SessionConfig::UNSPECIFIED_BACKEND),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(LlmExecutorUtilsTest, ConvertActivationDataTypeSuccess) {
  EXPECT_THAT(
      ConvertActivationDataType(SessionConfig::ACTIVATION_DATA_TYPE_F32),
      IsOkAndHolds(ActivationDataType::FLOAT32));
  EXPECT_THAT(
      ConvertActivationDataType(SessionConfig::ACTIVATION_DATA_TYPE_F16),
      IsOkAndHolds(ActivationDataType::FLOAT16));
  EXPECT_THAT(
      ConvertActivationDataType(SessionConfig::ACTIVATION_DATA_TYPE_I16),
      IsOkAndHolds(ActivationDataType::INT16));
  EXPECT_THAT(ConvertActivationDataType(SessionConfig::ACTIVATION_DATA_TYPE_I8),
              IsOkAndHolds(ActivationDataType::INT8));
}

}  // namespace
