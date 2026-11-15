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

#include "support/preprocessor/image_preprocessor.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "support/util/convert_tensor_buffer.h"
#include "support/util/io_types.h"
#include "support/util/test_utils.h"  // NOLINT

namespace litert::support {
namespace {

using ::litert::Dimensions;
using ::testing::HasSubstr;
using ::testing::Return;
// using ::testing::status::StatusIs;

// Mock implementation of ImagePreprocessor for testing.
class MockImagePreprocessor : public ImagePreprocessor {
 public:
  MOCK_METHOD(absl::StatusOr<InputImage>, Preprocess,
              (const InputImage& input_image,
               const ImagePreprocessParameter& parameter),
              (override));
};

TEST(ImagePreprocessorTest, Preprocess) {
  auto mock_preprocessor = std::make_unique<MockImagePreprocessor>();
  ImagePreprocessParameter parameter;
  parameter.SetTargetDimensions({1, 224, 224, 3});

  // Create a dummy InputImage. The content doesn't matter for the mock.
  std::vector<float> dummy_input_data(1 * 10 * 10 * 3, 0.0f);
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer dummy_input_tensor_buffer,
      CopyToTensorBuffer<float>(dummy_input_data, {1, 10, 10, 3}));
  InputImage test_input_image(std::move(dummy_input_tensor_buffer));

  // Create a dummy TensorBuffer to be returned *inside* the InputImage.
  // We'll use a float tensor of size {1, 224, 224, 3} as a common image tensor
  // shape.
  std::vector<float> dummy_data(1 * 224 * 224 * 3, 0.5f);
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer expected_tensor_buffer,
      CopyToTensorBuffer<float>(dummy_data, {1, 224, 224, 3}));
  InputImage expected_output_image(std::move(expected_tensor_buffer));

  // Set up the mock expectation.
  EXPECT_CALL(*mock_preprocessor,
              Preprocess(testing::_,  // Match any InputImage
                         testing::Ref(parameter)))
      .WillOnce(Return(std::move(expected_output_image)));

  // Call the Preprocess method.
  absl::StatusOr<InputImage> result =
      mock_preprocessor->Preprocess(test_input_image, parameter);

  // Assert that the result is OK.
  ASSERT_OK(result);

  // Get the TensorBuffer from the result InputImage.
  LITERT_ASSERT_OK_AND_ASSIGN(auto result_tensor_buffer,
                              result->GetPreprocessedImageTensor());

  // Verify the dimensions of the returned TensorBuffer.
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_type,
                              result_tensor_buffer->TensorType());
  EXPECT_EQ(tensor_type.Layout().Dimensions(), Dimensions({1, 224, 224, 3}));

  // Confirm the data in the result tensor buffer matches the dummy data.
  auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      *result_tensor_buffer, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(output_tensor_lock_and_addr.HasValue());
  const float* result_data =
      static_cast<const float*>(output_tensor_lock_and_addr->second);
  ASSERT_NE(result_data, nullptr);
  size_t num_elements = 1 * 224 * 224 * 3;
  EXPECT_THAT(absl::MakeConstSpan(result_data, num_elements),
              testing::ElementsAreArray(dummy_data));
}

TEST(ImagePreprocessorTest, PreprocessUnimplemented) {
  ImagePreprocessor preprocessor;
  ImagePreprocessParameter parameter;
  parameter.SetTargetDimensions({1, 224, 224, 3});

  // Create an InputImage from raw bytes, which is not currently supported
  // by the default ImagePreprocessor::Preprocess implementation.
  std::string dummy_image_data = "dummy_image_bytes";
  InputImage test_input_image(dummy_image_data);

  // Call the Preprocess method on the real ImagePreprocessor.
  absl::StatusOr<InputImage> result =
      preprocessor.Preprocess(test_input_image, parameter);

  // Assert that the result is an UnimplementedError.
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_THAT(std::string(result.status().message()),
              HasSubstr("Image preprocessor is not implemented."));
}

}  // namespace
}  // namespace litert::support
