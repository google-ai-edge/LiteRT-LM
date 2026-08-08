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

#include "support/preprocessor/audio_preprocessor.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
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
using ::testing::Return;

// Mock implementation of AudioPreprocessor for testing.
class MockAudioPreprocessor : public AudioPreprocessor {
 public:
  MOCK_METHOD(absl::StatusOr<InputAudio>, Preprocess,
              (const InputAudio& input_audio), (override));
  MOCK_METHOD(void, Reset, (), (override));
};

TEST(AudioPreprocessorTest, Preprocess) {
  auto mock_preprocessor = std::make_unique<MockAudioPreprocessor>();

  // Create a dummy InputAudio for the input. The content doesn't matter for
  // the mock.
  std::vector<float> dummy_input_data(1 * 10 * 128, 0.0f);
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer dummy_input_tensor_buffer,
      CopyToTensorBuffer<float>(dummy_input_data, {1, 10, 128}));
  InputAudio test_input_audio(std::move(dummy_input_tensor_buffer));

  // Create a dummy TensorBuffer to be returned *inside* the InputAudio.
  // We'll use a float tensor of size {1, 100, 128} as a common audio feature
  // shape (e.g., 100 time frames, 128 mel bins).
  std::vector<float> dummy_data(1 * 100 * 128, 0.1f);
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer expected_tensor_buffer,
      CopyToTensorBuffer<float>(dummy_data, {1, 100, 128}));
  InputAudio expected_output_audio(std::move(expected_tensor_buffer));

  // Set up the mock expectation.
  EXPECT_CALL(*mock_preprocessor,
              Preprocess(testing::Ref(
                  test_input_audio)))  // Match the specific test_input_audio
      .WillOnce(Return(std::move(expected_output_audio)));

  // Call the Preprocess method.
  absl::StatusOr<InputAudio> result =
      mock_preprocessor->Preprocess(test_input_audio);

  // Assert that the result is OK.
  ASSERT_OK(result);

  // Get the TensorBuffer from the result InputAudio.
  LITERT_ASSERT_OK_AND_ASSIGN(auto result_tensor_buffer,
                              result->GetPreprocessedAudioTensor());

  // Verify the dimensions and content of the returned TensorBuffer.
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_type,
                              result_tensor_buffer->TensorType());
  EXPECT_EQ(tensor_type.Layout().Dimensions(), Dimensions({1, 100, 128}));

  // Confirm the data in the result tensor buffer matches the dummy data.
  auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      *result_tensor_buffer, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(output_tensor_lock_and_addr.HasValue());
  const float* result_data =
      static_cast<const float*>(output_tensor_lock_and_addr->second);
  ASSERT_NE(result_data, nullptr);
  size_t num_elements = 1 * 100 * 128;
  EXPECT_THAT(absl::MakeConstSpan(result_data, num_elements),
              testing::ElementsAreArray(dummy_data));
}

}  // namespace
}  // namespace litert::support
