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

#include "support/preprocessor/by_pass_audio_preprocessor.h"

#include <cstddef>
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

TEST(ByPassAudioPreprocessorTest, PreprocessWithTensorBuffer) {
  ByPassAudioPreprocessor preprocessor;

  // Create an InputAudio with a TensorBuffer.
  std::vector<float> input_data(1 * 10 * 128, 0.5f);
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer input_tensor_buffer,
      CopyToTensorBuffer<float>(input_data, {1, 10, 128}));
  InputAudio test_input_audio(std::move(input_tensor_buffer));

  // Call the Preprocess method.
  absl::StatusOr<InputAudio> result = preprocessor.Preprocess(test_input_audio);

  // Assert that the result is OK.
  ASSERT_OK(result);

  // Get the TensorBuffer from the result InputAudio.
  LITERT_ASSERT_OK_AND_ASSIGN(auto result_tensor_buffer,
                              result->GetPreprocessedAudioTensor());

  // Verify the dimensions and content of the returned TensorBuffer match the
  // input.
  LITERT_ASSERT_OK_AND_ASSIGN(auto tensor_type,
                              result_tensor_buffer->TensorType());
  EXPECT_EQ(tensor_type.Layout().Dimensions(), Dimensions({1, 10, 128}));

  auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      *result_tensor_buffer, TensorBuffer::LockMode::kRead);
  ASSERT_TRUE(output_tensor_lock_and_addr.HasValue());
  const float* result_data =
      static_cast<const float*>(output_tensor_lock_and_addr->second);
  ASSERT_NE(result_data, nullptr);
  size_t num_elements = 1 * 10 * 128;
  EXPECT_THAT(absl::MakeConstSpan(result_data, num_elements),
              testing::ElementsAreArray(input_data));

  // Verify that the returned TensorBuffer is pointing to the same data as the
  // input.
  EXPECT_EQ(result_tensor_buffer->Get(),
            test_input_audio.GetPreprocessedAudioTensor().value()->Get());
}

TEST(ByPassAudioPreprocessorTest, PreprocessWithRawBytesFailed) {
  ByPassAudioPreprocessor preprocessor;

  // Create an InputAudio from raw bytes.
  std::string dummy_audio_data = "\x01\x02\x03\x04";
  InputAudio test_input_audio(dummy_audio_data);

  // Call the Preprocess method.
  absl::StatusOr<InputAudio> result = preprocessor.Preprocess(test_input_audio);

  // Assert that the result is an InvalidArgumentError.
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(result.status().message()),
              HasSubstr("Input audio is not preprocessed."));
}

}  // namespace
}  // namespace litert::support
