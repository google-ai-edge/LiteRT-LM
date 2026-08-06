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

#include "support/util/io_types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "support/util/convert_tensor_buffer.h"
#include "support/util/test_utils.h"  // NOLINT

namespace litert::support {
namespace {

using ::testing::ElementsAreArray;
using ::testing::litert::IsOkAndHolds;
using ::testing::status::StatusIs;

constexpr const float kTensorData[] = {10, 20, 30, 40};

constexpr const int32_t kTensorDimensions[] = {sizeof(kTensorData) /
                                               sizeof(kTensorData[0])};

constexpr const LiteRtRankedTensorType kTestTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    BuildLayout(kTensorDimensions)};

TEST(InputTextTest, GetRawText) {
  InputText input_text("Hello World!");
  EXPECT_FALSE(input_text.IsTensorBuffer());
  EXPECT_THAT(input_text.GetRawTextString(), IsOkAndHolds("Hello World!"));
  EXPECT_THAT(input_text.GetPreprocessedTextTensor(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(InputTextTest, GetPreprocessedTextTensor) {
  // Create a tensor buffer with kTensorData.
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  const ::litert::RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = ::litert::TensorBufferType::kHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);

  LITERT_ASSERT_OK_AND_ASSIGN(
      ::litert::TensorBuffer original_tensor_buffer,
      ::litert::TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                            kTensorSize));

  LITERT_ASSERT_OK(
      original_tensor_buffer.Write<float>(absl::MakeSpan(kTensorData, 4)));

  // Create an InputText from the tensor buffer. This InputText takes
  // ownership of the tensor buffer.
  InputText input_text(std::move(original_tensor_buffer));

  // Confirm the InputText is preprocessed.
  EXPECT_TRUE(input_text.IsTensorBuffer());

  // Confirm that GetRawTextString returns an error.
  EXPECT_THAT(input_text.GetRawTextString(),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Confirm the retrieved tensor buffer is identical to the original tensor
  // buffer.
  ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_buffer,
                       input_text.GetPreprocessedTextTensor());

  LITERT_ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_buffer_size,
                              retrieved_tensor_buffer->Size());
  EXPECT_EQ(retrieved_tensor_buffer_size, kTensorSize);
  LITERT_ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_buffer_type,
                              retrieved_tensor_buffer->BufferType());
  EXPECT_EQ(retrieved_tensor_buffer_type, kTensorBufferType);
  LITERT_ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_type,
                              retrieved_tensor_buffer->TensorType());
  EXPECT_EQ(retrieved_tensor_type, kTensorType);

  // Confirm the value of the retrieved_tensor_buffer is identical to
  // kTensorData.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto retrieved_data,
      ::litert::support::ReferTensorBufferAsSpan<float>(*retrieved_tensor_buffer));
  EXPECT_THAT(retrieved_data, ElementsAreArray(kTensorData));
}

TEST(InputImageTest, GetRawImageBytes) {
  InputImage input_image("Hello Image!");
  ASSERT_OK_AND_ASSIGN(auto raw_image_bytes, input_image.GetRawImageBytes());
  EXPECT_EQ(raw_image_bytes, "Hello Image!");
}

TEST(InputImageTest, GetPreprocessedImageTensor) {
  // Create a tensor buffer with kTensorData.
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  const ::litert::RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = ::litert::TensorBufferType::kHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);

  LITERT_ASSERT_OK_AND_ASSIGN(
      ::litert::TensorBuffer original_tensor_buffer,
      ::litert::TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                            kTensorSize));

  LITERT_ASSERT_OK(
      original_tensor_buffer.Write<float>(absl::MakeSpan(kTensorData, 4)));

  // Create an InputImage from the tensor buffer. This InputImage takes
  // ownership of the tensor buffer.
  InputImage input_image(std::move(original_tensor_buffer));

  // Confirm the InputImage is preprocessed.
  EXPECT_TRUE(input_image.IsTensorBuffer());

  // Confirm the retrieved tensor buffer is identical to the original tensor
  // buffer.
  ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_buffer,
                       input_image.GetPreprocessedImageTensor());

  LITERT_ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_buffer_size,
                              retrieved_tensor_buffer->Size());
  EXPECT_EQ(retrieved_tensor_buffer_size, kTensorSize);
  LITERT_ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_buffer_type,
                              retrieved_tensor_buffer->BufferType());
  EXPECT_EQ(retrieved_tensor_buffer_type, kTensorBufferType);
  LITERT_ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_type,
                              retrieved_tensor_buffer->TensorType());
  EXPECT_EQ(retrieved_tensor_type, kTensorType);

  // Confirm the value of the retrieved_tensor_buffer is identical to
  // kTensorData.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto retrieved_data,
      ::litert::support::ReferTensorBufferAsSpan<float>(*retrieved_tensor_buffer));
  EXPECT_THAT(retrieved_data, ElementsAreArray(kTensorData));
}

TEST(InputAudioTest, GetRawAudioBytes) {
  InputAudio input_audio("Hello Audio!");
  ASSERT_OK_AND_ASSIGN(auto raw_audio_bytes, input_audio.GetRawAudioBytes());
  EXPECT_EQ(raw_audio_bytes, "Hello Audio!");
}

TEST(InputAudioTest, GetPreprocessedAudioTensor) {
  // Create a tensor buffer with kTensorData.
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  const ::litert::RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = ::litert::TensorBufferType::kHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);

  LITERT_ASSERT_OK_AND_ASSIGN(
      ::litert::TensorBuffer original_tensor_buffer,
      ::litert::TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                            kTensorSize));

  LITERT_ASSERT_OK(
      original_tensor_buffer.Write<float>(absl::MakeSpan(kTensorData, 4)));

  // Create an InputAudio from the tensor buffer. This InputAudio takes
  // ownership of the tensor buffer.
  InputAudio input_audio(std::move(original_tensor_buffer));

  // Confirm the InputAudio is preprocessed.
  EXPECT_TRUE(input_audio.IsTensorBuffer());

  // Confirm the retrieved tensor buffer is identical to the original tensor
  // buffer.
  ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_buffer,
                       input_audio.GetPreprocessedAudioTensor());

  LITERT_ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_buffer_size,
                              retrieved_tensor_buffer->Size());
  EXPECT_EQ(retrieved_tensor_buffer_size, kTensorSize);
  LITERT_ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_buffer_type,
                              retrieved_tensor_buffer->BufferType());
  EXPECT_EQ(retrieved_tensor_buffer_type, kTensorBufferType);
  LITERT_ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_type,
                              retrieved_tensor_buffer->TensorType());
  EXPECT_EQ(retrieved_tensor_type, kTensorType);

  // Confirm the value of the retrieved_tensor_buffer is identical to
  // kTensorData.
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto retrieved_data,
      ::litert::support::ReferTensorBufferAsSpan<float>(*retrieved_tensor_buffer));
  EXPECT_THAT(retrieved_data, ElementsAreArray(kTensorData));
}

TEST(InputAudioTest, GetAudioFloatData) {
  std::vector<float> audio_data = {0.1, 0.2, 0.3, 0.4};
  InputAudio input_audio(audio_data);
  EXPECT_TRUE(input_audio.IsPcmFrames());
  ASSERT_OK_AND_ASSIGN(auto retrieved_audio_data, input_audio.GetPcmFrames());
  EXPECT_THAT(retrieved_audio_data, ElementsAreArray(audio_data));
}

TEST(InputTextTest, CreateCopyFromString) {
  InputText original_input_text("Hello World!");
  ASSERT_OK_AND_ASSIGN(InputText copied_input_text,
                       original_input_text.CreateCopy());

  EXPECT_FALSE(copied_input_text.IsTensorBuffer());
  EXPECT_THAT(copied_input_text.GetRawTextString(),
              IsOkAndHolds("Hello World!"));
}

TEST(InputTextTest, CreateCopyFromTensorBuffer) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  const ::litert::RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = ::litert::TensorBufferType::kHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);

  LITERT_ASSERT_OK_AND_ASSIGN(
      ::litert::TensorBuffer original_tensor_buffer,
      ::litert::TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                            kTensorSize));
  LITERT_ASSERT_OK(
      original_tensor_buffer.Write<float>(absl::MakeSpan(kTensorData, 4)));

  InputText original_input_text(std::move(original_tensor_buffer));
  ASSERT_OK_AND_ASSIGN(InputText copied_input_text,
                       original_input_text.CreateCopy());

  EXPECT_TRUE(copied_input_text.IsTensorBuffer());
  EXPECT_THAT(copied_input_text.GetRawTextString(),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_buffer,
                       copied_input_text.GetPreprocessedTextTensor());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto retrieved_data,
      ::litert::support::ReferTensorBufferAsSpan<float>(*retrieved_tensor_buffer));
  EXPECT_THAT(retrieved_data, ElementsAreArray(kTensorData));
}

TEST(InputImageTest, CreateCopyFromString) {
  InputImage original_input_image("Hello Image!");
  ASSERT_OK_AND_ASSIGN(InputImage copied_input_image,
                       original_input_image.CreateCopy());

  EXPECT_FALSE(copied_input_image.IsTensorBuffer());
  EXPECT_THAT(copied_input_image.GetRawImageBytes(),
              IsOkAndHolds("Hello Image!"));
}

TEST(InputImageTest, CreateCopyFromTensorBuffer) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  const ::litert::RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = ::litert::TensorBufferType::kHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);

  LITERT_ASSERT_OK_AND_ASSIGN(
      ::litert::TensorBuffer original_tensor_buffer,
      ::litert::TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                            kTensorSize));
  LITERT_ASSERT_OK(
      original_tensor_buffer.Write<float>(absl::MakeSpan(kTensorData, 4)));

  InputImage original_input_image(std::move(original_tensor_buffer));
  ASSERT_OK_AND_ASSIGN(InputImage copied_input_image,
                       original_input_image.CreateCopy());

  EXPECT_TRUE(copied_input_image.IsTensorBuffer());
  EXPECT_THAT(copied_input_image.GetRawImageBytes(),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_buffer,
                       copied_input_image.GetPreprocessedImageTensor());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto retrieved_data,
      ::litert::support::ReferTensorBufferAsSpan<float>(*retrieved_tensor_buffer));
  EXPECT_THAT(retrieved_data, ElementsAreArray(kTensorData));
}

TEST(InputAudioTest, CreateCopyFromString) {
  InputAudio original_input_audio("Hello Audio!");
  ASSERT_OK_AND_ASSIGN(InputAudio copied_input_audio,
                       original_input_audio.CreateCopy());

  EXPECT_FALSE(copied_input_audio.IsTensorBuffer());
  EXPECT_THAT(copied_input_audio.GetRawAudioBytes(),
              IsOkAndHolds("Hello Audio!"));
}

TEST(InputAudioTest, CreateCopyFromTensorBuffer) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  const ::litert::RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = ::litert::TensorBufferType::kHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);

  LITERT_ASSERT_OK_AND_ASSIGN(
      ::litert::TensorBuffer original_tensor_buffer,
      ::litert::TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                            kTensorSize));
  LITERT_ASSERT_OK(
      original_tensor_buffer.Write<float>(absl::MakeSpan(kTensorData, 4)));

  InputAudio original_input_audio(std::move(original_tensor_buffer));
  ASSERT_OK_AND_ASSIGN(InputAudio copied_input_audio,
                       original_input_audio.CreateCopy());

  EXPECT_TRUE(copied_input_audio.IsTensorBuffer());
  EXPECT_THAT(copied_input_audio.GetRawAudioBytes(),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  ASSERT_OK_AND_ASSIGN(auto retrieved_tensor_buffer,
                       copied_input_audio.GetPreprocessedAudioTensor());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto retrieved_data,
      ::litert::support::ReferTensorBufferAsSpan<float>(*retrieved_tensor_buffer));
  EXPECT_THAT(retrieved_data, ElementsAreArray(kTensorData));
}

TEST(InputAudioTest, CreateCopyFromFloatVector) {
  std::vector<float> audio_data = {0.1, 0.2, 0.3, 0.4};
  InputAudio original_input_audio(audio_data);
  ASSERT_OK_AND_ASSIGN(InputAudio copied_input_audio,
                       original_input_audio.CreateCopy());

  EXPECT_TRUE(copied_input_audio.IsPcmFrames());
  EXPECT_THAT(copied_input_audio.GetRawAudioBytes(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(copied_input_audio.GetPreprocessedAudioTensor(),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  ASSERT_OK_AND_ASSIGN(auto retrieved_audio_data,
                       copied_input_audio.GetPcmFrames());
  EXPECT_THAT(retrieved_audio_data, ElementsAreArray(audio_data));
}

TEST(CreateInputDataCopyTest, InputText) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));
  InputData original_data = InputText("Test Text");
  ASSERT_OK_AND_ASSIGN(InputData copied_data,
                       CreateInputDataCopy(original_data));
  ASSERT_TRUE(std::holds_alternative<InputText>(copied_data));
  EXPECT_THAT(std::get<InputText>(copied_data).GetRawTextString(),
              IsOkAndHolds("Test Text"));

  const ::litert::RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = ::litert::TensorBufferType::kHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);
  LITERT_ASSERT_OK_AND_ASSIGN(
      ::litert::TensorBuffer original_tensor_buffer,
      ::litert::TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                            kTensorSize));
  original_data = InputText(std::move(original_tensor_buffer));
  ASSERT_OK_AND_ASSIGN(copied_data, CreateInputDataCopy(original_data));
  ASSERT_TRUE(std::holds_alternative<InputText>(copied_data));
  EXPECT_TRUE(std::get<InputText>(copied_data).IsTensorBuffer());
}

TEST(CreateInputDataCopyTest, InputImage) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));

  InputData original_data = InputImage("Test Image");
  ASSERT_OK_AND_ASSIGN(InputData copied_data,
                       CreateInputDataCopy(original_data));
  ASSERT_TRUE(std::holds_alternative<InputImage>(copied_data));
  EXPECT_THAT(std::get<InputImage>(copied_data).GetRawImageBytes(),
              IsOkAndHolds("Test Image"));

  const ::litert::RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = ::litert::TensorBufferType::kHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);
  LITERT_ASSERT_OK_AND_ASSIGN(
      ::litert::TensorBuffer original_tensor_buffer,
      ::litert::TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                            kTensorSize));
  original_data = InputImage(std::move(original_tensor_buffer));
  ASSERT_OK_AND_ASSIGN(copied_data, CreateInputDataCopy(original_data));
  ASSERT_TRUE(std::holds_alternative<InputImage>(copied_data));
  EXPECT_TRUE(std::get<InputImage>(copied_data).IsTensorBuffer());

  absl::flat_hash_map<std::string, ::litert::TensorBuffer> original_tensor_map;
  LITERT_ASSERT_OK_AND_ASSIGN(
      ::litert::TensorBuffer original_tensor_buffer_1,
      ::litert::TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                            kTensorSize));
  LITERT_ASSERT_OK_AND_ASSIGN(
      ::litert::TensorBuffer original_tensor_buffer_2,
      ::litert::TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                            kTensorSize));
  original_tensor_map["test_key_1"] = std::move(original_tensor_buffer_1);
  original_tensor_map["test_key_2"] = std::move(original_tensor_buffer_2);
  original_data = InputImage(std::move(original_tensor_map));
  ASSERT_OK_AND_ASSIGN(copied_data, CreateInputDataCopy(original_data));
  ASSERT_TRUE(std::holds_alternative<InputImage>(copied_data));
  EXPECT_TRUE(std::get<InputImage>(copied_data).IsTensorBufferMap());
  EXPECT_TRUE(
      std::get<InputImage>(copied_data).GetPreprocessedImageTensorMap().ok());
  EXPECT_THAT(std::get<InputImage>(copied_data).GetPreprocessedImageTensor(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(CreateInputDataCopyTest, InputAudio) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, ::litert::Environment::Create({}));

  InputData original_data = InputAudio("Test Audio");
  ASSERT_OK_AND_ASSIGN(InputData copied_data,
                       CreateInputDataCopy(original_data));
  ASSERT_TRUE(std::holds_alternative<InputAudio>(copied_data));
  EXPECT_THAT(std::get<InputAudio>(copied_data).GetRawAudioBytes(),
              IsOkAndHolds("Test Audio"));

  const ::litert::RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = ::litert::TensorBufferType::kHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);
  LITERT_ASSERT_OK_AND_ASSIGN(
      ::litert::TensorBuffer original_tensor_buffer,
      ::litert::TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                            kTensorSize));
  original_data = InputAudio(std::move(original_tensor_buffer));
  ASSERT_OK_AND_ASSIGN(copied_data, CreateInputDataCopy(original_data));
  ASSERT_TRUE(std::holds_alternative<InputAudio>(copied_data));
  EXPECT_TRUE(std::get<InputAudio>(copied_data).IsTensorBuffer());
}

TEST(CreateInputDataCopyTest, InputAudioWithFloatVector) {
  std::vector<float> audio_data = {0.1, 0.2, 0.3, 0.4};
  InputData original_data = InputAudio(audio_data);
  ASSERT_OK_AND_ASSIGN(InputData copied_data,
                       CreateInputDataCopy(original_data));
  ASSERT_TRUE(std::holds_alternative<InputAudio>(copied_data));
  EXPECT_TRUE(std::get<InputAudio>(copied_data).IsPcmFrames());
  ASSERT_OK_AND_ASSIGN(auto retrieved_audio_data,
                       std::get<InputAudio>(copied_data).GetPcmFrames());
  EXPECT_THAT(retrieved_audio_data, ElementsAreArray(audio_data));
}

TEST(CreateInputDataCopyTest, InputAudioEnd) {
  InputData original_data = InputAudioEnd();
  ASSERT_OK_AND_ASSIGN(InputData copied_data,
                       CreateInputDataCopy(original_data));
  ASSERT_TRUE(std::holds_alternative<InputAudioEnd>(copied_data));
}

TEST(CreateInputDataCopyTest, InputImageEnd) {
  InputData original_data = InputImageEnd();
  ASSERT_OK_AND_ASSIGN(InputData copied_data,
                       CreateInputDataCopy(original_data));
  ASSERT_TRUE(std::holds_alternative<InputImageEnd>(copied_data));
}

}  // namespace
}  // namespace litert::support
