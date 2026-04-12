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

#include "runtime/engine/io_types.h"

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
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_model_types.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/constrained_decoding/fake_constraint.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using ::testing::ContainsRegex;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

constexpr const float kTensorData[] = {10, 20, 30, 40};

constexpr const int32_t kTensorDimensions[] = {sizeof(kTensorData) /
                                               sizeof(kTensorData[0])};

constexpr const LiteRtRankedTensorType kTestTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    BuildLayout(kTensorDimensions)};

std::string FloatToString(float val) {
  std::ostringstream oss;
  oss << val;
  return oss.str();
}

TEST(InputTextTest, GetRawText) {
  InputText input_text("Hello World!");
  EXPECT_FALSE(input_text.IsTensorBuffer());
  EXPECT_THAT(input_text.GetRawTextString(), IsOkAndHolds("Hello World!"));
  EXPECT_THAT(input_text.GetPreprocessedTextTensor(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(InputTextTest, GetPreprocessedTextTensor) {
  // Create a tensor buffer with kTensorData.
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = ::litert::TensorBufferType::kHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer original_tensor_buffer,
      TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
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
      ::litert::lm::ReferTensorBufferAsSpan<float>(*retrieved_tensor_buffer));
  EXPECT_THAT(retrieved_data, ElementsAreArray(kTensorData));
}

TEST(InputImageTest, GetRawImageBytes) {
  InputImage input_image("Hello Image!");
  ASSERT_OK_AND_ASSIGN(auto raw_image_bytes, input_image.GetRawImageBytes());
  EXPECT_EQ(raw_image_bytes, "Hello Image!");
}

TEST(InputImageTest, GetPreprocessedImageTensor) {
  // Create a tensor buffer with kTensorData.
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = ::litert::TensorBufferType::kHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer original_tensor_buffer,
      TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
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
      ::litert::lm::ReferTensorBufferAsSpan<float>(*retrieved_tensor_buffer));
  EXPECT_THAT(retrieved_data, ElementsAreArray(kTensorData));
}

TEST(InputAudioTest, GetRawAudioBytes) {
  InputAudio input_audio("Hello Audio!");
  ASSERT_OK_AND_ASSIGN(auto raw_audio_bytes, input_audio.GetRawAudioBytes());
  EXPECT_EQ(raw_audio_bytes, "Hello Audio!");
}

TEST(InputAudioTest, GetPreprocessedAudioTensor) {
  // Create a tensor buffer with kTensorData.
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = ::litert::TensorBufferType::kHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer original_tensor_buffer,
      TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
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
      ::litert::lm::ReferTensorBufferAsSpan<float>(*retrieved_tensor_buffer));
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
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = ::litert::TensorBufferType::kHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer original_tensor_buffer,
      TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
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
      ::litert::lm::ReferTensorBufferAsSpan<float>(*retrieved_tensor_buffer));
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
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = ::litert::TensorBufferType::kHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer original_tensor_buffer,
      TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
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
      ::litert::lm::ReferTensorBufferAsSpan<float>(*retrieved_tensor_buffer));
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
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = ::litert::TensorBufferType::kHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer original_tensor_buffer,
      TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
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
      ::litert::lm::ReferTensorBufferAsSpan<float>(*retrieved_tensor_buffer));
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
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));
  InputData original_data = InputText("Test Text");
  ASSERT_OK_AND_ASSIGN(InputData copied_data,
                       CreateInputDataCopy(original_data));
  ASSERT_TRUE(std::holds_alternative<InputText>(copied_data));
  EXPECT_THAT(std::get<InputText>(copied_data).GetRawTextString(),
              IsOkAndHolds("Test Text"));

  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = ::litert::TensorBufferType::kHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer original_tensor_buffer,
      TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                  kTensorSize));
  original_data = InputText(std::move(original_tensor_buffer));
  ASSERT_OK_AND_ASSIGN(copied_data, CreateInputDataCopy(original_data));
  ASSERT_TRUE(std::holds_alternative<InputText>(copied_data));
  EXPECT_TRUE(std::get<InputText>(copied_data).IsTensorBuffer());
}

TEST(CreateInputDataCopyTest, InputImage) {
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));

  InputData original_data = InputImage("Test Image");
  ASSERT_OK_AND_ASSIGN(InputData copied_data,
                       CreateInputDataCopy(original_data));
  ASSERT_TRUE(std::holds_alternative<InputImage>(copied_data));
  EXPECT_THAT(std::get<InputImage>(copied_data).GetRawImageBytes(),
              IsOkAndHolds("Test Image"));

  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = ::litert::TensorBufferType::kHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer original_tensor_buffer,
      TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                  kTensorSize));
  original_data = InputImage(std::move(original_tensor_buffer));
  ASSERT_OK_AND_ASSIGN(copied_data, CreateInputDataCopy(original_data));
  ASSERT_TRUE(std::holds_alternative<InputImage>(copied_data));
  EXPECT_TRUE(std::get<InputImage>(copied_data).IsTensorBuffer());

  absl::flat_hash_map<std::string, TensorBuffer> original_tensor_map;
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer original_tensor_buffer_1,
      TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
                                  kTensorSize));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer original_tensor_buffer_2,
      TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
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
  LITERT_ASSERT_OK_AND_ASSIGN(auto env, litert::Environment::Create({}));

  InputData original_data = InputAudio("Test Audio");
  ASSERT_OK_AND_ASSIGN(InputData copied_data,
                       CreateInputDataCopy(original_data));
  ASSERT_TRUE(std::holds_alternative<InputAudio>(copied_data));
  EXPECT_THAT(std::get<InputAudio>(copied_data).GetRawAudioBytes(),
              IsOkAndHolds("Test Audio"));

  const RankedTensorType kTensorType(kTestTensorType);
  constexpr auto kTensorBufferType = ::litert::TensorBufferType::kHostMemory;
  const size_t kTensorSize = sizeof(kTensorData);
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer original_tensor_buffer,
      TensorBuffer::CreateManaged(env, kTensorBufferType, kTensorType,
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

TEST(CreateInputDataVectorCopyTest, CopyVector) {
  std::vector<InputData> original_vector;
  original_vector.push_back(InputText("Test Text"));
  original_vector.push_back(InputImage("Test Image"));
  original_vector.push_back(InputAudio("Test Audio"));
  original_vector.push_back(InputAudioEnd());
  original_vector.push_back(InputImageEnd());

  ASSERT_OK_AND_ASSIGN(std::vector<InputData> copied_vector,
                       CreateInputDataVectorCopy(original_vector));

  ASSERT_EQ(copied_vector.size(), 5);
  ASSERT_TRUE(std::holds_alternative<InputText>(copied_vector[0]));
  EXPECT_THAT(std::get<InputText>(copied_vector[0]).GetRawTextString(),
              IsOkAndHolds("Test Text"));
  ASSERT_TRUE(std::holds_alternative<InputImage>(copied_vector[1]));
  EXPECT_THAT(std::get<InputImage>(copied_vector[1]).GetRawImageBytes(),
              IsOkAndHolds("Test Image"));
  ASSERT_TRUE(std::holds_alternative<InputAudio>(copied_vector[2]));
  EXPECT_THAT(std::get<InputAudio>(copied_vector[2]).GetRawAudioBytes(),
              IsOkAndHolds("Test Audio"));
  ASSERT_TRUE(std::holds_alternative<InputAudioEnd>(copied_vector[3]));
  ASSERT_TRUE(std::holds_alternative<InputImageEnd>(copied_vector[4]));
}

TEST(CreateInputDataVectorCopyTest, EmptyVector) {
  std::vector<InputData> original_vector;
  ASSERT_OK_AND_ASSIGN(std::vector<InputData> copied_vector,
                       CreateInputDataVectorCopy(original_vector));
  EXPECT_TRUE(copied_vector.empty());
}

TEST(ResponsesTest, GetTaskState) {
  {
    Responses responses(TaskState::kCreated, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kCreated);
  }
  {
    Responses responses(TaskState::kQueued, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kQueued);
  }
  {
    Responses responses(TaskState::kProcessing, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kProcessing);
  }
  {
    Responses responses(TaskState::kDone, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kDone);
  }
  {
    Responses responses(TaskState::kMaxNumTokensReached, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kMaxNumTokensReached);
  }
  {
    Responses responses(TaskState::kFailed, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kFailed);
  }
  {
    Responses responses(TaskState::kDependentTaskFailed, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kDependentTaskFailed);
  }
  {
    Responses responses(TaskState::kCancelled, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kCancelled);
  }
  {
    Responses responses(TaskState::kDependentTaskCancelled, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kDependentTaskCancelled);
  }
  {
    Responses responses(TaskState::kLastCallbackQueued, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kLastCallbackQueued);
  }
  {
    Responses responses(TaskState::kUnknown, {});
    EXPECT_EQ(responses.GetTaskState(), TaskState::kUnknown);
  }
}

TEST(TaskStateTest, TaskStateToString) {
  {
    std::stringstream ss;
    ss << TaskState::kCreated;
    EXPECT_EQ(ss.str(), "Created");
  }
  {
    std::stringstream ss;
    ss << TaskState::kQueued;
    EXPECT_EQ(ss.str(), "Queued");
  }
  {
    std::stringstream ss;
    ss << TaskState::kProcessing;
    EXPECT_EQ(ss.str(), "Processing");
  }
  {
    std::stringstream ss;
    ss << TaskState::kDone;
    EXPECT_EQ(ss.str(), "Done");
  }
  {
    std::stringstream ss;
    ss << TaskState::kMaxNumTokensReached;
    EXPECT_EQ(ss.str(), "MaxNumTokensReached");
  }
  {
    std::stringstream ss;
    ss << TaskState::kFailed;
    EXPECT_EQ(ss.str(), "Failed");
  }
  {
    std::stringstream ss;
    ss << TaskState::kDependentTaskFailed;
    EXPECT_EQ(ss.str(), "DependentTaskFailed");
  }
  {
    std::stringstream ss;
    ss << TaskState::kCancelled;
    EXPECT_EQ(ss.str(), "Cancelled");
  }
  {
    std::stringstream ss;
    ss << TaskState::kDependentTaskCancelled;
    EXPECT_EQ(ss.str(), "DependentTaskCancelled");
  }
  {
    std::stringstream ss;
    ss << TaskState::kLastCallbackQueued;
    EXPECT_EQ(ss.str(), "LastCallbackQueued");
  }
  {
    std::stringstream ss;
    ss << TaskState::kUnknown;
    EXPECT_EQ(ss.str(), "Unknown");
  }
}

TEST(ResponsesTest, SetTaskState) {
  Responses responses(TaskState::kCreated, {});
  EXPECT_EQ(responses.GetTaskState(), TaskState::kCreated);
  responses.SetTaskState(TaskState::kQueued);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kQueued);
  responses.SetTaskState(TaskState::kProcessing);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kProcessing);
  responses.SetTaskState(TaskState::kDone);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kDone);
  responses.SetTaskState(TaskState::kMaxNumTokensReached);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kMaxNumTokensReached);
  responses.SetTaskState(TaskState::kFailed);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kFailed);
  responses.SetTaskState(TaskState::kDependentTaskFailed);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kDependentTaskFailed);
  responses.SetTaskState(TaskState::kCancelled);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kCancelled);
  responses.SetTaskState(TaskState::kDependentTaskCancelled);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kDependentTaskCancelled);
  responses.SetTaskState(TaskState::kLastCallbackQueued);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kLastCallbackQueued);
  responses.SetTaskState(TaskState::kUnknown);
  EXPECT_EQ(responses.GetTaskState(), TaskState::kUnknown);
}

TEST(TaskStateTest, IsTaskEndState) {
  EXPECT_TRUE(IsTaskEndState(TaskState::kDone));
  EXPECT_TRUE(IsTaskEndState(TaskState::kMaxNumTokensReached));
  EXPECT_TRUE(IsTaskEndState(TaskState::kFailed));
  EXPECT_TRUE(IsTaskEndState(TaskState::kDependentTaskFailed));
  EXPECT_TRUE(IsTaskEndState(TaskState::kCancelled));
  EXPECT_TRUE(IsTaskEndState(TaskState::kDependentTaskCancelled));
  EXPECT_FALSE(IsTaskEndState(TaskState::kCreated));
  EXPECT_FALSE(IsTaskEndState(TaskState::kQueued));
  EXPECT_FALSE(IsTaskEndState(TaskState::kProcessing));
}

TEST(ResponsesTest, GetTexts) {
  Responses responses(TaskState::kProcessing,
                      {"Hello World!", "How's it going?"});

  EXPECT_THAT(responses.GetTexts(),
              ElementsAre("Hello World!", "How's it going?"));
}

TEST(ResponsesTest, GetScores) {
  Responses responses(TaskState::kProcessing, /*response_texts=*/{},
                      /*scores=*/{0.1f, 0.2f});

  EXPECT_THAT(responses.GetScores(), ElementsAre(0.1, 0.2));
}

TEST(ResponsesTest, GetMutableTexts) {
  Responses responses =
      Responses(TaskState::kProcessing, {"Hello World!", "How's it going?"});

  EXPECT_EQ(responses.GetMutableTexts().size(), 2);
  EXPECT_THAT(responses.GetMutableTexts()[0], "Hello World!");
  EXPECT_THAT(responses.GetMutableTexts()[1], "How's it going?");
}

TEST(ResponsesTest, GetMutableScores) {
  Responses responses = Responses(TaskState::kProcessing, /*response_texts=*/{},
                                  /*scores=*/{0.1f, 0.2f});

  EXPECT_EQ(responses.GetMutableScores().size(), 2);
  EXPECT_FLOAT_EQ(responses.GetMutableScores()[0], 0.1f);
  EXPECT_FLOAT_EQ(responses.GetMutableScores()[1], 0.2f);
}

TEST(ScoringResponsesTest, GetScorerOutputs) {
  ScorerOutput output;
  output.score = 0.5;
  output.option_text_token_length = 10;
  output.token_scores = std::vector<float>{0.1f, 0.2f};

  ScoringResponses responses(TaskState::kDone, {output});

  EXPECT_EQ(responses.GetScorerOutputs().size(), 1);
  EXPECT_DOUBLE_EQ(responses.GetScorerOutputs()[0].score, 0.5);
  EXPECT_EQ(*responses.GetScorerOutputs()[0].option_text_token_length, 10);
  EXPECT_THAT(*responses.GetScorerOutputs()[0].token_scores,
              ElementsAre(0.1f, 0.2f));
}

TEST(ResponsesTest, HandlesMultipleCandidatesWithTextAndScores) {
  litert::lm::Responses responses =
      Responses(TaskState::kProcessing, {"Hello", "World"}, {0.9f, -0.5f});

  std::stringstream ss;
  ss << responses;

  const std::string expected_output =
      "Task State: Processing\n"
      "Total candidates: 2:\n"
      "  Candidate 0 (score: " +
      FloatToString(0.9f) +
      "):\n"
      "    Text: \"Hello\"\n"
      "  Candidate 1 (score: " +
      FloatToString(-0.5f) +
      "):\n"
      "    Text: \"World\"\n";
  EXPECT_EQ(ss.str(), expected_output);
}

TEST(ResponsesTest, HandlesMultipleCandidatesWithTextAndNoScores) {
  litert::lm::Responses responses =
      Responses(TaskState::kProcessing, {"Hello", "World"});

  std::stringstream ss;
  ss << responses;

  const std::string expected_output =
      "Task State: Processing\n"
      "Total candidates: 2:\n"
      "  Candidate 0 (score: N/A):\n"
      "    Text: \"Hello\"\n"
      "  Candidate 1 (score: N/A):\n"
      "    Text: \"World\"\n";
  EXPECT_EQ(ss.str(), expected_output);
}

proto::BenchmarkParams GetBenchmarkParams() {
  proto::BenchmarkParams benchmark_params;
  benchmark_params.set_num_decode_tokens(100);
  benchmark_params.set_num_prefill_tokens(100);
  return benchmark_params;
}

// --- Test Init Phases ---
TEST(BenchmarkInfoTests, AddAndGetInitPhases) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeInitPhaseStart(
      BenchmarkInfo::InitPhase::kModelAssets));
  EXPECT_OK(
      benchmark_info.TimeInitPhaseStart(BenchmarkInfo::InitPhase::kTokenizer));
  absl::SleepFor(absl::Milliseconds(50));
  EXPECT_OK(
      benchmark_info.TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kTokenizer));
  absl::SleepFor(absl::Milliseconds(50));
  EXPECT_OK(
      benchmark_info.TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kModelAssets));

  const auto& phases = benchmark_info.GetInitPhases();
  ASSERT_EQ(phases.size(), 2);
  // The time should be greater than 50ms.
  EXPECT_GT(phases.at(std::string(BenchmarkInfo::InitPhaseToString(
                BenchmarkInfo::InitPhase::kTokenizer))),
            absl::Milliseconds(50));
  // The time should be greater than 50 + 50 = 100ms.
  EXPECT_GT(phases.at(std::string(BenchmarkInfo::InitPhaseToString(
                BenchmarkInfo::InitPhase::kModelAssets))),
            absl::Milliseconds(100));
}

TEST(BenchmarkInfoTests, AddInitPhaseTwice) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeInitPhaseStart(
      BenchmarkInfo::InitPhase::kModelAssets));
  // Starting the same phase twice should fail.
  EXPECT_THAT(
      benchmark_info.TimeInitPhaseStart(BenchmarkInfo::InitPhase::kModelAssets),
      StatusIs(absl::StatusCode::kInternal));

  // Ending a phase that has not started should fail.
  EXPECT_THAT(
      benchmark_info.TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kTokenizer),
      StatusIs(absl::StatusCode::kInternal));
}

TEST(BenchmarkInfoTests, CandDoInitPhaseRecord) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.InitPhaseRecord(
      BenchmarkInfo::InitPhase::kModelAssets, absl::Milliseconds(50)));
  EXPECT_OK(benchmark_info.InitPhaseRecord(BenchmarkInfo::InitPhase::kTokenizer,
                                           absl::Milliseconds(100)));

  EXPECT_EQ(benchmark_info.GetInitPhases().at(
                std::string(BenchmarkInfo::InitPhaseToString(
                    BenchmarkInfo::InitPhase::kModelAssets))),
            absl::Milliseconds(50));
  EXPECT_EQ(benchmark_info.GetInitPhases().at(
                std::string(BenchmarkInfo::InitPhaseToString(
                    BenchmarkInfo::InitPhase::kTokenizer))),
            absl::Milliseconds(100));
}

TEST(BenchmarkInfoTests, AddInitPhaseError) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  // Recording the same phase twice should fail.
  EXPECT_OK(benchmark_info.InitPhaseRecord(
      BenchmarkInfo::InitPhase::kModelAssets, absl::Milliseconds(50)));
  EXPECT_THAT(
      benchmark_info.InitPhaseRecord(BenchmarkInfo::InitPhase::kModelAssets,
                                     absl::Milliseconds(50)),
      StatusIs(absl::StatusCode::kInternal));
}

TEST(BenchmarkInfoTests, AddPrefillTurn) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimePrefillTurnStart());
  EXPECT_OK(benchmark_info.TimePrefillTurnEnd(100));
  EXPECT_OK(benchmark_info.TimePrefillTurnStart());
  EXPECT_OK(benchmark_info.TimePrefillTurnEnd(200));
  EXPECT_EQ(benchmark_info.GetTotalPrefillTurns(), 2);
}

TEST(BenchmarkInfoTests, AddPrefillTurnError) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimePrefillTurnStart());
  // Starting the prefill turn twice should fail.
  EXPECT_THAT(benchmark_info.TimePrefillTurnStart(),
              StatusIs(absl::StatusCode::kInternal));

  EXPECT_OK(benchmark_info.TimePrefillTurnEnd(100));
  // Ending a prefill turn that has not started should fail.
  EXPECT_THAT(benchmark_info.TimePrefillTurnEnd(200),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(BenchmarkInfoTests, AddDecodeTurn) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeDecodeTurnStart());
  EXPECT_OK(benchmark_info.TimeDecodeTurnEnd(100));
  EXPECT_OK(benchmark_info.TimeDecodeTurnStart());
  EXPECT_OK(benchmark_info.TimeDecodeTurnEnd(200));
  EXPECT_EQ(benchmark_info.GetTotalDecodeTurns(), 2);
}

TEST(BenchmarkInfoTests, AddDecodeTurnError) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeDecodeTurnStart());
  // Starting the decode turn twice should fail.
  EXPECT_THAT(benchmark_info.TimeDecodeTurnStart(),
              StatusIs(absl::StatusCode::kInternal));

  EXPECT_OK(benchmark_info.TimeDecodeTurnEnd(100));
  // Ending a decode turn that has not started should fail.
  EXPECT_THAT(benchmark_info.TimeDecodeTurnEnd(200),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(BenchmarkInfoTests, AddTextToTokenIdsTurn) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeTextToTokenIdsStart());
  EXPECT_OK(benchmark_info.TimeTextToTokenIdsEnd(10));
  EXPECT_OK(benchmark_info.TimeTextToTokenIdsStart());
  EXPECT_OK(benchmark_info.TimeTextToTokenIdsEnd(20));
  EXPECT_EQ(benchmark_info.GetTotalTextToTokenIdsTurns(), 2);
}

TEST(BenchmarkInfoTests, AddTextToTokenIdsTurnError) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeTextToTokenIdsStart());
  // Starting the turn twice should fail.
  EXPECT_THAT(benchmark_info.TimeTextToTokenIdsStart(),
              StatusIs(absl::StatusCode::kInternal));

  EXPECT_OK(benchmark_info.TimeTextToTokenIdsEnd(10));
  // Ending a turn that has not started should fail.
  EXPECT_THAT(benchmark_info.TimeTextToTokenIdsEnd(20),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(BenchmarkInfoTests, AddMarks) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeMarkDelta("sampling"));
  absl::SleepFor(absl::Milliseconds(200));
  EXPECT_OK(benchmark_info.TimeMarkDelta("sampling"));
  absl::SleepFor(absl::Milliseconds(200));
  EXPECT_OK(benchmark_info.TimeMarkDelta("sampling"));
  EXPECT_EQ(benchmark_info.GetMarkDurations().size(), 1);

  // The time should record the duration between the 2nd and 3rd calls, which
  // should be slightly more than 200ms.
  EXPECT_GT(benchmark_info.GetMarkDurations().at("sampling"),
            absl::Milliseconds(200));
  // Verify that the time doesn't record the duration between the 1st and 3nd
  // calls, which is less than 200ms + 200ms = 400ms.
  EXPECT_LT(benchmark_info.GetMarkDurations().at("sampling"),
            absl::Milliseconds(400));
}

TEST(BenchmarkInfoTests, AddTwoMarks) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimeMarkDelta("tokenize"));
  EXPECT_OK(benchmark_info.TimeMarkDelta("sampling"));
  absl::SleepFor(absl::Milliseconds(50));
  EXPECT_OK(benchmark_info.TimeMarkDelta("sampling"));
  absl::SleepFor(absl::Milliseconds(50));
  EXPECT_OK(benchmark_info.TimeMarkDelta("tokenize"));
  EXPECT_EQ(benchmark_info.GetMarkDurations().size(), 2);

  // Time between two sampling calls should be more than 50ms.
  EXPECT_GT(benchmark_info.GetMarkDurations().at("sampling"),
            absl::Milliseconds(50));
  // Time between two tokenize calls should be more than 50ms + 50ms = 100ms.
  EXPECT_GT(benchmark_info.GetMarkDurations().at("tokenize"),
            absl::Milliseconds(100));
}

TEST(BenchmarkInfoTests, GetTimeToFirstTokenInvalid) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(benchmark_info.TimePrefillTurnStart());
  EXPECT_OK(benchmark_info.TimePrefillTurnEnd(100));
  EXPECT_EQ(benchmark_info.GetTimeToFirstToken(), 0.0);
}

TEST(BenchmarkInfoTests, GetTimeToFirstTokenValid) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  // Simulating prefilling 100 tokens takes > 100ms.
  EXPECT_OK(benchmark_info.TimePrefillTurnStart());
  absl::SleepFor(absl::Milliseconds(100));
  EXPECT_OK(benchmark_info.TimePrefillTurnEnd(100));
  // Simulating decoding 50 tokens takes > 200ms.
  EXPECT_OK(benchmark_info.TimeDecodeTurnStart());
  absl::SleepFor(absl::Milliseconds(200));
  EXPECT_OK(benchmark_info.TimeDecodeTurnEnd(50));

  // The time to first token should be (larger than) 100ms + 200ms / 50 = 104ms.
  EXPECT_GT(benchmark_info.GetTimeToFirstToken(), 0.104);
}

TEST(BenchmarkInfoTests, OperatorOutputWithData) {
  BenchmarkInfo benchmark_info(GetBenchmarkParams());
  EXPECT_OK(
      benchmark_info.TimeInitPhaseStart(BenchmarkInfo::InitPhase::kTotal));
  EXPECT_OK(benchmark_info.TimeInitPhaseStart(
      BenchmarkInfo::InitPhase::kModelAssets));
  EXPECT_OK(
      benchmark_info.TimeInitPhaseStart(BenchmarkInfo::InitPhase::kTokenizer));
  EXPECT_OK(
      benchmark_info.TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kModelAssets));
  EXPECT_OK(
      benchmark_info.TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kTokenizer));
  EXPECT_OK(benchmark_info.TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kTotal));

  EXPECT_OK(benchmark_info.TimePrefillTurnStart());
  EXPECT_OK(benchmark_info.TimePrefillTurnEnd(100));
  EXPECT_OK(benchmark_info.TimePrefillTurnStart());
  EXPECT_OK(benchmark_info.TimePrefillTurnEnd(200));

  EXPECT_OK(benchmark_info.TimeDecodeTurnStart());
  EXPECT_OK(benchmark_info.TimeDecodeTurnEnd(100));

  EXPECT_OK(benchmark_info.TimeTextToTokenIdsStart());
  EXPECT_OK(benchmark_info.TimeTextToTokenIdsEnd(50));

  std::stringstream ss;
  ss << benchmark_info;
  const std::string expected_output = R"(BenchmarkInfo:
  Init Phases \(3\):
    - Init Model assets: .* ms
    - Init Tokenizer: .* ms
    - Init Total: .* ms
--------------------------------------------------
  Time to first token: .* s
--------------------------------------------------
  Prefill Turns \(Total 2 turns\):
    Prefill Turn 1: Processed 100 tokens in .* duration.
      Prefill Speed: .* tokens/sec.
    Prefill Turn 2: Processed 200 tokens in .* duration.
      Prefill Speed: .* tokens/sec.
--------------------------------------------------
  Decode Turns \(Total 1 turns\):
    Decode Turn 1: Processed 100 tokens in .* duration.
      Decode Speed: .* tokens/sec.
--------------------------------------------------
  TextToTokenIds Turns \(Total 1 turns\):
    Turn 1: .*, 50 tokens
--------------------------------------------------
--------------------------------------------------
)";
  EXPECT_THAT(ss.str(), ContainsRegex(expected_output));
}

TEST(DecodeConfigTest, CreateDefault) {
  DecodeConfig decode_config = DecodeConfig::CreateDefault();
  EXPECT_EQ(decode_config.GetConstraint(), nullptr);
  EXPECT_EQ(decode_config.GetMaxOutputTokens(), std::nullopt);
}

TEST(DecodeConfigTest, SetAndGetConstraint) {
  DecodeConfig decode_config = DecodeConfig::CreateDefault();
  auto constraint = FakeConstraint({1, 2, 3}, /*vocabulary_size=*/10);
  decode_config.SetConstraint(&constraint);
  EXPECT_EQ(decode_config.GetConstraint(), &constraint);
}

TEST(DecodeConfigTest, SetAndGetMaxOutputTokens) {
  DecodeConfig decode_config = DecodeConfig::CreateDefault();
  EXPECT_EQ(decode_config.GetMaxOutputTokens(), std::nullopt);

  decode_config.SetMaxOutputTokens(42);
  EXPECT_EQ(decode_config.GetMaxOutputTokens(), 42);
}

TEST(VisionExecutorPropertiesTest, OperatorOutput) {
  VisionExecutorProperties properties;
  properties.num_tokens_per_image = 128;
  properties.patch_num_shrink_factor = 4;

  std::stringstream ss;
  ss << properties;
  EXPECT_THAT(ss.str(), ContainsRegex("num_tokens_per_image: 128"));
  EXPECT_THAT(ss.str(), ContainsRegex("patch_num_shrink_factor: 4"));
}

TEST(VisionExecutorPropertiesTest, OperatorOutputDefault) {
  VisionExecutorProperties properties;

  std::stringstream ss;
  ss << properties;
  EXPECT_THAT(ss.str(), ContainsRegex("num_tokens_per_image: 256"));
  EXPECT_THAT(ss.str(), ContainsRegex("patch_num_shrink_factor: not set"));
}

TEST(AudioExecutorPropertiesTest, OperatorOutput) {
  AudioExecutorProperties properties;
  properties.is_streaming_model = true;
  properties.streaming_chunk_size = 1024;
  properties.streaming_chunk_overlap_size = 256;
  properties.audio_shrink_factor = 8;

  std::stringstream ss;
  ss << properties;
  EXPECT_THAT(ss.str(), ContainsRegex("is_streaming_model: 1"));
  EXPECT_THAT(ss.str(), ContainsRegex("streaming_chunk_size: 1024"));
  EXPECT_THAT(ss.str(), ContainsRegex("streaming_chunk_overlap_size: 256"));
  EXPECT_THAT(ss.str(), ContainsRegex("audio_shrink_factor: 8"));
}

}  // namespace
}  // namespace litert::lm
