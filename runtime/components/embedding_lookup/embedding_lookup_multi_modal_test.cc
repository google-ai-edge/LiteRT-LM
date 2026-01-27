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

#include "runtime/components/embedding_lookup/embedding_lookup_multi_modal.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert

namespace litert::lm {

class EmbeddingLookupMultiModalTest : public testing::Test {
 protected:
  std::unique_ptr<EmbeddingLookupMultiModal> GetEmbeddingLookupMultiModal() {
    static struct alignas(::litert::kHostMemoryBufferAlignment) {
      float d[24] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                     9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                     17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    } data;

    auto buffer = ::litert::TensorBuffer::CreateFromHostMemory(
        ::litert::RankedTensorType(
            ::litert::ElementType::Float32,
            ::litert::Layout(::litert::Dimensions({4, 2, 3}))),
        data.d, 24 * sizeof(float));
    EXPECT_TRUE(buffer.HasValue());
    buffer_ = std::move(buffer.Value());

    auto embedding_lookup =
        EmbeddingLookupMultiModal::Create(&buffer_, special_token_);
    EXPECT_TRUE(embedding_lookup.ok());
    return std::move(embedding_lookup.value());
  }

  Expected<TensorBuffer> GetTensorBuffer(
      Dimensions& dimensions, ElementType element_type = ElementType::Float32) {
    size_t buffer_size = sizeof(float);
    for (auto dim : dimensions) {
      buffer_size *= dim;
    }
    Layout layout(dimensions);
    RankedTensorType ranked_tensor_type(element_type, std::move(layout));

    LITERT_ASSIGN_OR_RETURN(auto buffer,
                            TensorBuffer::CreateManaged(
                                *env_, ::litert::TensorBufferType::kHostMemory,
                                ranked_tensor_type, buffer_size));

    // Clear the buffer to 0.
    auto buffer_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        buffer, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<float*>(buffer_lock_and_addr->second);
    memset(output_tensor_ptr, 0, buffer_size);

    return buffer;
  }

  Expected<Environment> env_ = Environment::Create({});
  int special_token_ = -1;
  litert::TensorBuffer buffer_;
};

TEST_F(EmbeddingLookupMultiModalTest, LookupPrefillAllSpecialTokens) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  ::litert::Dimensions dimensions({1, 4, 2, 3});
  LITERT_ASSERT_OK_AND_ASSIGN(litert::TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));
  std::vector<int> tokens = {special_token_, special_token_, special_token_,
                             special_token_};
  ASSERT_OK(
      embedding->LookupPrefill(absl::MakeConstSpan(tokens), &output_tensor, 0));

  auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      output_tensor, ::litert::TensorBuffer::LockMode::kRead);
  auto output_tensor_ptr =
      reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

  LITERT_ASSERT_OK_AND_ASSIGN(size_t output_tensor_size, output_tensor.Size());
  size_t num_floats = output_tensor_size / 4;
  for (size_t i = 0; i < num_floats; ++i) {
    ASSERT_EQ(output_tensor_ptr[i], i + 1.0);
  }
}

TEST_F(EmbeddingLookupMultiModalTest, LookupPrefillNoSpecialTokens) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  ::litert::Dimensions dimensions({1, 4, 2, 3});
  LITERT_ASSERT_OK_AND_ASSIGN(litert::TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));
  std::vector<int> tokens = {1, 2, 3, 4};
  ASSERT_OK(
      embedding->LookupPrefill(absl::MakeConstSpan(tokens), &output_tensor, 0));

  auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      output_tensor, ::litert::TensorBuffer::LockMode::kRead);
  auto output_tensor_ptr =
      reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

  LITERT_ASSERT_OK_AND_ASSIGN(size_t output_tensor_size, output_tensor.Size());
  size_t num_floats = output_tensor_size / 4;
  for (size_t i = 0; i < num_floats; ++i) {
    ASSERT_EQ(output_tensor_ptr[i], 0.0);
  }
}

TEST_F(EmbeddingLookupMultiModalTest, LookupPrefillSingleSpecialToken) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  ::litert::Dimensions dimensions({1, 4, 2, 3});
  LITERT_ASSERT_OK_AND_ASSIGN(litert::TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));
  std::vector<int> tokens = {1, special_token_, 3, 4};
  ASSERT_OK(
      embedding->LookupPrefill(absl::MakeConstSpan(tokens), &output_tensor, 0));

  auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      output_tensor, ::litert::TensorBuffer::LockMode::kRead);
  auto output_tensor_ptr =
      reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

  LITERT_ASSERT_OK_AND_ASSIGN(size_t output_tensor_size, output_tensor.Size());
  size_t num_floats = output_tensor_size / 4;
  float embedding_value = 1.0;
  for (size_t i = 0; i < num_floats; ++i) {
    // Only the second token out of four should have been updated.
    if (i >= 6 && i < 12) {
      ASSERT_EQ(output_tensor_ptr[i], embedding_value++);
    } else {
      ASSERT_EQ(output_tensor_ptr[i], 0.0);
    }
  }
}

TEST_F(EmbeddingLookupMultiModalTest, LookupPrefillVectorNoSpecialToken) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  std::vector<float> output_vector(2 * 3);
  ASSERT_OK(embedding->LookupPrefill(1, output_vector));

  for (float f : output_vector) {
    ASSERT_EQ(f, 0.0);
  }
}

TEST_F(EmbeddingLookupMultiModalTest, LookupPrefillVectorSpecialToken) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  std::vector<float> output_vector(2 * 3);
  ASSERT_OK(embedding->LookupPrefill(-1, output_vector));

  float embedding_value = 1.0;
  for (float f : output_vector) {
    ASSERT_EQ(f, embedding_value++);
  }
}

TEST_F(EmbeddingLookupMultiModalTest, LookupPrefillVectorTooSmall) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  std::vector<float> output_vector(4 * 2 * 3 + 1);
  ASSERT_THAT(
      embedding->LookupPrefill(-1, output_vector),
      testing::status::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("The embedding buffer is not large enough to "
                             "contain the number of requested tokens.")));
}

TEST_F(EmbeddingLookupMultiModalTest, LookupPrefillMultipleSpecialTokens) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  ::litert::Dimensions dimensions({1, 4, 2, 3});
  LITERT_ASSERT_OK_AND_ASSIGN(litert::TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));
  std::vector<int> tokens = {1, special_token_, 3, special_token_};
  ASSERT_OK(
      embedding->LookupPrefill(absl::MakeConstSpan(tokens), &output_tensor, 0));

  auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      output_tensor, ::litert::TensorBuffer::LockMode::kRead);
  auto output_tensor_ptr =
      reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

  LITERT_ASSERT_OK_AND_ASSIGN(size_t output_tensor_size, output_tensor.Size());
  size_t num_floats = output_tensor_size / 4;
  float embedding_value = 1.0;
  for (size_t i = 0; i < num_floats; ++i) {
    // Only the second token and fourth tokens out of four should have been
    // updated.
    if ((i >= 6 && i < 12) || (i >= 18 && i < 24)) {
      ASSERT_EQ(output_tensor_ptr[i], embedding_value++);
    } else {
      ASSERT_EQ(output_tensor_ptr[i], 0.0);
    }
  }
}

TEST_F(EmbeddingLookupMultiModalTest, LookupPrefillLargerOutputTensor) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  ::litert::Dimensions dimensions({1, 4, 2, 3});
  LITERT_ASSERT_OK_AND_ASSIGN(litert::TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));
  std::vector<int> tokens = {1, special_token_, special_token_};
  ASSERT_OK(
      embedding->LookupPrefill(absl::MakeConstSpan(tokens), &output_tensor, 0));

  auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      output_tensor, ::litert::TensorBuffer::LockMode::kRead);
  auto output_tensor_ptr =
      reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

  LITERT_ASSERT_OK_AND_ASSIGN(size_t output_tensor_size, output_tensor.Size());
  size_t num_floats = output_tensor_size / 4;
  float embedding_value = 1.0;
  for (size_t i = 0; i < num_floats; ++i) {
    // Only the second token and fourth tokens out of four should have been
    // updated.
    if ((i >= 6 && i < 18)) {
      ASSERT_EQ(output_tensor_ptr[i], embedding_value++);
    } else {
      ASSERT_EQ(output_tensor_ptr[i], 0.0);
    }
  }
}

TEST_F(EmbeddingLookupMultiModalTest, LookupPrefillMultipleCalls) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  ::litert::Dimensions dimensions({1, 2, 2, 3});
  LITERT_ASSERT_OK_AND_ASSIGN(litert::TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));
  std::vector<int> tokens = {1, special_token_};
  ASSERT_OK(
      embedding->LookupPrefill(absl::MakeConstSpan(tokens), &output_tensor, 0));

  LITERT_ASSERT_OK_AND_ASSIGN(size_t output_tensor_size, output_tensor.Size());
  size_t num_floats = output_tensor_size / 4;
  float embedding_value = 1.0;
  {
    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

    for (size_t i = 0; i < num_floats; ++i) {
      if ((i >= 6)) {
        ASSERT_EQ(output_tensor_ptr[i], embedding_value++);
      } else {
        ASSERT_EQ(output_tensor_ptr[i], 0.0);
      }
    }
  }

  ASSERT_OK(
      embedding->LookupPrefill(absl::MakeConstSpan(tokens), &output_tensor, 0));

  {
    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<float*>(output_tensor_lock_and_addr->second);
    for (size_t i = 0; i < num_floats; ++i) {
      if ((i >= 6)) {
        ASSERT_EQ(output_tensor_ptr[i], embedding_value++);
      } else {
        ASSERT_EQ(output_tensor_ptr[i], 0.0);
      }
    }
  }
}

TEST_F(EmbeddingLookupMultiModalTest, LookupPrefillWithOffset) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  ::litert::Dimensions dimensions({1, 4, 2, 3});
  LITERT_ASSERT_OK_AND_ASSIGN(litert::TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  std::vector<int> tokens = {1, special_token_, 3};
  const size_t float_offset = 2 * 3;
  const size_t byte_offset = float_offset * sizeof(float);

  {
    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kWrite);
    auto output_tensor_ptr =
        reinterpret_cast<uint8_t*>(output_tensor_lock_and_addr->second);
    memset(output_tensor_ptr, 99, byte_offset);
  }

  ASSERT_OK(
      embedding->LookupPrefill(absl::MakeConstSpan(tokens), &output_tensor, 0));

  auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      output_tensor, ::litert::TensorBuffer::LockMode::kRead);
  auto output_tensor_ptr =
      reinterpret_cast<uint8_t*>(output_tensor_lock_and_addr->second);

  for (int i = 0; i < byte_offset; ++i) {
    ASSERT_EQ(output_tensor_ptr[i], 99);
  }

  auto output_tensor_ptr_float = reinterpret_cast<float*>(output_tensor_ptr);

  LITERT_ASSERT_OK_AND_ASSIGN(size_t output_tensor_size, output_tensor.Size());
  size_t num_floats = output_tensor_size / 4;
  float embedding_value = 1.0;
  for (size_t i = float_offset; i < num_floats; ++i) {
    // Only the second token out of four should have been updated.
    if (i >= 6 && i < 12) {
      ASSERT_EQ(output_tensor_ptr_float[i], embedding_value++);
    } else {
      ASSERT_EQ(output_tensor_ptr_float[i], 0.0);
    }
  }
}

TEST_F(EmbeddingLookupMultiModalTest, LookupPrefillWithBadOffset) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  ::litert::Dimensions dimensions({1, 4, 2, 3});
  LITERT_ASSERT_OK_AND_ASSIGN(litert::TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  std::vector<int> tokens = {1, special_token_, 2, 3};
  const size_t float_offset = 2 * 3;
  const size_t byte_offset = float_offset * sizeof(float);

  {
    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kWrite);
    auto output_tensor_ptr =
        reinterpret_cast<uint8_t*>(output_tensor_lock_and_addr->second);

    memset(output_tensor_ptr, 99, byte_offset);
  }

  ASSERT_OK(
      embedding->LookupPrefill(absl::MakeConstSpan(tokens), &output_tensor, 0));

  ASSERT_THAT(
      embedding->LookupPrefill(tokens, &output_tensor, byte_offset),
      testing::status::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr(
              "The byte offset and the total number of bytes to be written "
              "must not exceed the size of the output tensor")));
}

TEST_F(EmbeddingLookupMultiModalTest, LookupPrefillNullOutputTensor) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  std::vector<int> tokens = {1, special_token_, 3, special_token_};
  ASSERT_THAT(
      embedding->LookupPrefill(absl::MakeConstSpan(tokens), nullptr, 0),
      testing::status::StatusIs(absl::StatusCode::kInvalidArgument,
                                testing::HasSubstr("Output tensor is null")));
}

TEST_F(EmbeddingLookupMultiModalTest, LookupPrefillBadOutputTensorTyp) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  std::vector<int> tokens = {1, special_token_, 3, special_token_};

  ::litert::Dimensions dimensions({1, 1, 4, 32});
  LITERT_ASSERT_OK_AND_ASSIGN(
      litert::TensorBuffer output_tensor,
      GetTensorBuffer(dimensions, ElementType::Float16));

  ASSERT_THAT(
      embedding->LookupPrefill(absl::MakeConstSpan(tokens), &output_tensor, 0),
      testing::status::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("The output tensor type for multimodal embedding "
                             "lookup must be float32")));
}

TEST_F(EmbeddingLookupMultiModalTest, LookupPrefillWrongDimensions) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  ::litert::Dimensions dimensions({1, 2});
  LITERT_ASSERT_OK_AND_ASSIGN(litert::TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));
  std::vector<int> tokens = {1, 2};
  ASSERT_THAT(
      embedding->LookupPrefill(absl::MakeConstSpan(tokens), &output_tensor, 0),
      testing::status::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("The output tensor provided to the "
                             "Embedding LookupPrefill function "
                             "must have at least 3 dimensions")));
}

TEST_F(EmbeddingLookupMultiModalTest, LookupPrefillBadDimension0) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  ::litert::Dimensions dimensions({2, 4, 2, 3});
  LITERT_ASSERT_OK_AND_ASSIGN(litert::TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));
  std::vector<int> tokens = {1, 2, 3, 4};
  ASSERT_THAT(
      embedding->LookupPrefill(absl::MakeConstSpan(tokens), &output_tensor, 0),
      testing::status::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("The output tensor to fill with the multimodal "
                             "embeddings must be have the 0th dimension as 1. "
                             "Other sizes are not supported yet")));
}

TEST_F(EmbeddingLookupMultiModalTest, LookupPrefillBadSize) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  ::litert::Dimensions dimensions({1, 4, 2, 3});
  LITERT_ASSERT_OK_AND_ASSIGN(litert::TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));
  std::vector<int> tokens = {1, 2, 3, 4, 5};
  ASSERT_THAT(
      embedding->LookupPrefill(absl::MakeConstSpan(tokens), &output_tensor, 0),
      testing::status::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr(
              "The output tensor to fill from the multimodal embeddings must "
              "have a 1st dimension that is at least the same size as the "
              "number of tokens")));
}

TEST_F(EmbeddingLookupMultiModalTest, LookupPrefillTooManySpecialTokens) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  ::litert::Dimensions dimensions({1, 5, 2, 3});
  LITERT_ASSERT_OK_AND_ASSIGN(litert::TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));
  std::vector<int> tokens = {special_token_, special_token_, special_token_,
                             special_token_, special_token_};
  ASSERT_THAT(
      embedding->LookupPrefill(absl::MakeConstSpan(tokens), &output_tensor, 0),
      testing::status::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("The embedding buffer is not large enough to "
                             "contain the number of requested tokens")));
}

TEST_F(EmbeddingLookupMultiModalTest, LookupDecode) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  ::litert::Dimensions dimensions({1, 4, 2, 3});
  LITERT_ASSERT_OK_AND_ASSIGN(litert::TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));
  int token = special_token_;
  ASSERT_THAT(
      embedding->LookupDecode(token, &output_tensor),
      testing::status::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("Multimodal embedding lookup is not supported for "
                             "single token decode case.")));
}

TEST_F(EmbeddingLookupMultiModalTest, LookupDecodeVectorNoSpecialToken) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  std::vector<float> output_vector(2 * 3);
  int token = 1;
  ASSERT_THAT(
      embedding->LookupDecode(token, output_vector),
      testing::status::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("Multimodal embedding lookup is not supported for "
                             "single token decode case.")));
}

TEST_F(EmbeddingLookupMultiModalTest, LookupDecodeVectorSpecialToken) {
  std::unique_ptr<EmbeddingLookupMultiModal> embedding =
      GetEmbeddingLookupMultiModal();
  ASSERT_NE(embedding, nullptr);

  std::vector<float> output_vector(2 * 3);
  int token = special_token_;
  ASSERT_THAT(
      embedding->LookupDecode(token, output_vector),
      testing::status::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("Multimodal embedding lookup is not supported for "
                             "single token decode case.")));
}

}  // namespace litert::lm
