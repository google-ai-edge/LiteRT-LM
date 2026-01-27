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

#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/util/status_macros.h"  //NOLINT

namespace litert::lm {

using ::litert::lm::ExecutorInputs;

constexpr char kTestdataDir[] =
    "litert_lm/runtime/components/testdata/";

class EmbeddingLookupManagerTest : public ::testing::Test {
 public:
  void SetUp() override {
    auto text_embedding_model_path =
        std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
        "dummy_embedding_cpu_model.tflite";
    auto text_embedding_model =
        Model::CreateFromFile(text_embedding_model_path.string());
    if (!text_embedding_model.HasValue()) {
      ABSL_LOG(ERROR) << "Failed to create text embedding model.";
      return;
    }
    text_embedding_model_ = std::move(*text_embedding_model);

    auto end_of_multi_modal_model_path =
        std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
        "dummy_end_of_multi_modal_model.tflite";
    auto end_of_multi_modal_model =
        Model::CreateFromFile(end_of_multi_modal_model_path.string());
    if (!end_of_multi_modal_model.HasValue()) {
      ABSL_LOG(ERROR) << "Failed to create end of multi-modal model.";
      return;
    }
    end_of_multi_modal_model_ = std::move(*end_of_multi_modal_model);

    absl::flat_hash_map<int, const Model*> end_of_multi_modal_embedding_models;
    end_of_multi_modal_embedding_models.insert(
        {-3, &end_of_multi_modal_model_});

    auto status = EmbeddingLookupManager::Create(
        &text_embedding_model_, end_of_multi_modal_embedding_models);
    ASSERT_OK(status);
    embedding_lookup_manager_ = std::move(status.value());
  }

  absl::Status UpdateMultiModalEmbeddings() {
    // 2 vision embedding columns, each with 128 elements.
    static struct alignas(::litert::kHostMemoryBufferAlignment) {
      float d[256] = {
          1.0,   2.0,   3.0,   4.0,   5.0,   6.0,   7.0,   8.0,   9.0,   10.0,
          11.0,  12.0,  13.0,  14.0,  15.0,  16.0,  17.0,  18.0,  19.0,  20.0,
          21.0,  22.0,  23.0,  24.0,  25.0,  26.0,  27.0,  28.0,  29.0,  30.0,
          31.0,  32.0,  33.0,  34.0,  35.0,  36.0,  37.0,  38.0,  39.0,  40.0,
          41.0,  42.0,  43.0,  44.0,  45.0,  46.0,  47.0,  48.0,  49.0,  50.0,
          51.0,  52.0,  53.0,  54.0,  55.0,  56.0,  57.0,  58.0,  59.0,  60.0,
          61.0,  62.0,  63.0,  64.0,  65.0,  66.0,  67.0,  68.0,  69.0,  70.0,
          71.0,  72.0,  73.0,  74.0,  75.0,  76.0,  77.0,  78.0,  79.0,  80.0,
          81.0,  82.0,  83.0,  84.0,  85.0,  86.0,  87.0,  88.0,  89.0,  90.0,
          91.0,  92.0,  93.0,  94.0,  95.0,  96.0,  97.0,  98.0,  99.0,  100.0,
          101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
          111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
          121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0,
          131.0, 132.0, 133.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0, 140.0,
          141.0, 142.0, 143.0, 144.0, 145.0, 146.0, 147.0, 148.0, 149.0, 150.0,
          151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0, 160.0,
          161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0, 169.0, 170.0,
          171.0, 172.0, 173.0, 174.0, 175.0, 176.0, 177.0, 178.0, 179.0, 180.0,
          181.0, 182.0, 183.0, 184.0, 185.0, 186.0, 187.0, 188.0, 189.0, 190.0,
          191.0, 192.0, 193.0, 194.0, 195.0, 196.0, 197.0, 198.0, 199.0, 200.0,
          201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0, 209.0, 210.0,
          211.0, 212.0, 213.0, 214.0, 215.0, 216.0, 217.0, 218.0, 219.0, 220.0,
          221.0, 222.0, 223.0, 224.0, 225.0, 226.0, 227.0, 228.0, 229.0, 230.0,
          231.0, 232.0, 233.0, 234.0, 235.0, 236.0, 237.0, 238.0, 239.0, 240.0,
          241.0, 242.0, 243.0, 244.0, 245.0, 246.0, 247.0, 248.0, 249.0, 250.0,
          251.0, 252.0, 253.0, 254.0, 255.0, 256.0};
    } vision_data;

    // 2 audio embedding columns, each with 128 elements.
    static struct alignas(::litert::kHostMemoryBufferAlignment) {
      float d[256] = {
          257.0, 258.0, 259.0, 260.0, 261.0, 262.0, 263.0, 264.0, 265.0, 266.0,
          267.0, 268.0, 269.0, 270.0, 271.0, 272.0, 273.0, 274.0, 275.0, 276.0,
          277.0, 278.0, 279.0, 280.0, 281.0, 282.0, 283.0, 284.0, 285.0, 286.0,
          287.0, 288.0, 289.0, 290.0, 291.0, 292.0, 293.0, 294.0, 295.0, 296.0,
          297.0, 298.0, 299.0, 300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0,
          307.0, 308.0, 309.0, 310.0, 311.0, 312.0, 313.0, 314.0, 315.0, 316.0,
          317.0, 318.0, 319.0, 320.0, 321.0, 322.0, 323.0, 324.0, 325.0, 326.0,
          327.0, 328.0, 329.0, 330.0, 331.0, 332.0, 333.0, 334.0, 335.0, 336.0,
          337.0, 338.0, 339.0, 340.0, 341.0, 342.0, 343.0, 344.0, 345.0, 346.0,
          347.0, 348.0, 349.0, 350.0, 351.0, 352.0, 353.0, 354.0, 355.0, 356.0,
          357.0, 358.0, 359.0, 360.0, 361.0, 362.0, 363.0, 364.0, 365.0, 366.0,
          367.0, 368.0, 369.0, 370.0, 371.0, 372.0, 373.0, 374.0, 375.0, 376.0,
          377.0, 378.0, 379.0, 380.0, 381.0, 382.0, 383.0, 384.0, 385.0, 386.0,
          387.0, 388.0, 389.0, 390.0, 391.0, 392.0, 393.0, 394.0, 395.0, 396.0,
          397.0, 398.0, 399.0, 400.0, 401.0, 402.0, 403.0, 404.0, 405.0, 406.0,
          407.0, 408.0, 409.0, 410.0, 411.0, 412.0, 413.0, 414.0, 415.0, 416.0,
          417.0, 418.0, 419.0, 420.0, 421.0, 422.0, 423.0, 424.0, 425.0, 426.0,
          427.0, 428.0, 429.0, 430.0, 431.0, 432.0, 433.0, 434.0, 435.0, 436.0,
          437.0, 438.0, 439.0, 440.0, 441.0, 442.0, 443.0, 444.0, 445.0, 446.0,
          447.0, 448.0, 449.0, 450.0, 451.0, 452.0, 453.0, 454.0, 455.0, 456.0,
          457.0, 458.0, 459.0, 460.0, 461.0, 462.0, 463.0, 464.0, 465.0, 466.0,
          467.0, 468.0, 469.0, 470.0, 471.0, 472.0, 473.0, 474.0, 475.0, 476.0,
          477.0, 478.0, 479.0, 480.0, 481.0, 482.0, 483.0, 484.0, 485.0, 486.0,
          487.0, 488.0, 489.0, 490.0, 491.0, 492.0, 493.0, 494.0, 495.0, 496.0,
          497.0, 498.0, 499.0, 500.0, 501.0, 502.0, 503.0, 504.0, 505.0, 506.0,
          507.0, 508.0, 509.0, 510.0, 511.0, 512.0};
    } audio_data;

    auto buffer = ::litert::TensorBuffer::CreateFromHostMemory(
        ::litert::RankedTensorType(
            ::litert::ElementType::Float32,
            ::litert::Layout(::litert::Dimensions({2, 4, 32}))),
        vision_data.d, 256 * sizeof(float));
    if (!buffer.HasValue()) {
      return absl::InternalError(
          "Failed to create multimodal embedding buffer.");
    }

    ::litert::lm::ExecutorVisionData vision_data_input(std::move(*buffer),
                                                       std::nullopt);

    auto audio_buffer = ::litert::TensorBuffer::CreateFromHostMemory(
        ::litert::RankedTensorType(
            ::litert::ElementType::Float32,
            ::litert::Layout(::litert::Dimensions({2, 4, 32}))),
        audio_data.d, 256 * sizeof(float));
    if (!audio_buffer.HasValue()) {
      return absl::InternalError(
          "Failed to create multimodal embedding buffer.");
    }

    ::litert::lm::ExecutorAudioData audio_data_input(std::move(*audio_buffer),
                                                     std::nullopt);

    // Construct ExecutorInputs with only text_data
    ExecutorInputs inputs(std::nullopt, std::move(vision_data_input),
                          std::move(audio_data_input));

    return embedding_lookup_manager_->UpdateMultiModalEmbeddings(inputs);
  }

  Expected<TensorBuffer> GetTensorBuffer(Dimensions& dimensions) {
    size_t buffer_size = sizeof(float);
    for (auto dim : dimensions) {
      buffer_size *= dim;
    }
    Layout layout(dimensions);
    RankedTensorType ranked_tensor_type(ElementType::Float32,
                                        std::move(layout));

    return TensorBuffer::CreateManaged(*env_,
                                       ::litert::TensorBufferType::kHostMemory,
                                       ranked_tensor_type, buffer_size);
  }

 protected:
  Expected<Environment> env_ = Environment::Create({});
  std::unique_ptr<EmbeddingLookupManager> embedding_lookup_manager_;
  Model text_embedding_model_;
  Model end_of_multi_modal_model_;
};

TEST_F(EmbeddingLookupManagerTest, LookupDecodeTextSingleTokenVector) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  std::vector<float> output_vector;
  int32_t token = 1;
  ASSERT_OK(embedding_lookup_manager_->LookupDecode(token, output_vector));
  ASSERT_EQ(output_vector.size(), 4 * 32);

  // Dimensions 0 and 1 both have size 1.
  for (int idx2 = 0; idx2 < 4; ++idx2) {
    for (int idx3 = 0; idx3 < 32; ++idx3) {
      // Dimensions 0 and 1 both have size 1 so offset and expected value
      // can ignore them.
      size_t offset = idx2 * 32 + idx3;
      float expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
      ASSERT_NEAR(output_vector[offset], expected_value, 1e-5);
    }
  }
}

TEST_F(EmbeddingLookupManagerTest, LookupDecodeTextSingleTokenVectorNonEmpty) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  std::vector<float> output_vector(4 * 32 * 2);
  memset(output_vector.data(), 99, output_vector.size() * sizeof(float));

  int32_t token = 1;
  ASSERT_OK(embedding_lookup_manager_->LookupDecode(token, output_vector));
  ASSERT_EQ(output_vector.size(), 4 * 32);

  // Dimensions 0 and 1 both have size 1.
  for (int idx2 = 0; idx2 < 4; ++idx2) {
    for (int idx3 = 0; idx3 < 32; ++idx3) {
      // Dimensions 0 and 1 both have size 1 so offset and expected value
      // can ignore them.
      size_t offset = idx2 * 32 + idx3;
      float expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
      ASSERT_NEAR(output_vector[offset], expected_value, 1e-5);
    }
  }
}

TEST_F(EmbeddingLookupManagerTest, LookupDecodeTextSingleNegativeTokenVector) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  std::vector<float> output_vector;
  int32_t token = -1;
  ASSERT_THAT(
      embedding_lookup_manager_->LookupDecode(token, output_vector),
      testing::status::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr(
              "Multimodal embeddings are not supported during decode")));

  token = -2;
  ASSERT_THAT(
      embedding_lookup_manager_->LookupDecode(token, output_vector),
      testing::status::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr(
              "Multimodal embeddings are not supported during decode")));
}

TEST_F(EmbeddingLookupManagerTest, LookupDecodeTextSingleToken) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  Dimensions dimensions({1, 1, 4, 32});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  int32_t token = 1;
  ASSERT_OK(embedding_lookup_manager_->LookupDecode(token, &output_tensor));

  auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      output_tensor, ::litert::TensorBuffer::LockMode::kRead);
  auto output_tensor_ptr =
      reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

  // Dimensions 0 and 1 both have size 1.
  for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
    for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
      // Dimensions 0 and 1 both have size 1 so offset and expected value
      // can ignore them.
      size_t offset = idx2 * dimensions[3] + idx3;
      float expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
      ASSERT_NEAR(output_tensor_ptr[offset], expected_value, 1e-5);
    }
  }
}

TEST_F(EmbeddingLookupManagerTest,
       LookupDecodeTextSingleTokenBadOutputTensorDimNum) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  Dimensions dimensions({1, 1, 4});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  ASSERT_THAT(embedding_lookup_manager_->LookupDecode(1, &output_tensor),
              testing::status::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("The output tensor from the Embedding "
                                     "model must be have the same "
                                     "number of dimensions")));
}

TEST_F(EmbeddingLookupManagerTest,
       LookupDecodeTextSingleTokenBadOutputTensorDimSize) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  Dimensions dimensions({1, 1, 4, 256});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  ASSERT_THAT(embedding_lookup_manager_->LookupDecode(1, &output_tensor),
              testing::status::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("The output tensor from the Embedding "
                                     "model must be have the same "
                                     "dimensions")));
}

TEST_F(EmbeddingLookupManagerTest,
       LookupDecodeTextSingleTokenNullOutputTensor) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  ASSERT_THAT(embedding_lookup_manager_->LookupDecode(1, nullptr),
              testing::status::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Decode output tensor buffer is null")));
}

TEST_F(EmbeddingLookupManagerTest, LookupPrefillTextSingleTokenVector) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  std::vector<float> output_vector;
  int32_t token = 1;
  ASSERT_OK(embedding_lookup_manager_->LookupPrefill(token, output_vector));
  ASSERT_EQ(output_vector.size(), 4 * 32);

  // Dimensions 0 and 1 both have size 1.
  for (int idx2 = 0; idx2 < 4; ++idx2) {
    for (int idx3 = 0; idx3 < 32; ++idx3) {
      // Dimensions 0 and 1 both have size 1 so offset and expected value
      // can ignore them.
      size_t offset = idx2 * 32 + idx3;
      float expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
      ASSERT_NEAR(output_vector[offset], expected_value, 1e-5);
    }
  }
}

TEST_F(EmbeddingLookupManagerTest, LookupPrefillTextSingleTokenVectorNonEmpty) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  std::vector<float> output_vector(4 * 32 * 2);
  memset(output_vector.data(), 99, output_vector.size() * sizeof(float));

  int32_t token = 1;
  ASSERT_OK(embedding_lookup_manager_->LookupPrefill(token, output_vector));
  ASSERT_EQ(output_vector.size(), 4 * 32);

  // Dimensions 0 and 1 both have size 1.
  for (int idx2 = 0; idx2 < 4; ++idx2) {
    for (int idx3 = 0; idx3 < 32; ++idx3) {
      // Dimensions 0 and 1 both have size 1 so offset and expected value
      // can ignore them.
      size_t offset = idx2 * 32 + idx3;
      float expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
      ASSERT_NEAR(output_vector[offset], expected_value, 1e-5);
    }
  }
}

TEST_F(EmbeddingLookupManagerTest, LookupPrefillSingleNegativeTokenVector) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  ASSERT_OK(UpdateMultiModalEmbeddings());

  std::vector<float> output_vector;

  {
    int token = -1;
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(token, output_vector));

    float multi_modal_embedding_value = 1.0;
    for (float value : output_vector) {
      ASSERT_EQ(value, multi_modal_embedding_value++);
    }

    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(token, output_vector));
    for (float value : output_vector) {
      ASSERT_EQ(value, multi_modal_embedding_value++);
    }
  }

  {
    int token = -2;
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(token, output_vector));

    float multi_modal_embedding_value = 257.0;
    for (float value : output_vector) {
      ASSERT_EQ(value, multi_modal_embedding_value++);
    }

    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(token, output_vector));
    for (float value : output_vector) {
      ASSERT_EQ(value, multi_modal_embedding_value++);
    }
  }

  ASSERT_OK(embedding_lookup_manager_->CleanupMultiModalEmbeddings());
}

TEST_F(EmbeddingLookupManagerTest, LookupPrefillTextMultipleTokens) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  Dimensions dimensions({1, 3, 4, 32});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  std::vector<int> tokens = {1, 2, 3};
  absl::Span<const int> tokens_span(tokens);
  ASSERT_OK(
      embedding_lookup_manager_->LookupPrefill(tokens_span, &output_tensor, 0));

  auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      output_tensor, ::litert::TensorBuffer::LockMode::kRead);
  auto output_tensor_ptr =
      reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

  for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
    int token = tokens[idx0];
    for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
      for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
        // Since dimension 1 is of size 1, the offset and expected value can
        // ignore it.
        size_t offset =
            idx0 * dimensions[2] * dimensions[3] + idx2 * dimensions[3] + idx3;
        float expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
        ASSERT_NEAR(output_tensor_ptr[offset], expected_value, 1e-5);
      }
    }
  }
}

TEST_F(EmbeddingLookupManagerTest,
       LookupPrefillTextMultipleTokensDecendingTokens) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  Dimensions dimensions({1, 3, 4, 32});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  std::vector<int> tokens = {3, 2, 1};
  absl::Span<const int> tokens_span(tokens);
  ASSERT_OK(
      embedding_lookup_manager_->LookupPrefill(tokens_span, &output_tensor, 0));

  auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      output_tensor, ::litert::TensorBuffer::LockMode::kRead);
  auto output_tensor_ptr =
      reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

  for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
    int token = tokens[idx0];
    for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
      for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
        // Since dimension 1 is of size 1, the offset and expected value can
        // ignore it.
        size_t offset =
            idx0 * dimensions[2] * dimensions[3] + idx2 * dimensions[3] + idx3;
        float expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
        ASSERT_NEAR(output_tensor_ptr[offset], expected_value, 1e-5);
      }
    }
  }
}

TEST_F(EmbeddingLookupManagerTest,
       LookupPrefillTextMultipleTokensRepeatedToken) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  Dimensions dimensions({1, 3, 4, 32});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  std::vector<int> tokens = {1, 1, 1};
  absl::Span<const int> tokens_span(tokens);
  ASSERT_OK(
      embedding_lookup_manager_->LookupPrefill(tokens_span, &output_tensor, 0));

  auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      output_tensor, ::litert::TensorBuffer::LockMode::kRead);
  auto output_tensor_ptr =
      reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

  for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
    int token = tokens[idx0];
    for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
      for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
        // Since dimension 1 is of size 1, the offset and expected value can
        // ignore it.
        size_t offset =
            idx0 * dimensions[2] * dimensions[3] + idx2 * dimensions[3] + idx3;
        float expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
        ASSERT_NEAR(output_tensor_ptr[offset], expected_value, 1e-5);
      }
    }
  }
}

TEST_F(EmbeddingLookupManagerTest,
       LookupPrefillTextMultipleTokensBadOutputTensorDimNum) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  Dimensions dimensions({1, 3, 4});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  std::vector<int> tokens = {1, 2, 3};
  absl::Span<const int> tokens_span(tokens);
  ASSERT_THAT(
      embedding_lookup_manager_->LookupPrefill(tokens_span, &output_tensor, 0),
      testing::status::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("The output tensor from the Embedding "
                             "model must be have the same "
                             "number of dimensions")));
}

TEST_F(EmbeddingLookupManagerTest,
       LookupPrefillTextMultipleTokensBadOutputTensorDimSize) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  Dimensions dimensions({1, 3, 4, 256});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  std::vector<int> tokens = {1, 2, 3};
  absl::Span<const int> tokens_span(tokens);
  ASSERT_THAT(
      embedding_lookup_manager_->LookupPrefill(tokens_span, &output_tensor, 0),
      testing::status::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("The output tensor from the Embedding "
                             "model must be have the same "
                             "dimensions")));
}

TEST_F(EmbeddingLookupManagerTest,
       LookupPrefillTextMultipleTokensBadOutputTensorDim0) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  Dimensions dimensions({3, 1, 4, 256});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  std::vector<int> tokens = {1, 2, 3};
  absl::Span<const int> tokens_span(tokens);
  ASSERT_THAT(
      embedding_lookup_manager_->LookupPrefill(tokens_span, &output_tensor, 0),
      testing::status::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("The output tensor to fill from the Embedding "
                             "model must be have the 0th dimension as 1.")));
}

TEST_F(EmbeddingLookupManagerTest,
       LookupPrefillTextMultipleTokensBadOutputTensorDim1) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  Dimensions dimensions({1, 1, 4, 256});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  std::vector<int> tokens = {1, 2, 3};
  absl::Span<const int> tokens_span(tokens);
  ASSERT_THAT(
      embedding_lookup_manager_->LookupPrefill(tokens_span, &output_tensor, 0),
      testing::status::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("The output tensor to fill from the Embedding "
                             "model must have a 1st dimension that is at least "
                             "the same size as the number of tokens")));
}

TEST_F(EmbeddingLookupManagerTest,
       LookupPrefillTextMultipleTokensNullOutputTensor) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  std::vector<int> tokens = {1, 2, 3};
  absl::Span<const int> tokens_span(tokens);
  ASSERT_THAT(embedding_lookup_manager_->LookupPrefill(tokens_span, nullptr, 0),
              testing::status::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Prefill output tensor buffer is null")));
}

TEST_F(EmbeddingLookupManagerTest,
       LookupPrefillTextMultipleTokensNegativeToken) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  Dimensions dimensions({1, 4, 4, 32});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  {
    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kWrite);
    auto output_tensor_ptr =
        reinterpret_cast<float*>(output_tensor_lock_and_addr->second);
    LITERT_ASSERT_OK_AND_ASSIGN(size_t output_tensor_size,
                                output_tensor.Size());
    float filler_value = 9999.0;
    for (int i = 0; i < output_tensor_size / sizeof(float); ++i) {
      output_tensor_ptr[i] = filler_value;
      ABSL_LOG(INFO) << "output_tensor_ptr[" << i
                     << "]: " << output_tensor_ptr[i];
    }
  }

  std::vector<int> tokens = {1, -1, -2, 2};
  absl::Span<const int> tokens_span(tokens);
  ASSERT_OK(
      embedding_lookup_manager_->LookupPrefill(tokens_span, &output_tensor, 0));

  auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      output_tensor, ::litert::TensorBuffer::LockMode::kRead);
  auto output_tensor_ptr =
      reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

  for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
    int token = tokens[idx0];
    for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
      for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
        float expected_value;
        if (token < 0) {
          // If the token is negative, the expected value is embedding for token
          // 0.
          expected_value = 100.0 * idx2 + idx3;
        } else {
          // Since dimension 1 is of size 1, the offset and expected value
          // can ignore it.
          expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
        }
        size_t offset =
            idx0 * dimensions[2] * dimensions[3] + idx2 * dimensions[3] + idx3;
        ASSERT_NEAR(output_tensor_ptr[offset], expected_value, 1e-5);
      }
    }
  }
}

TEST_F(EmbeddingLookupManagerTest,
       LookupPrefillTextAndMultimodalMultipleTokens) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  ASSERT_OK(UpdateMultiModalEmbeddings());

  Dimensions dimensions({1, 4, 4, 32});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  {
    std::vector<int> tokens = {1, -1, -1, 2};
    absl::Span<const int> tokens_span(tokens);
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(tokens_span,
                                                       &output_tensor, 0));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

    float multi_modal_embedding_value = 1.0;
    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      int token = tokens[idx0];
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          size_t offset = idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          if (token == -1) {
            ASSERT_EQ(output_tensor_ptr[offset], multi_modal_embedding_value++);
          } else {
            // Since dimension 1 is of size 1, the offset and expected value can
            // ignore it.
            float expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
            ASSERT_NEAR(output_tensor_ptr[offset], expected_value, 1e-5);
          }
        }
      }
    }
  }

  {
    std::vector<int> tokens = {1, -2, -2, 2};
    absl::Span<const int> tokens_span(tokens);
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(tokens_span,
                                                       &output_tensor, 0));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

    float multi_modal_embedding_value = 257.0;
    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      int token = tokens[idx0];
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          size_t offset = idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          if (token == -2) {
            ASSERT_EQ(output_tensor_ptr[offset], multi_modal_embedding_value++);
          } else {
            // Since dimension 1 is of size 1, the offset and expected value can
            // ignore it.
            float expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
            ASSERT_NEAR(output_tensor_ptr[offset], expected_value, 1e-5);
          }
        }
      }
    }
  }

  ASSERT_OK(embedding_lookup_manager_->CleanupMultiModalEmbeddings());
}

TEST_F(EmbeddingLookupManagerTest, LookupPrefillMultimodalMultipleTokens) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  ASSERT_OK(UpdateMultiModalEmbeddings());

  Dimensions dimensions({1, 2, 4, 32});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  {
    std::vector<int> tokens = {-1, -1};
    absl::Span<const int> tokens_span(tokens);
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(tokens_span,
                                                       &output_tensor, 0));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

    float multi_modal_embedding_value = 1.0;
    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          size_t offset = idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          ASSERT_EQ(output_tensor_ptr[offset], multi_modal_embedding_value++);
        }
      }
    }
  }

  {
    std::vector<int> tokens = {-2, -2};
    absl::Span<const int> tokens_span(tokens);
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(tokens_span,
                                                       &output_tensor, 0));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

    float multi_modal_embedding_value = 257.0;
    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          size_t offset = idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          ASSERT_EQ(output_tensor_ptr[offset], multi_modal_embedding_value++);
        }
      }
    }
  }

  ASSERT_OK(embedding_lookup_manager_->CleanupMultiModalEmbeddings());
}

TEST_F(EmbeddingLookupManagerTest, LookupPrefillMultimodalTooManyTokens) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  ASSERT_OK(UpdateMultiModalEmbeddings());

  Dimensions dimensions({1, 3, 4, 32});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  {
    // The provided vision embeddings only have 2 columns, mapping to 2 tokens,
    // so the request for 3 tokens will exceed the capacity of the
    // embeddings.
    std::vector<int> tokens = {-1, -1, -1};
    absl::Span<const int> tokens_span(tokens);
    ASSERT_THAT(
        embedding_lookup_manager_->LookupPrefill(tokens_span, &output_tensor,
                                                 0),
        testing::status::StatusIs(
            absl::StatusCode::kInvalidArgument,
            testing::HasSubstr("The embedding buffer is not large enough to "
                               "contain the number of requested tokens")));
  }

  {
    // The provided audio embeddings only have 2 columns, mapping to 2 tokens,
    // so the request for 3 tokens will exceed the capacity of the embeddings.
    std::vector<int> tokens = {-2, -2, -2};
    absl::Span<const int> tokens_span(tokens);
    ASSERT_THAT(
        embedding_lookup_manager_->LookupPrefill(tokens_span, &output_tensor,
                                                 0),
        testing::status::StatusIs(
            absl::StatusCode::kInvalidArgument,
            testing::HasSubstr("The embedding buffer is not large enough to "
                               "contain the number of requested tokens")));
  }
}

TEST_F(EmbeddingLookupManagerTest, LookupPrefillMultimodalMultipleLookups) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  ASSERT_OK(UpdateMultiModalEmbeddings());

  float multi_modal_embedding_value = 1.0;
  const size_t num_repeats = 2;
  for (int i = 0; i < num_repeats; ++i) {
    Dimensions dimensions({1, 1, 4, 32});
    LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                                GetTensorBuffer(dimensions));

    std::vector<int> tokens = {-1};
    absl::Span<const int> tokens_span(tokens);

    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(tokens_span,
                                                       &output_tensor, 0));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          size_t offset = idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          ASSERT_EQ(output_tensor_ptr[offset], multi_modal_embedding_value++);
        }
      }
    }
  }
  for (int i = 0; i < num_repeats; ++i) {
    Dimensions dimensions({1, 1, 4, 32});
    LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                                GetTensorBuffer(dimensions));

    std::vector<int> tokens = {-2};
    absl::Span<const int> tokens_span(tokens);

    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(tokens_span,
                                                       &output_tensor, 0));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          size_t offset = idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          ASSERT_EQ(output_tensor_ptr[offset], multi_modal_embedding_value++);
        }
      }
    }
  }
  ASSERT_OK(embedding_lookup_manager_->CleanupMultiModalEmbeddings());
}

TEST_F(EmbeddingLookupManagerTest, LookupPrefillMultimodalNotAllTokensUsed) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  ASSERT_OK(UpdateMultiModalEmbeddings());

  Dimensions dimensions({1, 1, 4, 32});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  {
    std::vector<int> tokens = {-1};
    absl::Span<const int> tokens_span(tokens);
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(tokens_span,
                                                       &output_tensor, 0));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

    float multi_modal_embedding_value = 1.0;
    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          size_t offset = idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          ASSERT_EQ(output_tensor_ptr[offset], multi_modal_embedding_value++);
        }
      }
    }
  }

  {
    std::vector<int> tokens = {-2};
    absl::Span<const int> tokens_span(tokens);
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(tokens_span,
                                                       &output_tensor, 0));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

    float multi_modal_embedding_value = 257.0;
    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          size_t offset = idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          ASSERT_EQ(output_tensor_ptr[offset], multi_modal_embedding_value++);
        }
      }
    }
  }

  ASSERT_OK(embedding_lookup_manager_->CleanupMultiModalEmbeddings());
}

TEST_F(EmbeddingLookupManagerTest,
       LookupPrefillTextAndMultimodalMultipleTokensWithOffset) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  ASSERT_OK(UpdateMultiModalEmbeddings());

  Dimensions dimensions({1, 4, 4, 32});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  {
    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kWrite);
    auto output_tensor_ptr =
        reinterpret_cast<uint8_t*>(output_tensor_lock_and_addr->second);
    LITERT_ASSERT_OK_AND_ASSIGN(size_t output_tensor_size,
                                output_tensor.Size());
    memset(output_tensor_ptr, 99, output_tensor_size);
  }

  {
    std::vector<int> tokens = {-1, -1, 2};
    absl::Span<const int> tokens_span(tokens);
    const size_t float_offset = 4 * 32;
    const size_t byte_offset = float_offset * sizeof(float);

    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(
        tokens_span, &output_tensor, /*token_offset=*/1));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<uint8_t*>(output_tensor_lock_and_addr->second);

    for (int i = 0; i < byte_offset; ++i) {
      ASSERT_EQ(output_tensor_ptr[i], 99);
    }

    auto output_tensor_float_ptr = reinterpret_cast<float*>(output_tensor_ptr);

    float multi_modal_embedding_value = 1.0;
    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      int token = tokens[idx0];
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          size_t offset = float_offset + idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          if (token == -1) {
            ASSERT_EQ(output_tensor_float_ptr[offset],
                      multi_modal_embedding_value++);
          } else {
            // Since dimension 1 is of size 1, the offset and expected value can
            // ignore it.
            float expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
            ASSERT_NEAR(output_tensor_float_ptr[offset], expected_value, 1e-5);
          }
        }
      }
    }
  }
  {
    std::vector<int> tokens = {-2, -2, 2};
    absl::Span<const int> tokens_span(tokens);
    const size_t float_offset = 4 * 32;
    const size_t byte_offset = float_offset * sizeof(float);

    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(
        tokens_span, &output_tensor, /*token_offset=*/1));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<uint8_t*>(output_tensor_lock_and_addr->second);

    for (int i = 0; i < byte_offset; ++i) {
      ASSERT_EQ(output_tensor_ptr[i], 99);
    }

    auto output_tensor_float_ptr = reinterpret_cast<float*>(output_tensor_ptr);

    float multi_modal_embedding_value = 257.0;
    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      int token = tokens[idx0];
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          size_t offset = float_offset + idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          if (token == -2) {
            ASSERT_EQ(output_tensor_float_ptr[offset],
                      multi_modal_embedding_value++);
          } else {
            // Since dimension 1 is of size 1, the offset and expected value can
            // ignore it.
            float expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
            ASSERT_NEAR(output_tensor_float_ptr[offset], expected_value, 1e-5);
          }
        }
      }
    }
  }
  ASSERT_OK(embedding_lookup_manager_->CleanupMultiModalEmbeddings());
}

TEST_F(EmbeddingLookupManagerTest,
       LookupPrefillTextAndMultimodalMultipleTokensWithBadOffset) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  ASSERT_OK(UpdateMultiModalEmbeddings());

  Dimensions dimensions({1, 4, 4, 32});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  {
    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kWrite);
    auto output_tensor_ptr =
        reinterpret_cast<uint8_t*>(output_tensor_lock_and_addr->second);
    LITERT_ASSERT_OK_AND_ASSIGN(size_t output_tensor_size,
                                output_tensor.Size());
    memset(output_tensor_ptr, 99, output_tensor_size);
  }

  {
    std::vector<int> tokens = {1, -1, -1, 2};
    absl::Span<const int> tokens_span(tokens);

    ASSERT_THAT(
        embedding_lookup_manager_->LookupPrefill(tokens_span, &output_tensor,
                                                 /*token_offset=*/1),
        testing::status::StatusIs(
            absl::StatusCode::kInvalidArgument,
            testing::HasSubstr(
                "The byte offset and the total number of bytes to be written "
                "must not exceed the size of the output tensor")));
  }
  {
    std::vector<int> tokens = {1, -2, -2, 2};
    absl::Span<const int> tokens_span(tokens);

    ASSERT_THAT(
        embedding_lookup_manager_->LookupPrefill(tokens_span, &output_tensor,
                                                 /*token_offset=*/1),
        testing::status::StatusIs(
            absl::StatusCode::kInvalidArgument,
            testing::HasSubstr(
                "The byte offset and the total number of bytes to be written "
                "must not exceed the size of the output tensor")));
  }
}

TEST_F(EmbeddingLookupManagerTest,
       LookupPrefillTextMultipleTokensWithPartialMultiModalSupport) {
  auto status = EmbeddingLookupManager::Create(
      &text_embedding_model_, /*fully_supports_multi_modal=*/false);
  ASSERT_OK(status);
  embedding_lookup_manager_ = std::move(status.value());
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  ExecutorInputs inputs(std::nullopt, std::nullopt, std::nullopt);
  ASSERT_OK(embedding_lookup_manager_->UpdateMultiModalEmbeddings(inputs));

  Dimensions dimensions({1, 3, 4, 32});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  {
    std::vector<int> tokens = {1, -1, 2};
    absl::Span<const int> tokens_span(tokens);
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(tokens_span,
                                                       &output_tensor, 0));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      int token = tokens[idx0];
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          float expected_value;
          if (token < 0) {
            // If the token is negative, the expected value is the embedding
            // value for token 0.
            expected_value = 100.0 * idx2 + idx3;
          } else {
            // Since dimension 1 is of size 1, the offset and expected value
            // can ignore it.
            expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
          }
          size_t offset = idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          ASSERT_NEAR(output_tensor_ptr[offset], expected_value, 1e-5);
        }
      }
    }
  }
  {
    std::vector<int> tokens = {1, -2, 2};
    absl::Span<const int> tokens_span(tokens);
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(tokens_span,
                                                       &output_tensor, 0));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      int token = tokens[idx0];
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          float expected_value;
          if (token < 0) {
            // If the token is negative, the expected value is the embedding
            // value for token 0.
            expected_value = 100.0 * idx2 + idx3;
          } else {
            // Since dimension 1 is of size 1, the offset and expected value
            // can ignore it.
            expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
          }
          size_t offset = idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          ASSERT_NEAR(output_tensor_ptr[offset], expected_value, 1e-5);
        }
      }
    }
  }
  ASSERT_OK(embedding_lookup_manager_->CleanupMultiModalEmbeddings());
}

TEST_F(EmbeddingLookupManagerTest,
       LookupPrefillMultimodalMultipleTokensWithPartialMultiModalSupport) {
  auto status = EmbeddingLookupManager::Create(
      &text_embedding_model_, /*fully_supports_multi_modal=*/false);
  ASSERT_OK(status);
  embedding_lookup_manager_ = std::move(status.value());
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  ExecutorInputs inputs(std::nullopt, std::nullopt, std::nullopt);
  ASSERT_OK(embedding_lookup_manager_->UpdateMultiModalEmbeddings(inputs));

  Dimensions dimensions({1, 2, 4, 32});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  {
    std::vector<int> tokens = {-1, -1};
    absl::Span<const int> tokens_span(tokens);
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(tokens_span,
                                                       &output_tensor, 0));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          // If the token is negative, the expected value is the embedding value
          // for token 0.
          float expected_value = 100.0 * idx2 + idx3;
          size_t offset = idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          ASSERT_EQ(output_tensor_ptr[offset], expected_value);
        }
      }
    }
  }
  {
    std::vector<int> tokens = {-2, -2};
    absl::Span<const int> tokens_span(tokens);
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(tokens_span,
                                                       &output_tensor, 0));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          // If the token is negative, the expected value is the embedding value
          // for token 0.
          float expected_value = 100.0 * idx2 + idx3;
          size_t offset = idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          ASSERT_EQ(output_tensor_ptr[offset], expected_value);
        }
      }
    }
  }
  ASSERT_OK(embedding_lookup_manager_->CleanupMultiModalEmbeddings());
}

TEST_F(EmbeddingLookupManagerTest,
       LookupPrefillTextMultipleTokensWithPartialMultiModalSupportWithOffset) {
  auto status = EmbeddingLookupManager::Create(
      &text_embedding_model_, /*fully_supports_multi_modal=*/false);
  ASSERT_OK(status);
  embedding_lookup_manager_ = std::move(status.value());
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  ExecutorInputs inputs(std::nullopt, std::nullopt, std::nullopt);
  ASSERT_OK(embedding_lookup_manager_->UpdateMultiModalEmbeddings(inputs));

  Dimensions dimensions({1, 4, 4, 32});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  {
    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kWrite);
    auto output_tensor_ptr =
        reinterpret_cast<uint8_t*>(output_tensor_lock_and_addr->second);
    LITERT_ASSERT_OK_AND_ASSIGN(size_t output_tensor_size,
                                output_tensor.Size());
    memset(output_tensor_ptr, 99, output_tensor_size);
  }

  const size_t float_offset = 4 * 32;
  const size_t byte_offset = float_offset * sizeof(float);

  {
    std::vector<int> tokens = {1, -1, 2};
    absl::Span<const int> tokens_span(tokens);
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(
        tokens_span, &output_tensor, /*token_offset=*/1));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<uint8_t*>(output_tensor_lock_and_addr->second);
    for (int i = 0; i < byte_offset; ++i) {
      ASSERT_EQ(output_tensor_ptr[i], 99);
    }

    auto output_tensor_float_ptr = reinterpret_cast<float*>(output_tensor_ptr);

    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      int token = tokens[idx0];
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          float expected_value;
          if (token < 0) {
            // If the token is negative, the expected value is the embedding
            // value for token 0.
            expected_value = 100.0 * idx2 + idx3;
          } else {
            // Since dimension 1 is of size 1, the offset and expected value
            // can ignore it.
            expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
          }
          size_t offset = float_offset + idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          ASSERT_NEAR(output_tensor_float_ptr[offset], expected_value, 1e-5);
        }
      }
    }
  }
  {
    std::vector<int> tokens = {1, -2, 2};
    absl::Span<const int> tokens_span(tokens);
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(
        tokens_span, &output_tensor, /*token_offset=*/1));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<uint8_t*>(output_tensor_lock_and_addr->second);

    for (int i = 0; i < byte_offset; ++i) {
      ASSERT_EQ(output_tensor_ptr[i], 99);
    }
    auto output_tensor_float_ptr = reinterpret_cast<float*>(output_tensor_ptr);

    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      int token = tokens[idx0];
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          float expected_value;
          if (token < 0) {
            // If the token is negative, the expected value is the embedding
            // value for token 0.
            expected_value = 100.0 * idx2 + idx3;
          } else {
            // Since dimension 1 is of size 1, the offset and expected value
            // can ignore it.
            expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
          }
          size_t offset = float_offset + idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          ASSERT_NEAR(output_tensor_float_ptr[offset], expected_value, 1e-5);
        }
      }
    }
  }
  ASSERT_OK(embedding_lookup_manager_->CleanupMultiModalEmbeddings());
}

TEST_F(EmbeddingLookupManagerTest,
       LookupPrefillTokenVectorWithPartialMultiModalSupport) {
  auto status = EmbeddingLookupManager::Create(
      &text_embedding_model_, /*fully_supports_multi_modal=*/false);
  ASSERT_OK(status);
  embedding_lookup_manager_ = std::move(status.value());
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  ExecutorInputs inputs(std::nullopt, std::nullopt, std::nullopt);
  ASSERT_OK(embedding_lookup_manager_->UpdateMultiModalEmbeddings(inputs));

  std::vector<float> output_vector;

  {
    int token = -1;
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(token, output_vector));

    for (int idx2 = 0; idx2 < 4; ++idx2) {
      for (int idx3 = 0; idx3 < 32; ++idx3) {
        float expected_value = 100.0 * idx2 + idx3;
        size_t offset = idx2 * 32 + idx3;
        ASSERT_NEAR(output_vector[offset], expected_value, 1e-5);
      }
    }
  }
  {
    int token = -2;
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(token, output_vector));

    for (int idx2 = 0; idx2 < 4; ++idx2) {
      for (int idx3 = 0; idx3 < 32; ++idx3) {
        float expected_value = 100.0 * idx2 + idx3;
        size_t offset = idx2 * 32 + idx3;
        ASSERT_NEAR(output_vector[offset], expected_value, 1e-5);
      }
    }
  }

  ASSERT_OK(embedding_lookup_manager_->CleanupMultiModalEmbeddings());
}

TEST_F(EmbeddingLookupManagerTest, LookupPrefillTokenSpecifySignatureKey) {
  auto status = EmbeddingLookupManager::Create(
      &text_embedding_model_, /*fully_supports_multi_modal=*/false,
      "serving_default");
  ASSERT_OK(status);
  embedding_lookup_manager_ = std::move(status.value());
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  ExecutorInputs inputs(std::nullopt, std::nullopt, std::nullopt);
  ASSERT_OK(embedding_lookup_manager_->UpdateMultiModalEmbeddings(inputs));

  std::vector<float> output_vector;

  {
    int token = -1;
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(token, output_vector));

    for (int idx2 = 0; idx2 < 4; ++idx2) {
      for (int idx3 = 0; idx3 < 32; ++idx3) {
        float expected_value = 100.0 * idx2 + idx3;
        size_t offset = idx2 * 32 + idx3;
        ASSERT_NEAR(output_vector[offset], expected_value, 1e-5);
      }
    }
  }
  {
    int token = -2;
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(token, output_vector));

    for (int idx2 = 0; idx2 < 4; ++idx2) {
      for (int idx3 = 0; idx3 < 32; ++idx3) {
        float expected_value = 100.0 * idx2 + idx3;
        size_t offset = idx2 * 32 + idx3;
        ASSERT_NEAR(output_vector[offset], expected_value, 1e-5);
      }
    }
  }

  ASSERT_OK(embedding_lookup_manager_->CleanupMultiModalEmbeddings());
}

TEST_F(EmbeddingLookupManagerTest,
       LookupPrefillPartialMultiModalSupportWithEmbeddings) {
  auto status = EmbeddingLookupManager::Create(
      &text_embedding_model_, /*fully_supports_multi_modal=*/false);
  ASSERT_OK(status);
  embedding_lookup_manager_ = std::move(status.value());
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  ASSERT_THAT(
      UpdateMultiModalEmbeddings(),
      testing::status::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("When fully_supports_multi_modal_ is false, "
                             "multimodal embeddings must not be provided")));
}

TEST_F(EmbeddingLookupManagerTest,
       CreateWithPartialMultiModalSupportAndEndOfMultiModalEmbeddings) {
  absl::flat_hash_map<int, const litert::Model*>
      end_of_multi_modal_embedding_models;
  end_of_multi_modal_embedding_models.insert({-3, &end_of_multi_modal_model_});
  auto status = EmbeddingLookupManager::Create(
      &text_embedding_model_, end_of_multi_modal_embedding_models,
      /*fully_supports_multi_modal=*/false);
  ASSERT_THAT(status,
              testing::status::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr(
                      "When fully_supports_multi_modal is false, "
                      "end_of_multi_modal_embedding_models must be empty")));
}

TEST_F(EmbeddingLookupManagerTest,
       LookupPrefillTextAndMultimodalMultipleTokensAndEndOfMultiModal) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  ASSERT_OK(UpdateMultiModalEmbeddings());

  Dimensions dimensions({1, 4, 4, 32});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  {
    std::vector<int> tokens = {1, -1, -1, -3};
    absl::Span<const int> tokens_span(tokens);
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(tokens_span,
                                                       &output_tensor, 0));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

    float multi_modal_embedding_value = 1.0;
    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      int token = tokens[idx0];
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          size_t offset = idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          if (token == -1) {
            ASSERT_EQ(output_tensor_ptr[offset], multi_modal_embedding_value++);
          } else if (token == -3) {
            float expected_value = (100.0 * idx2 + idx3) * 2;
            ASSERT_NEAR(output_tensor_ptr[offset], expected_value, 1e-5);
          } else {
            // Since dimension 1 is of size 1, the offset and expected value can
            // ignore it.
            float expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
            ASSERT_NEAR(output_tensor_ptr[offset], expected_value, 1e-5);
          }
        }
      }
    }
  }

  {
    std::vector<int> tokens = {1, -2, -2, -3};
    absl::Span<const int> tokens_span(tokens);
    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(tokens_span,
                                                       &output_tensor, 0));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

    float multi_modal_embedding_value = 257.0;
    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      int token = tokens[idx0];
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          size_t offset = idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          if (token == -2) {
            ASSERT_EQ(output_tensor_ptr[offset], multi_modal_embedding_value++);
          } else if (token == -3) {
            float expected_value = (100.0 * idx2 + idx3) * 2;
            ASSERT_NEAR(output_tensor_ptr[offset], expected_value, 1e-5);
          } else {
            // Since dimension 1 is of size 1, the offset and expected value can
            // ignore it.
            float expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
            ASSERT_NEAR(output_tensor_ptr[offset], expected_value, 1e-5);
          }
        }
      }
    }
  }

  ASSERT_OK(embedding_lookup_manager_->CleanupMultiModalEmbeddings());
}

// TODO: b/438462241 - Re-enable this test once the LiteRT model loading bug is
// fixed.
TEST_F(EmbeddingLookupManagerTest,
       LookupPrefillMultimodalMultipleTokensAndEndOfMultiModal) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  ASSERT_OK(UpdateMultiModalEmbeddings());

  Dimensions dimensions({1, 4, 4, 32});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  std::vector<int> tokens = {-1, -3, -1, -3};
  absl::Span<const int> tokens_span(tokens);
  ASSERT_OK(
      embedding_lookup_manager_->LookupPrefill(tokens_span, &output_tensor, 0));

  auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
      output_tensor, ::litert::TensorBuffer::LockMode::kRead);
  auto output_tensor_ptr =
      reinterpret_cast<float*>(output_tensor_lock_and_addr->second);

  float multi_modal_embedding_value = 1.0;
  for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
    int token = tokens[idx0];
    for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
      for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
        size_t offset =
            idx0 * dimensions[2] * dimensions[3] + idx2 * dimensions[3] + idx3;
        if (token == -1) {
          ASSERT_EQ(output_tensor_ptr[offset], multi_modal_embedding_value++);
        } else {
          // If the token is -3, the expected value is the end of multi-modal
          // embedding value.
          float expected_value = (100.0 * idx2 + idx3) * 2;
          ASSERT_NEAR(output_tensor_ptr[offset], expected_value, 1e-5);
        }
      }
    }
  }
}

TEST_F(EmbeddingLookupManagerTest,
       LookupPrefillTextAndMultimodalAndEndOfMultiModalWithOffset) {
  ASSERT_NE(embedding_lookup_manager_, nullptr);

  ASSERT_OK(UpdateMultiModalEmbeddings());

  Dimensions dimensions({1, 4, 4, 32});
  LITERT_ASSERT_OK_AND_ASSIGN(TensorBuffer output_tensor,
                              GetTensorBuffer(dimensions));

  {
    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kWrite);
    auto output_tensor_ptr =
        reinterpret_cast<uint8_t*>(output_tensor_lock_and_addr->second);
    LITERT_ASSERT_OK_AND_ASSIGN(size_t output_tensor_size,
                                output_tensor.Size());
    memset(output_tensor_ptr, 99, output_tensor_size);
  }

  {
    std::vector<int> tokens = {-1, -1, 3};
    absl::Span<const int> tokens_span(tokens);
    const size_t float_offset = 4 * 32;
    const size_t byte_offset = float_offset * sizeof(float);

    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(
        tokens_span, &output_tensor, /*token_offset=*/1));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<uint8_t*>(output_tensor_lock_and_addr->second);

    for (int i = 0; i < byte_offset; ++i) {
      ASSERT_EQ(output_tensor_ptr[i], 99);
    }

    auto output_tensor_float_ptr = reinterpret_cast<float*>(output_tensor_ptr);

    float multi_modal_embedding_value = 1.0;
    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      int token = tokens[idx0];
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          size_t offset = float_offset + idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          if (token == -1) {
            ASSERT_EQ(output_tensor_float_ptr[offset],
                      multi_modal_embedding_value++);
          } else if (token == -3) {
            float expected_value = (100.0 * idx2 + idx3) * 2;
            ASSERT_NEAR(output_tensor_float_ptr[offset], expected_value, 1e-5);
          } else {
            // Since dimension 1 is of size 1, the offset and expected value can
            // ignore it.
            float expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
            ASSERT_NEAR(output_tensor_float_ptr[offset], expected_value, 1e-5);
          }
        }
      }
    }
  }
  {
    std::vector<int> tokens = {-2, -2, -3};
    absl::Span<const int> tokens_span(tokens);
    const size_t float_offset = 4 * 32;
    const size_t byte_offset = float_offset * sizeof(float);

    ASSERT_OK(embedding_lookup_manager_->LookupPrefill(
        tokens_span, &output_tensor, /*token_offset=*/1));

    auto output_tensor_lock_and_addr = ::litert::TensorBufferScopedLock::Create(
        output_tensor, ::litert::TensorBuffer::LockMode::kRead);
    auto output_tensor_ptr =
        reinterpret_cast<uint8_t*>(output_tensor_lock_and_addr->second);

    for (int i = 0; i < byte_offset; ++i) {
      ASSERT_EQ(output_tensor_ptr[i], 99);
    }

    auto output_tensor_float_ptr = reinterpret_cast<float*>(output_tensor_ptr);

    float multi_modal_embedding_value = 257.0;
    for (int idx0 = 0; idx0 < tokens.size(); ++idx0) {
      int token = tokens[idx0];
      for (int idx2 = 0; idx2 < dimensions[2]; ++idx2) {
        for (int idx3 = 0; idx3 < dimensions[3]; ++idx3) {
          size_t offset = float_offset + idx0 * dimensions[2] * dimensions[3] +
                          idx2 * dimensions[3] + idx3;
          if (token == -2) {
            ASSERT_EQ(output_tensor_float_ptr[offset],
                      multi_modal_embedding_value++);
          } else if (token == -3) {
            float expected_value = (100.0 * idx2 + idx3) * 2;
            ASSERT_NEAR(output_tensor_float_ptr[offset], expected_value, 1e-5);
          } else {
            // Since dimension 1 is of size 1, the offset and expected value can
            // ignore it.
            float expected_value = 10000.0 * token + 100.0 * idx2 + idx3;
            ASSERT_NEAR(output_tensor_float_ptr[offset], expected_value, 1e-5);
          }
        }
      }
    }
  }
  ASSERT_OK(embedding_lookup_manager_->CleanupMultiModalEmbeddings());
}

}  // namespace litert::lm
