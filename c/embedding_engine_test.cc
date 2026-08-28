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

#include "c/embedding_engine.h"

#include <cstddef>
#include <string>

#include <gtest/gtest.h>
#include "c/engine.h"

namespace {

constexpr char kTestEmbeddingModelPath[] =
    "runtime/testdata/test_embedding.litertlm";

TEST(EmbeddingEngineCTest, CreateSettingsSuccess) {
  auto* settings = litert_lm_embedding_engine_settings_create(
      kTestEmbeddingModelPath, "cpu", nullptr, nullptr);
  ASSERT_NE(settings, nullptr);
  litert_lm_embedding_engine_settings_set_cache_dir(settings, "/tmp");
  litert_lm_embedding_engine_settings_delete(settings);
}

TEST(EmbeddingEngineCTest, CreateSettingsWithMaxInputLengthAndVisionTokens) {
  auto* settings = litert_lm_embedding_engine_settings_create(
      kTestEmbeddingModelPath, "cpu", nullptr, nullptr);
  ASSERT_NE(settings, nullptr);
  litert_lm_embedding_engine_settings_set_max_input_length(settings, 512);
  litert_lm_embedding_engine_settings_set_vision_tokens_per_image(settings,
                                                                  280);
  // Passing a non-positive value unsets the option.
  litert_lm_embedding_engine_settings_set_max_input_length(settings, 0);
  litert_lm_embedding_engine_settings_set_vision_tokens_per_image(settings, -1);
  litert_lm_embedding_engine_settings_delete(settings);
}

TEST(EmbeddingEngineCTest, CreateSettingsInvalidBackend) {
  auto* settings = litert_lm_embedding_engine_settings_create(
      kTestEmbeddingModelPath, "invalid_backend", nullptr, nullptr);
  EXPECT_EQ(settings, nullptr);
}

TEST(EmbeddingEngineCTest, OptionsNormalize) {
  auto* options = litert_lm_embedding_options_create();
  ASSERT_NE(options, nullptr);
  EXPECT_TRUE(litert_lm_embedding_options_get_normalize(options));

  litert_lm_embedding_options_set_normalize(options, false);
  EXPECT_FALSE(litert_lm_embedding_options_get_normalize(options));

  litert_lm_embedding_options_delete(options);
}

TEST(EmbeddingEngineCTest, OptionsInsertSpecialTokens) {
  auto* options = litert_lm_embedding_options_create();
  ASSERT_NE(options, nullptr);
  EXPECT_TRUE(litert_lm_embedding_options_get_insert_special_tokens(options));

  litert_lm_embedding_options_set_insert_special_tokens(options, false);
  EXPECT_FALSE(litert_lm_embedding_options_get_insert_special_tokens(options));

  litert_lm_embedding_options_delete(options);
}

TEST(EmbeddingEngineCTest, OptionsInputOverflowStrategy) {
  auto* options = litert_lm_embedding_options_create();
  ASSERT_NE(options, nullptr);
  EXPECT_EQ(litert_lm_embedding_options_get_input_overflow_strategy(options),
            kLiteRtLmInputOverflowStrategyError);

  litert_lm_embedding_options_set_input_overflow_strategy(
      options, kLiteRtLmInputOverflowStrategyTruncate);
  EXPECT_EQ(litert_lm_embedding_options_get_input_overflow_strategy(options),
            kLiteRtLmInputOverflowStrategyTruncate);

  litert_lm_embedding_options_set_input_overflow_strategy(
      options, kLiteRtLmInputOverflowStrategyChunkAndAverage);
  EXPECT_EQ(litert_lm_embedding_options_get_input_overflow_strategy(options),
            kLiteRtLmInputOverflowStrategyChunkAndAverage);

  litert_lm_embedding_options_delete(options);
}

TEST(EmbeddingEngineCTest, OptionsOutputSize) {
  auto* options = litert_lm_embedding_options_create();
  ASSERT_NE(options, nullptr);
  EXPECT_EQ(litert_lm_embedding_options_get_output_size(options), -1);

  litert_lm_embedding_options_set_output_size(options, 128);
  EXPECT_EQ(litert_lm_embedding_options_get_output_size(options), 128);

  litert_lm_embedding_options_set_output_size(options, 0);
  EXPECT_EQ(litert_lm_embedding_options_get_output_size(options), -1);

  litert_lm_embedding_options_set_output_size(options, 128);
  EXPECT_EQ(litert_lm_embedding_options_get_output_size(options), 128);

  litert_lm_embedding_options_set_output_size(options, -1);
  EXPECT_EQ(litert_lm_embedding_options_get_output_size(options), -1);

  litert_lm_embedding_options_delete(options);
}

TEST(EmbeddingEngineCTest, OptionsVisionTokensPerImage) {
  auto* options = litert_lm_embedding_options_create();
  ASSERT_NE(options, nullptr);
  EXPECT_EQ(litert_lm_embedding_options_get_vision_tokens_per_image(options),
            0);

  litert_lm_embedding_options_set_vision_tokens_per_image(options, 70);
  EXPECT_EQ(litert_lm_embedding_options_get_vision_tokens_per_image(options),
            70);

  litert_lm_embedding_options_set_vision_tokens_per_image(options, 0);
  EXPECT_EQ(litert_lm_embedding_options_get_vision_tokens_per_image(options),
            0);

  litert_lm_embedding_options_set_vision_tokens_per_image(options, 70);
  EXPECT_EQ(litert_lm_embedding_options_get_vision_tokens_per_image(options),
            70);

  litert_lm_embedding_options_set_vision_tokens_per_image(options, -1);
  EXPECT_EQ(litert_lm_embedding_options_get_vision_tokens_per_image(options),
            0);

  litert_lm_embedding_options_delete(options);
}
TEST(EmbeddingEngineCTest, ComputeEmbeddingSuccess) {
  auto* settings = litert_lm_embedding_engine_settings_create(
      kTestEmbeddingModelPath, "cpu", nullptr, nullptr);
  ASSERT_NE(settings, nullptr);

  auto* engine = litert_lm_embedding_engine_create(settings);
  litert_lm_embedding_engine_settings_delete(settings);
  ASSERT_NE(engine, nullptr);

  std::string prompt = "'s";
  auto* input_data = litert_lm_input_data_create(kLiteRtLmInputDataTypeText,
                                                 prompt.data(), prompt.size());
  ASSERT_NE(input_data, nullptr);

  const LiteRtLmInputData* inputs[] = {input_data};
  auto* options = litert_lm_embedding_options_create();
  litert_lm_embedding_options_set_normalize(options, true);

  auto* response =
      litert_lm_embedding_engine_compute_embedding(engine, inputs, 1, options);
  ASSERT_NE(response, nullptr);

  size_t dim = litert_lm_embedding_response_get_size(response);
  EXPECT_GT(dim, 0);

  const float* values = litert_lm_embedding_response_get_values(response);
  ASSERT_NE(values, nullptr);

  // Check L2 normalization (sum of squares should be ~1.0)
  float sum_sq = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    sum_sq += values[i] * values[i];
  }
  EXPECT_NEAR(sum_sq, 1.0f, 1e-4f);

  litert_lm_embedding_response_delete(response);
  litert_lm_embedding_options_delete(options);
  litert_lm_input_data_delete(input_data);
  litert_lm_embedding_engine_delete(engine);
}

TEST(EmbeddingEngineCTest, ComputeEmbeddingWithMaxInputLengthSuccess) {
  auto* settings = litert_lm_embedding_engine_settings_create(
      kTestEmbeddingModelPath, "cpu", nullptr, nullptr);
  ASSERT_NE(settings, nullptr);
  litert_lm_embedding_engine_settings_set_max_input_length(settings, 512);

  auto* engine = litert_lm_embedding_engine_create(settings);
  litert_lm_embedding_engine_settings_delete(settings);
  ASSERT_NE(engine, nullptr);

  std::string prompt = "'s";
  auto* input_data = litert_lm_input_data_create(kLiteRtLmInputDataTypeText,
                                                 prompt.data(), prompt.size());
  ASSERT_NE(input_data, nullptr);

  const LiteRtLmInputData* inputs[] = {input_data};
  auto* options = litert_lm_embedding_options_create();
  litert_lm_embedding_options_set_normalize(options, true);

  auto* response =
      litert_lm_embedding_engine_compute_embedding(engine, inputs, 1, options);
  ASSERT_NE(response, nullptr);

  size_t dim = litert_lm_embedding_response_get_size(response);
  EXPECT_GT(dim, 0);

  litert_lm_embedding_response_delete(response);
  litert_lm_embedding_options_delete(options);
  litert_lm_input_data_delete(input_data);
  litert_lm_embedding_engine_delete(engine);
}

TEST(EmbeddingEngineCTest, ComputeEmbeddingBatchSuccess) {
  auto* settings = litert_lm_embedding_engine_settings_create(
      kTestEmbeddingModelPath, "cpu", nullptr, nullptr);
  ASSERT_NE(settings, nullptr);

  auto* engine = litert_lm_embedding_engine_create(settings);
  litert_lm_embedding_engine_settings_delete(settings);
  ASSERT_NE(engine, nullptr);

  std::string prompt1 = "'s";
  std::string prompt2 = "'s";
  auto* input1 = litert_lm_input_data_create(kLiteRtLmInputDataTypeText,
                                             prompt1.data(), prompt1.size());
  auto* input2 = litert_lm_input_data_create(kLiteRtLmInputDataTypeText,
                                             prompt2.data(), prompt2.size());

  const LiteRtLmInputData* req1[] = {input1};
  const LiteRtLmInputData* req2[] = {input2};
  const LiteRtLmInputData* const* batch_inputs[] = {req1, req2};
  size_t num_inputs_per_batch[] = {1, 1};

  auto* options = litert_lm_embedding_options_create();
  litert_lm_embedding_options_set_normalize(options, true);

  auto* responses = litert_lm_embedding_engine_compute_embedding_batch(
      engine, batch_inputs, num_inputs_per_batch, 2, options);
  ASSERT_NE(responses, nullptr);

  EXPECT_EQ(litert_lm_embedding_responses_get_size(responses), 2);

  const auto* resp0 = litert_lm_embedding_responses_get_at(responses, 0);
  const auto* resp1 = litert_lm_embedding_responses_get_at(responses, 1);
  ASSERT_NE(resp0, nullptr);
  ASSERT_NE(resp1, nullptr);

  EXPECT_GT(litert_lm_embedding_response_get_size(resp0), 0);
  EXPECT_GT(litert_lm_embedding_response_get_size(resp1), 0);

  litert_lm_embedding_responses_delete(responses);
  litert_lm_embedding_options_delete(options);
  litert_lm_input_data_delete(input1);
  litert_lm_input_data_delete(input2);
  litert_lm_embedding_engine_delete(engine);
}

TEST(EmbeddingEngineCTest, ComputeEmbeddingWithOutputSize) {
  auto* settings = litert_lm_embedding_engine_settings_create(
      kTestEmbeddingModelPath, "cpu", nullptr, nullptr);
  ASSERT_NE(settings, nullptr);

  auto* engine = litert_lm_embedding_engine_create(settings);
  litert_lm_embedding_engine_settings_delete(settings);
  ASSERT_NE(engine, nullptr);

  std::string prompt = "'s";
  auto* input_data = litert_lm_input_data_create(kLiteRtLmInputDataTypeText,
                                                 prompt.data(), prompt.size());
  ASSERT_NE(input_data, nullptr);

  const LiteRtLmInputData* inputs[] = {input_data};
  auto* options = litert_lm_embedding_options_create();
  litert_lm_embedding_options_set_normalize(options, true);
  litert_lm_embedding_options_set_output_size(options, 64);

  auto* response =
      litert_lm_embedding_engine_compute_embedding(engine, inputs, 1, options);
  ASSERT_NE(response, nullptr);

  size_t dim = litert_lm_embedding_response_get_size(response);
  EXPECT_EQ(dim, 64);

  const float* values = litert_lm_embedding_response_get_values(response);
  ASSERT_NE(values, nullptr);

  float sum_sq = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    sum_sq += values[i] * values[i];
  }
  EXPECT_NEAR(sum_sq, 1.0f, 1e-4f);

  litert_lm_embedding_response_delete(response);
  litert_lm_embedding_options_delete(options);
  litert_lm_input_data_delete(input_data);
  litert_lm_embedding_engine_delete(engine);
}

TEST(EmbeddingEngineCTest, ComputeEmbeddingBatchWithOutputSize) {
  auto* settings = litert_lm_embedding_engine_settings_create(
      kTestEmbeddingModelPath, "cpu", nullptr, nullptr);
  ASSERT_NE(settings, nullptr);

  auto* engine = litert_lm_embedding_engine_create(settings);
  litert_lm_embedding_engine_settings_delete(settings);
  ASSERT_NE(engine, nullptr);

  std::string prompt1 = "'s";
  std::string prompt2 = "'s";
  auto* input1 = litert_lm_input_data_create(kLiteRtLmInputDataTypeText,
                                             prompt1.data(), prompt1.size());
  auto* input2 = litert_lm_input_data_create(kLiteRtLmInputDataTypeText,
                                             prompt2.data(), prompt2.size());

  const LiteRtLmInputData* req1[] = {input1};
  const LiteRtLmInputData* req2[] = {input2};
  const LiteRtLmInputData* const* batch_inputs[] = {req1, req2};
  size_t num_inputs_per_batch[] = {1, 1};

  auto* options = litert_lm_embedding_options_create();
  litert_lm_embedding_options_set_normalize(options, true);
  litert_lm_embedding_options_set_output_size(options, 64);

  auto* responses = litert_lm_embedding_engine_compute_embedding_batch(
      engine, batch_inputs, num_inputs_per_batch, 2, options);
  ASSERT_NE(responses, nullptr);

  EXPECT_EQ(litert_lm_embedding_responses_get_size(responses), 2);

  const auto* resp0 = litert_lm_embedding_responses_get_at(responses, 0);
  const auto* resp1 = litert_lm_embedding_responses_get_at(responses, 1);
  ASSERT_NE(resp0, nullptr);
  ASSERT_NE(resp1, nullptr);

  EXPECT_EQ(litert_lm_embedding_response_get_size(resp0), 64);
  EXPECT_EQ(litert_lm_embedding_response_get_size(resp1), 64);

  litert_lm_embedding_responses_delete(responses);
  litert_lm_embedding_options_delete(options);
  litert_lm_input_data_delete(input1);
  litert_lm_input_data_delete(input2);
  litert_lm_embedding_engine_delete(engine);
}

}  // namespace
