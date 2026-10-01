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
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "c/engine.h"
#include "c/error_reporter.h"

namespace {

using ::testing::HasSubstr;

constexpr char kTestEmbeddingModelPath[] =
    "runtime/testdata/test_embedding.litertlm";

// Creates input data through the status + out-parameter C API. Returns NULL on
// failure.
LiteRtLmInputData* CreateInputData(LiteRtLmInputDataType type, const void* data,
                                   size_t size) {
  LiteRtLmInputData* input_data = nullptr;
  EXPECT_EQ(litert_lm_input_data_create(type, data, size, &input_data),
            kLiteRtLmStatusOk);
  return input_data;
}

TEST(EmbeddingEngineCTest, CreateSettingsSuccess) {
  auto* settings = litert_lm_embedding_engine_settings_create(
      kTestEmbeddingModelPath, "cpu", nullptr, nullptr);
  ASSERT_NE(settings, nullptr);
  litert_lm_embedding_engine_settings_set_cache_dir(settings, "/tmp");
  litert_lm_embedding_engine_settings_delete(settings);
}

TEST(EmbeddingEngineCTest, CreateSettingsWithMinMaxInputLengthAndVisionTokens) {
  auto* settings = litert_lm_embedding_engine_settings_create(
      kTestEmbeddingModelPath, "cpu", nullptr, nullptr);
  ASSERT_NE(settings, nullptr);
  litert_lm_embedding_engine_settings_set_min_input_length(settings, 128);
  litert_lm_embedding_engine_settings_set_max_input_length(settings, 512);
  litert_lm_embedding_engine_settings_set_vision_tokens_per_image(settings,
                                                                  280);
  // Passing a negative value unsets min_input_length; non-positive unsets
  // max_input_length and vision_tokens_per_image.
  litert_lm_embedding_engine_settings_set_min_input_length(settings, -1);
  litert_lm_embedding_engine_settings_set_max_input_length(settings, 0);
  litert_lm_embedding_engine_settings_set_vision_tokens_per_image(settings, -1);
  litert_lm_embedding_engine_settings_delete(settings);
}

TEST(EmbeddingEngineCTest, CreateSettingsWithNumThreads) {
  auto* settings = litert_lm_embedding_engine_settings_create(
      kTestEmbeddingModelPath, "cpu", nullptr, nullptr);
  ASSERT_NE(settings, nullptr);
  litert_lm_embedding_engine_settings_set_num_threads(settings, 4);
  auto* engine = litert_lm_embedding_engine_create(settings);
  litert_lm_embedding_engine_settings_delete(settings);
  ASSERT_NE(engine, nullptr);
  litert_lm_embedding_engine_delete(engine);
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
  auto* input_data =
      CreateInputData(kLiteRtLmInputDataTypeText, prompt.data(), prompt.size());
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
  litert_lm_embedding_engine_settings_set_max_input_length(settings, 128);

  auto* engine = litert_lm_embedding_engine_create(settings);
  litert_lm_embedding_engine_settings_delete(settings);
  ASSERT_NE(engine, nullptr);

  std::string prompt = "'s";
  auto* input_data =
      CreateInputData(kLiteRtLmInputDataTypeText, prompt.data(), prompt.size());
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

TEST(EmbeddingEngineCTest,
     ComputeEmbeddingWithMaxInputLengthExceedingCapacityFails) {
  auto* settings = litert_lm_embedding_engine_settings_create(
      kTestEmbeddingModelPath, "cpu", nullptr, nullptr);
  ASSERT_NE(settings, nullptr);
  litert_lm_embedding_engine_settings_set_max_input_length(settings, 512);

  auto* engine = litert_lm_embedding_engine_create(settings);
  litert_lm_embedding_engine_settings_delete(settings);
  EXPECT_EQ(engine, nullptr);
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
  auto* input1 = CreateInputData(kLiteRtLmInputDataTypeText, prompt1.data(),
                                 prompt1.size());
  auto* input2 = CreateInputData(kLiteRtLmInputDataTypeText, prompt2.data(),
                                 prompt2.size());

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
  auto* input_data =
      CreateInputData(kLiteRtLmInputDataTypeText, prompt.data(), prompt.size());
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
  auto* input1 = CreateInputData(kLiteRtLmInputDataTypeText, prompt1.data(),
                                 prompt1.size());
  auto* input2 = CreateInputData(kLiteRtLmInputDataTypeText, prompt2.data(),
                                 prompt2.size());

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

TEST(EmbeddingEngineCTest, NullArgumentsSetError) {
  litert_lm_clear_last_error();
  EXPECT_EQ(
      litert_lm_embedding_engine_settings_set_max_input_length(nullptr, 16),
      kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(litert_lm_get_last_error_code(), kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              HasSubstr("Invalid embedding engine settings"));

  litert_lm_clear_last_error();
  EXPECT_FALSE(litert_lm_embedding_options_get_normalize(nullptr));
  EXPECT_EQ(litert_lm_get_last_error_code(), kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              HasSubstr("options must not be NULL"));

  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_embedding_responses_get_at(nullptr, 0), nullptr);
  EXPECT_EQ(litert_lm_get_last_error_code(), kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              HasSubstr("responses must not be NULL"));
}

TEST(EmbeddingEngineCTest, SetCacheDirNullSetsError) {
  auto* settings = litert_lm_embedding_engine_settings_create(
      kTestEmbeddingModelPath, "cpu", nullptr, nullptr);
  ASSERT_NE(settings, nullptr);
  litert_lm_clear_last_error();
  EXPECT_EQ(
      litert_lm_embedding_engine_settings_set_cache_dir(settings, nullptr),
      kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(litert_lm_get_last_error_code(), kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              HasSubstr("cache_dir must not be NULL"));
  litert_lm_embedding_engine_settings_delete(settings);
}

// Runs every setter in `setters` against `handle` (expecting OK) and against
// NULL (expecting kLiteRtLmStatusInvalidArgument, mirrored in the last error).
template <typename T>
void ExpectSettersReturnStatus(
    T* handle,
    const std::vector<std::pair<std::string, std::function<int(T*)>>>&
        setters) {
  for (const auto& [name, setter] : setters) {
    SCOPED_TRACE(name);
    EXPECT_EQ(setter(handle), kLiteRtLmStatusOk);
    litert_lm_clear_last_error();
    EXPECT_EQ(setter(nullptr), kLiteRtLmStatusInvalidArgument);
    EXPECT_EQ(litert_lm_get_last_error_code(), kLiteRtLmStatusInvalidArgument);
  }
}

TEST(EmbeddingEngineCStatusTest, SettingsSettersReturnStatus) {
  std::unique_ptr<LiteRtLmEmbeddingEngineSettings,
                  decltype(&litert_lm_embedding_engine_settings_delete)>
      settings(litert_lm_embedding_engine_settings_create(
                   kTestEmbeddingModelPath, "cpu", nullptr, nullptr),
               &litert_lm_embedding_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  using S = LiteRtLmEmbeddingEngineSettings;
  ExpectSettersReturnStatus<S>(
      settings.get(),
      {
          {"set_num_threads",
           [](S* s) {
             return litert_lm_embedding_engine_settings_set_num_threads(s, 2);
           }},
          // Non-positive values are ignored but still succeed.
          {"set_num_threads_ignored",
           [](S* s) {
             return litert_lm_embedding_engine_settings_set_num_threads(s, 0);
           }},
          {"set_audio_num_threads",
           [](S* s) {
             return litert_lm_embedding_engine_settings_set_audio_num_threads(
                 s, 2);
           }},
          {"set_cache_dir",
           [](S* s) {
             return litert_lm_embedding_engine_settings_set_cache_dir(
                 s, "test_cache_dir");
           }},
          {"set_litert_dispatch_lib_dir",
           [](S* s) {
             return litert_lm_embedding_engine_settings_set_litert_dispatch_lib_dir(  // NOLINT
                 s, "test_lib_dir");
           }},
          {"set_vision_litert_dispatch_lib_dir",
           [](S* s) {
             return litert_lm_embedding_engine_settings_set_vision_litert_dispatch_lib_dir(  // NOLINT
                 s, "test_lib_dir");
           }},
          {"set_audio_litert_dispatch_lib_dir",
           [](S* s) {
             return litert_lm_embedding_engine_settings_set_audio_litert_dispatch_lib_dir(  // NOLINT
                 s, "test_lib_dir");
           }},
          {"set_max_input_length",
           [](S* s) {
             return litert_lm_embedding_engine_settings_set_max_input_length(
                 s, 512);
           }},
          {"set_min_input_length",
           [](S* s) {
             return litert_lm_embedding_engine_settings_set_min_input_length(s,
                                                                             1);
           }},
          {"set_vision_tokens_per_image",
           [](S* s) {
             return litert_lm_embedding_engine_settings_set_vision_tokens_per_image(  // NOLINT
                 s, 280);
           }},
          {"set_activation_data_type",
           [](S* s) {
             return litert_lm_embedding_engine_settings_set_activation_data_type(  // NOLINT
                 s, kLiteRtLmActivationDataTypeFloat32);
           }},
      });

  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_embedding_engine_settings_set_litert_dispatch_lib_dir(
                settings.get(), nullptr),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              ::testing::HasSubstr("lib_dir must not be NULL"));
}

TEST(EmbeddingEngineCStatusTest, OptionsSettersReturnStatus) {
  std::unique_ptr<LiteRtLmEmbeddingOptions,
                  decltype(&litert_lm_embedding_options_delete)>
      options(litert_lm_embedding_options_create(),
              &litert_lm_embedding_options_delete);
  ASSERT_NE(options, nullptr);
  using O = LiteRtLmEmbeddingOptions;
  ExpectSettersReturnStatus<O>(
      options.get(),
      {
          {"set_normalize",
           [](O* o) {
             return litert_lm_embedding_options_set_normalize(o, false);
           }},
          {"set_insert_special_tokens",
           [](O* o) {
             return litert_lm_embedding_options_set_insert_special_tokens(
                 o, false);
           }},
          {"set_input_overflow_strategy",
           [](O* o) {
             return litert_lm_embedding_options_set_input_overflow_strategy(
                 o, kLiteRtLmInputOverflowStrategyTruncate);
           }},
          {"set_output_size",
           [](O* o) {
             return litert_lm_embedding_options_set_output_size(o, 8);
           }},
          {"set_vision_tokens_per_image",
           [](O* o) {
             return litert_lm_embedding_options_set_vision_tokens_per_image(o,
                                                                            16);
           }},
      });

  litert_lm_clear_last_error();
  // 3 is within the enum's value range but is not a declared enumerator.
  EXPECT_EQ(litert_lm_embedding_options_set_input_overflow_strategy(
                options.get(), static_cast<LiteRtLmInputOverflowStrategy>(3)),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(litert_lm_get_last_error_code(), kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              ::testing::HasSubstr("Unknown LiteRtLmInputOverflowStrategy"));
  EXPECT_EQ(
      litert_lm_embedding_options_get_input_overflow_strategy(options.get()),
      kLiteRtLmInputOverflowStrategyTruncate);
}

}  // namespace
