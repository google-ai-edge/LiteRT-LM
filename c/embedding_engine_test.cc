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
#include <optional>
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

using SettingsPtr =
    std::unique_ptr<LiteRtLmEmbeddingEngineSettings,
                    decltype(&litert_lm_embedding_engine_settings_delete)>;
using EnginePtr = std::unique_ptr<LiteRtLmEmbeddingEngine,
                                  decltype(&litert_lm_embedding_engine_delete)>;
using OptionsPtr =
    std::unique_ptr<LiteRtLmEmbeddingOptions,
                    decltype(&litert_lm_embedding_options_delete)>;
using ResponsePtr =
    std::unique_ptr<LiteRtLmEmbeddingResponse,
                    decltype(&litert_lm_embedding_response_delete)>;
using ResponsesPtr =
    std::unique_ptr<LiteRtLmEmbeddingResponses,
                    decltype(&litert_lm_embedding_responses_delete)>;
using InputDataPtr =
    std::unique_ptr<LiteRtLmInputData, decltype(&litert_lm_input_data_delete)>;

// Creates input data through the status + out-parameter C API. Returns NULL on
// failure.
LiteRtLmInputData* CreateInputData(LiteRtLmInputDataType type, const void* data,
                                   size_t size) {
  LiteRtLmInputData* input_data = nullptr;
  EXPECT_EQ(litert_lm_input_data_create(type, data, size, &input_data),
            kLiteRtLmStatusOk);
  return input_data;
}

// Creates CPU settings for the test model. Returns NULL on failure.
SettingsPtr CreateSettings() {
  LiteRtLmEmbeddingEngineSettings* settings = nullptr;
  EXPECT_EQ(litert_lm_embedding_engine_settings_create(
                kTestEmbeddingModelPath, "cpu", nullptr, nullptr, &settings),
            kLiteRtLmStatusOk);
  return SettingsPtr(settings, &litert_lm_embedding_engine_settings_delete);
}

// Creates an engine from `settings`. Returns NULL on failure.
EnginePtr CreateEngine(const LiteRtLmEmbeddingEngineSettings* settings) {
  LiteRtLmEmbeddingEngine* engine = nullptr;
  EXPECT_EQ(litert_lm_embedding_engine_create(settings, &engine),
            kLiteRtLmStatusOk);
  return EnginePtr(engine, &litert_lm_embedding_engine_delete);
}

// Creates an engine with default CPU settings. Returns NULL on failure.
EnginePtr CreateDefaultEngine() {
  SettingsPtr settings = CreateSettings();
  if (settings == nullptr) {
    return EnginePtr(nullptr, &litert_lm_embedding_engine_delete);
  }
  return CreateEngine(settings.get());
}

// Creates default embedding options. Returns NULL on failure.
OptionsPtr CreateOptions() {
  LiteRtLmEmbeddingOptions* options = nullptr;
  EXPECT_EQ(litert_lm_embedding_options_create(&options), kLiteRtLmStatusOk);
  return OptionsPtr(options, &litert_lm_embedding_options_delete);
}

// Computes the embedding of `inputs`. Returns NULL on failure.
ResponsePtr ComputeEmbedding(LiteRtLmEmbeddingEngine* engine,
                             const LiteRtLmInputData* const* inputs,
                             size_t num_inputs,
                             const LiteRtLmEmbeddingOptions* options) {
  LiteRtLmEmbeddingResponse* response = nullptr;
  EXPECT_EQ(litert_lm_embedding_engine_compute_embedding(
                engine, inputs, num_inputs, options, &response),
            kLiteRtLmStatusOk);
  return ResponsePtr(response, &litert_lm_embedding_response_delete);
}

// Computes the embeddings of a batch. Returns NULL on failure.
ResponsesPtr ComputeEmbeddingBatch(
    LiteRtLmEmbeddingEngine* engine,
    const LiteRtLmInputData* const* const* inputs_batch,
    const size_t* num_inputs_per_batch, size_t batch_size,
    const LiteRtLmEmbeddingOptions* options) {
  LiteRtLmEmbeddingResponses* responses = nullptr;
  EXPECT_EQ(litert_lm_embedding_engine_compute_embedding_batch(
                engine, inputs_batch, num_inputs_per_batch, batch_size, options,
                &responses),
            kLiteRtLmStatusOk);
  return ResponsesPtr(responses, &litert_lm_embedding_responses_delete);
}

size_t GetResponseSize(const LiteRtLmEmbeddingResponse* response) {
  size_t size = 0;
  EXPECT_EQ(litert_lm_embedding_response_get_size(response, &size),
            kLiteRtLmStatusOk);
  return size;
}

const float* GetResponseValues(const LiteRtLmEmbeddingResponse* response) {
  const float* values = nullptr;
  EXPECT_EQ(litert_lm_embedding_response_get_values(response, &values),
            kLiteRtLmStatusOk);
  return values;
}

size_t GetResponsesSize(const LiteRtLmEmbeddingResponses* responses) {
  size_t size = 0;
  EXPECT_EQ(litert_lm_embedding_responses_get_size(responses, &size),
            kLiteRtLmStatusOk);
  return size;
}

const LiteRtLmEmbeddingResponse* GetResponseAt(
    const LiteRtLmEmbeddingResponses* responses, size_t index) {
  const LiteRtLmEmbeddingResponse* response = nullptr;
  EXPECT_EQ(litert_lm_embedding_responses_get_at(responses, index, &response),
            kLiteRtLmStatusOk);
  return response;
}

bool GetNormalize(const LiteRtLmEmbeddingOptions* options) {
  bool normalize = false;
  EXPECT_EQ(litert_lm_embedding_options_get_normalize(options, &normalize),
            kLiteRtLmStatusOk);
  return normalize;
}

bool GetInsertSpecialTokens(const LiteRtLmEmbeddingOptions* options) {
  bool insert_special_tokens = false;
  EXPECT_EQ(litert_lm_embedding_options_get_insert_special_tokens(
                options, &insert_special_tokens),
            kLiteRtLmStatusOk);
  return insert_special_tokens;
}

LiteRtLmInputOverflowStrategy GetInputOverflowStrategy(
    const LiteRtLmEmbeddingOptions* options) {
  LiteRtLmInputOverflowStrategy strategy =
      kLiteRtLmInputOverflowStrategyChunkAndAverage;
  EXPECT_EQ(litert_lm_embedding_options_get_input_overflow_strategy(options,
                                                                    &strategy),
            kLiteRtLmStatusOk);
  return strategy;
}

// Value that an optional-int getter must leave untouched when it fails.
constexpr int kUntouchedOutValue = 12345;

// Returns the value of an optional-int getter, or std::nullopt if the getter
// reports kLiteRtLmStatusNotFound (value not set). In that case, also expects
// the out param to be untouched and the last error to describe the missing
// value.
std::optional<int> GetOptionalInt(const std::function<int(int*)>& getter,
                                  const std::string& not_set_error) {
  litert_lm_clear_last_error();
  int value = kUntouchedOutValue;
  int status = getter(&value);
  if (status == kLiteRtLmStatusNotFound) {
    EXPECT_THAT(litert_lm_get_last_error_message(),
                ::testing::HasSubstr(not_set_error));
    EXPECT_EQ(value, kUntouchedOutValue);
    return std::nullopt;
  }
  EXPECT_EQ(status, kLiteRtLmStatusOk);
  return value;
}

std::optional<int> GetOutputSize(const LiteRtLmEmbeddingOptions* options) {
  return GetOptionalInt(
      [options](int* out) {
        return litert_lm_embedding_options_get_output_size(options, out);
      },
      "output_size is not set");
}

std::optional<int> GetVisionTokensPerImage(
    const LiteRtLmEmbeddingOptions* options) {
  return GetOptionalInt(
      [options](int* out) {
        return litert_lm_embedding_options_get_vision_tokens_per_image(options,
                                                                       out);
      },
      "vision_tokens_per_image is not set");
}

// Expects `status` to be `expected_code` and a thread-local last error message
// to be recorded.
void ExpectFailureMatchesLastError(int status, int expected_code) {
  EXPECT_EQ(status, expected_code);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
}

TEST(EmbeddingEngineCTest, CreateSettingsSuccess) {
  SettingsPtr settings = CreateSettings();
  ASSERT_NE(settings, nullptr);
  EXPECT_EQ(
      litert_lm_embedding_engine_settings_set_cache_dir(settings.get(), "/tmp"),
      kLiteRtLmStatusOk);
}

TEST(EmbeddingEngineCTest, CreateSettingsWithMinMaxInputLengthAndVisionTokens) {
  SettingsPtr settings = CreateSettings();
  ASSERT_NE(settings, nullptr);
  EXPECT_EQ(litert_lm_embedding_engine_settings_set_min_input_length(
                settings.get(), 128),
            kLiteRtLmStatusOk);
  EXPECT_EQ(litert_lm_embedding_engine_settings_set_max_input_length(
                settings.get(), 512),
            kLiteRtLmStatusOk);
  EXPECT_EQ(litert_lm_embedding_engine_settings_set_vision_tokens_per_image(
                settings.get(), 280),
            kLiteRtLmStatusOk);
  // Passing a negative value unsets min_input_length; non-positive unsets
  // max_input_length and vision_tokens_per_image.
  EXPECT_EQ(litert_lm_embedding_engine_settings_set_min_input_length(
                settings.get(), -1),
            kLiteRtLmStatusOk);
  EXPECT_EQ(litert_lm_embedding_engine_settings_set_max_input_length(
                settings.get(), 0),
            kLiteRtLmStatusOk);
  EXPECT_EQ(litert_lm_embedding_engine_settings_set_vision_tokens_per_image(
                settings.get(), -1),
            kLiteRtLmStatusOk);
}

TEST(EmbeddingEngineCTest, CreateSettingsWithNumThreads) {
  SettingsPtr settings = CreateSettings();
  ASSERT_NE(settings, nullptr);
  EXPECT_EQ(
      litert_lm_embedding_engine_settings_set_num_threads(settings.get(), 4),
      kLiteRtLmStatusOk);
  EnginePtr engine = CreateEngine(settings.get());
  EXPECT_NE(engine, nullptr);
}

TEST(EmbeddingEngineCTest, CreateSettingsInvalidBackend) {
  litert_lm_clear_last_error();
  int dummy = 0;
  auto* settings = reinterpret_cast<LiteRtLmEmbeddingEngineSettings*>(&dummy);
  const int status = litert_lm_embedding_engine_settings_create(
      kTestEmbeddingModelPath, "invalid_backend", nullptr, nullptr, &settings);
  EXPECT_NE(status, kLiteRtLmStatusOk);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_EQ(settings, nullptr);
}

TEST(EmbeddingEngineCTest, CreateSettingsNullModelPath) {
  litert_lm_clear_last_error();
  LiteRtLmEmbeddingEngineSettings* settings = nullptr;
  ExpectFailureMatchesLastError(
      litert_lm_embedding_engine_settings_create(nullptr, "cpu", nullptr,
                                                 nullptr, &settings),
      kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(settings, nullptr);
}

TEST(EmbeddingEngineCTest, OptionsNormalize) {
  OptionsPtr options = CreateOptions();
  ASSERT_NE(options, nullptr);
  EXPECT_TRUE(GetNormalize(options.get()));

  EXPECT_EQ(litert_lm_embedding_options_set_normalize(options.get(), false),
            kLiteRtLmStatusOk);
  EXPECT_FALSE(GetNormalize(options.get()));
}

TEST(EmbeddingEngineCTest, OptionsInsertSpecialTokens) {
  OptionsPtr options = CreateOptions();
  ASSERT_NE(options, nullptr);
  EXPECT_TRUE(GetInsertSpecialTokens(options.get()));

  EXPECT_EQ(litert_lm_embedding_options_set_insert_special_tokens(options.get(),
                                                                  false),
            kLiteRtLmStatusOk);
  EXPECT_FALSE(GetInsertSpecialTokens(options.get()));
}

TEST(EmbeddingEngineCTest, OptionsInputOverflowStrategy) {
  OptionsPtr options = CreateOptions();
  ASSERT_NE(options, nullptr);
  EXPECT_EQ(GetInputOverflowStrategy(options.get()),
            kLiteRtLmInputOverflowStrategyError);

  EXPECT_EQ(litert_lm_embedding_options_set_input_overflow_strategy(
                options.get(), kLiteRtLmInputOverflowStrategyTruncate),
            kLiteRtLmStatusOk);
  EXPECT_EQ(GetInputOverflowStrategy(options.get()),
            kLiteRtLmInputOverflowStrategyTruncate);

  EXPECT_EQ(litert_lm_embedding_options_set_input_overflow_strategy(
                options.get(), kLiteRtLmInputOverflowStrategyChunkAndAverage),
            kLiteRtLmStatusOk);
  EXPECT_EQ(GetInputOverflowStrategy(options.get()),
            kLiteRtLmInputOverflowStrategyChunkAndAverage);
}

TEST(EmbeddingEngineCTest, OptionsOutputSize) {
  OptionsPtr options = CreateOptions();
  ASSERT_NE(options, nullptr);
  EXPECT_EQ(GetOutputSize(options.get()), std::nullopt);

  EXPECT_EQ(litert_lm_embedding_options_set_output_size(options.get(), 128),
            kLiteRtLmStatusOk);
  EXPECT_EQ(GetOutputSize(options.get()), 128);

  EXPECT_EQ(litert_lm_embedding_options_set_output_size(options.get(), 0),
            kLiteRtLmStatusOk);
  EXPECT_EQ(GetOutputSize(options.get()), std::nullopt);

  EXPECT_EQ(litert_lm_embedding_options_set_output_size(options.get(), 128),
            kLiteRtLmStatusOk);
  EXPECT_EQ(GetOutputSize(options.get()), 128);

  EXPECT_EQ(litert_lm_embedding_options_set_output_size(options.get(), -1),
            kLiteRtLmStatusOk);
  EXPECT_EQ(GetOutputSize(options.get()), std::nullopt);
}

TEST(EmbeddingEngineCTest, OptionsVisionTokensPerImage) {
  OptionsPtr options = CreateOptions();
  ASSERT_NE(options, nullptr);
  EXPECT_EQ(GetVisionTokensPerImage(options.get()), std::nullopt);

  EXPECT_EQ(litert_lm_embedding_options_set_vision_tokens_per_image(
                options.get(), 70),
            kLiteRtLmStatusOk);
  EXPECT_EQ(GetVisionTokensPerImage(options.get()), 70);

  EXPECT_EQ(
      litert_lm_embedding_options_set_vision_tokens_per_image(options.get(), 0),
      kLiteRtLmStatusOk);
  EXPECT_EQ(GetVisionTokensPerImage(options.get()), std::nullopt);

  EXPECT_EQ(litert_lm_embedding_options_set_vision_tokens_per_image(
                options.get(), 70),
            kLiteRtLmStatusOk);
  EXPECT_EQ(GetVisionTokensPerImage(options.get()), 70);

  EXPECT_EQ(litert_lm_embedding_options_set_vision_tokens_per_image(
                options.get(), -1),
            kLiteRtLmStatusOk);
  EXPECT_EQ(GetVisionTokensPerImage(options.get()), std::nullopt);
}

TEST(EmbeddingEngineCTest, ComputeEmbeddingSuccess) {
  EnginePtr engine = CreateDefaultEngine();
  ASSERT_NE(engine, nullptr);

  std::string prompt = "'s";
  InputDataPtr input_data(
      CreateInputData(kLiteRtLmInputDataTypeText, prompt.data(), prompt.size()),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);

  const LiteRtLmInputData* inputs[] = {input_data.get()};
  OptionsPtr options = CreateOptions();
  ASSERT_NE(options, nullptr);
  EXPECT_EQ(litert_lm_embedding_options_set_normalize(options.get(), true),
            kLiteRtLmStatusOk);

  ResponsePtr response =
      ComputeEmbedding(engine.get(), inputs, 1, options.get());
  ASSERT_NE(response, nullptr);

  size_t dim = GetResponseSize(response.get());
  EXPECT_GT(dim, 0);

  const float* values = GetResponseValues(response.get());
  ASSERT_NE(values, nullptr);

  // Check L2 normalization (sum of squares should be ~1.0)
  float sum_sq = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    sum_sq += values[i] * values[i];
  }
  EXPECT_NEAR(sum_sq, 1.0f, 1e-4f);
}

TEST(EmbeddingEngineCTest, ComputeEmbeddingWithNullOptionsUsesDefaults) {
  EnginePtr engine = CreateDefaultEngine();
  ASSERT_NE(engine, nullptr);

  std::string prompt = "'s";
  InputDataPtr input_data(
      CreateInputData(kLiteRtLmInputDataTypeText, prompt.data(), prompt.size()),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};

  ResponsePtr response = ComputeEmbedding(engine.get(), inputs, 1, nullptr);
  ASSERT_NE(response, nullptr);
  EXPECT_GT(GetResponseSize(response.get()), 0);
}

TEST(EmbeddingEngineCTest, ComputeEmbeddingWithMaxInputLengthSuccess) {
  SettingsPtr settings = CreateSettings();
  ASSERT_NE(settings, nullptr);
  EXPECT_EQ(litert_lm_embedding_engine_settings_set_max_input_length(
                settings.get(), 128),
            kLiteRtLmStatusOk);

  EnginePtr engine = CreateEngine(settings.get());
  settings.reset();
  ASSERT_NE(engine, nullptr);

  std::string prompt = "'s";
  InputDataPtr input_data(
      CreateInputData(kLiteRtLmInputDataTypeText, prompt.data(), prompt.size()),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);

  const LiteRtLmInputData* inputs[] = {input_data.get()};
  OptionsPtr options = CreateOptions();
  ASSERT_NE(options, nullptr);
  EXPECT_EQ(litert_lm_embedding_options_set_normalize(options.get(), true),
            kLiteRtLmStatusOk);

  ResponsePtr response =
      ComputeEmbedding(engine.get(), inputs, 1, options.get());
  ASSERT_NE(response, nullptr);

  EXPECT_GT(GetResponseSize(response.get()), 0);
}

TEST(EmbeddingEngineCTest,
     ComputeEmbeddingWithMaxInputLengthExceedingCapacityFails) {
  SettingsPtr settings = CreateSettings();
  ASSERT_NE(settings, nullptr);
  EXPECT_EQ(litert_lm_embedding_engine_settings_set_max_input_length(
                settings.get(), 512),
            kLiteRtLmStatusOk);

  litert_lm_clear_last_error();
  int dummy = 0;
  auto* engine = reinterpret_cast<LiteRtLmEmbeddingEngine*>(&dummy);
  const int status = litert_lm_embedding_engine_create(settings.get(), &engine);
  EXPECT_NE(status, kLiteRtLmStatusOk);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_EQ(engine, nullptr);
}

TEST(EmbeddingEngineCTest, ComputeEmbeddingBatchSuccess) {
  EnginePtr engine = CreateDefaultEngine();
  ASSERT_NE(engine, nullptr);

  std::string prompt1 = "'s";
  std::string prompt2 = "'s";
  InputDataPtr input1(CreateInputData(kLiteRtLmInputDataTypeText,
                                      prompt1.data(), prompt1.size()),
                      &litert_lm_input_data_delete);
  InputDataPtr input2(CreateInputData(kLiteRtLmInputDataTypeText,
                                      prompt2.data(), prompt2.size()),
                      &litert_lm_input_data_delete);

  const LiteRtLmInputData* req1[] = {input1.get()};
  const LiteRtLmInputData* req2[] = {input2.get()};
  const LiteRtLmInputData* const* batch_inputs[] = {req1, req2};
  size_t num_inputs_per_batch[] = {1, 1};

  OptionsPtr options = CreateOptions();
  ASSERT_NE(options, nullptr);
  EXPECT_EQ(litert_lm_embedding_options_set_normalize(options.get(), true),
            kLiteRtLmStatusOk);

  ResponsesPtr responses = ComputeEmbeddingBatch(
      engine.get(), batch_inputs, num_inputs_per_batch, 2, options.get());
  ASSERT_NE(responses, nullptr);

  EXPECT_EQ(GetResponsesSize(responses.get()), 2);

  const auto* resp0 = GetResponseAt(responses.get(), 0);
  const auto* resp1 = GetResponseAt(responses.get(), 1);
  ASSERT_NE(resp0, nullptr);
  ASSERT_NE(resp1, nullptr);

  EXPECT_GT(GetResponseSize(resp0), 0);
  EXPECT_GT(GetResponseSize(resp1), 0);
  EXPECT_NE(GetResponseValues(resp0), nullptr);
  EXPECT_NE(GetResponseValues(resp1), nullptr);

  // An index equal to the batch size is out of range.
  litert_lm_clear_last_error();
  const LiteRtLmEmbeddingResponse* out_of_range = resp0;
  ExpectFailureMatchesLastError(
      litert_lm_embedding_responses_get_at(responses.get(), 2, &out_of_range),
      kLiteRtLmStatusOutOfRange);
  EXPECT_EQ(out_of_range, nullptr);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              ::testing::HasSubstr("out of range"));
}

TEST(EmbeddingEngineCTest, ComputeEmbeddingWithOutputSize) {
  EnginePtr engine = CreateDefaultEngine();
  ASSERT_NE(engine, nullptr);

  std::string prompt = "'s";
  InputDataPtr input_data(
      CreateInputData(kLiteRtLmInputDataTypeText, prompt.data(), prompt.size()),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);

  const LiteRtLmInputData* inputs[] = {input_data.get()};
  OptionsPtr options = CreateOptions();
  ASSERT_NE(options, nullptr);
  EXPECT_EQ(litert_lm_embedding_options_set_normalize(options.get(), true),
            kLiteRtLmStatusOk);
  EXPECT_EQ(litert_lm_embedding_options_set_output_size(options.get(), 64),
            kLiteRtLmStatusOk);

  ResponsePtr response =
      ComputeEmbedding(engine.get(), inputs, 1, options.get());
  ASSERT_NE(response, nullptr);

  size_t dim = GetResponseSize(response.get());
  EXPECT_EQ(dim, 64);

  const float* values = GetResponseValues(response.get());
  ASSERT_NE(values, nullptr);

  float sum_sq = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    sum_sq += values[i] * values[i];
  }
  EXPECT_NEAR(sum_sq, 1.0f, 1e-4f);
}

TEST(EmbeddingEngineCTest, ComputeEmbeddingBatchWithOutputSize) {
  EnginePtr engine = CreateDefaultEngine();
  ASSERT_NE(engine, nullptr);

  std::string prompt1 = "'s";
  std::string prompt2 = "'s";
  InputDataPtr input1(CreateInputData(kLiteRtLmInputDataTypeText,
                                      prompt1.data(), prompt1.size()),
                      &litert_lm_input_data_delete);
  InputDataPtr input2(CreateInputData(kLiteRtLmInputDataTypeText,
                                      prompt2.data(), prompt2.size()),
                      &litert_lm_input_data_delete);

  const LiteRtLmInputData* req1[] = {input1.get()};
  const LiteRtLmInputData* req2[] = {input2.get()};
  const LiteRtLmInputData* const* batch_inputs[] = {req1, req2};
  size_t num_inputs_per_batch[] = {1, 1};

  OptionsPtr options = CreateOptions();
  ASSERT_NE(options, nullptr);
  EXPECT_EQ(litert_lm_embedding_options_set_normalize(options.get(), true),
            kLiteRtLmStatusOk);
  EXPECT_EQ(litert_lm_embedding_options_set_output_size(options.get(), 64),
            kLiteRtLmStatusOk);

  ResponsesPtr responses = ComputeEmbeddingBatch(
      engine.get(), batch_inputs, num_inputs_per_batch, 2, options.get());
  ASSERT_NE(responses, nullptr);

  EXPECT_EQ(GetResponsesSize(responses.get()), 2);

  const auto* resp0 = GetResponseAt(responses.get(), 0);
  const auto* resp1 = GetResponseAt(responses.get(), 1);
  ASSERT_NE(resp0, nullptr);
  ASSERT_NE(resp1, nullptr);

  EXPECT_EQ(GetResponseSize(resp0), 64);
  EXPECT_EQ(GetResponseSize(resp1), 64);
}

TEST(EmbeddingEngineCTest, NullArgumentsSetError) {
  litert_lm_clear_last_error();
  EXPECT_EQ(
      litert_lm_embedding_engine_settings_set_max_input_length(nullptr, 16),
      kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              HasSubstr("Invalid embedding engine settings"));

  litert_lm_clear_last_error();
  bool normalize = true;
  EXPECT_EQ(litert_lm_embedding_options_get_normalize(nullptr, &normalize),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              HasSubstr("options must not be NULL"));

  litert_lm_clear_last_error();
  const LiteRtLmEmbeddingResponse* response = nullptr;
  EXPECT_EQ(litert_lm_embedding_responses_get_at(nullptr, 0, &response),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              HasSubstr("responses must not be NULL"));
}

TEST(EmbeddingEngineCTest, SetCacheDirNullSetsError) {
  SettingsPtr settings = CreateSettings();
  ASSERT_NE(settings, nullptr);
  litert_lm_clear_last_error();
  EXPECT_EQ(litert_lm_embedding_engine_settings_set_cache_dir(settings.get(),
                                                              nullptr),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_THAT(litert_lm_get_last_error_message(),
              HasSubstr("cache_dir must not be NULL"));
}

// Runs every setter in `setters` against `handle` (expecting OK) and against
// NULL (expecting kLiteRtLmStatusInvalidArgument and a recorded error message).
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
    EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
  }
}

TEST(EmbeddingEngineCStatusTest, SettingsSettersReturnStatus) {
  SettingsPtr settings = CreateSettings();
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
  OptionsPtr options = CreateOptions();
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
  EXPECT_THAT(litert_lm_get_last_error_message(),
              ::testing::HasSubstr("Unknown LiteRtLmInputOverflowStrategy"));
  EXPECT_EQ(GetInputOverflowStrategy(options.get()),
            kLiteRtLmInputOverflowStrategyTruncate);
}

// Every status-returning producer rejects a NULL out-parameter with
// kLiteRtLmStatusInvalidArgument and records an error message.
TEST(EmbeddingEngineCStatusTest, NullOutParamReturnsInvalidArgument) {
  SettingsPtr settings = CreateSettings();
  ASSERT_NE(settings, nullptr);
  EnginePtr engine = CreateEngine(settings.get());
  ASSERT_NE(engine, nullptr);
  OptionsPtr options = CreateOptions();
  ASSERT_NE(options, nullptr);

  std::string prompt = "'s";
  InputDataPtr input_data(
      CreateInputData(kLiteRtLmInputDataTypeText, prompt.data(), prompt.size()),
      &litert_lm_input_data_delete);
  ASSERT_NE(input_data, nullptr);
  const LiteRtLmInputData* inputs[] = {input_data.get()};
  const LiteRtLmInputData* const* batch_inputs[] = {inputs};
  size_t num_inputs_per_batch[] = {1};

  ResponsePtr response =
      ComputeEmbedding(engine.get(), inputs, 1, options.get());
  ASSERT_NE(response, nullptr);
  ResponsesPtr responses = ComputeEmbeddingBatch(
      engine.get(), batch_inputs, num_inputs_per_batch, 1, options.get());
  ASSERT_NE(responses, nullptr);

  const std::vector<std::pair<std::string, std::function<int()>>> calls = {
      {"settings_create",
       [] {
         return litert_lm_embedding_engine_settings_create(
             kTestEmbeddingModelPath, "cpu", nullptr, nullptr, nullptr);
       }},
      {"options_create",
       [] { return litert_lm_embedding_options_create(nullptr); }},
      {"options_get_normalize",
       [&] {
         return litert_lm_embedding_options_get_normalize(options.get(),
                                                          nullptr);
       }},
      {"options_get_insert_special_tokens",
       [&] {
         return litert_lm_embedding_options_get_insert_special_tokens(
             options.get(), nullptr);
       }},
      {"options_get_input_overflow_strategy",
       [&] {
         return litert_lm_embedding_options_get_input_overflow_strategy(
             options.get(), nullptr);
       }},
      {"options_get_output_size",
       [&] {
         return litert_lm_embedding_options_get_output_size(options.get(),
                                                            nullptr);
       }},
      {"options_get_vision_tokens_per_image",
       [&] {
         return litert_lm_embedding_options_get_vision_tokens_per_image(
             options.get(), nullptr);
       }},
      {"response_get_size",
       [&] {
         return litert_lm_embedding_response_get_size(response.get(), nullptr);
       }},
      {"response_get_values",
       [&] {
         return litert_lm_embedding_response_get_values(response.get(),
                                                        nullptr);
       }},
      {"responses_get_size",
       [&] {
         return litert_lm_embedding_responses_get_size(responses.get(),
                                                       nullptr);
       }},
      {"responses_get_at",
       [&] {
         return litert_lm_embedding_responses_get_at(responses.get(), 0,
                                                     nullptr);
       }},
      {"engine_create",
       [&] {
         return litert_lm_embedding_engine_create(settings.get(), nullptr);
       }},
      {"compute_embedding",
       [&] {
         return litert_lm_embedding_engine_compute_embedding(
             engine.get(), inputs, 1, options.get(), nullptr);
       }},
      {"compute_embedding_batch",
       [&] {
         return litert_lm_embedding_engine_compute_embedding_batch(
             engine.get(), batch_inputs, num_inputs_per_batch, 1, options.get(),
             nullptr);
       }},
  };
  for (const auto& [name, call] : calls) {
    SCOPED_TRACE(name);
    litert_lm_clear_last_error();
    ExpectFailureMatchesLastError(call(), kLiteRtLmStatusInvalidArgument);
    EXPECT_THAT(litert_lm_get_last_error_message(),
                ::testing::HasSubstr("must not be NULL"));
  }
}

// A NULL handle is kLiteRtLmStatusInvalidArgument. Pointer out-parameters are
// reset to NULL; scalar out-parameters are left untouched.
TEST(EmbeddingEngineCStatusTest, NullHandleReturnsInvalidArgument) {
  int dummy = 0;
  void* const kSentinel = &dummy;

  // Pointer out-parameters.
  {
    litert_lm_clear_last_error();
    auto* engine = static_cast<LiteRtLmEmbeddingEngine*>(kSentinel);
    ExpectFailureMatchesLastError(
        litert_lm_embedding_engine_create(nullptr, &engine),
        kLiteRtLmStatusInvalidArgument);
    EXPECT_EQ(engine, nullptr);
  }
  {
    litert_lm_clear_last_error();
    auto* response = static_cast<LiteRtLmEmbeddingResponse*>(kSentinel);
    ExpectFailureMatchesLastError(litert_lm_embedding_engine_compute_embedding(
                                      nullptr, nullptr, 0, nullptr, &response),
                                  kLiteRtLmStatusInvalidArgument);
    EXPECT_EQ(response, nullptr);
  }
  {
    litert_lm_clear_last_error();
    auto* responses = static_cast<LiteRtLmEmbeddingResponses*>(kSentinel);
    ExpectFailureMatchesLastError(
        litert_lm_embedding_engine_compute_embedding_batch(
            nullptr, nullptr, nullptr, 0, nullptr, &responses),
        kLiteRtLmStatusInvalidArgument);
    EXPECT_EQ(responses, nullptr);
  }
  {
    litert_lm_clear_last_error();
    const auto* values = static_cast<const float*>(kSentinel);
    ExpectFailureMatchesLastError(
        litert_lm_embedding_response_get_values(nullptr, &values),
        kLiteRtLmStatusInvalidArgument);
    EXPECT_EQ(values, nullptr);
  }
  {
    litert_lm_clear_last_error();
    const auto* response =
        static_cast<const LiteRtLmEmbeddingResponse*>(kSentinel);
    ExpectFailureMatchesLastError(
        litert_lm_embedding_responses_get_at(nullptr, 0, &response),
        kLiteRtLmStatusInvalidArgument);
    EXPECT_EQ(response, nullptr);
  }

  // Scalar out-parameters.
  {
    litert_lm_clear_last_error();
    bool normalize = true;
    ExpectFailureMatchesLastError(
        litert_lm_embedding_options_get_normalize(nullptr, &normalize),
        kLiteRtLmStatusInvalidArgument);
    EXPECT_TRUE(normalize);
  }
  {
    litert_lm_clear_last_error();
    bool insert_special_tokens = true;
    ExpectFailureMatchesLastError(
        litert_lm_embedding_options_get_insert_special_tokens(
            nullptr, &insert_special_tokens),
        kLiteRtLmStatusInvalidArgument);
    EXPECT_TRUE(insert_special_tokens);
  }
  {
    litert_lm_clear_last_error();
    LiteRtLmInputOverflowStrategy strategy =
        kLiteRtLmInputOverflowStrategyTruncate;
    ExpectFailureMatchesLastError(
        litert_lm_embedding_options_get_input_overflow_strategy(nullptr,
                                                                &strategy),
        kLiteRtLmStatusInvalidArgument);
    EXPECT_EQ(strategy, kLiteRtLmInputOverflowStrategyTruncate);
  }
  {
    litert_lm_clear_last_error();
    int output_size = 42;
    ExpectFailureMatchesLastError(
        litert_lm_embedding_options_get_output_size(nullptr, &output_size),
        kLiteRtLmStatusInvalidArgument);
    EXPECT_EQ(output_size, 42);
  }
  {
    litert_lm_clear_last_error();
    int vision_tokens_per_image = 42;
    ExpectFailureMatchesLastError(
        litert_lm_embedding_options_get_vision_tokens_per_image(
            nullptr, &vision_tokens_per_image),
        kLiteRtLmStatusInvalidArgument);
    EXPECT_EQ(vision_tokens_per_image, 42);
  }
  {
    litert_lm_clear_last_error();
    size_t size = 42;
    ExpectFailureMatchesLastError(
        litert_lm_embedding_response_get_size(nullptr, &size),
        kLiteRtLmStatusInvalidArgument);
    EXPECT_EQ(size, 42);
  }
  {
    litert_lm_clear_last_error();
    size_t size = 42;
    ExpectFailureMatchesLastError(
        litert_lm_embedding_responses_get_size(nullptr, &size),
        kLiteRtLmStatusInvalidArgument);
    EXPECT_EQ(size, 42);
  }

  SettingsPtr settings = CreateSettings();
  ASSERT_NE(settings, nullptr);
  EnginePtr valid_engine = CreateEngine(settings.get());
  ASSERT_NE(valid_engine, nullptr);
  {
    litert_lm_clear_last_error();
    auto* response = static_cast<LiteRtLmEmbeddingResponse*>(kSentinel);
    ExpectFailureMatchesLastError(litert_lm_embedding_engine_compute_embedding(
                                      valid_engine.get(), /*inputs=*/nullptr,
                                      /*num_inputs=*/1, nullptr, &response),
                                  kLiteRtLmStatusInvalidArgument);
    EXPECT_EQ(response, nullptr);
  }
  {
    litert_lm_clear_last_error();
    auto* responses = static_cast<LiteRtLmEmbeddingResponses*>(kSentinel);
    ExpectFailureMatchesLastError(
        litert_lm_embedding_engine_compute_embedding_batch(
            valid_engine.get(), /*inputs_batch=*/nullptr,
            /*num_inputs_per_batch=*/nullptr, /*batch_size=*/1, nullptr,
            &responses),
        kLiteRtLmStatusInvalidArgument);
    EXPECT_EQ(responses, nullptr);
  }
}

}  // namespace
