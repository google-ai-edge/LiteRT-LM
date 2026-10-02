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

#include "c/model_info.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

#if defined(__APPLE__)
#include "engine.h"          // NOLINT
#include "error_reporter.h"  // NOLINT
#else
#include "c/engine.h"
#include "c/error_reporter.h"
#endif

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using ::testing::HasSubstr;

std::string GetRunfilePath(const std::string& relative_path) {
  std::string srcdir = ::testing::SrcDir();
  // On Windows, SrcDir() may return paths with backslashes. The LiteRT LM C API
  // expects forward slashes.
  std::replace(srcdir.begin(), srcdir.end(), '\\', '/');
  return srcdir + "/" + relative_path;
}

std::string GetTestdataPath(const std::string& file_name) {
  return GetRunfilePath(
      "litert_lm/runtime/testdata/" +
      file_name);
}

struct LoadedFileDeleter {
  void operator()(LiteRtLmLoadedFile* file) const {
    litert_lm_loaded_file_delete(file);
  }
};
using LoadedFilePtr = std::unique_ptr<LiteRtLmLoadedFile, LoadedFileDeleter>;

// Loads `file_name` from the test data directory, failing the test on error.
LoadedFilePtr LoadTestFile(const std::string& file_name) {
  LiteRtLmLoadedFile* file = nullptr;
  int status =
      litert_lm_loaded_file_create(GetTestdataPath(file_name).c_str(), &file);
  EXPECT_EQ(status, kLiteRtLmStatusOk)
      << (litert_lm_get_last_error_message() != nullptr
              ? litert_lm_get_last_error_message()
              : "");
  EXPECT_NE(file, nullptr);
  return LoadedFilePtr(file);
}

// Expects that `status` is `expected_code` and that the last error message
// contains `message`.
void ExpectError(int status, LiteRtLmStatusCode expected_code,
                 const std::string& message) {
  EXPECT_EQ(status, expected_code);
  ASSERT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_THAT(litert_lm_get_last_error_message(), HasSubstr(message));
}

TEST(ModelInfoCTest, InspectLoadedFile) {
  LoadedFilePtr file = LoadTestFile("test_lm.litertlm");
  ASSERT_NE(file, nullptr);

  // Verify core capabilities and fallback defaults.
  LiteRtLmModelType model_type = kLiteRtLmModelTypeUnknown;
  EXPECT_EQ(litert_lm_loaded_file_model_type(file.get(), &model_type),
            kLiteRtLmStatusOk);
  EXPECT_EQ(model_type, kLiteRtLmModelTypeLlm);

  bool value = true;
  EXPECT_EQ(litert_lm_loaded_file_has_speculative_decoding_support(file.get(),
                                                                   &value),
            kLiteRtLmStatusOk);
  EXPECT_FALSE(value);
  value = true;
  EXPECT_EQ(litert_lm_loaded_file_supports_thinking(file.get(), &value),
            kLiteRtLmStatusOk);
  EXPECT_FALSE(value);
  value = true;
  EXPECT_EQ(litert_lm_loaded_file_supports_function_calling(file.get(), &value),
            kLiteRtLmStatusOk);
  EXPECT_FALSE(value);

  const char* version = "unset";
  EXPECT_EQ(litert_lm_loaded_file_min_runtime_version(file.get(), &version),
            kLiteRtLmStatusOk);
  EXPECT_EQ(version, nullptr);

  // Verify modality-specific backends for text (default to CPU, GPU).
  LiteRtLmBackendType text_backends[3];
  int32_t text_backend_count = -1;
  EXPECT_EQ(litert_lm_loaded_file_modality_supported_backends(
                file.get(), kLiteRtLmModalityText, text_backends, 3,
                &text_backend_count),
            kLiteRtLmStatusOk);
  EXPECT_EQ(text_backend_count, 2);
  EXPECT_EQ(text_backends[0], kLiteRtLmBackendTypeCpu);
  EXPECT_EQ(text_backends[1], kLiteRtLmBackendTypeGpu);

  // Count-only query.
  int32_t count_only = -1;
  EXPECT_EQ(litert_lm_loaded_file_modality_supported_backends(
                file.get(), kLiteRtLmModalityText, nullptr, 0, &count_only),
            kLiteRtLmStatusOk);
  EXPECT_EQ(count_only, 2);

  // A buffer smaller than the count receives a prefix; the full count is
  // still reported.
  LiteRtLmBackendType one_backend[1];
  int32_t partial_count = -1;
  EXPECT_EQ(
      litert_lm_loaded_file_modality_supported_backends(
          file.get(), kLiteRtLmModalityText, one_backend, 1, &partial_count),
      kLiteRtLmStatusOk);
  EXPECT_EQ(partial_count, 2);
  EXPECT_EQ(one_backend[0], kLiteRtLmBackendTypeCpu);

  // Verify modality-specific backends for vision (not present -> OK, 0).
  int32_t vision_backend_count = -1;
  EXPECT_EQ(litert_lm_loaded_file_modality_supported_backends(
                file.get(), kLiteRtLmModalityVision, nullptr, 0,
                &vision_backend_count),
            kLiteRtLmStatusOk);
  EXPECT_EQ(vision_backend_count, 0);

  LiteRtLmNpuBrand brand = kLiteRtLmNpuBrandQualcomm;
  EXPECT_EQ(litert_lm_loaded_file_modality_npu_brand(
                file.get(), kLiteRtLmModalityText, &brand),
            kLiteRtLmStatusOk);
  EXPECT_EQ(brand, kLiteRtLmNpuBrandUnknown);

  const char* soc_name = "unset";
  EXPECT_EQ(litert_lm_loaded_file_modality_soc_name(
                file.get(), kLiteRtLmModalityText, &soc_name),
            kLiteRtLmStatusOk);
  EXPECT_EQ(soc_name, nullptr);

  // Verify modalities.
  bool text = false;
  bool vision = true;
  bool audio = true;
  bool video = true;
  EXPECT_EQ(litert_lm_loaded_file_supports_input_modality(
                file.get(), kLiteRtLmModalityText, &text),
            kLiteRtLmStatusOk);
  EXPECT_EQ(litert_lm_loaded_file_supports_input_modality(
                file.get(), kLiteRtLmModalityVision, &vision),
            kLiteRtLmStatusOk);
  EXPECT_EQ(litert_lm_loaded_file_supports_input_modality(
                file.get(), kLiteRtLmModalityAudio, &audio),
            kLiteRtLmStatusOk);
  EXPECT_EQ(litert_lm_loaded_file_supports_input_modality(
                file.get(), kLiteRtLmModalityVideo, &video),
            kLiteRtLmStatusOk);
  EXPECT_TRUE(text);
  EXPECT_FALSE(vision);
  EXPECT_FALSE(audio);
  EXPECT_FALSE(video);

  // Verify default sampler parameters (from model config).
  LiteRtLmSamplerType sampler_type = kLiteRtLmSamplerTypeUnspecified;
  float temperature = -1.0f;
  int32_t top_k = -1;
  float top_p = -1.0f;
  EXPECT_EQ(litert_lm_loaded_file_sampler_type(file.get(), &sampler_type),
            kLiteRtLmStatusOk);
  EXPECT_EQ(litert_lm_loaded_file_sampler_temperature(file.get(), &temperature),
            kLiteRtLmStatusOk);
  EXPECT_EQ(litert_lm_loaded_file_sampler_top_k(file.get(), &top_k),
            kLiteRtLmStatusOk);
  EXPECT_EQ(litert_lm_loaded_file_sampler_top_p(file.get(), &top_p),
            kLiteRtLmStatusOk);
  EXPECT_EQ(sampler_type, kLiteRtLmSamplerTypeTopP);
  EXPECT_FLOAT_EQ(temperature, 0.0f);
  EXPECT_EQ(top_k, 1);
  EXPECT_FLOAT_EQ(top_p, 0.7f);
}

TEST(ModelInfoCTest, LlmModelReportsAbsentOptionalValuesAsNotFound) {
  LoadedFilePtr file = LoadTestFile("test_lm.litertlm");
  ASSERT_NE(file, nullptr);

  // Scalar out parameters are not written when the value is absent.
  int32_t int_value = 42;
  litert_lm_clear_last_error();
  ExpectError(litert_lm_loaded_file_embedding_dimension(file.get(), &int_value),
              kLiteRtLmStatusNotFound, "embedding dimension");
  EXPECT_EQ(int_value, 42);

  ExpectError(litert_lm_loaded_file_embedding_signature_selection(
                  file.get(), nullptr, 0, &int_value),
              kLiteRtLmStatusNotFound, "embedding signature lengths");
  EXPECT_EQ(int_value, 42);

  ExpectError(
      litert_lm_loaded_file_max_vision_token_budget(file.get(), &int_value),
      kLiteRtLmStatusNotFound, "max vision token budget");
  EXPECT_EQ(int_value, 42);

  ExpectError(litert_lm_loaded_file_vision_signature_selection(
                  file.get(), nullptr, 0, &int_value),
              kLiteRtLmStatusNotFound, "vision signature selection");
  EXPECT_EQ(int_value, 42);
}

TEST(ModelInfoCTest, GetMaxContextTokens) {
  LoadedFilePtr file = LoadTestFile("test_lm.litertlm");
  ASSERT_NE(file, nullptr);

  uint32_t max_context_tokens = 0;
  EXPECT_EQ(
      litert_lm_loaded_file_max_context_tokens(file.get(), &max_context_tokens),
      kLiteRtLmStatusOk);
  EXPECT_EQ(max_context_tokens, 128);
  bool is_dynamic = true;
  EXPECT_EQ(litert_lm_loaded_file_is_dynamic_context(file.get(), &is_dynamic),
            kLiteRtLmStatusOk);
  EXPECT_FALSE(is_dynamic);
}

TEST(ModelInfoCTest, CreateInvalidPathFails) {
  LiteRtLmLoadedFile* file = reinterpret_cast<LiteRtLmLoadedFile*>(0x1);
  litert_lm_clear_last_error();
  int status =
      litert_lm_loaded_file_create("/invalid/path/that/does/not/exist", &file);
  EXPECT_NE(status, kLiteRtLmStatusOk);
  EXPECT_NE(litert_lm_get_last_error_message(), nullptr);
  EXPECT_EQ(file, nullptr);
}

TEST(ModelInfoCTest, CreateNullArgumentsFail) {
  litert_lm_clear_last_error();
  ExpectError(litert_lm_loaded_file_create(
                  GetTestdataPath("test_lm.litertlm").c_str(), nullptr),
              kLiteRtLmStatusInvalidArgument,
              "out_loaded_file must not be NULL");

  LiteRtLmLoadedFile* file = reinterpret_cast<LiteRtLmLoadedFile*>(0x1);
  ExpectError(litert_lm_loaded_file_create(nullptr, &file),
              kLiteRtLmStatusInvalidArgument, "litertlm_path must not be NULL");
  EXPECT_EQ(file, nullptr);
}

TEST(ModelInfoCTest, InspectMultimodalCapabilities) {
  LoadedFilePtr file = LoadTestFile("dummy_vision_with_adapter.litertlm");
  ASSERT_NE(file, nullptr);

  bool vision = false;
  EXPECT_EQ(litert_lm_loaded_file_supports_input_modality(
                file.get(), kLiteRtLmModalityVision, &vision),
            kLiteRtLmStatusOk);
  EXPECT_TRUE(vision);

  // 1. Get count only.
  int32_t count = -1;
  EXPECT_EQ(litert_lm_loaded_file_vision_signature_selection(
                file.get(), nullptr, 0, &count),
            kLiteRtLmStatusOk);
  EXPECT_EQ(count, 1);

  // 2. Fetch values.
  int32_t lengths[1] = {0};
  count = -1;
  EXPECT_EQ(litert_lm_loaded_file_vision_signature_selection(
                file.get(), lengths, 1, &count),
            kLiteRtLmStatusOk);
  EXPECT_EQ(count, 1);
  EXPECT_EQ(lengths[0], 5);

  // 3. A negative capacity is rejected and nothing is written.
  count = 42;
  litert_lm_clear_last_error();
  ExpectError(litert_lm_loaded_file_vision_signature_selection(
                  file.get(), lengths, -1, &count),
              kLiteRtLmStatusInvalidArgument, "max_size must not be negative");
  EXPECT_EQ(count, 42);
}

TEST(ModelInfoCTest, NullLoadedFileReturnsInvalidArgument) {
  bool bool_value = true;
  int32_t int_value = 42;
  uint32_t uint_value = 42;
  float float_value = 4.2f;
  LiteRtLmSamplerType sampler_type = kLiteRtLmSamplerTypeGreedy;
  LiteRtLmNpuBrand brand = kLiteRtLmNpuBrandQualcomm;
  LiteRtLmModelType model_type = kLiteRtLmModelTypeLlm;
  const char* str = "unset";
  constexpr char kMessage[] = "loaded_file must not be NULL";
  litert_lm_clear_last_error();

  ExpectError(litert_lm_loaded_file_has_speculative_decoding_support(
                  nullptr, &bool_value),
              kLiteRtLmStatusInvalidArgument, kMessage);
  ExpectError(litert_lm_loaded_file_supports_thinking(nullptr, &bool_value),
              kLiteRtLmStatusInvalidArgument, kMessage);
  ExpectError(
      litert_lm_loaded_file_supports_function_calling(nullptr, &bool_value),
      kLiteRtLmStatusInvalidArgument, kMessage);
  ExpectError(litert_lm_loaded_file_supports_input_modality(
                  nullptr, kLiteRtLmModalityText, &bool_value),
              kLiteRtLmStatusInvalidArgument, kMessage);
  ExpectError(litert_lm_loaded_file_is_dynamic_context(nullptr, &bool_value),
              kLiteRtLmStatusInvalidArgument, kMessage);
  EXPECT_TRUE(bool_value);

  ExpectError(litert_lm_loaded_file_sampler_type(nullptr, &sampler_type),
              kLiteRtLmStatusInvalidArgument, kMessage);
  EXPECT_EQ(sampler_type, kLiteRtLmSamplerTypeGreedy);
  ExpectError(litert_lm_loaded_file_sampler_temperature(nullptr, &float_value),
              kLiteRtLmStatusInvalidArgument, kMessage);
  ExpectError(litert_lm_loaded_file_sampler_top_p(nullptr, &float_value),
              kLiteRtLmStatusInvalidArgument, kMessage);
  EXPECT_FLOAT_EQ(float_value, 4.2f);

  ExpectError(litert_lm_loaded_file_sampler_top_k(nullptr, &int_value),
              kLiteRtLmStatusInvalidArgument, kMessage);
  ExpectError(
      litert_lm_loaded_file_max_vision_token_budget(nullptr, &int_value),
      kLiteRtLmStatusInvalidArgument, kMessage);
  ExpectError(litert_lm_loaded_file_vision_signature_selection(nullptr, nullptr,
                                                               0, &int_value),
              kLiteRtLmStatusInvalidArgument, kMessage);
  ExpectError(litert_lm_loaded_file_modality_supported_backends(
                  nullptr, kLiteRtLmModalityText, nullptr, 0, &int_value),
              kLiteRtLmStatusInvalidArgument, kMessage);
  ExpectError(litert_lm_loaded_file_embedding_dimension(nullptr, &int_value),
              kLiteRtLmStatusInvalidArgument, kMessage);
  ExpectError(litert_lm_loaded_file_embedding_signature_selection(
                  nullptr, nullptr, 0, &int_value),
              kLiteRtLmStatusInvalidArgument, kMessage);
  EXPECT_EQ(int_value, 42);

  ExpectError(litert_lm_loaded_file_max_context_tokens(nullptr, &uint_value),
              kLiteRtLmStatusInvalidArgument, kMessage);
  EXPECT_EQ(uint_value, 42);

  ExpectError(litert_lm_loaded_file_modality_npu_brand(
                  nullptr, kLiteRtLmModalityText, &brand),
              kLiteRtLmStatusInvalidArgument, kMessage);
  EXPECT_EQ(brand, kLiteRtLmNpuBrandQualcomm);

  ExpectError(litert_lm_loaded_file_model_type(nullptr, &model_type),
              kLiteRtLmStatusInvalidArgument, kMessage);
  EXPECT_EQ(model_type, kLiteRtLmModelTypeLlm);

  // Pointer out parameters are set to NULL on failure.
  ExpectError(litert_lm_loaded_file_modality_soc_name(
                  nullptr, kLiteRtLmModalityText, &str),
              kLiteRtLmStatusInvalidArgument, kMessage);
  EXPECT_EQ(str, nullptr);
  str = "unset";
  ExpectError(litert_lm_loaded_file_min_runtime_version(nullptr, &str),
              kLiteRtLmStatusInvalidArgument, kMessage);
  EXPECT_EQ(str, nullptr);

  litert_lm_loaded_file_delete(nullptr);
}

TEST(ModelInfoCTest, NullOutParameterReturnsInvalidArgument) {
  LoadedFilePtr file = LoadTestFile("test_lm.litertlm");
  ASSERT_NE(file, nullptr);
  LiteRtLmLoadedFile* f = file.get();
  litert_lm_clear_last_error();

  ExpectError(
      litert_lm_loaded_file_has_speculative_decoding_support(f, nullptr),
      kLiteRtLmStatusInvalidArgument, "out_supported must not be NULL");
  ExpectError(litert_lm_loaded_file_supports_thinking(f, nullptr),
              kLiteRtLmStatusInvalidArgument, "out_supported must not be NULL");
  ExpectError(litert_lm_loaded_file_supports_function_calling(f, nullptr),
              kLiteRtLmStatusInvalidArgument, "out_supported must not be NULL");
  ExpectError(litert_lm_loaded_file_supports_input_modality(
                  f, kLiteRtLmModalityText, nullptr),
              kLiteRtLmStatusInvalidArgument, "out_supported must not be NULL");
  ExpectError(litert_lm_loaded_file_sampler_type(f, nullptr),
              kLiteRtLmStatusInvalidArgument,
              "out_sampler_type must not be NULL");
  ExpectError(litert_lm_loaded_file_sampler_temperature(f, nullptr),
              kLiteRtLmStatusInvalidArgument,
              "out_temperature must not be NULL");
  ExpectError(litert_lm_loaded_file_sampler_top_k(f, nullptr),
              kLiteRtLmStatusInvalidArgument, "out_top_k must not be NULL");
  ExpectError(litert_lm_loaded_file_sampler_top_p(f, nullptr),
              kLiteRtLmStatusInvalidArgument, "out_top_p must not be NULL");
  ExpectError(litert_lm_loaded_file_max_vision_token_budget(f, nullptr),
              kLiteRtLmStatusInvalidArgument, "out_budget must not be NULL");
  ExpectError(litert_lm_loaded_file_max_context_tokens(f, nullptr),
              kLiteRtLmStatusInvalidArgument,
              "out_max_context_tokens must not be NULL");
  ExpectError(litert_lm_loaded_file_is_dynamic_context(f, nullptr),
              kLiteRtLmStatusInvalidArgument,
              "out_is_dynamic must not be NULL");
  ExpectError(
      litert_lm_loaded_file_vision_signature_selection(f, nullptr, 0, nullptr),
      kLiteRtLmStatusInvalidArgument, "out_count must not be NULL");
  ExpectError(litert_lm_loaded_file_modality_supported_backends(
                  f, kLiteRtLmModalityText, nullptr, 0, nullptr),
              kLiteRtLmStatusInvalidArgument, "out_count must not be NULL");
  ExpectError(litert_lm_loaded_file_modality_npu_brand(f, kLiteRtLmModalityText,
                                                       nullptr),
              kLiteRtLmStatusInvalidArgument, "out_npu_brand must not be NULL");
  ExpectError(litert_lm_loaded_file_modality_soc_name(f, kLiteRtLmModalityText,
                                                      nullptr),
              kLiteRtLmStatusInvalidArgument, "out_soc_name must not be NULL");
  ExpectError(litert_lm_loaded_file_min_runtime_version(f, nullptr),
              kLiteRtLmStatusInvalidArgument, "out_version must not be NULL");
  ExpectError(litert_lm_loaded_file_model_type(f, nullptr),
              kLiteRtLmStatusInvalidArgument,
              "out_model_type must not be NULL");
  ExpectError(litert_lm_loaded_file_embedding_dimension(f, nullptr),
              kLiteRtLmStatusInvalidArgument, "out_dimension must not be NULL");
  ExpectError(litert_lm_loaded_file_embedding_signature_selection(f, nullptr, 0,
                                                                  nullptr),
              kLiteRtLmStatusInvalidArgument, "out_count must not be NULL");
}

TEST(ModelInfoCTest, NegativeMaxSizeReturnsInvalidArgument) {
  LoadedFilePtr file = LoadTestFile("test_lm.litertlm");
  ASSERT_NE(file, nullptr);
  int32_t count = 42;
  litert_lm_clear_last_error();
  ExpectError(litert_lm_loaded_file_modality_supported_backends(
                  file.get(), kLiteRtLmModalityText, nullptr, -1, &count),
              kLiteRtLmStatusInvalidArgument, "max_size must not be negative");
  EXPECT_EQ(count, 42);
}

TEST(ModelInfoCTest, InspectAudioCapabilities) {
  LoadedFilePtr file = LoadTestFile("dummy_audio_only.litertlm");
  ASSERT_NE(file, nullptr);

  bool audio = false;
  EXPECT_EQ(litert_lm_loaded_file_supports_input_modality(
                file.get(), kLiteRtLmModalityAudio, &audio),
            kLiteRtLmStatusOk);
  EXPECT_TRUE(audio);

  LiteRtLmBackendType audio_backends[3];
  int32_t count = -1;
  EXPECT_EQ(litert_lm_loaded_file_modality_supported_backends(
                file.get(), kLiteRtLmModalityAudio, audio_backends, 3, &count),
            kLiteRtLmStatusOk);
  EXPECT_EQ(count, 2);
  EXPECT_EQ(audio_backends[0], kLiteRtLmBackendTypeCpu);
  EXPECT_EQ(audio_backends[1], kLiteRtLmBackendTypeGpu);
}

TEST(ModelInfoCTest, InspectEmbeddingCapabilities) {
  LoadedFilePtr file = LoadTestFile("test_embedding.litertlm");
  ASSERT_NE(file, nullptr);

  LiteRtLmModelType model_type = kLiteRtLmModelTypeUnknown;
  EXPECT_EQ(litert_lm_loaded_file_model_type(file.get(), &model_type),
            kLiteRtLmStatusOk);
  EXPECT_EQ(model_type, kLiteRtLmModelTypeEmbedding);

  int32_t dimension = -1;
  EXPECT_EQ(litert_lm_loaded_file_embedding_dimension(file.get(), &dimension),
            kLiteRtLmStatusOk);
  EXPECT_EQ(dimension, 768);

  uint32_t max_context_tokens = 0;
  EXPECT_EQ(
      litert_lm_loaded_file_max_context_tokens(file.get(), &max_context_tokens),
      kLiteRtLmStatusOk);
  EXPECT_EQ(max_context_tokens, 128);
  bool is_dynamic = true;
  EXPECT_EQ(litert_lm_loaded_file_is_dynamic_context(file.get(), &is_dynamic),
            kLiteRtLmStatusOk);
  EXPECT_FALSE(is_dynamic);

  bool text = false;
  bool vision = true;
  EXPECT_EQ(litert_lm_loaded_file_supports_input_modality(
                file.get(), kLiteRtLmModalityText, &text),
            kLiteRtLmStatusOk);
  EXPECT_EQ(litert_lm_loaded_file_supports_input_modality(
                file.get(), kLiteRtLmModalityVision, &vision),
            kLiteRtLmStatusOk);
  EXPECT_TRUE(text);
  EXPECT_FALSE(vision);

  int32_t int_value = 42;
  litert_lm_clear_last_error();
  ExpectError(
      litert_lm_loaded_file_max_vision_token_budget(file.get(), &int_value),
      kLiteRtLmStatusNotFound, "max vision token budget");
  ExpectError(litert_lm_loaded_file_vision_signature_selection(
                  file.get(), nullptr, 0, &int_value),
              kLiteRtLmStatusNotFound, "vision signature selection");
  EXPECT_EQ(int_value, 42);

  // An embedding model has no default sampler parameters.
  LiteRtLmSamplerType sampler_type = kLiteRtLmSamplerTypeGreedy;
  float float_value = 4.2f;
  ExpectError(litert_lm_loaded_file_sampler_type(file.get(), &sampler_type),
              kLiteRtLmStatusNotFound, "default sampler parameters");
  EXPECT_EQ(sampler_type, kLiteRtLmSamplerTypeGreedy);
  ExpectError(
      litert_lm_loaded_file_sampler_temperature(file.get(), &float_value),
      kLiteRtLmStatusNotFound, "default sampler parameters");
  ExpectError(litert_lm_loaded_file_sampler_top_k(file.get(), &int_value),
              kLiteRtLmStatusNotFound, "default sampler parameters");
  ExpectError(litert_lm_loaded_file_sampler_top_p(file.get(), &float_value),
              kLiteRtLmStatusNotFound, "default sampler parameters");
  EXPECT_FLOAT_EQ(float_value, 4.2f);
  EXPECT_EQ(int_value, 42);

  // LLM-only predicates report false for an embedding model.
  bool thinking = true;
  EXPECT_EQ(litert_lm_loaded_file_supports_thinking(file.get(), &thinking),
            kLiteRtLmStatusOk);
  EXPECT_FALSE(thinking);

  LiteRtLmBackendType text_backends[3];
  int32_t text_count = -1;
  EXPECT_EQ(
      litert_lm_loaded_file_modality_supported_backends(
          file.get(), kLiteRtLmModalityText, text_backends, 3, &text_count),
      kLiteRtLmStatusOk);
  EXPECT_EQ(text_count, 2);
  EXPECT_EQ(text_backends[0], kLiteRtLmBackendTypeCpu);
  EXPECT_EQ(text_backends[1], kLiteRtLmBackendTypeGpu);

  int32_t sig_count = -1;
  EXPECT_EQ(litert_lm_loaded_file_embedding_signature_selection(
                file.get(), nullptr, 0, &sig_count),
            kLiteRtLmStatusOk);
  EXPECT_EQ(sig_count, 1);
  int32_t lengths[1] = {0};
  sig_count = -1;
  EXPECT_EQ(litert_lm_loaded_file_embedding_signature_selection(
                file.get(), lengths, 1, &sig_count),
            kLiteRtLmStatusOk);
  EXPECT_EQ(sig_count, 1);
  EXPECT_EQ(lengths[0], 128);

  sig_count = 42;
  ExpectError(litert_lm_loaded_file_embedding_signature_selection(
                  file.get(), lengths, -1, &sig_count),
              kLiteRtLmStatusInvalidArgument, "max_size must not be negative");
  EXPECT_EQ(sig_count, 42);
}

}  // namespace
