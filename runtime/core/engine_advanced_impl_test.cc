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

#include <algorithm>
#include <cstdlib>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
#ifdef ENGINE_ADVANCED
#include <cstdint>

#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "runtime/executor/executor_backend_registry.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/util/file_data_stream.h"
#include "schema/core/litertlm_header.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "schema/core/litertlm_read.h"
#include "tflite/schema/schema_generated.h"  // from @litert
#endif
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

#if defined(__ANDROID__)
// Must be longer than # of prefill tokens which is 4.
constexpr int kMaxNumTokens = 8;
#else
constexpr int kMaxNumTokens = 16;
#endif

absl::StatusOr<std::unique_ptr<Engine>> CreateEngine(
    EngineSettings engine_settings) {
  ABSL_ASSIGN_OR_RETURN(std::vector<EngineFactory::EngineType> engine_types,
                        EngineFactory::Instance().ListEngineTypes());
  RET_CHECK_EQ(engine_types.size(), 1);
  return EngineFactory::CreateDefault(std::move(engine_settings));
}

#if defined(ENGINE_ADVANCED)
// Registered (custom) backends are only supported by EngineAdvancedImpl.
TEST(EngineTest, CreateEngine_RegisteredBackendReturningNullExecutorFails) {
  ASSERT_OK_AND_ASSIGN(
      const Backend backend,
      ExecutorBackendRegistry::Instance().TryRegister(
          "engine_test_null_executor_backend",
          [](const LlmExecutorSettings&,
             ModelResources&) -> absl::StatusOr<std::unique_ptr<LlmExecutor>> {
            return nullptr;
          }));
  absl::Cleanup unregister = [backend] {
    ExecutorBackendRegistry::Instance().Unregister(backend).IgnoreError();
  };

  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";
  ASSERT_OK_AND_ASSIGN(auto model_assets,
                       ModelAssets::Create(task_path.string()));
  ASSERT_OK_AND_ASSIGN(auto engine_settings,
                       EngineSettings::CreateDefault(model_assets, backend));

  EXPECT_THAT(
      CreateEngine(std::move(engine_settings)),
      testing::status::StatusIs(absl::StatusCode::kInternal,
                                ::testing::HasSubstr("null llm_executor")));
}
#endif  // ENGINE_ADVANCED

TEST(EngineTest, CreateEngine_WithoutCache) {
  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";
  auto model_assets = ModelAssets::Create(task_path.string());
  ASSERT_OK(model_assets);
  auto engine_settings =
      EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  ASSERT_OK(engine_settings);
  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  engine_settings->GetMutableMainExecutorSettings().SetCacheDir(":nocache");

  absl::StatusOr<std::unique_ptr<Engine>> llm = CreateEngine(*engine_settings);
  ABSL_CHECK_OK(llm);

  const Tokenizer& default_tok = (*llm)->GetTokenizer();
  auto prefill_tok =
      (*llm)->GetTokenizer(proto::LlmMetadata::TF_LITE_PREFILL_DECODE);
  ASSERT_OK(prefill_tok);
  EXPECT_EQ(prefill_tok.value(), &default_tok);

  absl::StatusOr<std::unique_ptr<Engine::Session>> session =
      (*llm)->CreateSession(SessionConfig::CreateDefault());
  ABSL_CHECK_OK(session);

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello world!"));
  ABSL_CHECK_OK((*session)->RunPrefill(inputs));

  auto responses = (*session)->RunDecode();
  ASSERT_OK(responses);
  EXPECT_EQ(responses->GetTexts().size(), 1);
  EXPECT_FALSE(responses->GetTexts()[0].empty());

  // 2nd run with the same engine.
  session->reset();  // Destroy the previous first.
  session = (*llm)->CreateSession(SessionConfig::CreateDefault());
  ABSL_CHECK_OK(session);

  ABSL_CHECK_OK((*session)->RunPrefill(inputs));

  responses = (*session)->RunDecode();
  ASSERT_OK(responses);
  EXPECT_EQ(responses->GetTexts().size(), 1);
  EXPECT_FALSE(responses->GetTexts()[0].empty());
}

TEST(EngineTest, CreateEngine_WithNoParallelFileSectionLoading_RunsInference) {
  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";
  auto model_assets = ModelAssets::Create(task_path.string());
  ASSERT_OK(model_assets);
  auto engine_settings =
      EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  ASSERT_OK(engine_settings);
  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  engine_settings->GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings->SetParallelFileSectionLoading(false);

  absl::StatusOr<std::unique_ptr<Engine>> llm = CreateEngine(*engine_settings);
  ABSL_CHECK_OK(llm);

  absl::StatusOr<std::unique_ptr<Engine::Session>> session =
      (*llm)->CreateSession(SessionConfig::CreateDefault());
  ABSL_CHECK_OK(session);

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello world!"));
  ABSL_CHECK_OK((*session)->RunPrefill(inputs));

  auto responses = (*session)->RunDecode();
  ASSERT_OK(responses);
  EXPECT_EQ(responses->GetTexts().size(), 1);
  EXPECT_FALSE(responses->GetTexts()[0].empty());
}

TEST(EngineTest, CreateEngine_WithSingleThreadedExecution_RunsInference) {
  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";
  auto model_assets = ModelAssets::Create(task_path.string());
  ASSERT_OK(model_assets);
  auto engine_settings =
      EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  ASSERT_OK(engine_settings);
  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  engine_settings->GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings->SetSingleThreadedExecution(true);

  absl::StatusOr<std::unique_ptr<Engine>> llm = CreateEngine(*engine_settings);
  ABSL_CHECK_OK(llm);

  absl::StatusOr<std::unique_ptr<Engine::Session>> session =
      (*llm)->CreateSession(SessionConfig::CreateDefault());
  ABSL_CHECK_OK(session);

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello world!"));
  ABSL_CHECK_OK((*session)->RunPrefill(inputs));

  auto responses = (*session)->RunDecode();
  ASSERT_OK(responses);
  EXPECT_EQ(responses->GetTexts().size(), 1);
  EXPECT_FALSE(responses->GetTexts()[0].empty());
}

TEST(EngineTest, CreateEngine_WithCache) {
  auto cache_path = std::filesystem::path(::testing::TempDir()) /
                    absl::StrCat("cache-", std::rand());
  std::filesystem::remove_all(cache_path);
  std::filesystem::create_directories(cache_path);
  absl::Cleanup remove_cache = [cache_path] {
    std::filesystem::remove_all(cache_path);
  };

  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";
  auto model_assets = ModelAssets::Create(task_path.string());
  ASSERT_OK(model_assets);
  auto engine_settings =
      EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  ASSERT_OK(engine_settings);
  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  engine_settings->GetMutableMainExecutorSettings().SetCacheDir(
      cache_path.string());

  // 1st run to populate the cache.
  absl::StatusOr<std::unique_ptr<Engine>> llm = CreateEngine(*engine_settings);
  ABSL_CHECK_OK(llm);

  absl::StatusOr<std::unique_ptr<Engine::Session>> session =
      (*llm)->CreateSession(SessionConfig::CreateDefault());
  ABSL_CHECK_OK(session);

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello world!"));
  ABSL_CHECK_OK((*session)->RunPrefill(inputs));

  auto responses = (*session)->RunDecode();
  ASSERT_OK(responses);
  EXPECT_EQ(responses->GetTexts().size(), 1);
  EXPECT_FALSE(responses->GetTexts()[0].empty());

  // 2nd run with the same engine and the same cache.
  session->reset();  // Destroy the previous first.
  session = (*llm)->CreateSession(SessionConfig::CreateDefault());
  ABSL_CHECK_OK(session);

  ABSL_CHECK_OK((*session)->RunPrefill(inputs));

  responses = (*session)->RunDecode();
  ASSERT_OK(responses);
  EXPECT_EQ(responses->GetTexts().size(), 1);
  EXPECT_FALSE(responses->GetTexts()[0].empty());

  // 3rd run with a new engine and the same cache.
  session->reset();  // Destroy the previous first.
  llm->reset();
  llm = CreateEngine(*engine_settings);
  ABSL_CHECK_OK(llm);

  session = (*llm)->CreateSession(SessionConfig::CreateDefault());
  ABSL_CHECK_OK(session);

  ABSL_CHECK_OK((*session)->RunPrefill(inputs));

  responses = (*session)->RunDecode();
  ASSERT_OK(responses);
  EXPECT_EQ(responses->GetTexts().size(), 1);
  EXPECT_FALSE(responses->GetTexts()[0].empty());
}

TEST(EngineTest, CreateEngine_WithModelAndCacheFromFileDescriptor) {
  auto cache_path = std::filesystem::path(::testing::TempDir()) /
                    absl::StrCat("cache-", std::rand(), ".cache");
  std::filesystem::remove_all(cache_path);
  {
    // Create an empty file - ScopedFile expects the file to exist.
    std::ofstream cache_file(cache_path.string());
  }
  absl::Cleanup remove_cache = [cache_path] {
    std::filesystem::remove_all(cache_path);
  };
  ASSERT_OK_AND_ASSIGN(auto scoped_cache_file,
                       ScopedFile::OpenWritable(cache_path.string()));
  auto shared_scoped_cache_file =
      std::make_shared<ScopedFile>(std::move(scoped_cache_file));

  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";
  ASSERT_OK_AND_ASSIGN(auto task_descriptor,
                       ScopedFile::Open(task_path.string()));
  auto shared_task_descriptor =
      std::make_shared<ScopedFile>(std::move(task_descriptor));
  auto model_assets = ModelAssets::Create(shared_task_descriptor);
  ASSERT_OK(model_assets);
  auto engine_settings =
      EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  ASSERT_OK(engine_settings);
  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  engine_settings->GetMutableMainExecutorSettings().SetScopedCacheFile(
      shared_scoped_cache_file);

  absl::StatusOr<std::unique_ptr<Engine>> llm = CreateEngine(*engine_settings);
  ABSL_CHECK_OK(llm);

  absl::StatusOr<std::unique_ptr<Engine::Session>> session =
      (*llm)->CreateSession(SessionConfig::CreateDefault());
  ABSL_CHECK_OK(session);

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello world!"));
  ABSL_CHECK_OK((*session)->RunPrefill(inputs));

  auto responses = (*session)->RunDecode();
  ASSERT_OK(responses);
  EXPECT_EQ(responses->GetTexts().size(), 1);
  EXPECT_FALSE(responses->GetTexts()[0].empty());
}

TEST(EngineTest, CreateEngine_AsyncTokenizer_ValidatesConcurrency) {
  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";
  auto model_assets = ModelAssets::Create(task_path.string());
  ASSERT_OK(model_assets);
  auto engine_settings =
      EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  ASSERT_OK(engine_settings);

  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  engine_settings->GetMutableMainExecutorSettings().SetCacheDir(":nocache");

  // Enable Benchmark to measure the phases and prove concurrency
  engine_settings->GetMutableBenchmarkParams();

  absl::StatusOr<std::unique_ptr<Engine>> llm = CreateEngine(*engine_settings);
  ABSL_CHECK_OK(llm);

  absl::StatusOr<std::unique_ptr<Engine::Session>> session =
      (*llm)->CreateSession(SessionConfig::CreateDefault());
  ABSL_CHECK_OK(session);

  auto benchmark_info = (*session)->GetMutableBenchmarkInfo();
  ASSERT_OK(benchmark_info);

  const auto& init_phases = (*benchmark_info)->GetInitPhases();

  auto total_time = init_phases.at(std::string(
      BenchmarkInfo::InitPhaseToString(BenchmarkInfo::InitPhase::kTotal)));
  auto executor_time = init_phases.at(std::string(
      BenchmarkInfo::InitPhaseToString(BenchmarkInfo::InitPhase::kExecutor)));
  auto tokenizer_time = init_phases.at(std::string(
      BenchmarkInfo::InitPhaseToString(BenchmarkInfo::InitPhase::kTokenizer)));

  // The total duration should be greater than or equal to the longest
  // concurrent branch.
  EXPECT_GE(total_time, executor_time);
  EXPECT_GE(total_time, tokenizer_time);
  // The total duration (minus the sequential part) should be less than the sum
  // of the two parallel branches. This is to prove that the tokenizer and
  // executor are loaded concurrently.
  auto rest_time = total_time - std::max(executor_time, tokenizer_time);
  EXPECT_LE(total_time - rest_time, executor_time + tokenizer_time);

  // Verifying tokenizer resolves tokens successfully without data bounds errors
  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello concurrent world!"));
  ABSL_CHECK_OK((*session)->RunPrefill(inputs));

  auto responses = (*session)->RunDecode();
  ASSERT_OK(responses);
  EXPECT_EQ(responses->GetTexts().size(), 1);
  EXPECT_FALSE(responses->GetTexts()[0].empty());
}

TEST(EngineTest, CreateEngine_WithBenchmark) {
  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";
  auto model_assets = ModelAssets::Create(task_path.string());
  ASSERT_OK(model_assets);
  auto engine_settings =
      EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  ASSERT_OK(engine_settings);

  // Enable Benchmark
  engine_settings->GetMutableBenchmarkParams();

  absl::StatusOr<std::unique_ptr<Engine>> llm = CreateEngine(*engine_settings);
  ABSL_CHECK_OK(llm);

  absl::StatusOr<std::unique_ptr<Engine::Session>> session =
      (*llm)->CreateSession(SessionConfig::CreateDefault());
  ABSL_CHECK_OK(session);

  auto benchmark_info = (*session)->GetMutableBenchmarkInfo();
  ASSERT_OK(benchmark_info);

  const auto& init_phases = (*benchmark_info)->GetInitPhases();

  EXPECT_TRUE(init_phases.contains(std::string(
      BenchmarkInfo::InitPhaseToString(BenchmarkInfo::InitPhase::kTokenizer))));
  EXPECT_TRUE(init_phases.contains(std::string(
      BenchmarkInfo::InitPhaseToString(BenchmarkInfo::InitPhase::kExecutor))));
  EXPECT_TRUE(init_phases.contains(std::string(
      BenchmarkInfo::InitPhaseToString(BenchmarkInfo::InitPhase::kTotal))));
}

TEST(EngineTest, CreateEngine_FailsNoVisionModel) {
  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";
  auto model_assets = ModelAssets::Create(task_path.string());
  ASSERT_OK(model_assets);
  auto engine_settings = EngineSettings::CreateDefault(
      *model_assets, /*backend=*/Backend::CPU, /*vision_backend=*/Backend::CPU,
      /*audio_backend=*/std::nullopt);
  ASSERT_OK(engine_settings);
  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  engine_settings->GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  ASSERT_OK_AND_ASSIGN(auto llm, CreateEngine(*engine_settings));
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetVisionModalityEnabled(true);
  EXPECT_THAT(
      llm->CreateSession(session_config),
      testing::AnyOf(testing::status::StatusIs(
                         absl::StatusCode::kNotFound,
                         "TF_LITE_VISION_ENCODER not found in the model."),
                     testing::status::StatusIs(
                         absl::StatusCode::kNotFound,
                         testing::HasSubstr(
                             "No file with name: TF_LITE_VISION_ENCODER."))));
}

TEST(EngineTest, CreateEngine_FailsNoAudioModel) {
  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";
  auto model_assets = ModelAssets::Create(task_path.string());
  ASSERT_OK(model_assets);
  auto engine_settings = EngineSettings::CreateDefault(
      *model_assets, /*backend=*/Backend::CPU, /*vision_backend=*/std::nullopt,
      /*audio_backend=*/Backend::CPU);
  ASSERT_OK(engine_settings);
  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  engine_settings->GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  ASSERT_OK_AND_ASSIGN(auto llm, CreateEngine(*engine_settings));
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetAudioModalityEnabled(true);
  EXPECT_THAT(llm->CreateSession(session_config),
              testing::status::StatusIs(
                  absl::StatusCode::kNotFound,
                  "TF_LITE_AUDIO_ENCODER_HW not found in the model."));
}

TEST(EngineTest,
     CreateEngine_MaxVisionTokensPerImage_NoVisionSettingsSucceeds) {
  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";
  auto model_assets = ModelAssets::Create(task_path.string());
  ASSERT_OK(model_assets);
  auto engine_settings =
      EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  ASSERT_OK(engine_settings);
  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  engine_settings->GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings->SetMaxVisionTokensPerImage(200);

  EXPECT_OK(CreateEngine(*engine_settings));
}

TEST(EngineTest, CreateEngine_MaxVisionTokensPerImage_InvalidValueFails) {
  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";
  auto model_assets = ModelAssets::Create(task_path.string());
  ASSERT_OK(model_assets);
  auto engine_settings = EngineSettings::CreateDefault(
      *model_assets, Backend::CPU, /*vision_backend=*/Backend::CPU);
  ASSERT_OK(engine_settings);
  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  engine_settings->GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings->SetMaxVisionTokensPerImage(-5);

  EXPECT_THAT(
      CreateEngine(*engine_settings),
      testing::status::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("max_vision_tokens_per_image must be positive")));
}

TEST(EngineTest,
     CreateEngine_VisionExecutorSettings_WithoutMaxVisionTokensPerImage) {
  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";
  auto model_assets = ModelAssets::Create(task_path.string());
  ASSERT_OK(model_assets);
  auto engine_settings = EngineSettings::CreateDefault(
      *model_assets, Backend::CPU, /*vision_backend=*/Backend::CPU);
  ASSERT_OK(engine_settings);
  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  engine_settings->GetMutableMainExecutorSettings().SetCacheDir(":nocache");

  ASSERT_OK_AND_ASSIGN(auto llm, CreateEngine(*engine_settings));
  EXPECT_NE(llm, nullptr);
}

TEST(EngineTest,
     CreateEngine_MaxVisionTokensPerImageIgnoredForUnsupportedModel) {
  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";
  auto model_assets = ModelAssets::Create(task_path.string());
  ASSERT_OK(model_assets);
  auto engine_settings = EngineSettings::CreateDefault(
      *model_assets, Backend::CPU, /*vision_backend=*/Backend::CPU);
  ASSERT_OK(engine_settings);
  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  engine_settings->GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings->SetMaxVisionTokensPerImage(280);

  ASSERT_OK_AND_ASSIGN(auto llm, CreateEngine(*engine_settings));
  EXPECT_NE(llm, nullptr);
}

#ifdef ENGINE_ADVANCED
TEST(EngineTest, CreateEngine_StreamedModelLoading) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";

  std::shared_ptr<FileDataStream> file_stream;
  ASSERT_OK_AND_ASSIGN(file_stream,
                       FileDataStream::Create(model_path.string()));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(std::move(file_stream)));

  auto engine_settings =
      EngineSettings::CreateDefault(std::move(model_assets), Backend::CPU);
  ASSERT_OK(engine_settings);
  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  engine_settings->GetMutableMainExecutorSettings().SetCacheDir(":nocache");

  absl::StatusOr<std::unique_ptr<Engine>> llm = CreateEngine(*engine_settings);
  ASSERT_OK(llm);

  absl::StatusOr<std::unique_ptr<Engine::Session>> session =
      (*llm)->CreateSession(SessionConfig::CreateDefault());
  ASSERT_OK(session);

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello world!"));
  ASSERT_OK((*session)->RunPrefill(inputs));

  auto responses = (*session)->RunDecode();
  ASSERT_OK(responses);
  EXPECT_EQ(responses->GetTexts().size(), 1);
  EXPECT_FALSE(responses->GetTexts()[0].empty());
}

// Rewrites `test_lm_external_weights.litertlm` to `output_path` with its legacy
// `"weights.bin"` external buffer group renamed to `kModelWeightGroupName`
// (`"tflite_weights"`), matching what `litert-lm pack` produces today.
absl::Status CreateExternalWeightsTestModel(
    const std::filesystem::path& output_path) {
  const auto src_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_external_weights.litertlm";

  schema::LitertlmHeader header;
  ABSL_RETURN_IF_ERROR(
      schema::ReadHeaderFromLiteRTLM(src_path.string(), &header));
  RET_CHECK(header.metadata != nullptr);
  RET_CHECK(header.metadata->section_metadata() != nullptr);
  const auto* objects = header.metadata->section_metadata()->objects();
  RET_CHECK(objects != nullptr);

  std::vector<const schema::SectionObject*> sorted_sections(objects->begin(),
                                                            objects->end());
  std::sort(sorted_sections.begin(), sorted_sections.end(),
            [](const schema::SectionObject* a, const schema::SectionObject* b) {
              return a->begin_offset() < b->begin_offset();
            });

  std::ifstream in(src_path.string(), std::ios::binary);
  RET_CHECK(in.is_open());

  struct RepackedSection {
    schema::AnySectionDataType data_type;
    bool has_prefill_decode_model_type = false;
    std::vector<uint8_t> bytes;
  };
  std::vector<RepackedSection> repacked_sections;
  repacked_sections.reserve(sorted_sections.size());

  for (const schema::SectionObject* sec : sorted_sections) {
    const uint64_t size = sec->end_offset() - sec->begin_offset();
    std::vector<uint8_t> sec_bytes(size);
    in.seekg(sec->begin_offset());
    in.read(reinterpret_cast<char*>(sec_bytes.data()), size);
    RET_CHECK(in.good());

    const bool is_model_or_weights =
        sec->data_type() == schema::AnySectionDataType_TFLiteModel ||
        sec->data_type() == schema::AnySectionDataType_TFLiteWeights;
    if (sec->data_type() == schema::AnySectionDataType_TFLiteModel) {
      std::unique_ptr<tflite::ModelT> model =
          tflite::UnPackModel(sec_bytes.data());
      RET_CHECK(model != nullptr);
      for (auto& group : model->external_buffer_groups) {
        if (group != nullptr && group->name == "weights.bin") {
          group->name = std::string(kModelWeightGroupName);
        }
      }
      flatbuffers::FlatBufferBuilder model_fbb;
      tflite::FinishModelBuffer(model_fbb,
                                tflite::Model::Pack(model_fbb, model.get()));
      sec_bytes.assign(model_fbb.GetBufferPointer(),
                       model_fbb.GetBufferPointer() + model_fbb.GetSize());
    }
    repacked_sections.push_back(RepackedSection{
        .data_type = sec->data_type(),
        .has_prefill_decode_model_type = is_model_or_weights,
        .bytes = std::move(sec_bytes),
    });
  }

  constexpr uint64_t kBlockAlignment = 16384;
  auto align_up = [](uint64_t offset) {
    return (offset + kBlockAlignment - 1) & ~(kBlockAlignment - 1);
  };

  flatbuffers::FlatBufferBuilder header_fbb;
  std::vector<flatbuffers::Offset<schema::SectionObject>> fb_sections;
  uint64_t current_offset = kBlockAlignment;
  std::vector<uint64_t> section_offsets;
  for (const RepackedSection& sec : repacked_sections) {
    current_offset = align_up(current_offset);
    section_offsets.push_back(current_offset);
    flatbuffers::Offset<
        flatbuffers::Vector<flatbuffers::Offset<schema::KeyValuePair>>>
        items_offset = 0;
    if (sec.has_prefill_decode_model_type) {
      const std::string model_type_val = "tf_lite_prefill_decode";
      auto kv =
          schema::CreateKeyValuePair(header_fbb, "model_type", model_type_val);
      items_offset = header_fbb.CreateVector(
          std::vector<flatbuffers::Offset<schema::KeyValuePair>>{kv});
    }
    fb_sections.push_back(schema::CreateSectionObject(
        header_fbb, items_offset, current_offset,
        current_offset + sec.bytes.size(), sec.data_type));
    current_offset += sec.bytes.size();
  }
  auto section_metadata = schema::CreateSectionMetadata(
      header_fbb, header_fbb.CreateVector(fb_sections));
  auto metadata =
      schema::CreateLiteRTLMMetaData(header_fbb, 0, section_metadata);
  header_fbb.Finish(metadata);

  const uint64_t header_end_offset = 32 + header_fbb.GetSize();
  RET_CHECK_LE(header_end_offset, kBlockAlignment);

  std::ofstream out(output_path.string(), std::ios::binary);
  RET_CHECK(out.is_open());
  out.write("LITERTLM", 8);
  const uint32_t major = 1;
  const uint32_t minor = 5;
  const uint32_t patch = 0;
  const uint32_t padding = 0;
  out.write(reinterpret_cast<const char*>(&major), sizeof(major));
  out.write(reinterpret_cast<const char*>(&minor), sizeof(minor));
  out.write(reinterpret_cast<const char*>(&patch), sizeof(patch));
  out.write(reinterpret_cast<const char*>(&padding), sizeof(padding));
  out.write(reinterpret_cast<const char*>(&header_end_offset),
            sizeof(header_end_offset));
  out.write(reinterpret_cast<const char*>(header_fbb.GetBufferPointer()),
            header_fbb.GetSize());

  uint64_t pos = header_end_offset;
  for (size_t i = 0; i < repacked_sections.size(); ++i) {
    if (section_offsets[i] > pos) {
      std::vector<char> zeros(section_offsets[i] - pos, 0);
      out.write(zeros.data(), zeros.size());
      pos = section_offsets[i];
    }
    out.write(reinterpret_cast<const char*>(repacked_sections[i].bytes.data()),
              repacked_sections[i].bytes.size());
    pos += repacked_sections[i].bytes.size();
  }
  out.flush();
  RET_CHECK(out.good());
  return absl::OkStatus();
}

// Regression test: on the CPU backend, the main model's external weights must
// be read into host memory rather than left as a weights stream, which only the
// GPU backends can consume.
TEST(EngineTest,
     CreateEngine_StreamedModelLoading_ExternalWeightsOnCpuMatchesFileBacked) {
  const auto model_path =
      std::filesystem::path(::testing::TempDir()) /
      absl::StrCat("test_lm_tflite_weights_", std::rand(), ".litertlm");
  absl::Cleanup remove_model = [&model_path] {
    std::filesystem::remove_all(model_path);
  };
  ASSERT_OK(CreateExternalWeightsTestModel(model_path));

  ASSERT_OK_AND_ASSIGN(ModelAssets file_assets,
                       ModelAssets::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(
      EngineSettings file_settings,
      EngineSettings::CreateDefault(std::move(file_assets), Backend::CPU));
  file_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(kMaxNumTokens);
  file_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Engine> file_engine,
                       CreateEngine(std::move(file_settings)));

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<FileDataStream> file_stream,
                       FileDataStream::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(ModelAssets streamed_assets,
                       ModelAssets::Create(std::move(file_stream)));
  ASSERT_OK_AND_ASSIGN(
      EngineSettings streamed_settings,
      EngineSettings::CreateDefault(std::move(streamed_assets), Backend::CPU));
  streamed_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  streamed_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Engine> streamed_engine,
                       CreateEngine(std::move(streamed_settings)));

  std::vector<InputData> file_inputs;
  file_inputs.emplace_back(InputText("Hello world!"));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Engine::Session> file_session,
      file_engine->CreateSession(SessionConfig::CreateDefault()));
  ASSERT_OK(file_session->RunPrefill(file_inputs));
  ASSERT_OK_AND_ASSIGN(Responses file_responses, file_session->RunDecode());

  std::vector<InputData> streamed_inputs;
  streamed_inputs.emplace_back(InputText("Hello world!"));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Engine::Session> streamed_session,
      streamed_engine->CreateSession(SessionConfig::CreateDefault()));
  ASSERT_OK(streamed_session->RunPrefill(streamed_inputs));
  ASSERT_OK_AND_ASSIGN(Responses streamed_responses,
                       streamed_session->RunDecode());

  ASSERT_EQ(streamed_responses.GetTexts().size(), 1);
  EXPECT_FALSE(streamed_responses.GetTexts()[0].empty());
  EXPECT_EQ(streamed_responses.GetTexts(), file_responses.GetTexts());
}

TEST(EngineTest, CreateEngine_StreamedModelLoading_EnableDebuggerDisabled) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<FileDataStream> file_stream,
                       FileDataStream::Create(model_path.string()));
  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(std::move(file_stream)));
  ASSERT_OK_AND_ASSIGN(
      auto engine_settings,
      EngineSettings::CreateDefault(std::move(model_assets), Backend::CPU));
  engine_settings.GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  engine_settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  engine_settings.SetEnableDebugger(true);

  ASSERT_OK_AND_ASSIGN(auto llm, CreateEngine(engine_settings));
  EXPECT_FALSE(llm->GetEngineSettings().GetEnableDebugger());
}

TEST(EngineTest, CreateEngine_StreamedModelLoading_VisionWithoutVisionModel) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";

  std::shared_ptr<FileDataStream> file_stream;
  ASSERT_OK_AND_ASSIGN(file_stream,
                       FileDataStream::Create(model_path.string()));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(std::move(file_stream)));

  auto engine_settings = EngineSettings::CreateDefault(
      std::move(model_assets), Backend::CPU, /*vision_backend=*/Backend::CPU);
  ASSERT_OK(engine_settings);
  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  engine_settings->GetMutableMainExecutorSettings().SetCacheDir(":nocache");

  // The engine is still created, matching the non-streamed path, but vision
  // is unavailable.
  ASSERT_OK_AND_ASSIGN(auto llm, CreateEngine(*engine_settings));

  // The engine doesn't advertise vision, so callers (e.g. the Web SDK, which
  // enables the vision modality by default when vision settings are present)
  // don't create sessions that fail.
  EXPECT_FALSE(
      llm->GetEngineSettings().GetVisionExecutorSettings().has_value());

  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetVisionModalityEnabled(true);
  EXPECT_FALSE(llm->CreateSession(session_config).ok());

  // Text-only sessions still work.
  EXPECT_OK(llm->CreateSession(SessionConfig::CreateDefault()));
}

TEST(EngineTest, CreateEngine_StreamedModelLoading_AudioWithoutAudioModel) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";

  std::shared_ptr<FileDataStream> file_stream;
  ASSERT_OK_AND_ASSIGN(file_stream,
                       FileDataStream::Create(model_path.string()));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(std::move(file_stream)));

  auto engine_settings = EngineSettings::CreateDefault(
      std::move(model_assets), Backend::CPU, /*vision_backend=*/std::nullopt,
      /*audio_backend=*/Backend::CPU);
  ASSERT_OK(engine_settings);
  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  engine_settings->GetMutableMainExecutorSettings().SetCacheDir(":nocache");

  ASSERT_OK_AND_ASSIGN(auto llm, CreateEngine(*engine_settings));

  EXPECT_FALSE(llm->GetEngineSettings().GetAudioExecutorSettings().has_value());

  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetAudioModalityEnabled(true);
  EXPECT_FALSE(llm->CreateSession(session_config).ok());

  EXPECT_OK(llm->CreateSession(SessionConfig::CreateDefault()));
}

#ifdef ENABLE_HUGGINGFACE_TOKENIZER
TEST(EngineTest, CreateEngine_StreamedModelLoading_HfTokenizer) {
  auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_hf_tokenizer.litertlm";

  std::shared_ptr<FileDataStream> file_stream;
  ASSERT_OK_AND_ASSIGN(file_stream,
                       FileDataStream::Create(model_path.string()));

  ASSERT_OK_AND_ASSIGN(ModelAssets model_assets,
                       ModelAssets::Create(std::move(file_stream)));

  auto engine_settings =
      EngineSettings::CreateDefault(std::move(model_assets), Backend::CPU);
  ASSERT_OK(engine_settings);
  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  engine_settings->GetMutableMainExecutorSettings().SetCacheDir(":nocache");

  absl::StatusOr<std::unique_ptr<Engine>> llm = CreateEngine(*engine_settings);
  // test_hf_tokenizer.litertlm only has HF_Tokenizer_Zlib section, so tokenizer
  // loading succeeds but engine creation fails because no TFLiteModel section
  // was provided to initialize the environment.
  EXPECT_THAT(llm, testing::status::StatusIs(
                       absl::StatusCode::kInternal,
                       "Environment was not initialized during streaming."));
}
#endif  // ENABLE_HUGGINGFACE_TOKENIZER
#endif  // ENGINE_ADVANCED

// TODO (b/397975034): Add more tests for Engine.

TEST(EngineTest, UpdateGpuEnableMetalResidencySet) {
  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";
  auto model_assets = ModelAssets::Create(task_path.string());
  ASSERT_OK(model_assets);
  auto engine_settings =
      EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  ASSERT_OK(engine_settings);
  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  engine_settings->GetMutableMainExecutorSettings().SetCacheDir(":nocache");

  absl::StatusOr<std::unique_ptr<Engine>> llm = CreateEngine(*engine_settings);
  ASSERT_OK(llm);

  EXPECT_OK((*llm)->UpdateGpuEnableMetalResidencySet(true));
  EXPECT_OK((*llm)->UpdateGpuEnableMetalResidencySet(false));
}

TEST(EngineTest, GetEnvironmentReturnsValidEnvironment) {
  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";
  auto model_assets = ModelAssets::Create(task_path.string());
  ASSERT_OK(model_assets);
  auto engine_settings =
      EngineSettings::CreateDefault(*model_assets, Backend::CPU);
  ASSERT_OK(engine_settings);
  engine_settings->GetMutableMainExecutorSettings().SetMaxNumTokens(
      kMaxNumTokens);
  engine_settings->GetMutableMainExecutorSettings().SetCacheDir(":nocache");

  ASSERT_OK_AND_ASSIGN(auto engine, CreateEngine(*engine_settings));
  ASSERT_OK_AND_ASSIGN(const auto* env, engine->GetEnvironment());
  EXPECT_NE(env, nullptr);

  ASSERT_OK_AND_ASSIGN(auto session,
                       engine->CreateSession(SessionConfig::CreateDefault()));
  ASSERT_OK_AND_ASSIGN(const auto* session_env, session->GetEnvironment());
  EXPECT_EQ(session_env, env);
}

}  // namespace
}  // namespace litert::lm
