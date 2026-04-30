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
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
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
  ASSIGN_OR_RETURN(std::vector<EngineFactory::EngineType> engine_types,
                   EngineFactory::Instance().ListEngineTypes());
  RET_CHECK_EQ(engine_types.size(), 1);
  return EngineFactory::CreateAny(std::move(engine_settings));
}

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

// TODO (b/397975034): Add more tests for Engine.

}  // namespace
}  // namespace litert::lm
