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

#include "runtime/engine/litert_lm_lib.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/escaping.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/mtp_gating_policy.h"
#include "runtime/executor/speculative_decoding_mode.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert {
namespace lm {
namespace {

using ::nlohmann::json;
using ::testing::Optional;
using ::testing::status::StatusIs;
class CooptStdout {
 public:
  CooptStdout() : old_buf_(std::cout.rdbuf(buffer_.rdbuf())) {}
  ~CooptStdout() { std::cout.rdbuf(old_buf_); }
  std::string GetOutput() const { return buffer_.str(); }

 private:
  std::stringstream buffer_;
  std::streambuf* old_buf_;
};

std::string ColorBlue(const std::string& s) {
  if (UseColor()) {
    return std::string("\033[34m") + s + "\033[0m";
  }
  return s;
}

std::string ColorYellow(const std::string& s) {
  if (UseColor()) {
    return std::string("\033[33m") + s + "\033[0m";
  }
  return s;
}

TEST(BuildContentListTest, TextOnly) {
  LiteRtLmSettings settings;
  std::vector<InputData> input_data;
  input_data.push_back(InputText("Hello world"));
  ASSERT_OK_AND_ASSIGN(json content_list,
                       BuildContentList(input_data, settings));
  ASSERT_EQ(content_list.size(), 1);
  EXPECT_EQ(content_list[0]["type"], "text");
  EXPECT_EQ(content_list[0]["text"], "Hello world");
}

TEST(BuildContentListTest, MediaTagsSuccess) {
  // Create a temporary file to mock a media file.
  const std::string temp_dir = testing::TempDir();
  const std::string image_path = temp_dir + "/test_image.jpg";
  std::ofstream(image_path) << "dummy image data";

  LiteRtLmSettings settings;
  settings.vision_backend = "cpu";
  const std::string prompt =
      absl::StrCat("Describe this [image:", image_path, "].");
  std::vector<InputData> input_data;
  input_data.push_back(InputText(prompt));
  ASSERT_OK_AND_ASSIGN(json content_list,
                       BuildContentList(input_data, settings));

  ASSERT_EQ(content_list.size(), 3);
  EXPECT_EQ(content_list[0]["type"], "text");
  EXPECT_EQ(content_list[0]["text"], "Describe this ");
  EXPECT_EQ(content_list[1]["type"], "image");
  EXPECT_EQ(content_list[1]["path"], image_path);
  EXPECT_EQ(content_list[2]["type"], "text");
  EXPECT_EQ(content_list[2]["text"], ".");
}

TEST(BuildContentListTest, MediaTagsMissingBackend) {
  const std::string temp_dir = testing::TempDir();
  const std::string image_path = temp_dir + "/test_image.jpg";
  std::ofstream(image_path) << "dummy image data";

  LiteRtLmSettings settings;
  settings.vision_backend = std::nullopt;  // Explicitly missing
  std::string prompt = absl::StrCat("Describe this [image:", image_path, "].");
  std::vector<InputData> input_data;
  input_data.push_back(InputText(prompt));
  EXPECT_THAT(BuildContentList(input_data, settings).status(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BuildContentListTest, ImageDataSuccess) {
  LiteRtLmSettings settings;
  const std::vector<std::string> images = {"image_blob_1", "image_blob_2"};

  std::vector<InputData> input_data;
  input_data.push_back(InputText("Image 1: "));
  input_data.push_back(InputImage(images[0]));
  input_data.push_back(InputText(", Image 2: "));
  input_data.push_back(InputImage(images[1]));
  input_data.push_back(InputText("."));

  ASSERT_OK_AND_ASSIGN(json content_list,
                       BuildContentList(input_data, settings));

  ASSERT_EQ(content_list.size(), 5);
  EXPECT_EQ(content_list[0]["text"], "Image 1: ");
  EXPECT_EQ(content_list[1]["type"], "image");
  EXPECT_EQ(content_list[1]["blob"], absl::Base64Escape("image_blob_1"));
  EXPECT_EQ(content_list[2]["text"], ", Image 2: ");
  EXPECT_EQ(content_list[3]["type"], "image");
  EXPECT_EQ(content_list[3]["blob"], absl::Base64Escape("image_blob_2"));
  EXPECT_EQ(content_list[4]["text"], ".");
}

TEST(BuildContentListTest, AudioDataSuccess) {
  LiteRtLmSettings settings;
  std::vector<std::string> audios = {"audio_blob_1", "audio_blob_2"};

  std::vector<InputData> input_data;
  input_data.push_back(InputText("Audio 1: "));
  input_data.push_back(InputAudio(audios[0]));
  input_data.push_back(InputText(", Audio 2: "));
  input_data.push_back(InputAudio(audios[1]));
  input_data.push_back(InputText("."));

  ASSERT_OK_AND_ASSIGN(json content_list,
                       BuildContentList(input_data, settings));

  ASSERT_EQ(content_list.size(), 5);
  EXPECT_EQ(content_list[0]["text"], "Audio 1: ");
  EXPECT_EQ(content_list[1]["type"], "audio");
  EXPECT_EQ(content_list[1]["blob"], absl::Base64Escape("audio_blob_1"));
  EXPECT_EQ(content_list[2]["text"], ", Audio 2: ");
  EXPECT_EQ(content_list[3]["type"], "audio");
  EXPECT_EQ(content_list[3]["blob"], absl::Base64Escape("audio_blob_2"));
  EXPECT_EQ(content_list[4]["text"], ".");
}

TEST(BuildContentListTest, MixedModality) {
  const std::string temp_dir = testing::TempDir();
  const std::string audio_path = temp_dir + "/test_audio.wav";
  std::ofstream(audio_path) << "dummy audio data";

  LiteRtLmSettings settings;
  settings.audio_backend = "cpu";
  std::vector<std::string> images = {"image_blob_1"};
  std::vector<std::string> audios = {"audio_blob_1"};

  std::string prompt =
      absl::StrCat("Listen to [audio:", audio_path, "] and look at ");
  std::vector<InputData> input_data;
  input_data.push_back(InputText(prompt));
  input_data.push_back(InputImage(images[0]));
  input_data.push_back(InputText(" and listen to "));
  input_data.push_back(InputAudio(audios[0]));
  input_data.push_back(InputText("."));
  ASSERT_OK_AND_ASSIGN(json content_list,
                       BuildContentList(input_data, settings));

  ASSERT_EQ(content_list.size(), 7);
  EXPECT_EQ(content_list[0]["text"], "Listen to ");
  EXPECT_EQ(content_list[1]["type"], "audio");
  EXPECT_EQ(content_list[1]["path"], audio_path);
  EXPECT_EQ(content_list[2]["text"], " and look at ");
  EXPECT_EQ(content_list[3]["type"], "image");
  EXPECT_EQ(content_list[3]["blob"], absl::Base64Escape("image_blob_1"));
  EXPECT_EQ(content_list[4]["text"], " and listen to ");
  EXPECT_EQ(content_list[5]["type"], "audio");
  EXPECT_EQ(content_list[5]["blob"], absl::Base64Escape("audio_blob_1"));
  EXPECT_EQ(content_list[6]["text"], ".");
}

TEST(LiteRtLmLibTest, RunLiteRtLmWithEmptyModelPathReturnsError) {
  LiteRtLmSettings settings;
  settings.model_path = "";
  EXPECT_THAT(RunLiteRtLm(settings),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

// Returns a LitertLmMetrics with one prefill and one decode turn, an "Init
// Executor" phase of `init_phase` and the given peak memory usage.
LitertLmMetrics CreateMetrics(
    absl::Duration init_phase, float peak_mem_mb, float peak_private_mb,
    std::optional<float> peak_accelerator_mb = std::nullopt) {
  BenchmarkInfo benchmark_info((proto::BenchmarkParams()));
  ABSL_CHECK_OK(benchmark_info.InitPhaseRecord(
      BenchmarkInfo::InitPhase::kExecutor, init_phase));
  ABSL_CHECK_OK(benchmark_info.TimePrefillTurnStart());
  ABSL_CHECK_OK(benchmark_info.TimePrefillTurnEnd(/*num_prefill_tokens=*/128));
  ABSL_CHECK_OK(benchmark_info.TimeDecodeTurnStart());
  ABSL_CHECK_OK(benchmark_info.TimeDecodeTurnEnd(/*num_decode_tokens=*/32));

  LitertLmMetrics metrics;
  metrics.benchmark_info = benchmark_info;
  metrics.peak_mem_mb = peak_mem_mb;
  metrics.peak_private_mb = peak_private_mb;
  metrics.peak_accelerator_mb = peak_accelerator_mb;
  return metrics;
}

// Returns the median of the given values, computed independently from the
// implementation under test.
double MedianOf(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values.size() % 2 == 1
             ? values[values.size() / 2]
             : (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2;
}

TEST(ComputeMedianMetricsTest, NoMetrics) {
  const AggregatedLitertLmMetrics aggregated = ComputeMedianMetrics({});

  EXPECT_EQ(aggregated.num_iterations, 0);
  EXPECT_THAT(aggregated.init_phases, testing::IsEmpty());
  EXPECT_THAT(aggregated.prefill_tokens_per_sec, testing::IsEmpty());
  EXPECT_THAT(aggregated.decode_tokens_per_sec, testing::IsEmpty());
  EXPECT_FALSE(aggregated.time_to_first_token_sec.has_value());
  EXPECT_FALSE(aggregated.peak_mem_mb.has_value());
  EXPECT_FALSE(aggregated.peak_private_mb.has_value());
  EXPECT_FALSE(aggregated.peak_accelerator_mb.has_value());
  EXPECT_THAT(aggregated.peak_accelerator_breakdown_mb, testing::IsEmpty());
}

TEST(ComputeMedianMetricsTest, OddNumberOfIterations) {
  const std::vector<LitertLmMetrics> metrics = {
      CreateMetrics(absl::Milliseconds(300), /*peak_mem_mb=*/300.0f,
                    /*peak_private_mb=*/30.0f, /*peak_accelerator_mb=*/30.0f),
      CreateMetrics(absl::Milliseconds(100), /*peak_mem_mb=*/100.0f,
                    /*peak_private_mb=*/10.0f, /*peak_accelerator_mb=*/10.0f),
      CreateMetrics(absl::Milliseconds(200), /*peak_mem_mb=*/200.0f,
                    /*peak_private_mb=*/20.0f, /*peak_accelerator_mb=*/20.0f),
  };

  const AggregatedLitertLmMetrics aggregated = ComputeMedianMetrics(metrics);

  EXPECT_EQ(aggregated.num_iterations, 3);
  EXPECT_THAT(
      aggregated.init_phases,
      testing::ElementsAre(testing::Pair(
          BenchmarkInfo::InitPhaseToString(BenchmarkInfo::InitPhase::kExecutor),
          absl::Milliseconds(200))));
  EXPECT_EQ(aggregated.peak_mem_mb, 200.0f);
  EXPECT_EQ(aggregated.peak_private_mb, 20.0f);
  EXPECT_EQ(aggregated.peak_accelerator_mb, 20.0f);

  std::vector<double> prefill_speeds;
  std::vector<double> decode_speeds;
  std::vector<double> times_to_first_token;
  for (const LitertLmMetrics& metric : metrics) {
    prefill_speeds.push_back(metric.benchmark_info->GetPrefillTokensPerSec(0));
    decode_speeds.push_back(metric.benchmark_info->GetDecodeTokensPerSec(0));
    times_to_first_token.push_back(
        metric.benchmark_info->GetTimeToFirstToken());
  }
  EXPECT_THAT(aggregated.prefill_tokens_per_sec,
              testing::ElementsAre(MedianOf(prefill_speeds)));
  EXPECT_THAT(aggregated.decode_tokens_per_sec,
              testing::ElementsAre(MedianOf(decode_speeds)));
  ASSERT_TRUE(aggregated.time_to_first_token_sec.has_value());
  EXPECT_EQ(*aggregated.time_to_first_token_sec,
            MedianOf(times_to_first_token));
}

TEST(ComputeMedianMetricsTest, EvenNumberOfIterationsAveragesMiddleValues) {
  const std::vector<LitertLmMetrics> metrics = {
      CreateMetrics(absl::Milliseconds(100), /*peak_mem_mb=*/100.0f,
                    /*peak_private_mb=*/10.0f, /*peak_accelerator_mb=*/10.0f),
      CreateMetrics(absl::Milliseconds(400), /*peak_mem_mb=*/400.0f,
                    /*peak_private_mb=*/40.0f, /*peak_accelerator_mb=*/40.0f),
      CreateMetrics(absl::Milliseconds(200), /*peak_mem_mb=*/200.0f,
                    /*peak_private_mb=*/20.0f, /*peak_accelerator_mb=*/20.0f),
      CreateMetrics(absl::Milliseconds(300), /*peak_mem_mb=*/300.0f,
                    /*peak_private_mb=*/30.0f, /*peak_accelerator_mb=*/30.0f),
  };

  const AggregatedLitertLmMetrics aggregated = ComputeMedianMetrics(metrics);

  EXPECT_EQ(aggregated.num_iterations, 4);
  EXPECT_THAT(
      aggregated.init_phases,
      testing::ElementsAre(testing::Pair(
          BenchmarkInfo::InitPhaseToString(BenchmarkInfo::InitPhase::kExecutor),
          absl::Milliseconds(250))));
  EXPECT_EQ(aggregated.peak_mem_mb, 250.0f);
  EXPECT_EQ(aggregated.peak_private_mb, 25.0f);
  EXPECT_EQ(aggregated.peak_accelerator_mb, 25.0f);
}

TEST(ComputeMedianMetricsTest, MetricsWithoutBenchmarkInfoAreSkipped) {
  std::vector<LitertLmMetrics> metrics = {
      CreateMetrics(absl::Milliseconds(100), /*peak_mem_mb=*/0.0f,
                    /*peak_private_mb=*/0.0f),
      CreateMetrics(absl::Milliseconds(200), /*peak_mem_mb=*/0.0f,
                    /*peak_private_mb=*/0.0f),
  };
  metrics.push_back(LitertLmMetrics());

  const AggregatedLitertLmMetrics aggregated = ComputeMedianMetrics(metrics);

  EXPECT_EQ(aggregated.num_iterations, 3);
  EXPECT_THAT(
      aggregated.init_phases,
      testing::ElementsAre(testing::Pair(
          BenchmarkInfo::InitPhaseToString(BenchmarkInfo::InitPhase::kExecutor),
          absl::Milliseconds(150))));
  // Peak memory is not reported unless --report_peak_memory_footprint is set.
  EXPECT_FALSE(aggregated.peak_mem_mb.has_value());
  EXPECT_FALSE(aggregated.peak_private_mb.has_value());
  EXPECT_FALSE(aggregated.peak_accelerator_mb.has_value());
  EXPECT_THAT(aggregated.peak_accelerator_breakdown_mb, testing::IsEmpty());
}

// Returns a LitertLmMetrics with only the accelerator memory fields populated.
LitertLmMetrics CreateAcceleratorMetrics(float peak_accelerator_mb,
                                         float local_mb, float non_local_mb) {
  LitertLmMetrics metrics;
  metrics.peak_accelerator_mb = peak_accelerator_mb;
  metrics.peak_accelerator_breakdown_mb = {{"local", local_mb},
                                           {"non-local", non_local_mb}};
  return metrics;
}

TEST(ComputeMedianMetricsTest, AcceleratorMemoryMedianIncludesBreakdown) {
  const std::vector<LitertLmMetrics> metrics = {
      CreateAcceleratorMetrics(/*peak_accelerator_mb=*/300.0f,
                               /*local_mb=*/250.0f, /*non_local_mb=*/50.0f),
      CreateAcceleratorMetrics(/*peak_accelerator_mb=*/100.0f,
                               /*local_mb=*/80.0f, /*non_local_mb=*/20.0f),
      CreateAcceleratorMetrics(/*peak_accelerator_mb=*/200.0f,
                               /*local_mb=*/150.0f, /*non_local_mb=*/50.0f),
  };

  const AggregatedLitertLmMetrics aggregated = ComputeMedianMetrics(metrics);

  EXPECT_EQ(aggregated.num_iterations, 3);
  EXPECT_EQ(aggregated.peak_accelerator_mb, 200.0f);
  // Each category is aggregated independently.
  EXPECT_THAT(aggregated.peak_accelerator_breakdown_mb,
              testing::ElementsAre(testing::Pair("local", 150.0f),
                                   testing::Pair("non-local", 50.0f)));
}

TEST(ComputeMedianMetricsTest, AcceleratorMemoryIgnoresUnsupportedIterations) {
  std::vector<LitertLmMetrics> metrics = {
      CreateAcceleratorMetrics(/*peak_accelerator_mb=*/400.0f,
                               /*local_mb=*/400.0f, /*non_local_mb=*/0.0f),
      CreateAcceleratorMetrics(/*peak_accelerator_mb=*/200.0f,
                               /*local_mb=*/200.0f, /*non_local_mb=*/0.0f),
  };
  // An iteration on which accelerator memory tracking was unsupported leaves
  // the metric unset and must not drag the median down.
  metrics.push_back(LitertLmMetrics());

  const AggregatedLitertLmMetrics aggregated = ComputeMedianMetrics(metrics);

  EXPECT_EQ(aggregated.num_iterations, 3);
  EXPECT_EQ(aggregated.peak_accelerator_mb, 300.0f);
  EXPECT_THAT(aggregated.peak_accelerator_breakdown_mb,
              testing::ElementsAre(testing::Pair("local", 300.0f),
                                   testing::Pair("non-local", 0.0f)));
}

TEST(ComputeMedianMetricsTest, AcceleratorMemoryAggregatesSupportedZero) {
  // Zero is a real measurement on a supported platform (e.g. a CPU-only run)
  // and must be aggregated rather than treated as unsupported.
  const std::vector<LitertLmMetrics> metrics = {
      CreateAcceleratorMetrics(/*peak_accelerator_mb=*/0.0f,
                               /*local_mb=*/0.0f, /*non_local_mb=*/0.0f),
      CreateAcceleratorMetrics(/*peak_accelerator_mb=*/0.0f,
                               /*local_mb=*/0.0f, /*non_local_mb=*/0.0f),
      CreateAcceleratorMetrics(/*peak_accelerator_mb=*/90.0f,
                               /*local_mb=*/90.0f, /*non_local_mb=*/0.0f),
  };

  const AggregatedLitertLmMetrics aggregated = ComputeMedianMetrics(metrics);

  EXPECT_EQ(aggregated.peak_accelerator_mb, 0.0f);
  EXPECT_THAT(aggregated.peak_accelerator_breakdown_mb,
              testing::ElementsAre(testing::Pair("local", 0.0f),
                                   testing::Pair("non-local", 0.0f)));
}

// Following tests are for various model file metadata and tokenizer types.
// They are not exhaustive, but designed to test a variety of scenarios.
// If metadata or tokenizer types are not handled properly, these tests could
// fail.
TEST(LiteRtLmLibTest, RunLiteRtLmWithValidModelPath) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.backend = "cpu";
  // To save time on testing, and make sure we can end gracefully with this
  // test litertlm file, we only run 32 tokens.
  settings.max_num_tokens = 32;
  EXPECT_OK(RunLiteRtLm(settings));
}

TEST(LiteRtLmLibTest, CreateEngineSuccess) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.backend = "cpu";

  auto engine_settings_or = CreateEngineSettings(settings);
  ASSERT_OK(engine_settings_or.status());

  auto engine_or = CreateEngine(settings, *engine_settings_or);
  ASSERT_OK(engine_or.status());
  EXPECT_NE(*engine_or, nullptr);
}

TEST(LiteRtLmLibTest, CreateEngineSettings_Backends) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.backend = "gpu";
  settings.vision_backend = "cpu";
  settings.audio_backend = "gpu";

  auto engine_settings_or = CreateEngineSettings(settings);
  ASSERT_OK(engine_settings_or.status());
  const auto& engine_settings = *engine_settings_or;

  EXPECT_EQ(engine_settings.GetMainExecutorSettings().GetBackend(),
            Backend::GPU);
  ASSERT_TRUE(engine_settings.GetVisionExecutorSettings().has_value());
  EXPECT_EQ(engine_settings.GetVisionExecutorSettings()->GetBackend(),
            Backend::CPU);
  ASSERT_TRUE(engine_settings.GetAudioExecutorSettings().has_value());
  EXPECT_EQ(engine_settings.GetAudioExecutorSettings()->GetBackend(),
            Backend::GPU);
}

TEST(LiteRtLmLibTest, CreateEngineSettings_ActivationDataType) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.force_f32 = true;
  settings.vision_backend = "cpu";
  settings.audio_backend = "cpu";

  auto engine_settings_or = CreateEngineSettings(settings);
  ASSERT_OK(engine_settings_or.status());
  const auto& engine_settings = *engine_settings_or;

  EXPECT_EQ(engine_settings.GetMainExecutorSettings().GetActivationDataType(),
            ActivationDataType::FLOAT32);
  EXPECT_EQ(
      engine_settings.GetVisionExecutorSettings()->GetActivationDataType(),
      ActivationDataType::FLOAT32);
  EXPECT_EQ(engine_settings.GetAudioExecutorSettings()->GetActivationDataType(),
            ActivationDataType::FLOAT32);
}

TEST(LiteRtLmLibTest, CreateEngineSettings_MaxNumTokens) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.max_num_tokens = 1024;

  auto engine_settings_or = CreateEngineSettings(settings);
  ASSERT_OK(engine_settings_or.status());
  EXPECT_EQ(engine_settings_or->GetMainExecutorSettings().GetMaxNumTokens(),
            1024);
}

TEST(LiteRtLmLibTest, CreateEngineSettings_VisualTokenBudget) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.visual_token_budget = 200;

  auto engine_settings = CreateEngineSettings(settings);
  ASSERT_OK(engine_settings.status());
  EXPECT_EQ(engine_settings->GetMaxVisionTokensPerImage(), 200);
}

TEST(LiteRtLmLibTest, CreateEngineSettings_CacheConfig) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.cache_dir = "/tmp/cache";

  auto engine_settings_or = CreateEngineSettings(settings);
  ASSERT_OK(engine_settings_or.status());
  EXPECT_EQ(engine_settings_or->GetMainExecutorSettings().GetCacheDir(),
            "/tmp/cache");

  settings.disable_cache = true;
  engine_settings_or = CreateEngineSettings(settings);
  ASSERT_OK(engine_settings_or.status());
  EXPECT_EQ(engine_settings_or->GetMainExecutorSettings().GetCacheDir(),
            ":nocache");
}

TEST(LiteRtLmLibTest, CreateEngineSettings_DisableProgramCache) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.backend = "gpu";
  settings.vision_backend = "cpu";
  settings.audio_backend = "gpu";
  settings.disable_gpu_program_cache = true;

  auto engine_settings_or = CreateEngineSettings(settings);
  ASSERT_OK(engine_settings_or.status());
  const auto& engine_settings = *engine_settings_or;

  EXPECT_TRUE(
      engine_settings.GetMainExecutorSettings().IsProgramCacheDisabled());
  EXPECT_FALSE(
      engine_settings.GetMainExecutorSettings().IsWeightCacheDisabled());

  ASSERT_TRUE(engine_settings.GetVisionExecutorSettings().has_value());
  EXPECT_TRUE(
      engine_settings.GetVisionExecutorSettings()->IsProgramCacheDisabled());
  EXPECT_FALSE(
      engine_settings.GetVisionExecutorSettings()->IsWeightCacheDisabled());

  ASSERT_TRUE(engine_settings.GetAudioExecutorSettings().has_value());
  EXPECT_TRUE(
      engine_settings.GetAudioExecutorSettings()->IsProgramCacheDisabled());
  EXPECT_FALSE(
      engine_settings.GetAudioExecutorSettings()->IsWeightCacheDisabled());
}

TEST(LiteRtLmLibTest, CreateEngineSettings_DisableWeightCache) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.backend = "gpu";
  settings.vision_backend = "cpu";
  settings.audio_backend = "gpu";
  settings.disable_weight_cache = true;

  auto engine_settings_or = CreateEngineSettings(settings);
  ASSERT_OK(engine_settings_or.status());
  const auto& engine_settings = *engine_settings_or;

  EXPECT_TRUE(
      engine_settings.GetMainExecutorSettings().IsWeightCacheDisabled());
  EXPECT_FALSE(
      engine_settings.GetMainExecutorSettings().IsProgramCacheDisabled());

  ASSERT_TRUE(engine_settings.GetVisionExecutorSettings().has_value());
  EXPECT_TRUE(
      engine_settings.GetVisionExecutorSettings()->IsWeightCacheDisabled());
  EXPECT_FALSE(
      engine_settings.GetVisionExecutorSettings()->IsProgramCacheDisabled());

  ASSERT_TRUE(engine_settings.GetAudioExecutorSettings().has_value());
  EXPECT_TRUE(
      engine_settings.GetAudioExecutorSettings()->IsWeightCacheDisabled());
  EXPECT_FALSE(
      engine_settings.GetAudioExecutorSettings()->IsProgramCacheDisabled());
}

TEST(LiteRtLmLibTest, CreateEngineSettings_GpuUseIrModel) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.backend = "gpu";
  settings.vision_backend = "gpu";
  settings.audio_backend = "gpu";
  settings.gpu_use_ir_model = true;

  auto engine_settings_or = CreateEngineSettings(settings);
  ASSERT_OK(engine_settings_or.status());
  const auto& engine_settings = *engine_settings_or;

  EXPECT_THAT(engine_settings.GetMainExecutorSettings().UseIrModel(),
              Optional(true));
  ASSERT_TRUE(engine_settings.GetVisionExecutorSettings().has_value());
  EXPECT_THAT(engine_settings.GetVisionExecutorSettings()->UseIrModel(),
              Optional(true));
  ASSERT_TRUE(engine_settings.GetAudioExecutorSettings().has_value());
  EXPECT_THAT(engine_settings.GetAudioExecutorSettings()->UseIrModel(),
              Optional(true));
}

TEST(LiteRtLmLibTest, CreateEngineSettings_GpuUseIrModelDefault) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.backend = "gpu";

  auto engine_settings_or = CreateEngineSettings(settings);
  ASSERT_OK(engine_settings_or.status());
  EXPECT_EQ(engine_settings_or->GetMainExecutorSettings().UseIrModel(),
            std::nullopt);
}

TEST(LiteRtLmLibTest, CreateEngineSettings_CpuConfig) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.backend = "cpu";
  settings.num_cpu_threads = 8;
  settings.enable_ynnpack = true;
  settings.prefill_chunk_size = 512;

  auto engine_settings_or = CreateEngineSettings(settings);
  ASSERT_OK(engine_settings_or.status());
  auto cpu_config_or = engine_settings_or->GetMainExecutorSettings()
                           .GetBackendConfig<CpuConfig>();
  ASSERT_OK(cpu_config_or.status());
  EXPECT_EQ(cpu_config_or->number_of_threads, 8);
  EXPECT_TRUE(cpu_config_or->enable_ynnpack);
  EXPECT_EQ(cpu_config_or->prefill_chunk_size, 512);
}

TEST(LiteRtLmLibTest, CreateEngineSettings_GpuConfig) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.backend = "gpu";
  settings.gpu_external_tensor_mode = true;

  auto engine_settings_or = CreateEngineSettings(settings);
  ASSERT_OK(engine_settings_or.status());
  auto gpu_config_or = engine_settings_or->GetMainExecutorSettings()
                           .GetBackendConfig<GpuConfig>();
  ASSERT_OK(gpu_config_or.status());
  EXPECT_TRUE(gpu_config_or->external_tensor_mode);
}

TEST(LiteRtLmLibTest, CreateEngineSettings_AdvancedSettings) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.num_output_candidates = 4;
  settings.conv_type = ConvType::kInt8;
  settings.prefill_batch_sizes = {1, 4, 16};
  settings.enable_profiling = true;

  auto engine_settings_or = CreateEngineSettings(settings);
  ASSERT_OK(engine_settings_or.status());
  const auto& advanced =
      engine_settings_or->GetMainExecutorSettings().GetAdvancedSettings();
  ASSERT_TRUE(advanced.has_value());
  EXPECT_EQ(advanced->num_output_candidates, 4);
  EXPECT_THAT(advanced->prefill_batch_sizes, ::testing::ElementsAre(1, 4, 16));
  EXPECT_TRUE(advanced->allow_src_quantized_fc_conv_ops.value_or(false));
  EXPECT_TRUE(advanced->enable_profiling);
}

TEST(LiteRtLmLibTest, CreateEngineSettings_SpeculativeDecoding) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();

  // By default, the engine's mode is unset (off) and the policy keeps its
  // defaults.
  ASSERT_OK_AND_ASSIGN(EngineSettings engine_settings,
                       CreateEngineSettings(settings));
  EXPECT_EQ(engine_settings.GetSpeculativeDecodingConfig().mode, std::nullopt);
  EXPECT_EQ(engine_settings.GetMainExecutorSettings()
                .GetEffectiveSpeculativeDecodingMode(),
            SpeculativeDecodingMode::kOff);
  EXPECT_EQ(engine_settings.GetMainExecutorSettings()
                .GetAdvancedSettings()
                .value_or(AdvancedSettings())
                .mtp_gating_config,
            std::nullopt);

  // The mode is the engine's default.
  settings.speculative_decoding_mode = SpeculativeDecodingMode::kAdaptive;
  ASSERT_OK_AND_ASSIGN(engine_settings, CreateEngineSettings(settings));
  EXPECT_EQ(engine_settings.GetSpeculativeDecodingConfig().mode,
            SpeculativeDecodingMode::kAdaptive);

  // The deprecated boolean means kOn, unless a mode is set.
  settings.enable_speculative_decoding = true;
  ASSERT_OK_AND_ASSIGN(engine_settings, CreateEngineSettings(settings));
  EXPECT_EQ(engine_settings.GetSpeculativeDecodingConfig().mode,
            SpeculativeDecodingMode::kAdaptive);
  settings.speculative_decoding_mode = std::nullopt;
  ASSERT_OK_AND_ASSIGN(engine_settings, CreateEngineSettings(settings));
  EXPECT_EQ(engine_settings.GetSpeculativeDecodingConfig().mode,
            SpeculativeDecodingMode::kOn);

  settings.mtp_gating_half_life = 8;
  settings.mtp_gating_forced_switch_period = 16;
  ASSERT_OK_AND_ASSIGN(engine_settings, CreateEngineSettings(settings));
  const auto& advanced =
      engine_settings.GetMainExecutorSettings().GetAdvancedSettings();
  ASSERT_TRUE(advanced.has_value());
  // Unset flags keep the policy defaults.
  MtpGatingConfig expected_config;
  expected_config.ema_half_life_steps = 8;
  expected_config.forced_switch_period = 16;
  EXPECT_EQ(advanced->mtp_gating_config, expected_config);
}

TEST(LiteRtLmLibTest, CreateEngineSettings_InvalidMtpGatingOverride) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.mtp_gating_probe_length = 0;
  EXPECT_THAT(CreateEngineSettings(settings),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(LiteRtLmLibTest, CreateEngineSettings_SamplerSeedRequiresSamplerOverride) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  // On its own, the seed would be replaced by the model's default sampler
  // parameters, so it is rejected rather than silently ignored.
  settings.sampler_seed = 7;
  EXPECT_THAT(CreateEngineSettings(settings),
              StatusIs(absl::StatusCode::kInvalidArgument));
  settings.sampler_top_k = 40;
  EXPECT_OK(CreateEngineSettings(settings));
}

TEST(LiteRtLmLibTest, CreateEngineSettings_RejectsInvalidSamplerOverrides) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings base;
  base.model_path = model_path.string();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();

  for (int top_k : {0, -1}) {
    LiteRtLmSettings settings = base;
    settings.sampler_top_k = top_k;
    EXPECT_THAT(CreateEngineSettings(settings),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << top_k;
  }
  for (double top_p : {-0.1, 1.1, nan}) {
    LiteRtLmSettings settings = base;
    settings.sampler_top_p = top_p;
    EXPECT_THAT(CreateEngineSettings(settings),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << top_p;
  }
  for (double temperature : {-0.1, nan, inf}) {
    LiteRtLmSettings settings = base;
    settings.sampler_temperature = temperature;
    EXPECT_THAT(CreateEngineSettings(settings),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << temperature;
  }

  // Boundary values are accepted.
  LiteRtLmSettings settings = base;
  settings.sampler_top_k = 1;
  settings.sampler_top_p = 0.0;
  settings.sampler_temperature = 0.0;
  EXPECT_OK(CreateEngineSettings(settings));
  settings.sampler_top_p = 1.0;
  EXPECT_OK(CreateEngineSettings(settings));
}

TEST(LiteRtLmLibTest, CreateEngineSettings_SamplerTopKRaisesGpuMaxTopK) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.backend = "gpu";

  ASSERT_OK_AND_ASSIGN(EngineSettings engine_settings,
                       CreateEngineSettings(settings));
  ASSERT_OK_AND_ASSIGN(
      GpuConfig gpu_config,
      engine_settings.GetMainExecutorSettings().GetBackendConfig<GpuConfig>());
  const uint32_t default_max_top_k = gpu_config.max_top_k;

  // A smaller k keeps the default capacity.
  settings.sampler_top_k = 1;
  ASSERT_OK_AND_ASSIGN(engine_settings, CreateEngineSettings(settings));
  ASSERT_OK_AND_ASSIGN(
      gpu_config,
      engine_settings.GetMainExecutorSettings().GetBackendConfig<GpuConfig>());
  EXPECT_EQ(gpu_config.max_top_k, default_max_top_k);

  settings.sampler_top_k = default_max_top_k + 36;
  ASSERT_OK_AND_ASSIGN(engine_settings, CreateEngineSettings(settings));
  ASSERT_OK_AND_ASSIGN(
      gpu_config,
      engine_settings.GetMainExecutorSettings().GetBackendConfig<GpuConfig>());
  EXPECT_EQ(gpu_config.max_top_k, default_max_top_k + 36);
}

TEST(LiteRtLmLibTest, CreateEngineSettings_InvalidSpeculativeLenience) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  for (const float lenience :
       {-0.1f, 1.1f, std::numeric_limits<float>::quiet_NaN()}) {
    settings.speculative_lenience = lenience;
    EXPECT_THAT(CreateEngineSettings(settings),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << lenience;
  }
  for (const float lenience : {0.0f, 0.5f, 1.0f}) {
    settings.speculative_lenience = lenience;
    EXPECT_OK(CreateEngineSettings(settings)) << lenience;
  }
}

TEST(LiteRtLmLibTest, CreateEngineSettings_BenchmarkParams) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.benchmark = true;
  settings.benchmark_prefill_tokens = 128;
  settings.benchmark_decode_tokens = 64;

  auto engine_settings_or = CreateEngineSettings(settings);
  ASSERT_OK(engine_settings_or.status());
  ASSERT_TRUE(engine_settings_or->IsBenchmarkEnabled());
  ASSERT_TRUE(engine_settings_or->GetBenchmarkParams().has_value());
  EXPECT_EQ(engine_settings_or->GetBenchmarkParams()->num_prefill_tokens(),
            128);
  EXPECT_EQ(engine_settings_or->GetBenchmarkParams()->num_decode_tokens(), 64);
}

TEST(LiteRtLmLibTest, CreateSessionConfig_Default) {
  LiteRtLmSettings settings;
  SessionConfig config = CreateSessionConfig(settings);

  EXPECT_EQ(config.GetNumOutputCandidates(), 1);
  EXPECT_EQ(config.GetSamplerBackend(), Backend::UNSPECIFIED);
  EXPECT_FALSE(config.VisionModalityEnabled());
  EXPECT_FALSE(config.AudioModalityEnabled());
}

TEST(LiteRtLmLibTest, CreateSessionConfig_InheritsSpeculativeDecodingMode) {
  // The mode is the engine's default; the session inherits it.
  LiteRtLmSettings settings;
  settings.speculative_decoding_mode = SpeculativeDecodingMode::kAdaptive;
  EXPECT_EQ(CreateSessionConfig(settings).GetSpeculativeDecodingConfig().mode,
            std::nullopt);
}

TEST(LiteRtLmLibTest, CreateEngineSettings_SpeculativeLenience) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  ASSERT_OK_AND_ASSIGN(EngineSettings engine_settings,
                       CreateEngineSettings(settings));
  EXPECT_EQ(engine_settings.GetSpeculativeDecodingConfig().lenience,
            std::nullopt);

  // The lenience is the engine's default; the session inherits it.
  settings.speculative_lenience = 0.25f;
  ASSERT_OK_AND_ASSIGN(engine_settings, CreateEngineSettings(settings));
  EXPECT_EQ(engine_settings.GetSpeculativeDecodingConfig().lenience, 0.25f);
  EXPECT_EQ(
      CreateSessionConfig(settings).GetSpeculativeDecodingConfig().lenience,
      std::nullopt);
}

TEST(LiteRtLmLibTest, CreateSessionConfig_SamplerOverrides) {
  LiteRtLmSettings settings;
  EXPECT_EQ(CreateSessionConfig(settings).GetSamplerParams().type(),
            proto::SamplerParameters::TYPE_UNSPECIFIED);

  settings.sampler_top_k = 40;
  settings.sampler_temperature = 0.8;
  settings.sampler_seed = 7;
  const proto::SamplerParameters params =
      CreateSessionConfig(settings).GetSamplerParams();
  EXPECT_EQ(params.type(), proto::SamplerParameters::TOP_P);
  EXPECT_EQ(params.k(), 40);
  EXPECT_FLOAT_EQ(params.p(), 1.0f);
  EXPECT_FLOAT_EQ(params.temperature(), 0.8f);
  EXPECT_EQ(params.seed(), 7);

  // `--sampler_top_p` / `--sampler_temperature` without `--sampler_top_k`
  // request stochastic sampling, so top-k must not default to 1 (greedy).
  LiteRtLmSettings temperature_only;
  temperature_only.sampler_temperature = 0.8;
  const proto::SamplerParameters temperature_params =
      CreateSessionConfig(temperature_only).GetSamplerParams();
  EXPECT_EQ(temperature_params.type(), proto::SamplerParameters::TOP_P);
  EXPECT_EQ(temperature_params.k(), 40);
  EXPECT_FLOAT_EQ(temperature_params.p(), 1.0f);
  EXPECT_FLOAT_EQ(temperature_params.temperature(), 0.8f);
}

TEST(LiteRtLmLibTest, RunLiteRtLmWithInferredGemma3ModelType) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_no_model_type.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.backend = "cpu";
  // To save time on testing, and make sure we can end gracefully with this
  // test litertlm file, we only run 32 tokens.
  settings.max_num_tokens = 32;
  EXPECT_OK(RunLiteRtLm(settings));
}

TEST(LiteRtLmLibTest, RunLiteRtLmWithDeepseekMetadataTokenizer) {
  const auto model_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_deepseek_metadata_tokenizer.litertlm";
  LiteRtLmSettings settings;
  settings.model_path = model_path.string();
  settings.backend = "cpu";
  // To save time on testing, and make sure we can end gracefully with this
  // test litertlm file, we only run 32 tokens.
  settings.max_num_tokens = 32;
  EXPECT_OK(RunLiteRtLm(settings));
}
TEST(PrintMessageTest, ContentArray) {
  json message = {
      {"role", "assistant"},
      {"content", json::array({{{"type", "text"}, {"text", "Hello "}},
                               {{"type", "text"}, {"text", "world!"}}})}};
  std::stringstream captured_output;
  CooptStdout coopt;
  EXPECT_OK(PrintMessage(message, captured_output));
  EXPECT_EQ(captured_output.str(), "Hello world!\n");
  EXPECT_EQ(coopt.GetOutput(),
            ColorYellow("Hello ") + ColorYellow("world!") + "\n");
}

TEST(PrintMessageTest, ContentObject) {
  json message = {{"role", "assistant"},
                  {"content", {{"type", "text"}, {"text", "Hello object!"}}}};
  std::stringstream captured_output;
  CooptStdout coopt;
  EXPECT_OK(PrintMessage(message, captured_output));
  EXPECT_EQ(captured_output.str(), "Hello object!\n");
  EXPECT_EQ(coopt.GetOutput(), ColorYellow("Hello object!") + "\n");
}

TEST(PrintMessageTest, ContentString) {
  json message = {{"role", "assistant"}, {"content", "Hello string!"}};
  std::stringstream captured_output;
  CooptStdout coopt;
  EXPECT_OK(PrintMessage(message, captured_output));
  EXPECT_EQ(captured_output.str(), "Hello string!\n");
  EXPECT_EQ(coopt.GetOutput(), ColorYellow("Hello string!") + "\n");
}

TEST(PrintMessageTest, ChannelsStreaming) {
  json message = {{"role", "assistant"},
                  {"channels", {{"thought", "Thinking... "}}}};
  std::stringstream captured_output;
  std::string active_channel;
  CooptStdout coopt;
  EXPECT_OK(PrintMessage(message, captured_output, &active_channel,
                         /*streaming=*/true));
  EXPECT_EQ(captured_output.str(), "Thinking... ");
  EXPECT_EQ(coopt.GetOutput(),
            ColorBlue("[thought] ") + ColorBlue("Thinking... "));
  EXPECT_EQ(active_channel, "thought");

  // Next chunk in same channel
  json message2 = {{"role", "assistant"},
                   {"channels", {{"thought", "more thinking."}}}};
  EXPECT_OK(PrintMessage(message2, captured_output, &active_channel,
                         /*streaming=*/true));
  EXPECT_EQ(captured_output.str(), "Thinking... more thinking.");
  EXPECT_EQ(coopt.GetOutput(), ColorBlue("[thought] ") +
                                   ColorBlue("Thinking... ") +
                                   ColorBlue("more thinking."));
  EXPECT_EQ(active_channel, "thought");

  // Switch channel
  json message3 = {{"role", "assistant"},
                   {"channels", {{"answer", "Final answer."}}}};
  EXPECT_OK(PrintMessage(message3, captured_output, &active_channel,
                         /*streaming=*/true));
  EXPECT_EQ(captured_output.str(), "Thinking... more thinking.Final answer.");
  EXPECT_EQ(coopt.GetOutput(),
            ColorBlue("[thought] ") + ColorBlue("Thinking... ") +
                ColorBlue("more thinking.") + ColorBlue("[/thought]") + "\n" +
                ColorBlue("[answer] ") + ColorBlue("Final answer."));
  EXPECT_EQ(active_channel, "answer");
}

TEST(PrintMessageTest, ChannelsNonStreaming) {
  json message = {{"role", "assistant"},
                  {"channels", {{"thought", "Thinking process."}}}};
  std::stringstream captured_output;
  CooptStdout coopt;
  EXPECT_OK(
      PrintMessage(message, captured_output, nullptr, /*streaming=*/false));
  EXPECT_EQ(captured_output.str(), "Thinking process.\n");
  EXPECT_EQ(coopt.GetOutput(),
            ColorBlue("[thought] Thinking process.[/thought]") + "\n");
}

TEST(PrintMessageTest, BothContentAndChannels) {
  json message = {{"role", "assistant"},
                  {"content", "Final answer."},
                  {"channels", {{"thought", "Thinking process."}}}};
  std::stringstream captured_output;
  CooptStdout coopt;
  EXPECT_OK(
      PrintMessage(message, captured_output, nullptr, /*streaming=*/false));
  EXPECT_EQ(captured_output.str(), "Thinking process.Final answer.\n");
  EXPECT_EQ(coopt.GetOutput(),
            ColorBlue("[thought] Thinking process.[/thought]") + "\n" +
                ColorYellow("Final answer.") + "\n");
}

TEST(PrintMessageTest, ToolCall) {
  json message = {{"role", "assistant"}, {"tool_calls", json::array()}};
  std::stringstream captured_output;
  CooptStdout coopt;
  EXPECT_OK(PrintMessage(message, captured_output));
  EXPECT_EQ(captured_output.str(), "");
  EXPECT_EQ(coopt.GetOutput(), "");
}

TEST(PrintMessageTest, InvalidMessage) {
  json message = {{"role", "assistant"}};
  std::stringstream captured_output;
  EXPECT_THAT(PrintMessage(message, captured_output),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(PrintMessageTest, StreamingTransitionChannelToContent) {
  std::stringstream captured_output;
  std::string active_channel = "thought";
  CooptStdout coopt;

  json message = {{"role", "assistant"}, {"content", "Final answer."}};
  EXPECT_OK(PrintMessage(message, captured_output, &active_channel,
                         /*streaming=*/true));
  EXPECT_EQ(captured_output.str(), "Final answer.");
  EXPECT_EQ(coopt.GetOutput(),
            ColorBlue("[/thought]") + "\n" + ColorYellow("Final answer."));
  EXPECT_EQ(active_channel, "");
}

}  // namespace
}  // namespace lm
}  // namespace litert
