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

#include <filesystem>  // NOLINT
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/escaping.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert {
namespace lm {
namespace {

using ::nlohmann::json;
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
