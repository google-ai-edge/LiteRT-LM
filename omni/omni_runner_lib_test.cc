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

#include "omni/omni_runner_lib.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>  // NOLINT: Required for std::error_code.
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/reflection.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/model_utils.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_runner_flags.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/proto/any2any_session_config.pb.h"
#include "runtime/proto/speech2text_session_config.pb.h"
#include "runtime/proto/text2image_session_config.pb.h"
#include "runtime/proto/text2speech_session_config.pb.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK, ASSERT_OK_AND_ASSIGN

namespace litert::omni {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Optional;
using ::testing::StartsWith;

// Writes a minimal mono 16-bit PCM WAV file with `num_samples` samples.
absl::Status WriteTestWavFile(absl::string_view path, int sample_rate_hz,
                              int num_samples) {
  const uint16_t num_channels = 1;
  const uint16_t bits_per_sample = 16;
  const uint32_t byte_rate = static_cast<uint32_t>(sample_rate_hz) *
                             num_channels * bits_per_sample / 8;
  const uint16_t block_align = num_channels * bits_per_sample / 8;
  const uint32_t data_size = static_cast<uint32_t>(num_samples) * block_align;
  const uint32_t riff_size = 36 + data_size;

  std::ofstream out(std::string(path), std::ios::binary);
  if (!out.is_open()) {
    return absl::InternalError(
        absl::StrCat("Failed to open test WAV file for writing: ", path));
  }
  out.write("RIFF", 4);
  out.write(reinterpret_cast<const char*>(&riff_size), 4);
  out.write("WAVE", 4);
  out.write("fmt ", 4);
  const uint32_t fmt_size = 16;
  const uint16_t audio_format = 1;  // PCM.
  out.write(reinterpret_cast<const char*>(&fmt_size), 4);
  out.write(reinterpret_cast<const char*>(&audio_format), 2);
  out.write(reinterpret_cast<const char*>(&num_channels), 2);
  const uint32_t sr = static_cast<uint32_t>(sample_rate_hz);
  out.write(reinterpret_cast<const char*>(&sr), 4);
  out.write(reinterpret_cast<const char*>(&byte_rate), 4);
  out.write(reinterpret_cast<const char*>(&block_align), 2);
  out.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
  out.write("data", 4);
  out.write(reinterpret_cast<const char*>(&data_size), 4);
  std::vector<int16_t> samples(num_samples, 1000);
  out.write(reinterpret_cast<const char*>(samples.data()), data_size);
  out.flush();
  if (!out.good()) {
    return absl::InternalError(
        absl::StrCat("Failed to write test WAV file: ", path));
  }
  return absl::OkStatus();
}

struct CapturedRun {
  std::string model_name;
  OmniEngine::Settings settings;
  std::vector<OmniSession::Input> inputs;
  lm::proto::Any2AnySessionConfig any2any_config;
  lm::proto::Speech2TextSessionConfig asr_config;
  lm::proto::Text2SpeechSessionConfig tts_config;
  lm::proto::Text2ImageSessionConfig t2i_config;
};

struct FakeSessionBehavior {
  std::vector<absl::StatusOr<OmniSession::Output>> process_results;
  absl::StatusOr<OmniSession::Output> flush_result = TextOutput{};
};

class FakeOmniSession : public OmniSession {
 public:
  FakeOmniSession(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
      FakeSessionBehavior behavior, CapturedRun* captured)
      : input_source_(std::move(input_source)),
        behavior_(std::move(behavior)),
        captured_(captured) {}

  void Reset() override { next_index_ = 0; }

  absl::StatusOr<Output> ProcessNext() override {
    DrainInputs();
    if (next_index_ < behavior_.process_results.size()) {
      return behavior_.process_results[next_index_++];
    }
    return absl::OutOfRangeError("End of fake stream.");
  }

  absl::StatusOr<Output> Flush() override {
    DrainInputs();
    return behavior_.flush_result;
  }

  absl::Status ProcessAsync(OutputCallback /*callback*/) override {
    return absl::UnimplementedError("ProcessAsync not used by runner.");
  }

 private:
  void DrainInputs() {
    if (captured_ == nullptr) {
      return;
    }
    while (true) {
      auto in = input_source_->GetOutput();
      if (!in.ok()) {
        break;
      }
      captured_->inputs.push_back(*std::move(in));
    }
  }

  std::unique_ptr<OmniSession::InputSource> input_source_;
  FakeSessionBehavior behavior_;
  CapturedRun* captured_;
  size_t next_index_ = 0;
};

class FakeOmniSessionFactory : public OmniSessionFactory {
 public:
  FakeOmniSessionFactory(FakeSessionBehavior behavior, CapturedRun* captured)
      : behavior_(std::move(behavior)), captured_(captured) {}

  absl::StatusOr<std::unique_ptr<OmniSession>> Create(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source)
      override {
    return std::make_unique<FakeOmniSession>(std::move(input_source), behavior_,
                                             captured_);
  }

 private:
  FakeSessionBehavior behavior_;
  CapturedRun* captured_;
};

absl::Status RegisterFakeModel(
    const std::string& model_name, absl::string_view session_config_type_id,
    FakeSessionBehavior behavior, CapturedRun* captured,
    std::optional<std::string> model_remote_url = std::nullopt,
    std::optional<std::string> metadata_type_id = std::nullopt) {
  return OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {model_name},
      .get_session_factory =
          [behavior = std::move(behavior), captured](
              absl::string_view resolved_name,
              const OmniEngine::Settings& settings,
              const MessageLite* absl_nullable session_cfg,
              const MessageLite* absl_nullable /*metadata*/,
              ::litert::Environment& /*env*/, ::litert::Options& /*options*/,
              std::shared_ptr<lm::ModelResources> absl_nullable
              /*lm_resources*/)
          -> absl::StatusOr<std::unique_ptr<OmniSessionFactory>> {
        if (captured != nullptr) {
          captured->model_name = std::string(resolved_name);
          captured->settings = settings;
          captured->inputs.clear();
          if (session_cfg != nullptr) {
            const std::string type_name(session_cfg->GetTypeName());
            if (type_name == "litert.lm.proto.Any2AnySessionConfig") {
              captured->any2any_config.CheckTypeAndMergeFrom(*session_cfg);
            } else if (type_name ==
                       "litert.lm.proto.Speech2TextSessionConfig") {
              captured->asr_config.CheckTypeAndMergeFrom(*session_cfg);
            } else if (type_name ==
                       "litert.lm.proto.Text2SpeechSessionConfig") {
              captured->tts_config.CheckTypeAndMergeFrom(*session_cfg);
            } else if (type_name == "litert.lm.proto.Text2ImageSessionConfig") {
              captured->t2i_config.CheckTypeAndMergeFrom(*session_cfg);
            }
          }
        }
        return std::make_unique<FakeOmniSessionFactory>(behavior, captured);
      },
      .session_config_type_id = std::string(session_config_type_id),
      .metadata_type_id = std::move(metadata_type_id),
      .model_remote_url = std::move(model_remote_url),
  });
}

std::filesystem::path MakeTestDir(absl::string_view subdir) {
  std::filesystem::path dir =
      std::filesystem::path(::testing::TempDir()) / std::string(subdir);
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

TEST(OmniRunnerLibTest, RejectsMissingInputsAndBasicOptionErrors) {
  std::ostringstream out;

  // Missing audio_path, image_path, and text (also exercises default `out`).
  OmniRunnerOptions empty_options;
  EXPECT_THAT(
      RunOmniRunner(empty_options),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("--audio_path")));

  // Invalid cache_dir (parent path is a regular file).
  const std::filesystem::path dir = MakeTestDir("omni_runner_basic_errors");
  const std::filesystem::path blocking_file = dir / "regular_file";
  std::ofstream(blocking_file) << "x";
  OmniRunnerOptions bad_cache_options;
  bad_cache_options.text = "hello";
  bad_cache_options.model_name = "unused";
  bad_cache_options.cache_dir = (blocking_file / "child").string();
  EXPECT_THAT(
      RunOmniRunner(bad_cache_options, out),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("--cache_dir")));

  // Invalid backend.
  OmniRunnerOptions bad_backend_options;
  bad_backend_options.text = "hello";
  bad_backend_options.model_name = "unused";
  bad_backend_options.cache_dir = ":nocache";
  bad_backend_options.backend = "tpu";
  EXPECT_THAT(RunOmniRunner(bad_backend_options, out),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Missing both model_name and model_path.
  OmniRunnerOptions missing_model_options;
  missing_model_options.text = "hello";
  missing_model_options.cache_dir = ":nocache";
  EXPECT_THAT(
      RunOmniRunner(missing_model_options, out),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("--model_path")));

  // Unregistered model name.
  OmniRunnerOptions unknown_model_options;
  unknown_model_options.text = "hello";
  unknown_model_options.model_name = "nonexistent-omni-runner-test-model";
  unknown_model_options.cache_dir = ":nocache";
  EXPECT_THAT(RunOmniRunner(unknown_model_options, out),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST(OmniRunnerLibTest, RejectsUnsupportedMetadataProtoType) {
  CapturedRun captured;
  ASSERT_OK(RegisterFakeModel(
      "runner-unsupported-meta-model", "custom.pkg.CustomSessionConfig",
      FakeSessionBehavior{}, &captured, /*model_remote_url=*/std::nullopt,
      /*metadata_type_id=*/"custom.pkg.CustomMetadata"));
  absl::Cleanup cleanup = [] {
    OmniRegistry::Get().UnregisterModel("runner-unsupported-meta-model");
  };

  OmniRunnerOptions options;
  options.model_name = "runner-unsupported-meta-model";
  options.text = "hello";
  options.cache_dir = ":nocache";
  std::ostringstream out;
  EXPECT_THAT(RunOmniRunner(options, out),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported metadata proto type")));
}

TEST(OmniRunnerLibTest, InfersModelNameFromPathAndUsesCachedRemoteModel) {
  const std::filesystem::path dir = MakeTestDir("omni_runner_infer");
  CapturedRun captured;
  FakeSessionBehavior behavior;
  behavior.process_results = {TextOutput{.confirmed_text = "ok"}};

  ASSERT_OK(RegisterFakeModel(
      "runner-infer-model", "Any2Any", behavior, &captured,
      "https://example.com/models/runner-infer-model.tflite"));
  absl::Cleanup cleanup = [] {
    OmniRegistry::Get().UnregisterModel("runner-infer-model");
  };

  // 1. Exact stem inference from file path.
  {
    OmniRunnerOptions options;
    options.model_path = (dir / "runner-infer-model.litertlm").string();
    options.text = "hi";
    options.cache_dir = ":nocache";
    std::ostringstream out;
    ASSERT_OK(RunOmniRunner(options, out));
    EXPECT_EQ(captured.model_name, "runner-infer-model");
    EXPECT_EQ(out.str(), "ok\n");
  }

  // 2. Trailing-slash directory path falls back to parent_path().filename().
  {
    OmniRunnerOptions options;
    options.model_path = (dir / "runner-infer-model").string() + "/";
    options.text = "hi";
    options.cache_dir = ":nocache";
    std::ostringstream out;
    ASSERT_OK(RunOmniRunner(options, out));
    EXPECT_EQ(captured.model_name, "runner-infer-model");
  }

  // 3. Variant suffix fallback via FindModelNameForPath.
  {
    OmniRunnerOptions options;
    options.model_path =
        (dir / "runner_infer_model_int8_gpu.litertlm").string();
    options.text = "hi";
    options.cache_dir = ":nocache";
    std::ostringstream out;
    ASSERT_OK(RunOmniRunner(options, out));
    EXPECT_EQ(captured.model_name, "runner-infer-model");
  }

  // 3b. Generic filename (`model.litertlm`) falls back to parent directory.
  {
    OmniRunnerOptions options;
    options.model_path =
        (dir / "runner_infer_model_fixed" / "model.litertlm").string();
    options.text = "hi";
    options.cache_dir = ":nocache";
    std::ostringstream out;
    ASSERT_OK(RunOmniRunner(options, out));
    EXPECT_EQ(captured.model_name, "runner-infer-model");
  }

  // 3c. Mixed-case explicit model_name resolves case-insensitively.
  {
    OmniRunnerOptions options;
    options.model_name = "Runner-Infer-Model";
    options.model_path = (dir / "custom.litertlm").string();
    options.text = "hi";
    options.cache_dir = ":nocache";
    std::ostringstream out;
    ASSERT_OK(RunOmniRunner(options, out));
    EXPECT_EQ(captured.model_name, "runner-infer-model");
  }

  // 3d. Version-dotted directory path without trailing slash preserves the
  // full versioned model name over a shorter prefix alias.
  {
    ASSERT_OK(RegisterFakeModel("runner2", "Any2Any", behavior, &captured));
    ASSERT_OK(
        RegisterFakeModel("runner2.5-omni-3b", "Any2Any", behavior, &captured));
    absl::Cleanup c = [] {
      OmniRegistry::Get().UnregisterModel("runner2");
      OmniRegistry::Get().UnregisterModel("runner2.5-omni-3b");
    };
    OmniRunnerOptions options;
    options.model_path = (dir / "runner2.5-omni-3b").string();
    options.text = "hi";
    options.cache_dir = ":nocache";
    std::ostringstream out;
    ASSERT_OK(RunOmniRunner(options, out));
    EXPECT_EQ(captured.model_name, "runner2.5-omni-3b");
  }

  // 4. Remote URL resolves into cache_dir and skips download when cached.
  {
    const std::filesystem::path cache_dir = dir / "cache";
    std::filesystem::create_directories(cache_dir);
    const std::filesystem::path cached_file =
        cache_dir / "runner-infer-model.tflite";
    std::ofstream(cached_file) << "placeholder";

    OmniRunnerOptions options;
    options.model_name = "runner-infer-model";
    options.text = "hi";
    options.cache_dir = cache_dir.string();
    std::ostringstream out;
    ASSERT_OK(RunOmniRunner(options, out));
    EXPECT_EQ(captured.settings.model_path, cached_file.string());
  }

  // 5. Extensionless remote URL defaults to `.litertlm` under cache_dir.
  {
    ASSERT_OK(RegisterFakeModel("runner-infer-noext", "Any2Any", behavior,
                                &captured, "https://example.com/models/noext"));
    absl::Cleanup c = [] {
      OmniRegistry::Get().UnregisterModel("runner-infer-noext");
    };
    const std::filesystem::path cache_dir = dir / "cache_noext";
    std::filesystem::create_directories(cache_dir);
    std::ofstream(cache_dir / "runner-infer-noext.litertlm") << "not-a-bundle";

    OmniRunnerOptions options;
    options.model_name = "runner-infer-noext";
    options.text = "hi";
    options.cache_dir = cache_dir.string();
    std::ostringstream out;
    // Reaches LitertLmLoader directly without attempting network download.
    EXPECT_THAT(RunOmniRunner(options, out),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST(OmniRunnerLibTest, RejectsUninspectableCachedRemoteModelPath) {
  const std::filesystem::path dir = MakeTestDir("omni_runner_symlink_loop");
  const std::filesystem::path looping_link =
      dir / "runner-symlink-loop-model.tflite";
  std::error_code ec;
  std::filesystem::create_symlink(looping_link, looping_link, ec);
  if (ec) {
    GTEST_SKIP() << "Symlink creation unsupported in this environment: "
                 << ec.message();
  }

  CapturedRun captured;
  ASSERT_OK(RegisterFakeModel(
      "runner-symlink-loop-model", "Any2Any", FakeSessionBehavior{}, &captured,
      "https://example.com/models/runner-symlink-loop-model.tflite"));
  absl::Cleanup cleanup = [] {
    OmniRegistry::Get().UnregisterModel("runner-symlink-loop-model");
  };

  OmniRunnerOptions options;
  options.model_name = "runner-symlink-loop-model";
  options.text = "hi";
  options.cache_dir = dir.string();
  std::ostringstream out;
  EXPECT_THAT(RunOmniRunner(options, out),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Failed to check cached model path")));
}

TEST(OmniRunnerLibTest, Speech2TextValidatesInputsReadsWavAndWritesTranscript) {
  const std::filesystem::path dir = MakeTestDir("omni_runner_asr");
  const std::string wav_path = (dir / "input.wav").string();
  // > 16000 samples exercises multi-chunk ReadWavFile and kPushChunkSamples.
  ASSERT_OK(WriteTestWavFile(wav_path, /*sample_rate_hz=*/16000,
                             /*num_samples=*/20000));

  CapturedRun captured;
  FakeSessionBehavior behavior;
  behavior.process_results = {
      TextOutput{.confirmed_text = ""},  // Ignored empty chunk.
      TextOutput{.confirmed_text = "hello"},
      TextOutput{.confirmed_text = "world"},
      absl::NotFoundError("need flush"),
  };
  behavior.flush_result = TextOutput{.confirmed_text = "done"};

  ASSERT_OK(RegisterFakeModel("runner-asr-model", "Speech2Text", behavior,
                              &captured));
  absl::Cleanup cleanup = [] {
    OmniRegistry::Get().UnregisterModel("runner-asr-model");
  };

  std::ostringstream out;

  // Missing audio_path.
  {
    OmniRunnerOptions options;
    options.model_name = "runner-asr-model";
    options.text = "unexpected";
    options.cache_dir = ":nocache";
    EXPECT_THAT(RunOmniRunner(options, out),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("requires --audio_path")));
  }

  // Extra text input rejected.
  {
    OmniRunnerOptions options;
    options.model_name = "runner-asr-model";
    options.audio_path = wav_path;
    options.text = "extra";
    options.cache_dir = ":nocache";
    EXPECT_THAT(RunOmniRunner(options, out),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("only accepts --audio_path")));
  }

  // Nonexistent WAV file rejected.
  {
    OmniRunnerOptions options;
    options.model_name = "runner-asr-model";
    options.audio_path = (dir / "missing.wav").string();
    options.cache_dir = ":nocache";
    EXPECT_THAT(RunOmniRunner(options, out),
                StatusIs(absl::StatusCode::kNotFound,
                         HasSubstr("Failed to open or read WAV file")));
  }

  // Happy path with levenshtein merger + transcript file output.
  {
    const std::string transcript_path = (dir / "transcript.txt").string();
    OmniRunnerOptions options;
    options.model_name = "runner-asr-model";
    options.audio_path = wav_path;
    options.output_path = transcript_path;
    options.overlap_ratio = 0.25f;
    options.text_merger_type = "levenshtein";
    options.cache_dir = ":nocache";
    out.str("");
    ASSERT_OK(RunOmniRunner(options, out));
    EXPECT_EQ(out.str(), "hello world done\n");
    EXPECT_THAT(LoadFile(transcript_path), IsOkAndHolds("hello world done\n"));
    EXPECT_FLOAT_EQ(captured.asr_config.overlap_ratio(), 0.25f);
    EXPECT_EQ(
        captured.asr_config.text_merger_type(),
        lm::proto::Speech2TextSessionConfig::TEXT_MERGER_TYPE_LEVENSHTEIN);
    // 1 AudioInputMetadata + 2 AudioInput chunks (16000 + 4000) + 1 EndOfInput.
    ASSERT_EQ(captured.inputs.size(), 4);
    EXPECT_TRUE(std::holds_alternative<AudioInputMetadata>(captured.inputs[0]));
    ASSERT_TRUE(std::holds_alternative<AudioInput>(captured.inputs[1]));
    EXPECT_EQ(std::get<AudioInput>(captured.inputs[1]).pcm_samples.size(),
              16000);
    ASSERT_TRUE(std::holds_alternative<AudioInput>(captured.inputs[2]));
    EXPECT_EQ(std::get<AudioInput>(captured.inputs[2]).pcm_samples.size(),
              4000);
    EXPECT_TRUE(std::holds_alternative<EndOfInput>(captured.inputs[3]));
  }

  // Default timestamp merger + unwritable transcript file error.
  {
    OmniRunnerOptions options;
    options.model_name = "runner-asr-model";
    options.audio_path = wav_path;
    options.output_path = (dir / "no_such_dir" / "transcript.txt").string();
    options.text_merger_type = "timestamp";
    options.cache_dir = ":nocache";
    out.str("");
    EXPECT_THAT(RunOmniRunner(options, out),
                StatusIs(absl::StatusCode::kInternal,
                         HasSubstr("Failed to open transcript file")));
    EXPECT_EQ(captured.asr_config.text_merger_type(),
              lm::proto::Speech2TextSessionConfig::TEXT_MERGER_TYPE_TIMESTAMP);
  }
}

TEST(OmniRunnerLibTest, PropagatesNonEofProcessNextError) {
  CapturedRun captured;
  FakeSessionBehavior behavior;
  behavior.process_results = {absl::DataLossError("corrupted frame")};
  ASSERT_OK(
      RegisterFakeModel("runner-err-model", "Any2Any", behavior, &captured));
  absl::Cleanup cleanup = [] {
    OmniRegistry::Get().UnregisterModel("runner-err-model");
  };

  OmniRunnerOptions options;
  options.model_name = "runner-err-model";
  options.text = "hi";
  options.cache_dir = ":nocache";
  std::ostringstream out;
  EXPECT_THAT(
      RunOmniRunner(options, out),
      StatusIs(absl::StatusCode::kDataLoss, HasSubstr("corrupted frame")));
}

TEST(OmniRunnerLibTest, Text2SpeechValidatesInputsForwardsConfigAndWritesWav) {
  const std::filesystem::path dir = MakeTestDir("omni_runner_tts");
  CapturedRun captured;
  FakeSessionBehavior behavior;
  behavior.process_results = {
      AudioOutput{.pcm_samples = {0.1f, -0.1f, 0.2f}, .sample_rate_hz = 22050},
  };
  behavior.flush_result =
      AudioOutput{.pcm_samples = {0.3f, -0.3f}, .sample_rate_hz = 22050};

  ASSERT_OK(RegisterFakeModel("runner-tts-model", "Text2Speech", behavior,
                              &captured));
  absl::Cleanup cleanup = [] {
    OmniRegistry::Get().UnregisterModel("runner-tts-model");
  };

  std::ostringstream out;

  // Missing text.
  {
    OmniRunnerOptions options;
    options.model_name = "runner-tts-model";
    options.audio_path = "dummy.wav";
    options.cache_dir = ":nocache";
    EXPECT_THAT(RunOmniRunner(options, out),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("requires --text")));
  }

  // Extra audio_path rejected.
  {
    OmniRunnerOptions options;
    options.model_name = "runner-tts-model";
    options.text = "hello";
    options.audio_path = "dummy.wav";
    options.cache_dir = ":nocache";
    EXPECT_THAT(RunOmniRunner(options, out),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("only accepts --text")));
  }

  // Default empty language falls back to "en-US" for TTS.
  {
    const std::string wav_out = (dir / "default_lang.wav").string();
    OmniRunnerOptions options;
    options.model_name = "runner-tts-model";
    options.text = "speak default";
    options.output_path = wav_out;
    options.cache_dir = ":nocache";
    captured.tts_config.Clear();
    out.str("");
    ASSERT_OK(RunOmniRunner(options, out));
    EXPECT_TRUE(captured.tts_config.has_language());
    EXPECT_EQ(captured.tts_config.language(), "en-US");
  }

  // Happy path writes valid RIFF/WAVE file.
  {
    const std::string wav_out = (dir / "out.wav").string();
    OmniRunnerOptions options;
    options.model_name = "runner-tts-model";
    options.text = "speak this";
    options.language = "fr-FR";
    options.voice = "af_heart";
    options.output_path = wav_out;
    options.cache_dir = ":nocache";
    captured.tts_config.Clear();
    out.str("");
    ASSERT_OK(RunOmniRunner(options, out));
    EXPECT_EQ(captured.tts_config.language(), "fr-FR");
    EXPECT_EQ(captured.tts_config.voice(), "af_heart");
    EXPECT_THAT(out.str(), HasSubstr("5 samples @ 22050 Hz"));
    ASSERT_OK_AND_ASSIGN(const std::string wav_bytes, LoadFile(wav_out));
    ASSERT_GT(wav_bytes.size(), 44);
    EXPECT_THAT(wav_bytes, StartsWith("RIFF"));
    EXPECT_EQ(wav_bytes.substr(8, 4), "WAVE");
  }

  // Unwritable WAV output path.
  {
    OmniRunnerOptions options;
    options.model_name = "runner-tts-model";
    options.text = "speak this";
    options.output_path = (dir / "no_such_dir" / "out.wav").string();
    options.cache_dir = ":nocache";
    out.str("");
    EXPECT_THAT(RunOmniRunner(options, out),
                StatusIs(absl::StatusCode::kInternal,
                         HasSubstr("Failed to open WAV file for writing")));
  }
}

TEST(OmniRunnerLibTest, Text2ImageValidatesInputsWritesPngAndChecksBounds) {
  const std::filesystem::path dir = MakeTestDir("omni_runner_t2i");
  CapturedRun captured;
  ImageOutput valid_image{
      .rgb_data = std::vector<uint8_t>(2 * 2 * 3, 128),
      .width = 2,
      .height = 2,
      .channels = 3,
  };
  FakeSessionBehavior ok_behavior;
  ok_behavior.process_results = {valid_image};

  ASSERT_OK(RegisterFakeModel("runner-t2i-model", "Text2Image", ok_behavior,
                              &captured));
  absl::Cleanup cleanup = [] {
    OmniRegistry::Get().UnregisterModel("runner-t2i-model");
  };

  std::ostringstream out;

  // Missing text.
  {
    OmniRunnerOptions options;
    options.model_name = "runner-t2i-model";
    options.image_path = "dummy.png";
    options.cache_dir = ":nocache";
    EXPECT_THAT(RunOmniRunner(options, out),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("requires --text")));
  }

  // Extra image_path rejected.
  {
    OmniRunnerOptions options;
    options.model_name = "runner-t2i-model";
    options.text = "a cat";
    options.image_path = "dummy.png";
    options.cache_dir = ":nocache";
    EXPECT_THAT(RunOmniRunner(options, out),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("only accepts --text")));
  }

  // Happy path with img_size > 0 writes valid PNG file.
  {
    const std::string png_out = (dir / "out.png").string();
    OmniRunnerOptions options;
    options.model_name = "runner-t2i-model";
    options.text = "a mountain";
    options.img_size = 256;
    options.steps = 8;
    options.seed = 12345;
    options.output_path = png_out;
    options.cache_dir = ":nocache";
    out.str("");
    ASSERT_OK(RunOmniRunner(options, out));
    EXPECT_EQ(captured.t2i_config.width(), 256);
    EXPECT_EQ(captured.t2i_config.height(), 256);
    EXPECT_EQ(captured.t2i_config.num_inference_steps(), 8);
    EXPECT_EQ(captured.t2i_config.seed(), 12345);
    EXPECT_THAT(out.str(), HasSubstr("Saved 2x2 image to"));
    ASSERT_OK_AND_ASSIGN(const std::string png_bytes, LoadFile(png_out));
    ASSERT_GT(png_bytes.size(), 8);
    EXPECT_EQ(png_bytes.substr(0, 8), "\x89PNG\r\n\x1a\n");
  }

  // Unwritable PNG path.
  {
    OmniRunnerOptions options;
    options.model_name = "runner-t2i-model";
    options.text = "a mountain";
    options.output_path = (dir / "no_such_dir" / "out.png").string();
    options.cache_dir = ":nocache";
    EXPECT_THAT(RunOmniRunner(options, out),
                StatusIs(absl::StatusCode::kInternal,
                         HasSubstr("Failed to write PNG file")));
  }

  // Invalid image dimensions rejected.
  {
    FakeSessionBehavior bad_dims;
    bad_dims.process_results = {
        ImageOutput{.rgb_data = {}, .width = 0, .height = 2, .channels = 3}};
    ASSERT_OK(RegisterFakeModel("runner-t2i-bad-dims", "Text2Image", bad_dims,
                                &captured));
    absl::Cleanup c = [] {
      OmniRegistry::Get().UnregisterModel("runner-t2i-bad-dims");
    };
    OmniRunnerOptions options;
    options.model_name = "runner-t2i-bad-dims";
    options.text = "x";
    options.output_path = (dir / "bad_dims.png").string();
    options.cache_dir = ":nocache";
    EXPECT_THAT(RunOmniRunner(options, out),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Invalid image dimensions")));
  }

  // Truncated RGB buffer rejected before calling stbi_write_png.
  {
    FakeSessionBehavior short_buf;
    short_buf.process_results = {ImageOutput{
        .rgb_data = {1, 2, 3}, .width = 2, .height = 2, .channels = 3}};
    ASSERT_OK(RegisterFakeModel("runner-t2i-short-buf", "Text2Image", short_buf,
                                &captured));
    absl::Cleanup c = [] {
      OmniRegistry::Get().UnregisterModel("runner-t2i-short-buf");
    };
    OmniRunnerOptions options;
    options.model_name = "runner-t2i-short-buf";
    options.text = "x";
    options.output_path = (dir / "short_buf.png").string();
    options.cache_dir = ":nocache";
    EXPECT_THAT(RunOmniRunner(options, out),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Image buffer too small")));
  }
}

TEST(OmniRunnerLibTest, Any2AnyForwardsMultimodalInputsConfigAndProtectsMedia) {
  const std::filesystem::path dir = MakeTestDir("omni_runner_any2any");
  const std::string img_in = (dir / "in.bin").string();
  std::ofstream(img_in, std::ios::binary) << "PNG_BYTES";
  const std::string wav_in = (dir / "in.wav").string();
  // > 16000 samples verifies Any-to-Any pushes the full audio clip in one
  // AudioInput chunk rather than splitting into 1-second ASR chunks.
  ASSERT_OK(WriteTestWavFile(wav_in, /*sample_rate_hz=*/16000,
                             /*num_samples=*/20000));

  CapturedRun captured;

  // 1. Text + image + audio inputs -> text output + transcript file + optional
  // Any2AnySessionConfig fields forwarded.
  {
    FakeSessionBehavior text_behavior;
    text_behavior.process_results = {
        TextOutput{.confirmed_text = "Hel"},
        TextOutput{.confirmed_text = "lo!"},
    };
    ASSERT_OK(RegisterFakeModel("runner-a2a-text", "Any2Any", text_behavior,
                                &captured));
    absl::Cleanup c = [] {
      OmniRegistry::Get().UnregisterModel("runner-a2a-text");
    };

    const std::string transcript_path = (dir / "a2a_text.txt").string();
    OmniRunnerOptions options;
    options.model_name = "runner-a2a-text";
    options.image_path = img_in;
    options.audio_path = wav_in;
    options.text = "Describe";
    options.output_path = transcript_path;
    options.max_new_tokens = 64;
    options.temperature = 0.7f;
    options.cache_dir = ":nocache";
    std::ostringstream out;
    ASSERT_OK(RunOmniRunner(options, out));
    EXPECT_EQ(out.str(), "Hello!\n");
    EXPECT_THAT(LoadFile(transcript_path), IsOkAndHolds("Hello!\n"));
    EXPECT_FALSE(captured.any2any_config.generate_audio());
    EXPECT_FALSE(captured.any2any_config.has_generate_image());
    EXPECT_FALSE(captured.any2any_config.generate_image());
    EXPECT_FALSE(captured.any2any_config.has_system_prompt());
    EXPECT_EQ(captured.any2any_config.max_new_tokens(), 64);
    EXPECT_FALSE(captured.any2any_config.has_max_tts_tokens());
    EXPECT_FLOAT_EQ(captured.any2any_config.temperature(), 0.7f);
    EXPECT_FALSE(captured.any2any_config.has_tts_temperature());
    EXPECT_FALSE(captured.any2any_config.has_tts_top_k());
    EXPECT_FALSE(captured.any2any_config.has_voice());
    EXPECT_FALSE(captured.any2any_config.has_language());
    EXPECT_FALSE(captured.any2any_config.has_seed());
    // ImageInput + AudioInputMetadata + 1 full-clip AudioInput + TextInput +
    // EndOfInput.
    ASSERT_EQ(captured.inputs.size(), 5);
    EXPECT_TRUE(std::holds_alternative<ImageInput>(captured.inputs[0]));
    EXPECT_TRUE(std::holds_alternative<AudioInputMetadata>(captured.inputs[1]));
    ASSERT_TRUE(std::holds_alternative<AudioInput>(captured.inputs[2]));
    EXPECT_EQ(std::get<AudioInput>(captured.inputs[2]).pcm_samples.size(),
              20000);
    EXPECT_TRUE(std::holds_alternative<TextInput>(captured.inputs[3]));
    EXPECT_TRUE(std::holds_alternative<EndOfInput>(captured.inputs[4]));
  }

  // 2. generate_image=true forwards cfg_weight, image_temperature, seed and
  // does not overwrite PNG output with text transcript.
  {
    FakeSessionBehavior img_behavior;
    img_behavior.process_results = {
        TextOutput{.confirmed_text = "caption"},
        ImageOutput{.rgb_data = std::vector<uint8_t>(2 * 2 * 3, 200),
                    .width = 2,
                    .height = 2,
                    .channels = 3},
    };
    ASSERT_OK(RegisterFakeModel("runner-a2a-img", "Any2Any", img_behavior,
                                &captured));
    absl::Cleanup c = [] {
      OmniRegistry::Get().UnregisterModel("runner-a2a-img");
    };

    const std::string png_out = (dir / "a2a_out.png").string();
    OmniRunnerOptions options;
    options.model_name = "runner-a2a-img";
    options.text = "Draw a sunset";
    options.output_path = png_out;
    options.generate_image = true;
    options.cfg_weight = 5.0f;
    options.image_temperature = 1.1f;
    options.seed = 777;
    options.cache_dir = ":nocache";
    std::ostringstream out;
    ASSERT_OK(RunOmniRunner(options, out));
    EXPECT_TRUE(captured.any2any_config.has_generate_image());
    EXPECT_TRUE(captured.any2any_config.generate_image());
    EXPECT_FLOAT_EQ(captured.any2any_config.cfg_weight(), 5.0f);
    EXPECT_FLOAT_EQ(captured.any2any_config.image_temperature(), 1.1f);
    EXPECT_EQ(captured.any2any_config.seed(), 777);
    ASSERT_OK_AND_ASSIGN(const std::string png_bytes, LoadFile(png_out));
    ASSERT_GT(png_bytes.size(), 8);
    EXPECT_EQ(png_bytes.substr(0, 8), "\x89PNG\r\n\x1a\n");
  }

  // 3. Even when generate_image=false, emitting ImageOutput + TextOutput
  // preserves the PNG file and does not overwrite it with text.
  {
    FakeSessionBehavior implicit_img_behavior;
    implicit_img_behavior.process_results = {
        ImageOutput{.rgb_data = std::vector<uint8_t>(2 * 2 * 3, 50),
                    .width = 2,
                    .height = 2,
                    .channels = 3},
        TextOutput{.confirmed_text = "should not overwrite png"},
    };
    ASSERT_OK(RegisterFakeModel("runner-a2a-implicit-img", "Any2Any",
                                implicit_img_behavior, &captured));
    absl::Cleanup c = [] {
      OmniRegistry::Get().UnregisterModel("runner-a2a-implicit-img");
    };

    const std::string png_out = (dir / "a2a_implicit.png").string();
    OmniRunnerOptions options;
    options.model_name = "runner-a2a-implicit-img";
    options.text = "Draw";
    options.output_path = png_out;
    options.cache_dir = ":nocache";
    std::ostringstream out;
    ASSERT_OK(RunOmniRunner(options, out));
    ASSERT_OK_AND_ASSIGN(const std::string png_bytes, LoadFile(png_out));
    ASSERT_GT(png_bytes.size(), 8);
    EXPECT_EQ(png_bytes.substr(0, 8), "\x89PNG\r\n\x1a\n");
  }

  // 4. generate_audio=true writes WAV and does not overwrite with text.
  {
    FakeSessionBehavior audio_behavior;
    audio_behavior.process_results = {
        TextOutput{.confirmed_text = "spoken"},
        AudioOutput{.pcm_samples = {0.25f, -0.25f}, .sample_rate_hz = 24000},
    };
    ASSERT_OK(RegisterFakeModel("runner-a2a-audio", "Any2Any", audio_behavior,
                                &captured));
    absl::Cleanup c = [] {
      OmniRegistry::Get().UnregisterModel("runner-a2a-audio");
    };

    const std::string wav_out = (dir / "a2a_out.wav").string();
    OmniRunnerOptions options;
    options.model_name = "runner-a2a-audio";
    options.text = "Say hi";
    options.output_path = wav_out;
    options.generate_audio = true;
    options.system_prompt = "Be concise.";
    options.max_tts_tokens = 128;
    options.tts_temperature = 0.6f;
    options.tts_top_k = 20;
    options.voice = "Chelsie";
    options.language = "zh-CN";
    options.cache_dir = ":nocache";
    std::ostringstream out;
    ASSERT_OK(RunOmniRunner(options, out));
    EXPECT_TRUE(captured.any2any_config.generate_audio());
    EXPECT_EQ(captured.any2any_config.system_prompt(), "Be concise.");
    EXPECT_EQ(captured.any2any_config.max_tts_tokens(), 128);
    EXPECT_FLOAT_EQ(captured.any2any_config.tts_temperature(), 0.6f);
    EXPECT_EQ(captured.any2any_config.tts_top_k(), 20);
    EXPECT_EQ(captured.any2any_config.voice(), "Chelsie");
    EXPECT_EQ(captured.any2any_config.language(), "zh-CN");
    ASSERT_OK_AND_ASSIGN(const std::string wav_bytes, LoadFile(wav_out));
    ASSERT_GT(wav_bytes.size(), 44);
    EXPECT_THAT(wav_bytes, StartsWith("RIFF"));
  }
}

TEST(OmniRunnerFlagsTest, BuildsOptionsAndRunsFromFlags) {
  absl::FlagSaver saver;

  EXPECT_TRUE(BuildOmniRunnerOptionsFromFlags().language.empty());

  absl::SetFlag(&FLAGS_model_name, "flag-model");
  absl::SetFlag(&FLAGS_model_path, "/tmp/flag.litertlm");
  absl::SetFlag(&FLAGS_backend, "gpu");
  absl::SetFlag(&FLAGS_force_fp32_on_gpu, true);
  absl::SetFlag(&FLAGS_num_threads, 6);
  absl::SetFlag(&FLAGS_cache_dir, ":nocache");
  absl::SetFlag(&FLAGS_litert_dispatch_lib_dir, "/tmp/dispatch");
  absl::SetFlag(&FLAGS_audio_path, "/tmp/in.wav");
  absl::SetFlag(&FLAGS_image_path, "/tmp/in.png");
  absl::SetFlag(&FLAGS_text, "prompt");
  absl::SetFlag(&FLAGS_output_path, "/tmp/out.txt");
  absl::SetFlag(&FLAGS_generate_audio, true);
  absl::SetFlag(&FLAGS_generate_image, true);
  absl::SetFlag(&FLAGS_system_prompt, "You are helpful.");
  absl::SetFlag(&FLAGS_max_new_tokens, 256);
  absl::SetFlag(&FLAGS_max_tts_tokens, 512);
  absl::SetFlag(&FLAGS_temperature, 0.8f);
  absl::SetFlag(&FLAGS_tts_temperature, 0.7f);
  absl::SetFlag(&FLAGS_tts_top_k, 32);
  absl::SetFlag(&FLAGS_cfg_weight, 4.5f);
  absl::SetFlag(&FLAGS_image_temperature, 0.9f);
  absl::SetFlag(&FLAGS_overlap_ratio, 0.3f);
  absl::SetFlag(&FLAGS_text_merger_type, "levenshtein");
  absl::SetFlag(&FLAGS_language, "es");
  absl::SetFlag(&FLAGS_voice, "bf_emma");
  absl::SetFlag(&FLAGS_img_size, 512);
  absl::SetFlag(&FLAGS_steps, 12);
  absl::SetFlag(&FLAGS_seed, 99);

  const OmniRunnerOptions opts = BuildOmniRunnerOptionsFromFlags();
  EXPECT_EQ(opts.model_name, "flag-model");
  EXPECT_EQ(opts.model_path, "/tmp/flag.litertlm");
  EXPECT_EQ(opts.backend, "gpu");
  EXPECT_TRUE(opts.force_fp32_on_gpu);
  EXPECT_EQ(opts.num_threads, 6);
  EXPECT_EQ(opts.cache_dir, ":nocache");
  EXPECT_EQ(opts.litert_dispatch_lib_dir, "/tmp/dispatch");
  EXPECT_EQ(opts.audio_path, "/tmp/in.wav");
  EXPECT_EQ(opts.image_path, "/tmp/in.png");
  EXPECT_EQ(opts.text, "prompt");
  EXPECT_EQ(opts.output_path, "/tmp/out.txt");
  EXPECT_TRUE(opts.generate_audio);
  EXPECT_TRUE(opts.generate_image);
  EXPECT_THAT(opts.system_prompt, Optional(Eq("You are helpful.")));
  EXPECT_THAT(opts.max_new_tokens, Optional(Eq(256)));
  EXPECT_THAT(opts.max_tts_tokens, Optional(Eq(512)));
  EXPECT_THAT(opts.temperature, Optional(Eq(0.8f)));
  EXPECT_THAT(opts.tts_temperature, Optional(Eq(0.7f)));
  EXPECT_THAT(opts.tts_top_k, Optional(Eq(32)));
  EXPECT_THAT(opts.cfg_weight, Optional(Eq(4.5f)));
  EXPECT_THAT(opts.image_temperature, Optional(Eq(0.9f)));
  EXPECT_FLOAT_EQ(opts.overlap_ratio, 0.3f);
  EXPECT_EQ(opts.text_merger_type, "levenshtein");
  EXPECT_EQ(opts.language, "es");
  EXPECT_EQ(opts.voice, "bf_emma");
  EXPECT_EQ(opts.img_size, 512);
  EXPECT_EQ(opts.steps, 12);
  EXPECT_EQ(opts.seed, 99);

  // Failure path returns exit code 1 and prints ERROR to err stream.
  absl::SetFlag(&FLAGS_text, "");
  absl::SetFlag(&FLAGS_audio_path, "");
  absl::SetFlag(&FLAGS_image_path, "");
  std::ostringstream out;
  std::ostringstream err;
  EXPECT_EQ(RunOmniRunnerFromFlags(out, err), 1);
  EXPECT_THAT(err.str(), HasSubstr("ERROR:"));

  // Success path returns exit code 0 (both explicit streams and default
  // `std::cout`/`std::cerr` arguments with empty output so stdout stays clean).
  CapturedRun captured;
  FakeSessionBehavior quiet_behavior;
  ASSERT_OK(RegisterFakeModel("runner-flags-ok-model", "Any2Any",
                              quiet_behavior, &captured));
  absl::Cleanup c = [] {
    OmniRegistry::Get().UnregisterModel("runner-flags-ok-model");
  };
  absl::SetFlag(&FLAGS_model_name, "runner-flags-ok-model");
  absl::SetFlag(&FLAGS_model_path, "");
  absl::SetFlag(&FLAGS_backend, "cpu");
  absl::SetFlag(&FLAGS_text, "hello");
  absl::SetFlag(&FLAGS_output_path, "");
  absl::SetFlag(&FLAGS_generate_audio, false);
  absl::SetFlag(&FLAGS_generate_image, false);
  out.str("");
  err.str("");
  EXPECT_EQ(RunOmniRunnerFromFlags(out, err), 0);
  EXPECT_TRUE(err.str().empty());
  EXPECT_EQ(RunOmniRunnerFromFlags(), 0);
}

}  // namespace
}  // namespace litert::omni
