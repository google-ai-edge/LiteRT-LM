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

#include "omni/tts/tts_engine.h"

#include <filesystem>  // NOLINT: Required for std::filesystem.
#include <fstream>
#include <memory>
#include <string>
#include <system_error>  // NOLINT: Required for std::error_code.
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/tts/kokoro/kokoro_factory.h"
#include "omni/tts/kokoro/kokoro_model_config.h"
#include "omni/tts/qwen3_tts/qwen3_tts_model_config.h"
#include "omni/tts/stream_text_source.h"
#include "omni/tts/tts_session.h"
#include "omni/tts/vocoder.h"
#include "runtime/framework/threadpool.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts {
struct TtsEngineTestingPeer {
  static std::unique_ptr<TtsEngine> CreateTestEngine(
      const TtsEngineSettings& settings,
      std::vector<std::string> available_voices = {}) {
    auto env = Environment::Create({});
    auto shared_env = std::make_shared<Environment>(std::move(*env));
    auto resources = std::make_shared<ModelResources>(shared_env);
    auto thread_pool = std::make_unique<lm::ThreadPool>("test_pool", 1);
    return std::unique_ptr<TtsEngine>(
        new TtsEngine(settings, std::move(available_voices), resources,
                      std::move(thread_pool)));
  }
};

namespace {

struct DummyFrontendOutput {
  std::vector<int> token_ids;
};

struct DummyAcousticOutput {
  std::vector<std::vector<int>> rvq_frames;
};

struct DummyLatentOutput {
  std::vector<float> codec_features;
  std::vector<std::vector<int>> rvq_frames;
};

class DummyTextFrontend
    : public SingleThreadedStageWithDeque<DummyFrontendOutput> {
 public:
  explicit DummyTextFrontend(Stage<std::string>* text_source)
      : text_source_(*text_source) {}

 protected:
  bool NeedScheduleInternal() const override {
    return text_source_.HasOutput();
  }

  absl::Status ScheduleInternal() override {
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
    auto text_chunk = text_source_.GetOutput();
    if (absl::IsNotFound(text_chunk.status())) {
      return absl::OkStatus();
    } else if (!text_chunk.ok()) {
      return text_chunk.status();
    }
    DummyFrontendOutput out;
    out.token_ids = {10, 20};
    PushOutput(std::move(out));
    return absl::OkStatus();
  }

 private:
  Stage<std::string>& text_source_;
};

class DummyAcousticPredictor
    : public SingleThreadedStageWithDeque<DummyAcousticOutput> {
 public:
  explicit DummyAcousticPredictor(Stage<DummyFrontendOutput>* text_frontend)
      : text_frontend_(*text_frontend) {}

 protected:
  bool NeedScheduleInternal() const override {
    return text_frontend_.HasOutput();
  }

  absl::Status ScheduleInternal() override {
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
    auto frontend_out = text_frontend_.GetOutput();
    if (absl::IsNotFound(frontend_out.status())) {
      return absl::OkStatus();
    } else if (!frontend_out.ok()) {
      return frontend_out.status();
    }
    DummyAcousticOutput out;
    out.rvq_frames.push_back({1, 2, 3});
    PushOutput(std::move(out));
    return absl::OkStatus();
  }

 private:
  Stage<DummyFrontendOutput>& text_frontend_;
};

class DummyLatentDecoder
    : public SingleThreadedStageWithDeque<DummyLatentOutput> {
 public:
  explicit DummyLatentDecoder(Stage<DummyAcousticOutput>* acoustic_predictor)
      : acoustic_predictor_(*acoustic_predictor) {}

 protected:
  bool NeedScheduleInternal() const override {
    return acoustic_predictor_.HasOutput();
  }

  absl::Status ScheduleInternal() override {
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
    auto acoustic_out = acoustic_predictor_.GetOutput();
    if (absl::IsNotFound(acoustic_out.status())) {
      return absl::OkStatus();
    } else if (!acoustic_out.ok()) {
      return acoustic_out.status();
    }
    DummyLatentOutput out;
    out.codec_features = {0.1f, 0.2f};
    out.rvq_frames = acoustic_out->rvq_frames;
    PushOutput(std::move(out));
    return absl::OkStatus();
  }

 private:
  Stage<DummyAcousticOutput>& acoustic_predictor_;
};

class DummyVocoder : public Vocoder {
 public:
  explicit DummyVocoder(Stage<DummyLatentOutput>* latent_decoder)
      : latent_decoder_(*latent_decoder) {}

  absl::Status Flush() override {
    if (has_pending_audio_) {
      PushOutput({{0.5f, -0.5f}, 24000});
      has_pending_audio_ = false;
    }
    return absl::OkStatus();
  }

 protected:
  void ResetInternal() override { has_pending_audio_ = false; }

  bool NeedScheduleInternal() const override {
    return latent_decoder_.HasOutput();
  }

  absl::Status ScheduleInternal() override {
    absl::Cleanup cleanup = [this] { SetState(State::kIdle); };
    auto latent_out = latent_decoder_.GetOutput();
    if (absl::IsNotFound(latent_out.status())) {
      return absl::OkStatus();
    } else if (!latent_out.ok()) {
      return latent_out.status();
    }
    AudioOutput out;
    out.pcm_samples = {0.1f, 0.2f, 0.3f};
    out.sample_rate_hz = 24000;
    PushOutput(std::move(out));
    has_pending_audio_ = true;
    return absl::OkStatus();
  }

 private:
  Stage<DummyLatentOutput>& latent_decoder_;
  bool has_pending_audio_ = false;
};

TtsSession::Components CreateDummyComponents() {
  TtsSession::Components components;
  components.text_source = std::make_unique<StreamTextSource>();
  auto frontend =
      std::make_unique<DummyTextFrontend>(components.text_source.get());
  auto acoustic = std::make_unique<DummyAcousticPredictor>(frontend.get());
  auto latent = std::make_unique<DummyLatentDecoder>(acoustic.get());
  auto vocoder = std::make_unique<DummyVocoder>(latent.get());

  components.intermediate_stages.push_back(std::move(frontend));
  components.intermediate_stages.push_back(std::move(acoustic));
  components.intermediate_stages.push_back(std::move(latent));
  components.vocoder = std::move(vocoder);

  return components;
}

// Creates a temporary directory holding the given (empty) model files.
class ScopedModelDir {
 public:
  ScopedModelDir(absl::string_view dir_name,
                 const std::vector<std::string>& file_names) {
    model_dir_ =
        std::filesystem::path(testing::TempDir()) / std::string(dir_name);
    std::error_code ec;
    std::filesystem::create_directories(model_dir_, ec);
    for (const auto& file_name : file_names) {
      std::ofstream(model_dir_ / file_name).close();
    }
  }

  ~ScopedModelDir() {
    std::error_code ec;
    std::filesystem::remove_all(model_dir_, ec);
  }

  std::string path() const { return model_dir_.string(); }

 private:
  std::filesystem::path model_dir_;
};

TEST(TtsEngineTest, DetectModelTypeKokoro) {
  ScopedModelDir model_dir(
      "detect_kokoro_dir",
      {"kokoro_acoustic.tflite", "kokoro_vocoder.tflite", "notes.txt"});

  ASSERT_OK_AND_ASSIGN(ModelType model_type, DetectModelType(model_dir.path()));
  EXPECT_EQ(model_type, ModelType::KOKORO);
}

TEST(TtsEngineTest, DetectModelTypeQwen3Tts) {
  ScopedModelDir model_dir(
      "detect_qwen3_dir",
      {"talker_int4.tflite", "codec_decoder_fp32.tflite", "tokenizer.json"});

  ASSERT_OK_AND_ASSIGN(ModelType model_type, DetectModelType(model_dir.path()));
  EXPECT_EQ(model_type, ModelType::QWEN3_TTS);
}

TEST(TtsEngineTest, DetectModelTypeFailsWithoutKnownModelFiles) {
  ScopedModelDir model_dir("detect_unknown_dir", {"unknown_model.tflite"});

  auto model_type = DetectModelType(model_dir.path());
  EXPECT_FALSE(model_type.ok());
  EXPECT_EQ(model_type.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(model_type.status().message(),
              testing::HasSubstr("Unable to determine the TTS model type"));
}

TEST(TtsEngineTest, DetectModelTypeFailsOnEmptyOrMissingFolder) {
  EXPECT_EQ(DetectModelType("").status().code(),
            absl::StatusCode::kInvalidArgument);

  const std::string missing_dir =
      (std::filesystem::path(testing::TempDir()) / "no_such_model_dir")
          .string();
  EXPECT_EQ(DetectModelType(missing_dir).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(TtsEngineTest, CreateFailsWhenModelTypeCannotBeDetected) {
  TtsEngineSettings settings;
  settings.model_config = std::monostate{};

  auto engine = TtsEngine::Create(settings);
  EXPECT_FALSE(engine.ok());
  EXPECT_EQ(engine.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(TtsEngineTest, SynthesizeSyncForceFlushOnSession) {
  lm::ThreadPool thread_pool("test_pool", 2);
  ASSERT_OK_AND_ASSIGN(
      auto session, TtsSession::Create(CreateDummyComponents(), &thread_pool));

  ASSERT_OK_AND_ASSIGN(auto audio, session->Synthesize("Hello world "));
  EXPECT_EQ(audio.sample_rate_hz, 24000);
  EXPECT_GE(audio.pcm_samples.size(), 3);
}

TEST(TtsEngineTest, SynthesizeAsyncStreamingOnSession) {
  lm::ThreadPool thread_pool("test_pool", 4);
  ASSERT_OK_AND_ASSIGN(
      auto session, TtsSession::Create(CreateDummyComponents(), &thread_pool));

  absl::Notification done;
  int chunk_count = 0;
  absl::Status final_status;

  absl::Status status = session->SynthesizeAsync(
      "Hello world.", [&](absl::StatusOr<AudioOutput> result) -> absl::Status {
        if (!result.ok()) {
          final_status = result.status();
          done.Notify();
          return result.status();
        }
        chunk_count++;
        return absl::OkStatus();
      });
  ASSERT_OK(status);

  done.WaitForNotification();
  EXPECT_TRUE(absl::IsOutOfRange(final_status))
      << "final_status was: " << final_status;
  EXPECT_GT(chunk_count, 0);
}

TEST(TtsEngineTest, SequentialSynthesizeCallsOnSession) {
  lm::ThreadPool thread_pool("test_pool", 2);
  ASSERT_OK_AND_ASSIGN(
      auto session, TtsSession::Create(CreateDummyComponents(), &thread_pool));

  ASSERT_OK_AND_ASSIGN(auto audio1, session->Synthesize("First chunk "));
  EXPECT_GE(audio1.pcm_samples.size(), 3);

  ASSERT_OK_AND_ASSIGN(auto audio2, session->Synthesize("Second chunk "));
  EXPECT_GE(audio2.pcm_samples.size(), 3);
}

TEST(TtsEngineTest, TtsSessionConfigDefault) {
  TtsSessionConfig config;
  EXPECT_EQ(config.language, "en-US");
  EXPECT_FALSE(config.voice.has_value());
  EXPECT_EQ(config.text_chunk_config.max_buffer_size, 0);
  EXPECT_FALSE(config.text_chunk_config.delimiters.empty());
}

TEST(TtsEngineTest, TtsSessionConfigCustomLanguageAndVoice) {
  TtsSessionConfig config;
  config.language = "es";
  config.voice = "ef_dora";
  config.text_chunk_config.max_buffer_size = 120;
  config.text_chunk_config.delimiters = {".", "!", "?"};

  EXPECT_EQ(config.language, "es");
  EXPECT_EQ(config.voice, "ef_dora");
  EXPECT_EQ(config.text_chunk_config.max_buffer_size, 120);
  EXPECT_EQ(config.text_chunk_config.delimiters.size(), 3);
}

TEST(TtsEngineTest, ToKokoroLanguageCodeMapping) {
  EXPECT_EQ(ToKokoroLanguageCode(""), "");
  EXPECT_EQ(ToKokoroLanguageCode("en"), "en-us");
  EXPECT_EQ(ToKokoroLanguageCode("en-US"), "en-us");
  EXPECT_EQ(ToKokoroLanguageCode("en_US"), "en-us");
  EXPECT_EQ(ToKokoroLanguageCode("en-AU"), "en-us");
  EXPECT_EQ(ToKokoroLanguageCode("en-GB"), "en-gb");
  EXPECT_EQ(ToKokoroLanguageCode("en-UK"), "en-gb");
  EXPECT_EQ(ToKokoroLanguageCode("es"), "es");
  EXPECT_EQ(ToKokoroLanguageCode("es-MX"), "es");
  EXPECT_EQ(ToKokoroLanguageCode("es-419"), "es");
  EXPECT_EQ(ToKokoroLanguageCode("hi"), "hi");
  EXPECT_EQ(ToKokoroLanguageCode("hi-IN"), "hi");
  EXPECT_EQ(ToKokoroLanguageCode("zh"), "cmn");
  EXPECT_EQ(ToKokoroLanguageCode("zh-CN"), "cmn");
  EXPECT_EQ(ToKokoroLanguageCode("zh-Hant-TW"), "cmn");
  EXPECT_EQ(ToKokoroLanguageCode("ja"), "ja");
  EXPECT_EQ(ToKokoroLanguageCode("ja-JP"), "ja");
  EXPECT_EQ(ToKokoroLanguageCode("fr"), "fr-fr");
  EXPECT_EQ(ToKokoroLanguageCode("fr-CA"), "fr-fr");
  EXPECT_EQ(ToKokoroLanguageCode("it"), "it");
  EXPECT_EQ(ToKokoroLanguageCode("it-IT"), "it");
  EXPECT_EQ(ToKokoroLanguageCode("pt"), "pt-br");
  EXPECT_EQ(ToKokoroLanguageCode("pt-BR"), "pt-br");
  EXPECT_EQ(ToKokoroLanguageCode("de"), "");
  EXPECT_EQ(ToKokoroLanguageCode("unknown_lang"), "");
}

TEST(TtsEngineTest, KokoroCodeToBcp47Mapping) {
  EXPECT_EQ(KokoroCodeToBcp47("en-us"), "en-US");
  EXPECT_EQ(KokoroCodeToBcp47("en-gb"), "en-GB");
  EXPECT_EQ(KokoroCodeToBcp47("es"), "es");
  EXPECT_EQ(KokoroCodeToBcp47("fr-fr"), "fr");
  EXPECT_EQ(KokoroCodeToBcp47("hi"), "hi");
  EXPECT_EQ(KokoroCodeToBcp47("it"), "it");
  EXPECT_EQ(KokoroCodeToBcp47("pt-br"), "pt-BR");
  EXPECT_EQ(KokoroCodeToBcp47("ja"), "ja");
  EXPECT_EQ(KokoroCodeToBcp47("cmn"), "zh-CN");
  EXPECT_EQ(KokoroCodeToBcp47(""), "");
  EXPECT_EQ(KokoroCodeToBcp47("de"), "");
}

class ScopedMockKokoroModelDir {
 public:
  ScopedMockKokoroModelDir() {
    model_dir_ =
        std::filesystem::path(testing::TempDir()) / "mock_kokoro_voices_dir";
    std::filesystem::path voices_dir = model_dir_ / "voices";
    std::error_code ec;
    std::filesystem::create_directories(voices_dir, ec);
    const std::vector<std::string> voice_names = {
        "af_alloy", "af_heart", "bf_alice",   "ef_dora",     "em_alex",
        "em_santa", "ff_siwis", "hf_alpha",   "hf_beta",     "hm_omega",
        "hm_psi",   "if_sara",  "im_nicola",  "pf_dora",     "pm_alex",
        "pm_santa", "jf_alpha", "zf_xiaobei", "zf_xiaoxiao",
    };
    for (const auto& name : voice_names) {
      std::ofstream(voices_dir / (name + ".bin")).close();
    }
  }

  ~ScopedMockKokoroModelDir() {
    std::error_code ec;
    std::filesystem::remove_all(model_dir_, ec);
  }

  std::string path() const { return model_dir_.string(); }

 private:
  std::filesystem::path model_dir_;
};

TEST(TtsEngineTest, EmptyModelFolderReturnsNoVoices) {
  auto voices = GetAvailableKokoroVoices("");
  EXPECT_TRUE(voices.empty());
}

TEST(TtsEngineTest, DiscoversKokoroVoicesFromDirectory) {
  std::filesystem::path model_dir =
      std::filesystem::path(testing::TempDir()) / "kokoro_test_dir";
  std::filesystem::path voices_dir = model_dir / "voices";
  std::error_code ec;
  std::filesystem::create_directories(voices_dir, ec);

  std::ofstream(voices_dir / "custom_voice1.bin").close();
  std::ofstream(voices_dir / "custom_voice2.bin").close();
  std::ofstream(voices_dir / "readme.txt").close();

  auto discovered = GetAvailableKokoroVoices(model_dir.string());
  EXPECT_THAT(discovered,
              testing::ElementsAre("custom_voice1", "custom_voice2"));

  std::filesystem::remove_all(model_dir, ec);
}

TEST(TtsEngineTest, DefaultKokoroVoicesPerLanguage) {
  EXPECT_EQ(GetDefaultKokoroVoice(""), "af_heart");
  EXPECT_EQ(GetDefaultKokoroVoice("en-us"), "af_heart");
  EXPECT_EQ(GetDefaultKokoroVoice("es"), "ef_dora");
  EXPECT_EQ(GetDefaultKokoroVoice("fr-fr"), "ff_siwis");
  EXPECT_EQ(GetDefaultKokoroVoice("hi"), "hf_alpha");
  EXPECT_EQ(GetDefaultKokoroVoice("it"), "if_sara");
  EXPECT_EQ(GetDefaultKokoroVoice("pt-br"), "pf_dora");
  EXPECT_EQ(GetDefaultKokoroVoice("en-gb"), "bf_alice");
  EXPECT_EQ(GetDefaultKokoroVoice("ja"), "jf_alpha");
  EXPECT_EQ(GetDefaultKokoroVoice("cmn"), "zf_xiaobei");
}

TEST(TtsEngineTest, TtsEngineAvailableLanguages) {
  ScopedMockKokoroModelDir mock_dir;
  TtsEngineSettings settings;
  settings.model_folder = mock_dir.path();
  settings.model_config = KokoroModelConfig{};
  auto engine = TtsEngineTestingPeer::CreateTestEngine(
      settings, GetAvailableKokoroVoices(mock_dir.path()));

  auto languages = engine->GetAvailableLanguages();
  EXPECT_THAT(languages,
              testing::ElementsAre("en-GB", "en-US", "es", "fr", "hi", "it",
                                   "ja", "pt-BR", "zh-CN"));

  // Engine with a subset of voices returns only those languages.
  auto subset_engine = TtsEngineTestingPeer::CreateTestEngine(
      settings, {"af_heart", "ef_dora", "em_alex"});
  EXPECT_THAT(subset_engine->GetAvailableLanguages(),
              testing::ElementsAre("en-US", "es"));

  // Qwen3-TTS returns en-US and zh-CN.
  TtsEngineSettings qwen_settings;
  qwen_settings.model_config = Qwen3TtsModelConfig{};
  auto qwen_engine = TtsEngineTestingPeer::CreateTestEngine(qwen_settings, {});
  EXPECT_THAT(qwen_engine->GetAvailableLanguages(),
              testing::ElementsAre("en-US", "zh-CN"));
}

TEST(TtsEngineTest, TtsEngineAvailableVoicesAll) {
  ScopedMockKokoroModelDir mock_dir;
  TtsEngineSettings settings;
  settings.model_folder = mock_dir.path();
  settings.model_config = KokoroModelConfig{};
  auto engine = TtsEngineTestingPeer::CreateTestEngine(
      settings, GetAvailableKokoroVoices(mock_dir.path()));

  auto voices = engine->GetAvailableVoices();
  EXPECT_EQ(voices.size(), 19);
  EXPECT_THAT(voices, testing::Contains("af_heart"));
  EXPECT_THAT(voices, testing::Contains("ef_dora"));
  EXPECT_THAT(voices, testing::Contains("zf_xiaoxiao"));
}

TEST(TtsEngineTest, TtsEngineAvailableVoicesFilteredByLanguage) {
  ScopedMockKokoroModelDir mock_dir;
  TtsEngineSettings settings;
  settings.model_folder = mock_dir.path();
  settings.model_config = KokoroModelConfig{};
  auto engine = TtsEngineTestingPeer::CreateTestEngine(
      settings, GetAvailableKokoroVoices(mock_dir.path()));

  auto spanish_voices = engine->GetAvailableVoices("es");
  EXPECT_THAT(spanish_voices,
              testing::ElementsAre("ef_dora", "em_alex", "em_santa"));

  auto italian_voices = engine->GetAvailableVoices("it");
  EXPECT_THAT(italian_voices, testing::ElementsAre("if_sara", "im_nicola"));

  auto french_voices = engine->GetAvailableVoices("fr");
  EXPECT_THAT(french_voices, testing::ElementsAre("ff_siwis"));

  auto hindi_voices = engine->GetAvailableVoices("hi");
  EXPECT_THAT(hindi_voices, testing::ElementsAre("hf_alpha", "hf_beta",
                                                 "hm_omega", "hm_psi"));

  auto pt_voices = engine->GetAvailableVoices("pt-BR");
  EXPECT_THAT(pt_voices,
              testing::ElementsAre("pf_dora", "pm_alex", "pm_santa"));

  auto british_voices = engine->GetAvailableVoices("en-GB");
  EXPECT_THAT(british_voices, testing::ElementsAre("bf_alice"));

  auto us_voices = engine->GetAvailableVoices("en-US");
  EXPECT_THAT(us_voices, testing::ElementsAre("af_alloy", "af_heart"));

  auto chinese_voices = engine->GetAvailableVoices("zh-CN");
  EXPECT_THAT(chinese_voices,
              testing::ElementsAre("zf_xiaobei", "zf_xiaoxiao"));

  auto japanese_voices = engine->GetAvailableVoices("ja");
  EXPECT_THAT(japanese_voices, testing::ElementsAre("jf_alpha"));

  auto unsupported_voices = engine->GetAvailableVoices("de");
  EXPECT_TRUE(unsupported_voices.empty());
}

TEST(TtsEngineTest, TtsEngineHasVoice) {
  ScopedMockKokoroModelDir mock_dir;
  TtsEngineSettings settings;
  settings.model_folder = mock_dir.path();
  settings.model_config = KokoroModelConfig{};
  auto engine = TtsEngineTestingPeer::CreateTestEngine(
      settings, GetAvailableKokoroVoices(mock_dir.path()));

  EXPECT_TRUE(engine->HasVoice("af_heart"));
  EXPECT_TRUE(engine->HasVoice("af_heart.bin"));
  EXPECT_TRUE(engine->HasVoice("voices/af_heart.bin"));
  EXPECT_TRUE(engine->HasVoice("ef_dora"));
  EXPECT_TRUE(engine->HasVoice("if_sara"));
  EXPECT_FALSE(engine->HasVoice("non_existent_voice"));
  EXPECT_FALSE(engine->HasVoice(""));
}

TEST(TtsEngineTest, TtsEngineGetDefaultVoice) {
  ScopedMockKokoroModelDir mock_dir;
  TtsEngineSettings settings;
  settings.model_folder = mock_dir.path();
  settings.model_config = KokoroModelConfig{};
  auto engine = TtsEngineTestingPeer::CreateTestEngine(
      settings, GetAvailableKokoroVoices(mock_dir.path()));

  EXPECT_EQ(engine->GetDefaultVoice("es"), "ef_dora");
  EXPECT_EQ(engine->GetDefaultVoice("it"), "if_sara");
  EXPECT_EQ(engine->GetDefaultVoice("fr"), "ff_siwis");
  EXPECT_EQ(engine->GetDefaultVoice("hi"), "hf_alpha");
  EXPECT_EQ(engine->GetDefaultVoice("pt-BR"), "pf_dora");
  EXPECT_EQ(engine->GetDefaultVoice("en-GB"), "bf_alice");
  EXPECT_EQ(engine->GetDefaultVoice("zh-CN"), "zf_xiaobei");
  EXPECT_EQ(engine->GetDefaultVoice("ja"), "jf_alpha");
  EXPECT_EQ(engine->GetDefaultVoice("en-US"), "af_heart");
  EXPECT_EQ(engine->GetDefaultVoice(""), "af_heart");
}

TEST(TtsEngineTest, CreateSessionVoiceValidation) {
  ScopedMockKokoroModelDir mock_dir;
  TtsEngineSettings settings;
  settings.model_folder = mock_dir.path();
  settings.model_config = KokoroModelConfig{};
  auto engine = TtsEngineTestingPeer::CreateTestEngine(
      settings, GetAvailableKokoroVoices(mock_dir.path()));

  // 1. Invalid voice returns InvalidArgumentError.
  TtsSessionConfig invalid_voice_config;
  invalid_voice_config.voice = "unknown_speaker_xyz";
  auto status1 = engine->CreateSession(invalid_voice_config);
  EXPECT_FALSE(status1.ok());
  EXPECT_EQ(status1.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status1.status().message(),
              testing::HasSubstr("is not available in TtsEngine"));

  // 2. Conflicting voice and language returns InvalidArgumentError.
  TtsSessionConfig mismatch_config;
  mismatch_config.language = "es";
  mismatch_config.voice = "af_heart";  // American English voice
  auto status2 = engine->CreateSession(mismatch_config);
  EXPECT_FALSE(status2.ok());
  EXPECT_EQ(status2.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status2.status().message(),
              testing::HasSubstr("conflicts with requested language"));

  // 3. Unsupported BCP-47 language returns InvalidArgumentError.
  TtsSessionConfig unsupported_lang_config;
  unsupported_lang_config.language = "de";
  auto status3 = engine->CreateSession(unsupported_lang_config);
  EXPECT_FALSE(status3.ok());
  EXPECT_EQ(status3.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status3.status().message(),
              testing::HasSubstr("Unsupported language"));

  // 4. Empty language returns InvalidArgumentError.
  TtsSessionConfig empty_lang_config;
  empty_lang_config.language = "";
  auto status4 = engine->CreateSession(empty_lang_config);
  EXPECT_FALSE(status4.ok());
  EXPECT_EQ(status4.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status4.status().message(),
              testing::HasSubstr("language must not be empty"));
}

TEST(TtsEngineTest, GetDefaultVoiceStaysWithinRequestedLanguage) {
  TtsEngineSettings settings;
  settings.model_config = KokoroModelConfig{};
  auto engine = TtsEngineTestingPeer::CreateTestEngine(
      settings, {"af_heart", "am_adam", "af_bella"});

  // No Spanish voice is available: the Spanish default is returned rather than
  // an unrelated English voice.
  EXPECT_EQ(engine->GetDefaultVoice("es"), "ef_dora");
  // Without a requested language, any available voice is acceptable.
  EXPECT_EQ(engine->GetDefaultVoice(""), "af_heart");
}

TEST(TtsEngineTest, CreateSessionFailsWhenNoVoiceForRequestedLanguage) {
  TtsEngineSettings settings;
  settings.model_config = KokoroModelConfig{};
  auto engine =
      TtsEngineTestingPeer::CreateTestEngine(settings, {"af_heart", "am_adam"});

  TtsSessionConfig config;
  config.language = "es";
  auto session = engine->CreateSession(config);
  EXPECT_FALSE(session.ok());
  EXPECT_EQ(session.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(session.status().message(),
              testing::HasSubstr("No voice is available for language"));
}

}  // namespace
}  // namespace litert::omni::tts
