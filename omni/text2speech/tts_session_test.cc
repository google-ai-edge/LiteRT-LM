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

#include "omni/text2speech/tts_session.h"

#include <filesystem>  // NOLINT: Required for std::filesystem.
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <system_error>  // NOLINT: Required for std::error_code.
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/text2speech/kokoro.h"
#include "omni/text2speech/kokoro/common.h"
#include "omni/text2speech/kokoro/kokoro_factory.h"
#include "omni/text2speech/qwen3_tts/common.h"
#include "omni/text2speech/stream_text_source.h"
#include "omni/text2speech/text_chunk_utils.h"
#include "omni/text2speech/text_source.h"
#include "omni/text2speech/vocoder.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/text2speech_metadata.pb.h"
#include "runtime/proto/text2speech_session_config.pb.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::tts {

class TtsSessionTest : public ::testing::Test {
 public:
  static std::unique_ptr<StreamTextSource> CreateTextInputSource(
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
      TextChunkConfig config = {}) {
    return TtsSessionFactory::CreateTextInputSource(std::move(input_source),
                                                    std::move(config));
  }
};

namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

// Fake Vocoder that encodes the length of each text chunk as a single float
// sample in `AudioOutput`.
class FakeLengthVocoder : public Vocoder {
 public:
  explicit FakeLengthVocoder(TextSource* text_source)
      : text_source_(*text_source) {}

 protected:
  bool NeedScheduleInternal() const override {
    return text_source_.HasOutput();
  }

  absl::Status ScheduleInternal() override {
    SetState(State::kRunning);
    auto text = text_source_.GetOutput();
    if (absl::IsNotFound(text.status())) {
      SetState(State::kIdle);
      return absl::OkStatus();
    }
    if (!text.ok()) {
      SetState(State::kIdle);
      return text.status();
    }
    PushOutput(AudioOutput{
        .pcm_samples = {static_cast<float>(text->size())},
        .sample_rate_hz = 24000,
    });
    SetState(State::kIdle);
    return absl::OkStatus();
  }

 private:
  TextSource& text_source_;
};

TEST_F(TtsSessionTest, SequentialSynthesizeCallsWithResetAndFlush) {
  auto text_source = std::make_unique<StreamTextSource>();
  StreamTextSource* raw_text_source = text_source.get();
  auto vocoder = std::make_unique<FakeLengthVocoder>(raw_text_source);
  Stage<Output>* raw_vocoder = vocoder.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(text_source));
  stages.push_back(std::move(vocoder));

  ASSERT_OK_AND_ASSIGN(
      auto session, MultiStagedSession::Create(std::move(stages), raw_vocoder));

  // First synthesize call (mirrors JNI nativeTtsSessionSynthesize & mediapipe).
  session->Reset();
  ASSERT_OK(raw_text_source->PushText("Hello"));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out1, session->Flush());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(out1));
  EXPECT_THAT(std::get<AudioOutput>(out1).pcm_samples, ElementsAre(5.0f));

  // Second synthesize call on the same session after Reset().
  session->Reset();
  ASSERT_OK(raw_text_source->PushText("Second call!"));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out2, session->Flush());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(out2));
  EXPECT_THAT(std::get<AudioOutput>(out2).pcm_samples, ElementsAre(12.0f));
}

TEST_F(TtsSessionTest, TextInputSourceProcessNextFlushAndReset) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input_source = input_source.get();
  auto text_source = CreateTextInputSource(std::move(input_source));
  auto vocoder = std::make_unique<FakeLengthVocoder>(text_source.get());
  Stage<Output>* raw_vocoder = vocoder.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(text_source));
  stages.push_back(std::move(vocoder));

  ASSERT_OK_AND_ASSIGN(
      auto session, MultiStagedSession::Create(std::move(stages), raw_vocoder));

  // Non-TextInput is rejected with InvalidArgumentError.
  ASSERT_OK(raw_input_source->PushInput(
      OmniSession::AudioInput{.pcm_samples = {0.1f}}));
  EXPECT_THAT(session->ProcessNext(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // First sentence has a delimiter; trailing fragment does not.
  ASSERT_OK(
      raw_input_source->PushInput(OmniSession::TextInput{.text = "Hi. Tail"}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output chunk1, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(chunk1));
  EXPECT_THAT(std::get<AudioOutput>(chunk1).pcm_samples, ElementsAre(3.0f));

  // Flush() finishes TextInputSource and synthesizes " Tail" (length 5).
  ASSERT_OK_AND_ASSIGN(OmniSession::Output flushed, session->Flush());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(flushed));
  EXPECT_THAT(std::get<AudioOutput>(flushed).pcm_samples, ElementsAre(5.0f));

  // Reset() clears the finished state so a new stream can run on the session.
  session->Reset();
  ASSERT_OK(
      raw_input_source->PushInput(OmniSession::TextInput{.text = "Again"}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output flushed2, session->Flush());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(flushed2));
  EXPECT_THAT(std::get<AudioOutput>(flushed2).pcm_samples, ElementsAre(5.0f));
}

TEST_F(TtsSessionTest,
       PartialFragmentsAndCoalescedInputsWaitUntilDelimiterOrEndOfInput) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input_source = input_source.get();
  auto text_source = CreateTextInputSource(std::move(input_source));
  auto vocoder = std::make_unique<FakeLengthVocoder>(text_source.get());
  Stage<Output>* raw_vocoder = vocoder.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(text_source));
  stages.push_back(std::move(vocoder));

  ASSERT_OK_AND_ASSIGN(
      auto session, MultiStagedSession::Create(std::move(stages), raw_vocoder));

  // Partial fragment without sentence delimiter before EndOfInput returns
  // NotFoundError rather than prematurely finishing or returning OutOfRange.
  ASSERT_OK(
      raw_input_source->PushInput(OmniSession::TextInput{.text = "Hello "}));
  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kNotFound));

  // Multiple consecutive TextInputs pushed before ProcessNext() are coalesced.
  ASSERT_OK(
      raw_input_source->PushInput(OmniSession::TextInput{.text = "world"}));
  ASSERT_OK(
      raw_input_source->PushInput(OmniSession::TextInput{.text = "! Tail"}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output chunk1, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(chunk1));
  // "Hello world!" has 12 characters.
  EXPECT_THAT(std::get<AudioOutput>(chunk1).pcm_samples, ElementsAre(12.0f));

  // Remaining " Tail" has no delimiter yet and waits for more input or Finish.
  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kNotFound));

  raw_input_source->Finish();
  ASSERT_OK_AND_ASSIGN(OmniSession::Output chunk2, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(chunk2));
  // " Tail" has 5 characters.
  EXPECT_THAT(std::get<AudioOutput>(chunk2).pcm_samples, ElementsAre(5.0f));

  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kOutOfRange));
}

TEST_F(TtsSessionTest, TextInputSourceProcessAsyncStreamsAndFlushesTail) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input_source = input_source.get();
  auto text_source = CreateTextInputSource(std::move(input_source));
  auto vocoder = std::make_unique<FakeLengthVocoder>(text_source.get());
  Stage<Output>* raw_vocoder = vocoder.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(text_source));
  stages.push_back(std::move(vocoder));

  ::litert::lm::ThreadPool pool("tts_omni_test_pool", 2);
  ASSERT_OK_AND_ASSIGN(
      auto session,
      MultiStagedSession::Create(std::move(stages), raw_vocoder, &pool));

  ASSERT_OK(
      raw_input_source->PushInput(OmniSession::TextInput{.text = "One. Two"}));
  raw_input_source->Finish();

  std::vector<float> lengths;
  absl::Status final_status;
  absl::Notification done;

  ASSERT_OK(session->ProcessAsync(
      [&](absl::StatusOr<OmniSession::Output> res) -> absl::Status {
        if (!res.ok()) {
          final_status = res.status();
          done.Notify();
          return res.status();
        }
        if (const auto* audio = std::get_if<AudioOutput>(&*res)) {
          lengths.insert(lengths.end(), audio->pcm_samples.begin(),
                         audio->pcm_samples.end());
        }
        return absl::OkStatus();
      }));

  done.WaitForNotification();
  EXPECT_THAT(final_status, StatusIs(absl::StatusCode::kOutOfRange));
  // "One." (4 chars) and " Two" (4 chars).
  EXPECT_THAT(lengths, ElementsAre(4.0f, 4.0f));
}

TEST_F(TtsSessionTest, KokoroAndQwen3TtsRegisteredInOmniRegistry) {
  EXPECT_THAT(OmniRegistry::Get().GetRegisteredModels(),
              testing::IsSupersetOf({"kokoro", "kokoro-82m", "qwen3-tts"}));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("kokoro"));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("qwen3-tts"));
  EXPECT_FALSE(OmniRegistry::Get().IsModelRegistered("qwen3"));
  EXPECT_FALSE(OmniRegistry::Get().IsModelRegistered("unknown-tts-model"));

  ASSERT_OK_AND_ASSIGN(const auto kokoro_reg,
                       OmniRegistry::Get().GetRegisteredModel("kokoro"));
  EXPECT_EQ(kokoro_reg.model_remote_url,
            "https://huggingface.co/litert-community/Kokoro-82M/resolve/main/"
            "kokoro_int8.litertlm");
  ASSERT_OK_AND_ASSIGN(const auto qwen3_reg,
                       OmniRegistry::Get().GetRegisteredModel("qwen3-tts"));
  EXPECT_EQ(qwen3_reg.model_remote_url,
            "https://huggingface.co/litert-community/"
            "Qwen3-TTS-12Hz-0.6B-Base/resolve/main/qwen3_tts_int8.litertlm");

  OmniEngine::Settings settings;
  settings.model_path = "/nonexistent/tts_model_dir";
  for (const auto& model_name :
       {"kokoro", "kokoro-82m", "qwen3-tts", "qwen3"}) {
    EXPECT_THAT(OmniRegistry::Get().CreateSessionFactory(model_name, settings),
                StatusIs(absl::StatusCode::kNotFound));
  }
}

TEST_F(TtsSessionTest, LitertLmModelPathIsNotTreatedAsModelDir) {
  OmniEngine::Settings settings;
  settings.model_path = "/nonexistent/tts_model.litertlm";
  for (const auto& model_name : {"kokoro", "qwen3-tts"}) {
    auto status_or =
        OmniRegistry::Get().CreateSessionFactory(model_name, settings);
    EXPECT_THAT(status_or, StatusIs(absl::StatusCode::kNotFound));
    EXPECT_THAT(status_or.status().message(),
                testing::HasSubstr("/nonexistent/tts_model.litertlm"));
    EXPECT_THAT(status_or.status().message(),
                testing::Not(testing::HasSubstr(".tflite")));
  }
}

TEST_F(TtsSessionTest, Text2SpeechSessionConfigProtoConversion) {
  lm::proto::Text2SpeechSessionConfig config;
  config.set_language("es");
  config.set_voice("ef_dora");
  auto* tc = config.mutable_text_chunk_config();
  tc->set_max_buffer_size(120);
  tc->add_delimiters(".");
  tc->add_delimiters("!");
  tc->add_delimiters("?");

  EXPECT_EQ(config.language(), "es");
  EXPECT_EQ(config.voice(), "ef_dora");
  TextChunkConfig converted = ToTextChunkConfig(config.text_chunk_config());
  EXPECT_EQ(converted.max_buffer_size, 120);
  EXPECT_THAT(converted.delimiters, ElementsAre(".", "!", "?"));
}

TEST_F(TtsSessionTest, ToKokoroLanguageCodeMapping) {
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

TEST_F(TtsSessionTest, KokoroCodeToBcp47Mapping) {
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

// Writes a zero-filled Kokoro voice pack (`kVoicePackSize` floats) to `path`.
void WriteMockVoicePack(const std::filesystem::path& path) {
  std::ofstream file(path, std::ios::binary);
  const std::string payload(kokoro::kVoicePackSize * sizeof(float), '\0');
  file.write(payload.data(), payload.size());
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
      WriteMockVoicePack(voices_dir / (name + ".bin"));
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

TEST_F(TtsSessionTest, EmptyModelFolderReturnsNoVoices) {
  auto voices = GetAvailableKokoroVoices("");
  EXPECT_TRUE(voices.empty());
}

TEST_F(TtsSessionTest, DiscoversKokoroVoicesFromDirectory) {
  std::filesystem::path model_dir =
      std::filesystem::path(testing::TempDir()) / "kokoro_test_dir";
  std::filesystem::path voices_dir = model_dir / "voices";
  std::error_code ec;
  std::filesystem::create_directories(voices_dir, ec);

  WriteMockVoicePack(voices_dir / "af_custom.bin");
  WriteMockVoicePack(voices_dir / "zm_custom.bin");
  // Not voice packs: wrong extension, unconventional name, or wrong size.
  WriteMockVoicePack(voices_dir / "bf_readme.txt");
  WriteMockVoicePack(voices_dir / "custom_voice.bin");
  std::ofstream(voices_dir / "am_empty.bin").close();

  auto discovered = GetAvailableKokoroVoices(model_dir.string());
  EXPECT_THAT(discovered, ElementsAre("af_custom", "zm_custom"));

  std::filesystem::remove_all(model_dir, ec);
}

TEST_F(TtsSessionTest, DefaultKokoroVoicesPerLanguage) {
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

TEST_F(TtsSessionTest, KokoroSessionFactoryLanguagesAndVoices) {
  ScopedMockKokoroModelDir mock_dir;
  KokoroSessionFactory factory(OmniEngine::Settings{},
                               /*session_config=*/nullptr,
                               lm::proto::Text2SpeechMetadata{},
                               GetAvailableKokoroVoices(mock_dir.path()),
                               /*model_resources=*/nullptr);

  EXPECT_THAT(factory.GetAvailableLanguages(),
              ElementsAre("en-GB", "en-US", "es", "fr", "hi", "it", "ja",
                          "pt-BR", "zh-CN"));

  KokoroSessionFactory subset_factory(
      OmniEngine::Settings{}, /*session_config=*/nullptr,
      lm::proto::Text2SpeechMetadata{}, {"af_heart", "ef_dora", "em_alex"},
      /*model_resources=*/nullptr);
  EXPECT_THAT(subset_factory.GetAvailableLanguages(),
              ElementsAre("en-US", "es"));

  lm::proto::Text2SpeechMetadata meta;
  meta.add_supported_languages("en-US");
  meta.add_supported_languages("zh-CN");
  meta.add_supported_languages("ja");
  KokoroSessionFactory meta_factory(OmniEngine::Settings{},
                                    /*session_config=*/nullptr, meta, {},
                                    /*model_resources=*/nullptr);
  EXPECT_THAT(meta_factory.GetAvailableLanguages(),
              ElementsAre("en-US", "zh-CN", "ja"));

  auto voices = factory.GetAvailableVoices();
  EXPECT_EQ(voices.size(), 19);
  EXPECT_THAT(voices, testing::Contains("af_heart"));
  EXPECT_THAT(voices, testing::Contains("ef_dora"));
  EXPECT_THAT(voices, testing::Contains("zf_xiaoxiao"));

  EXPECT_THAT(factory.GetAvailableVoices("es"),
              ElementsAre("ef_dora", "em_alex", "em_santa"));
  EXPECT_THAT(factory.GetAvailableVoices("it"),
              ElementsAre("if_sara", "im_nicola"));
  EXPECT_THAT(factory.GetAvailableVoices("fr"), ElementsAre("ff_siwis"));
  EXPECT_THAT(factory.GetAvailableVoices("hi"),
              ElementsAre("hf_alpha", "hf_beta", "hm_omega", "hm_psi"));
  EXPECT_THAT(factory.GetAvailableVoices("pt-BR"),
              ElementsAre("pf_dora", "pm_alex", "pm_santa"));
  EXPECT_THAT(factory.GetAvailableVoices("en-GB"), ElementsAre("bf_alice"));
  EXPECT_THAT(factory.GetAvailableVoices("en-US"),
              ElementsAre("af_alloy", "af_heart"));
  EXPECT_THAT(factory.GetAvailableVoices("zh-CN"),
              ElementsAre("zf_xiaobei", "zf_xiaoxiao"));
  EXPECT_THAT(factory.GetAvailableVoices("ja"), ElementsAre("jf_alpha"));
  EXPECT_TRUE(factory.GetAvailableVoices("de").empty());

  EXPECT_TRUE(factory.HasVoice("af_heart"));
  EXPECT_TRUE(factory.HasVoice("af_heart.bin"));
  EXPECT_TRUE(factory.HasVoice("voices/af_heart.bin"));
  EXPECT_FALSE(factory.HasVoice("non_existent_voice"));
  EXPECT_FALSE(factory.HasVoice(""));

  EXPECT_EQ(factory.GetDefaultVoice("es"), "ef_dora");
  EXPECT_EQ(factory.GetDefaultVoice("it"), "if_sara");
  EXPECT_EQ(factory.GetDefaultVoice("fr"), "ff_siwis");
  EXPECT_EQ(factory.GetDefaultVoice("hi"), "hf_alpha");
  EXPECT_EQ(factory.GetDefaultVoice("pt-BR"), "pf_dora");
  EXPECT_EQ(factory.GetDefaultVoice("en-GB"), "bf_alice");
  EXPECT_EQ(factory.GetDefaultVoice("zh-CN"), "zf_xiaobei");
  EXPECT_EQ(factory.GetDefaultVoice("ja"), "jf_alpha");
  EXPECT_EQ(factory.GetDefaultVoice("en-US"), "af_heart");
  EXPECT_EQ(factory.GetDefaultVoice(""), "af_heart");
}

TEST_F(TtsSessionTest, KokoroSessionFactoryVoiceAndLanguageValidation) {
  ScopedMockKokoroModelDir mock_dir;
  const auto voices = GetAvailableKokoroVoices(mock_dir.path());

  // 1. Invalid voice returns InvalidArgumentError.
  lm::proto::Text2SpeechSessionConfig invalid_voice_config;
  invalid_voice_config.set_voice("unknown_speaker_xyz");
  KokoroSessionFactory factory1(OmniEngine::Settings{}, &invalid_voice_config,
                                lm::proto::Text2SpeechMetadata{}, voices,
                                /*model_resources=*/nullptr);
  auto status1 = factory1.Create(std::make_unique<PushInputSource>());
  EXPECT_THAT(status1, StatusIs(absl::StatusCode::kInvalidArgument,
                                testing::HasSubstr("is not available")));

  // 2. Conflicting voice and language returns InvalidArgumentError.
  lm::proto::Text2SpeechSessionConfig mismatch_config;
  mismatch_config.set_language("es");
  mismatch_config.set_voice("af_heart");  // American English voice
  KokoroSessionFactory factory2(OmniEngine::Settings{}, &mismatch_config,
                                lm::proto::Text2SpeechMetadata{}, voices,
                                /*model_resources=*/nullptr);
  auto status2 = factory2.Create(std::make_unique<PushInputSource>());
  EXPECT_THAT(
      status2,
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr("conflicts with requested language")));

  // 3. Unsupported BCP-47 language returns InvalidArgumentError.
  lm::proto::Text2SpeechSessionConfig unsupported_lang_config;
  unsupported_lang_config.set_language("de");
  KokoroSessionFactory factory3(OmniEngine::Settings{},
                                &unsupported_lang_config,
                                lm::proto::Text2SpeechMetadata{}, voices,
                                /*model_resources=*/nullptr);
  auto status3 = factory3.Create(std::make_unique<PushInputSource>());
  EXPECT_THAT(status3, StatusIs(absl::StatusCode::kInvalidArgument,
                                testing::HasSubstr("Unsupported language")));

  // 4. Empty language returns InvalidArgumentError.
  lm::proto::Text2SpeechSessionConfig empty_lang_config;
  empty_lang_config.set_language("");
  KokoroSessionFactory factory4(OmniEngine::Settings{}, &empty_lang_config,
                                lm::proto::Text2SpeechMetadata{}, voices,
                                /*model_resources=*/nullptr);
  auto status4 = factory4.Create(std::make_unique<PushInputSource>());
  EXPECT_THAT(status4,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("language must not be empty")));

  // 5. No voice available for requested language returns InvalidArgumentError.
  lm::proto::Text2SpeechSessionConfig es_config;
  es_config.set_language("es");
  KokoroSessionFactory factory5(OmniEngine::Settings{}, &es_config,
                                lm::proto::Text2SpeechMetadata{},
                                {"af_heart", "am_adam"},
                                /*model_resources=*/nullptr);
  EXPECT_EQ(factory5.GetDefaultVoice("es"), "ef_dora");
  EXPECT_EQ(factory5.GetDefaultVoice(""), "af_heart");
  auto status5 = factory5.Create(std::make_unique<PushInputSource>());
  EXPECT_THAT(
      status5,
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::HasSubstr("No voice is available for language")));
}

TEST_F(TtsSessionTest, Qwen3TtsLanguageResolutionAndIds) {
  EXPECT_THAT(ResolveQwen3TtsLanguage("en-US"),
              absl_testing::IsOkAndHolds("english"));
  EXPECT_THAT(ResolveQwen3TtsLanguage("zh-CN"),
              absl_testing::IsOkAndHolds("chinese"));
  EXPECT_THAT(ResolveQwen3TtsLanguage("cmn"),
              absl_testing::IsOkAndHolds("chinese"));
  EXPECT_THAT(ResolveQwen3TtsLanguage("ja-JP"),
              absl_testing::IsOkAndHolds("japanese"));
  EXPECT_THAT(ResolveQwen3TtsLanguage("ko"),
              absl_testing::IsOkAndHolds("korean"));
  EXPECT_THAT(ResolveQwen3TtsLanguage("pt-BR"),
              absl_testing::IsOkAndHolds("portuguese"));
  EXPECT_THAT(ResolveQwen3TtsLanguage("auto"),
              absl_testing::IsOkAndHolds("auto"));
  EXPECT_THAT(ResolveQwen3TtsLanguage("english"),
              absl_testing::IsOkAndHolds("english"));
  EXPECT_THAT(ResolveQwen3TtsLanguage("hi"),
              StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(GetLanguageId("english"), absl_testing::IsOkAndHolds(2050));
  EXPECT_THAT(GetLanguageId("chinese"), absl_testing::IsOkAndHolds(2055));
  EXPECT_THAT(GetLanguageId("japanese"), absl_testing::IsOkAndHolds(2058));
  EXPECT_THAT(GetLanguageId("korean"), absl_testing::IsOkAndHolds(2064));
  EXPECT_THAT(GetLanguageId("unknown"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::omni::tts
