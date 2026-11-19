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

#include "omni/any2any/any2any_session.h"

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "omni/any2any/any2any_test_utils.h"
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/any2any_model_type.pb.h"
#include "runtime/proto/any2any_session_config.pb.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK
#include "tflite/schema/schema_generated.h"  // from @litert

namespace litert::omni::any2any {

class Any2AnySessionTest : public ::testing::Test {
 protected:
  static absl::StatusOr<std::unique_ptr<MultiStagedSession>>
  CallCreateWithInputSource(
      Any2AnySessionFactory& factory,
      const lm::proto::Any2AnySessionConfig& session_config,
      std::unique_ptr<OmniSession::InputSource> input_source) {
    return factory.CreateWithInputSource(session_config,
                                         std::move(input_source));
  }
};

namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

constexpr absl::string_view kMinimalTokenizerJson = R"json({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 5, "content": "\n", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<|endoftext|>": 0,
      "<|im_start|>": 1,
      "<|im_end|>": 2,
      "Hello": 3,
      "world": 4,
      "\n": 5,
      "t6": 6, "t7": 7, "t8": 8, "t9": 9, "t10": 10, "t11": 11, "t12": 12,
      "t13": 13, "t14": 14, "t15": 15, "t16": 16, "t17": 17, "t18": 18,
      "t19": 19, "t20": 20, "t21": 21, "t22": 22, "t23": 23, "t24": 24,
      "t25": 25, "t26": 26, "t27": 27, "t28": 28, "t29": 29, "t30": 30,
      "t31": 31
    },
    "unk_token": "<|endoftext|>"
  }
})json";

absl::StatusOr<std::string> WriteMiniCpmOBundleWithEmbeddedMetadata(
    absl::string_view filename,
    const std::vector<std::string>& supported_languages = {}) {
  lm::proto::Any2AnyMetadata any2any_meta;
  for (const std::string& lang : supported_languages) {
    any2any_meta.add_supported_languages(lang);
  }
  any2any_meta.set_input_sample_rate_hz(24000);
  any2any_meta.set_output_sample_rate_hz(24000);
  any2any_meta.set_system_prompt("Embedded system prompt");
  auto* mcpm = any2any_meta.mutable_any2any_model_type()->mutable_minicpm_o();
  mcpm->set_llm_hidden_size(4);
  mcpm->set_llm_vocab_size(32);
  mcpm->set_tts_hidden_size(4);
  mcpm->set_tts_vocab_size(8);
  mcpm->set_tts_num_mel_bins(80);
  mcpm->set_vision_scale_resolution(448);
  mcpm->set_vision_patch_size(2);
  mcpm->set_vision_max_slice_nums(2);
  mcpm->set_num_image_queries(2);
  mcpm->set_audio_pool_step(5);
  mcpm->set_generate_audio(false);
  mcpm->set_max_new_tokens(1);
  mcpm->set_max_tts_tokens(1);
  mcpm->set_voice("embedded_voice");
  mcpm->set_language("en-US");

  Any2AnyBundleOptions options{
      .metadata = std::move(any2any_meta),
      .tokenizer_json = std::string(kMinimalTokenizerJson),
      .extra_sections =
          {
              BundleSectionPayload{
                  .data_type = lm::schema::AnySectionDataType_TFLiteModel,
                  .model_type = "tf_lite_vision_encoder",
                  .bytes = BuildMinimalTfliteModelBuffer(
                      {{"images", {1, 4, 12}, tflite::TensorType_FLOAT32},
                       {"positions_xy", {1, 4, 2}, tflite::TensorType_INT32},
                       {"vit_positions", {1, 4}, tflite::TensorType_INT32}},
                      {{"features", {1, 2, 4}, tflite::TensorType_FLOAT32}}),
              },
              BundleSectionPayload{
                  .data_type = lm::schema::AnySectionDataType_TFLiteModel,
                  .model_type = "tf_lite_audio_encoder",
                  .bytes = BuildMinimalTfliteModelBuffer(
                      {{"mel", {1, 80, 10}, tflite::TensorType_FLOAT32}},
                      {{"features", {1, 2, 4}, tflite::TensorType_FLOAT32}}),
              },
          },
  };
  return WriteAny2AnyTestBundle(filename, std::move(options));
}

absl::StatusOr<std::string> WriteQwenOmniBundleWithEmbeddedMetadata(
    absl::string_view filename, int llm_vocab_size = 32,
    const std::vector<std::string>& supported_languages = {}) {
  lm::proto::Any2AnyMetadata any2any_meta;
  for (const std::string& lang : supported_languages) {
    any2any_meta.add_supported_languages(lang);
  }
  any2any_meta.set_input_sample_rate_hz(24000);
  any2any_meta.set_output_sample_rate_hz(24000);
  any2any_meta.set_system_prompt("Embedded Qwen system prompt");
  auto* qwen = any2any_meta.mutable_any2any_model_type()->mutable_qwen_omni();
  qwen->set_llm_hidden_size(4);
  qwen->set_llm_vocab_size(llm_vocab_size);
  qwen->set_tts_hidden_size(4);
  qwen->set_tts_vocab_size(8);
  qwen->set_tts_num_mel_bins(80);
  qwen->set_tts_num_codebooks(1);
  qwen->set_vision_image_size(28);
  qwen->set_vision_patch_size(14);
  qwen->set_vision_spatial_merge_size(2);
  qwen->set_audio_num_mel_bins(128);
  qwen->set_audio_chunk_frames(10);
  qwen->set_audio_pool_step(2);
  qwen->set_generate_audio(false);
  qwen->set_max_new_tokens(1);
  qwen->set_max_tts_tokens(1);
  qwen->set_voice("Chelsie");
  qwen->set_language("en-US");
  qwen->set_tts_text_bos_token_id(15);

  Any2AnyBundleOptions options{
      .metadata = std::move(any2any_meta),
      .tokenizer_json = std::string(kMinimalTokenizerJson),
      .extra_sections =
          {
              BundleSectionPayload{
                  .data_type = lm::schema::AnySectionDataType_TFLiteModel,
                  .model_type = "tf_lite_vision_encoder",
                  .bytes = BuildMinimalTfliteModelBuffer(
                      {{"pixel_values",
                        {1, 3, 28, 28},
                        tflite::TensorType_FLOAT32}},
                      {{"vision_embeds",
                        {1, 1, 4},
                        tflite::TensorType_FLOAT32}}),
              },
              BundleSectionPayload{
                  .data_type = lm::schema::AnySectionDataType_TFLiteModel,
                  .model_type = "tf_lite_audio_encoder",
                  .bytes = BuildMinimalTfliteModelBuffer(
                      {{"mel_features",
                        {1, 128, 10},
                        tflite::TensorType_FLOAT32}},
                      {{"audio_embeds",
                        {1, 2, 4},
                        tflite::TensorType_FLOAT32}}),
              },
          },
  };
  return WriteAny2AnyTestBundle(filename, std::move(options));
}

// Fake Any2Any processing stage that pulls `Input` items from `InputSource`
// until `EndOfInputTurn`, and emits `TextOutput`, optionally `AudioOutput`, and
// `EndOfOutputTurn`.
class FakeAny2AnyStage : public SingleThreadedStageWithDeque<Output> {
 public:
  explicit FakeAny2AnyStage(Stage<Input>* input_source, bool emit_audio = false)
      : input_source_(*input_source), emit_audio_(emit_audio) {}

 protected:
  bool NeedScheduleInternal() const override {
    return !is_finished_ &&
           (input_source_.NeedSchedule() || input_source_.HasOutput());
  }

  absl::Status ScheduleInternal() override {
    SetState(State::kRunning);

    while (!is_finished_) {
      if (!input_source_.HasOutput()) {
        if (!input_source_.NeedSchedule()) {
          break;
        }
        absl::Status status = input_source_.Schedule();
        if (absl::IsOutOfRange(status)) {
          is_finished_ = true;
          break;
        }
        if (absl::IsNotFound(status)) {
          break;
        }
        ABSL_RETURN_IF_ERROR(status);
        if (!input_source_.HasOutput()) {
          break;
        }
      }

      ABSL_ASSIGN_OR_RETURN(Input input, input_source_.GetOutput());
      if (std::holds_alternative<EndOfInput>(input)) {
        is_finished_ = true;
        break;
      }
      if (std::holds_alternative<EndOfInputTurn>(input)) {
        EmitTurnOutput();
        SetState(State::kIdle);
        return absl::OkStatus();
      }
      if (const auto* aud_meta = std::get_if<AudioInputMetadata>(&input)) {
        current_sample_rate_ = aud_meta->sample_rate_hz;
      } else if (const auto* txt = std::get_if<TextInput>(&input)) {
        if (!summary_.empty()) summary_ += '|';
        absl::StrAppend(&summary_, "text:", txt->text);
      } else if (const auto* img = std::get_if<ImageInput>(&input)) {
        if (!summary_.empty()) summary_ += '|';
        absl::StrAppend(&summary_, "img:", img->image_bytes.size(), "x",
                        img->width, "x", img->height);
      } else if (const auto* aud = std::get_if<AudioInput>(&input)) {
        if (!summary_.empty()) summary_ += '|';
        absl::StrAppend(&summary_, "aud:", aud->pcm_samples.size(), "@",
                        current_sample_rate_);
      }
    }

    if (is_finished_) {
      if (!summary_.empty()) {
        EmitTurnOutput();
      }
      SetState(State::kIdle);
      if (HasOutput()) {
        return absl::OkStatus();
      }
      return absl::OutOfRangeError("End of input reached.");
    }

    SetState(State::kIdle);
    return absl::OkStatus();
  }

  void ResetInternal() override {
    summary_.clear();
    current_sample_rate_ = kDefaultInputSampleRateHz;
    is_finished_ = false;
  }

 private:
  void EmitTurnOutput() {
    PushOutput(TextOutput{.confirmed_text = std::move(summary_)});
    summary_.clear();
    if (emit_audio_) {
      PushOutput(AudioOutput{
          .pcm_samples = {0.25f, -0.25f},
          .sample_rate_hz = 24000,
      });
    }
    PushOutput(EndOfOutputTurn{});
  }

  Stage<Input>& input_source_;
  bool emit_audio_ = false;
  std::string summary_;
  int current_sample_rate_ = kDefaultInputSampleRateHz;
  bool is_finished_ = false;
};

TEST_F(Any2AnySessionTest,
       PushInputSourceEndOfTurnEnablesMultiTurnWithoutReset) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  auto any2any_stage = std::make_unique<FakeAny2AnyStage>(input_source.get(),
                                                          /*emit_audio=*/true);
  Stage<Output>* raw_out = any2any_stage.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(input_source));
  stages.push_back(std::move(any2any_stage));

  ASSERT_OK_AND_ASSIGN(auto session,
                       MultiStagedSession::Create(std::move(stages), raw_out));

  // Before EndOfInputTurn or Finish(), ProcessNext() waits for turn completion
  // and returns kNotFound.
  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "Describe "}));
  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "this:"}));
  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kNotFound));

  ASSERT_OK(raw_input->PushInput(OmniSession::ImageInput{
      .image_bytes = {255, 0, 0, 0, 255, 0},
      .width = 2,
      .height = 1,
      .channels = 3,
  }));
  ASSERT_OK(raw_input->PushInput(
      OmniSession::AudioInputMetadata{.sample_rate_hz = 24000}));
  ASSERT_OK(raw_input->PushInput(
      OmniSession::AudioInput{.pcm_samples = {0.1f, 0.2f, 0.3f}}));

  // Signal end of turn with EndOfInputTurn control input.
  ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));

  // First ProcessNext() yields the TextOutput for turn 1.
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out1, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out1));
  EXPECT_EQ(std::get<TextOutput>(out1).confirmed_text,
            "text:Describe |text:this:|img:6x2x1|aud:3@24000");

  // Second ProcessNext() yields the AudioOutput for turn 1.
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out2, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(out2));
  EXPECT_EQ(std::get<AudioOutput>(out2).sample_rate_hz, 24000);
  EXPECT_THAT(std::get<AudioOutput>(out2).pcm_samples,
              ElementsAre(0.25f, -0.25f));

  // Third ProcessNext() yields EndOfOutputTurn for turn 1.
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out_turn1, session->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutputTurn>(out_turn1));

  // Turn 2 on the SAME session (no Reset() called):
  ASSERT_OK(raw_input->PushInput(
      OmniSession::TextInput{.text = "Followup question"}));
  ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));

  ASSERT_OK_AND_ASSIGN(OmniSession::Output out3, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out3));
  EXPECT_EQ(std::get<TextOutput>(out3).confirmed_text,
            "text:Followup question");
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out4, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(out4));

  ASSERT_OK_AND_ASSIGN(OmniSession::Output out_turn2, session->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutputTurn>(out_turn2));

  // Calling Finish() ends the session.
  raw_input->Finish();
  EXPECT_THAT(session->ProcessNext(), StatusIs(absl::StatusCode::kOutOfRange));
}

TEST_F(Any2AnySessionTest, PushInputSourceProcessAsyncStreamsTextAndAudio) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  auto any2any_stage = std::make_unique<FakeAny2AnyStage>(input_source.get(),
                                                          /*emit_audio=*/true);
  Stage<Output>* raw_out = any2any_stage.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(input_source));
  stages.push_back(std::move(any2any_stage));

  ::litert::lm::ThreadPool pool("any2any_test_pool", 2);
  ASSERT_OK_AND_ASSIGN(auto session, MultiStagedSession::Create(
                                         std::move(stages), raw_out, &pool));

  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "Async"}));
  ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));
  raw_input->Finish();

  std::vector<std::string> texts;
  std::vector<float> audio_samples;
  int end_of_turn_count = 0;
  absl::Status final_status;
  absl::Notification done;

  ASSERT_OK(session->ProcessAsync(
      [&](absl::StatusOr<OmniSession::Output> res) -> absl::Status {
        if (!res.ok()) {
          final_status = res.status();
          done.Notify();
          return res.status();
        }
        if (const auto* txt = std::get_if<TextOutput>(&*res)) {
          texts.push_back(txt->confirmed_text);
        } else if (const auto* aud = std::get_if<AudioOutput>(&*res)) {
          audio_samples.insert(audio_samples.end(), aud->pcm_samples.begin(),
                               aud->pcm_samples.end());
        } else if (std::holds_alternative<OmniSession::EndOfOutputTurn>(*res)) {
          ++end_of_turn_count;
        }
        return absl::OkStatus();
      }));

  done.WaitForNotification();
  EXPECT_THAT(final_status, StatusIs(absl::StatusCode::kOutOfRange));
  EXPECT_THAT(texts, ElementsAre("text:Async"));
  EXPECT_THAT(audio_samples, ElementsAre(0.25f, -0.25f));
  EXPECT_EQ(end_of_turn_count, 1);
}

class FakeAny2AnySessionFactory : public Any2AnySessionFactory {
 public:
  FakeAny2AnySessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const lm::proto::Any2AnyMetadata& metadata,
      std::vector<std::string> available_voices = {},
      std::shared_ptr<ModelResources> absl_nullable model_resources = nullptr)
      : Any2AnySessionFactory(settings, session_config, metadata,
                              std::move(available_voices),
                              std::move(model_resources)) {}

 protected:
  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateWithInputSource(
      const lm::proto::Any2AnySessionConfig& session_config,
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source)
      override {
    auto any2any_stage = std::make_unique<FakeAny2AnyStage>(
        input_source.get(), session_config.generate_audio());
    Stage<Output>* raw_output_stage = any2any_stage.get();
    std::vector<std::unique_ptr<internal::StageBase>> stages;
    stages.push_back(std::move(input_source));
    stages.push_back(std::move(any2any_stage));
    return MultiStagedSession::Create(std::move(stages), raw_output_stage,
                                      thread_pool());
  }
};

TEST_F(Any2AnySessionTest, FactoryMetadataAndConfigMerging) {
  lm::proto::Any2AnyMetadata metadata;
  metadata.add_supported_languages("en-US");
  metadata.add_supported_languages("zh-CN");
  metadata.set_input_sample_rate_hz(48000);

  lm::proto::Any2AnySessionConfig base_cfg;
  base_cfg.set_language("en-US");
  base_cfg.set_generate_audio(true);
  base_cfg.set_temperature(0.7f);

  OmniEngine::Settings settings;
  settings.num_threads = 2;

  FakeAny2AnySessionFactory factory(settings, &base_cfg, metadata,
                                    {"voice_a", "voice_b"},
                                    /*model_resources=*/nullptr);

  EXPECT_THAT(factory.GetAvailableLanguages(), ElementsAre("en-US", "zh-CN"));
  EXPECT_THAT(factory.available_voices(), ElementsAre("voice_a", "voice_b"));
  EXPECT_TRUE(factory.HasVoice("voice_a"));
  EXPECT_TRUE(factory.HasVoice("VOICE_A"));
  EXPECT_TRUE(factory.HasVoice("voice_b"));
  EXPECT_TRUE(factory.HasVoice("Voice_B"));
  EXPECT_FALSE(factory.HasVoice("unknown_voice"));
  EXPECT_EQ(factory.GetDefaultVoice("en-US"), "voice_a");
  EXPECT_EQ(factory.GetInputSampleRateHz(), 48000);
  EXPECT_EQ(factory.engine_settings().num_threads, 2);
  EXPECT_EQ(factory.model_resources(), nullptr);

  lm::proto::Any2AnySessionConfig override_cfg;
  override_cfg.set_generate_audio(false);
  override_cfg.set_temperature(0.2f);
  lm::proto::Any2AnySessionConfig merged =
      factory.MergeSessionConfig(override_cfg);
  EXPECT_EQ(merged.language(), "en-US");
  EXPECT_FALSE(merged.generate_audio());
  EXPECT_FLOAT_EQ(merged.temperature(), 0.2f);
}

TEST_F(Any2AnySessionTest, FactoryDefaultsWhenEmpty) {
  lm::proto::Any2AnyMetadata metadata;
  FakeAny2AnySessionFactory factory(OmniEngine::Settings{},
                                    /*session_config=*/nullptr, metadata);

  EXPECT_TRUE(factory.GetAvailableLanguages().empty());
  EXPECT_TRUE(factory.available_voices().empty());
  EXPECT_FALSE(factory.HasVoice("voice_a"));
  EXPECT_EQ(factory.GetDefaultVoice("en-US"), "");
  EXPECT_EQ(factory.GetInputSampleRateHz(), kDefaultInputSampleRateHz);
  EXPECT_EQ(factory.session_config().language(), "");
}

TEST_F(Any2AnySessionTest, FactoryCreateAndExecuteSession) {
  lm::proto::Any2AnyMetadata metadata;
  lm::proto::Any2AnySessionConfig session_config;
  session_config.set_generate_audio(true);

  FakeAny2AnySessionFactory factory(OmniEngine::Settings{}, &session_config,
                                    metadata, {"voice_a"});

  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();

  ASSERT_OK_AND_ASSIGN(auto session, factory.Create(std::move(input_source)));

  ASSERT_OK(
      raw_input->PushInput(OmniSession::TextInput{.text = "Test via Factory"}));
  ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));

  ASSERT_OK_AND_ASSIGN(OmniSession::Output text_out, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(text_out));
  EXPECT_EQ(std::get<TextOutput>(text_out).confirmed_text,
            "text:Test via Factory");

  ASSERT_OK_AND_ASSIGN(OmniSession::Output audio_out, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<AudioOutput>(audio_out));

  ASSERT_OK_AND_ASSIGN(OmniSession::Output end_out, session->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutputTurn>(end_out));
}

TEST_F(Any2AnySessionTest, MiniCpmORegisteredInOmniRegistry) {
  EXPECT_THAT(OmniRegistry::Get().GetRegisteredModels(),
              testing::IsSupersetOf(
                  {"minicpm-o-4.5", "minicpm-o-4_5", "minicpm-o", "minicpm_o",
                   "openbmb/minicpm-o-4.5", "openbmb/MiniCPM-o-4_5",
                   "openbmb/MiniCPM-o-4.5"}));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("minicpm-o-4.5"));
  EXPECT_FALSE(OmniRegistry::Get().IsModelRegistered("unknown-any2any-model"));

  ASSERT_OK_AND_ASSIGN(const auto reg,
                       OmniRegistry::Get().GetRegisteredModel("minicpm-o-4.5"));
  EXPECT_EQ(reg.model_remote_url,
            "https://huggingface.co/litert-community/MiniCPM-o-4_5/resolve/"
            "main/minicpm_o_4_5_int8.litertlm");

  OmniEngine::Settings dir_settings;
  dir_settings.model_path = "/nonexistent/minicpm_o_dir";
  EXPECT_THAT(
      OmniRegistry::Get().CreateSessionFactory("minicpm-o-4.5", dir_settings),
      StatusIs(absl::StatusCode::kInvalidArgument));

  OmniEngine::Settings file_settings;
  file_settings.model_path = "/nonexistent/minicpm_o.litertlm";
  EXPECT_FALSE(OmniRegistry::Get()
                   .CreateSessionFactory("minicpm-o-4.5", file_settings)
                   .ok());
}

TEST_F(Any2AnySessionTest,
       MiniCpmOSessionFactoryResolvesEmbeddedMetadataEnforcesLeaseAndRunsTurn) {
  // Arrange
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteMiniCpmOBundleWithEmbeddedMetadata(
                           "minicpm_o_session_test.litertlm"));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = ":nocache";
  settings.backend = OmniEngine::Settings::Backend::kCpu;
  settings.num_threads = 1;
  settings.force_fp32_on_gpu = true;

  lm::proto::Any2AnySessionConfig init_cfg;
  init_cfg.set_generate_audio(false);
  init_cfg.set_max_new_tokens(1);
  init_cfg.set_max_tts_tokens(1);
  init_cfg.set_temperature(0.0f);
  init_cfg.set_top_p(0.9f);
  init_cfg.set_top_k(10);
  init_cfg.set_tts_temperature(0.0f);
  init_cfg.set_tts_top_p(0.85f);
  init_cfg.set_tts_top_k(15);
  init_cfg.set_system_prompt("Custom system prompt");
  init_cfg.set_voice("override_voice");
  init_cfg.set_language("zh-CN");
  init_cfg.set_seed(12345);

  lm::proto::Any2AnyMetadata caller_meta;
  caller_meta.set_input_sample_rate_hz(16000);

  // Act: create factory through OmniRegistry with both embedded and caller
  // metadata plus initial Any2AnySessionConfig.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniSessionFactory> base_factory,
      OmniRegistry::Get().CreateSessionFactory(
          "openbmb/MiniCPM-o-4_5", settings, &init_cfg, &caller_meta));
  auto* factory = dynamic_cast<Any2AnySessionFactory*>(base_factory.get());
  ASSERT_NE(factory, nullptr);

  // Assert: embedded metadata fields were loaded and merged with caller_meta,
  // and default supported_languages were populated.
  EXPECT_THAT(factory->GetAvailableLanguages(), ElementsAre("en-US", "zh-CN"));
  EXPECT_EQ(factory->GetInputSampleRateHz(), 16000);
  EXPECT_EQ(factory->GetDefaultVoice("zh-CN"), "override_voice");

  // Create session 1.
  auto input_source1 = std::make_unique<PushInputSource>();
  PushInputSource* raw_input1 = input_source1.get();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session1,
                       factory->Create(std::move(input_source1)));

  // Attempting to create a second concurrent session while session1 is alive
  // must fail with kFailedPrecondition to protect the shared stateful
  // llm_backbone runner.
  EXPECT_THAT(factory->Create(std::make_unique<PushInputSource>()),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Execute a turn on session1.
  ASSERT_OK(raw_input1->PushInput(OmniSession::TextInput{.text = "Hello"}));
  ASSERT_OK(raw_input1->PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output text_out, session1->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<TextOutput>(text_out));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output end_out, session1->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutputTurn>(end_out));

  // Destroying session1 releases the active-session lease, allowing a new
  // session to be created and executed.
  session1.reset();

  lm::proto::Any2AnySessionConfig session2_cfg;
  session2_cfg.set_generate_audio(true);
  // Creating with generate_audio=true when TTS models are absent fails in
  // CreateMiniCpmOComponents AND releases the lease via absl::Cleanup!
  EXPECT_THAT(CallCreateWithInputSource(*factory, session2_cfg,
                                        std::make_unique<PushInputSource>()),
              StatusIs(absl::StatusCode::kNotFound));

  // Verify the lease was released after the failed creation above.
  auto input_source2 = std::make_unique<PushInputSource>();
  PushInputSource* raw_input2 = input_source2.get();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session2,
                       factory->Create(std::move(input_source2)));
  ASSERT_OK(raw_input2->PushInput(OmniSession::TextInput{.text = "world"}));
  ASSERT_OK(raw_input2->PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output text_out2, session2->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<TextOutput>(text_out2));

  // Passing the sentinel "default" voice preserves the bundle's embedded voice.
  lm::proto::Any2AnySessionConfig default_voice_cfg;
  default_voice_cfg.set_voice("default");
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSessionFactory> default_base_factory,
                       OmniRegistry::Get().CreateSessionFactory(
                           "openbmb/MiniCPM-o-4_5", settings,
                           &default_voice_cfg, /*metadata=*/nullptr));
  auto* default_factory =
      dynamic_cast<Any2AnySessionFactory*>(default_base_factory.get());
  ASSERT_NE(default_factory, nullptr);
  EXPECT_EQ(default_factory->GetDefaultVoice("en-US"), "embedded_voice");
  EXPECT_THAT(default_factory->available_voices(),
              ElementsAre("embedded_voice"));
  EXPECT_FALSE(default_factory->HasVoice("default"));
}

TEST_F(Any2AnySessionTest, QwenOmniRegisteredInOmniRegistry) {
  EXPECT_THAT(OmniRegistry::Get().GetRegisteredModels(),
              testing::IsSupersetOf({"qwen2.5-omni-3b", "qwen2.5-omni",
                                     "qwen2_5_omni", "qwen-omni", "qwen_omni",
                                     "Qwen/Qwen2.5-Omni-3B"}));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("qwen2.5-omni-3b"));

  ASSERT_OK_AND_ASSIGN(const auto reg, OmniRegistry::Get().GetRegisteredModel(
                                           "qwen2.5-omni-3b"));
  EXPECT_EQ(reg.model_remote_url,
            "https://huggingface.co/litert-community/Qwen2.5-Omni-3B/resolve/"
            "main/qwen2_5_omni_3b_int8.litertlm");

  OmniEngine::Settings dir_settings;
  dir_settings.model_path = "/nonexistent/qwen_omni_dir";
  EXPECT_THAT(
      OmniRegistry::Get().CreateSessionFactory("qwen2.5-omni-3b", dir_settings),
      StatusIs(absl::StatusCode::kInvalidArgument));

  OmniEngine::Settings file_settings;
  file_settings.model_path = "/nonexistent/qwen_omni.litertlm";
  EXPECT_FALSE(OmniRegistry::Get()
                   .CreateSessionFactory("qwen2.5-omni-3b", file_settings)
                   .ok());
}

TEST_F(Any2AnySessionTest,
       QwenOmniSessionFactoryResolvesEmbeddedMetadataEnforcesLeaseAndRunsTurn) {
  // Arrange
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteQwenOmniBundleWithEmbeddedMetadata(
                           "qwen_omni_session_test.litertlm"));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = ":nocache";
  settings.backend = OmniEngine::Settings::Backend::kCpu;
  settings.num_threads = 1;
  settings.force_fp32_on_gpu = true;

  lm::proto::Any2AnySessionConfig init_cfg;
  init_cfg.set_generate_audio(false);
  init_cfg.set_max_new_tokens(1);
  init_cfg.set_max_tts_tokens(1);
  init_cfg.set_temperature(0.0f);
  init_cfg.set_top_p(0.9f);
  init_cfg.set_top_k(10);
  init_cfg.set_tts_temperature(0.0f);
  init_cfg.set_tts_top_p(0.85f);
  init_cfg.set_tts_top_k(15);
  init_cfg.set_system_prompt("Custom Qwen system prompt");
  init_cfg.set_voice("Ethan");
  init_cfg.set_language("zh-CN");

  lm::proto::Any2AnyMetadata caller_meta;
  caller_meta.set_input_sample_rate_hz(16000);

  // Act: create factory through OmniRegistry with both embedded and caller
  // metadata plus initial Any2AnySessionConfig.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniSessionFactory> base_factory,
      OmniRegistry::Get().CreateSessionFactory("Qwen/Qwen2.5-Omni-3B", settings,
                                               &init_cfg, &caller_meta));
  auto* factory = dynamic_cast<Any2AnySessionFactory*>(base_factory.get());
  ASSERT_NE(factory, nullptr);

  // Assert: embedded metadata fields were merged with caller_meta, default
  // supported_languages were populated, and both supported voices are listed
  // with the configured voice first.
  EXPECT_THAT(factory->GetAvailableLanguages(), ElementsAre("en-US", "zh-CN"));
  EXPECT_EQ(factory->GetInputSampleRateHz(), 16000);
  EXPECT_EQ(factory->GetDefaultVoice("zh-CN"), "Ethan");
  EXPECT_THAT(factory->available_voices(), ElementsAre("Ethan", "Chelsie"));

  // Create session 1.
  auto input_source1 = std::make_unique<PushInputSource>();
  PushInputSource* raw_input1 = input_source1.get();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session1,
                       factory->Create(std::move(input_source1)));

  // Attempting to create a second concurrent session while session1 is alive
  // must fail with kFailedPrecondition to protect the shared stateful
  // llm_backbone runner.
  EXPECT_THAT(factory->Create(std::make_unique<PushInputSource>()),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // Execute a turn on session1.
  ASSERT_OK(raw_input1->PushInput(OmniSession::TextInput{.text = "Hello t15"}));
  ASSERT_OK(raw_input1->PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output text_out, session1->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<TextOutput>(text_out));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output end_out, session1->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutputTurn>(end_out));

  // Destroying session1 releases the active-session lease, allowing a new
  // session to be created and executed.
  session1.reset();

  lm::proto::Any2AnySessionConfig session2_cfg;
  session2_cfg.set_generate_audio(true);
  // Creating with generate_audio=true when TTS models are absent fails in
  // CreateQwenOmniComponents and releases the lease via absl::Cleanup.
  EXPECT_THAT(CallCreateWithInputSource(*factory, session2_cfg,
                                        std::make_unique<PushInputSource>()),
              StatusIs(absl::StatusCode::kNotFound));

  // Verify the lease was released after the failed creation above.
  auto input_source2 = std::make_unique<PushInputSource>();
  PushInputSource* raw_input2 = input_source2.get();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session2,
                       factory->Create(std::move(input_source2)));
  ASSERT_OK(raw_input2->PushInput(OmniSession::TextInput{.text = "world t14"}));
  ASSERT_OK(raw_input2->PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output text_out2, session2->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<TextOutput>(text_out2));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output end_out2, session2->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutputTurn>(end_out2));
}

TEST_F(Any2AnySessionTest,
       QwenOmniSessionFactoryAvailableVoicesAndSessionOverrides) {
  // Use the full Qwen2.5-Omni vocab size when exercising per-session voice
  // overrides through the registry factory; direct `tts_text_bos_token_id`
  // mapping assertions live in `qwen_omni/qwen_omni_test.cc`.
  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteQwenOmniBundleWithEmbeddedMetadata("qwen_omni_voices_test.litertlm",
                                              /*llm_vocab_size=*/151936));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = ":nocache";
  settings.backend = OmniEngine::Settings::Backend::kCpu;
  settings.num_threads = 1;

  // Default embedded voice ("Chelsie"): lists Chelsie first, Ethan second.
  {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<OmniSessionFactory> base_factory,
        OmniRegistry::Get().CreateSessionFactory("qwen2.5-omni-3b", settings));
    auto* factory = dynamic_cast<Any2AnySessionFactory*>(base_factory.get());
    ASSERT_NE(factory, nullptr);
    EXPECT_EQ(factory->GetDefaultVoice("en-US"), "Chelsie");
    EXPECT_THAT(factory->available_voices(), ElementsAre("Chelsie", "Ethan"));

    lm::proto::Any2AnySessionConfig ethan_cfg;
    ethan_cfg.set_voice("Ethan");
    ASSERT_OK_AND_ASSIGN(
        auto ethan_session,
        CallCreateWithInputSource(*factory, ethan_cfg,
                                  std::make_unique<PushInputSource>()));
    ethan_session.reset();

    lm::proto::Any2AnySessionConfig chelsie_cfg;
    chelsie_cfg.set_voice("Chelsie");
    ASSERT_OK_AND_ASSIGN(
        auto chelsie_session,
        CallCreateWithInputSource(*factory, chelsie_cfg,
                                  std::make_unique<PushInputSource>()));
    chelsie_session.reset();
  }

  // Sentinel "default" voice (case-insensitive) preserves the embedded voice
  // instead of polluting available_voices().
  for (absl::string_view sentinel : {"default", "DEFAULT"}) {
    lm::proto::Any2AnySessionConfig default_voice_cfg;
    default_voice_cfg.set_voice(sentinel);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSessionFactory> base_factory,
                         OmniRegistry::Get().CreateSessionFactory(
                             "qwen2.5-omni-3b", settings, &default_voice_cfg));
    auto* factory = dynamic_cast<Any2AnySessionFactory*>(base_factory.get());
    ASSERT_NE(factory, nullptr);
    EXPECT_EQ(factory->GetDefaultVoice("en-US"), "Chelsie");
    EXPECT_THAT(factory->available_voices(), ElementsAre("Chelsie", "Ethan"));
    EXPECT_TRUE(factory->HasVoice("Chelsie"));
    EXPECT_TRUE(factory->HasVoice("chelsie"));
    EXPECT_TRUE(factory->HasVoice("Ethan"));
    EXPECT_TRUE(factory->HasVoice("ethan"));
    EXPECT_FALSE(factory->HasVoice("default"));
  }

  // Known voices are canonicalized to their official casing and matched
  // case-insensitively.
  {
    lm::proto::Any2AnySessionConfig lower_ethan_cfg;
    lower_ethan_cfg.set_voice("ethan");
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSessionFactory> base_factory,
                         OmniRegistry::Get().CreateSessionFactory(
                             "qwen2.5-omni-3b", settings, &lower_ethan_cfg));
    auto* factory = dynamic_cast<Any2AnySessionFactory*>(base_factory.get());
    ASSERT_NE(factory, nullptr);
    EXPECT_EQ(factory->GetDefaultVoice("en-US"), "Ethan");
    EXPECT_THAT(factory->available_voices(), ElementsAre("Ethan", "Chelsie"));
    EXPECT_TRUE(factory->HasVoice("Ethan"));
    EXPECT_TRUE(factory->HasVoice("ethan"));
  }

  {
    lm::proto::Any2AnySessionConfig lower_chelsie_cfg;
    lower_chelsie_cfg.set_voice("chelsie");
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSessionFactory> base_factory,
                         OmniRegistry::Get().CreateSessionFactory(
                             "qwen2.5-omni-3b", settings, &lower_chelsie_cfg));
    auto* factory = dynamic_cast<Any2AnySessionFactory*>(base_factory.get());
    ASSERT_NE(factory, nullptr);
    EXPECT_EQ(factory->GetDefaultVoice("en-US"), "Chelsie");
    EXPECT_THAT(factory->available_voices(), ElementsAre("Chelsie", "Ethan"));
    EXPECT_TRUE(factory->HasVoice("Chelsie"));
    EXPECT_TRUE(factory->HasVoice("chelsie"));
  }

  // Custom voice is placed first while keeping both built-in supported voices.
  {
    lm::proto::Any2AnySessionConfig custom_cfg;
    custom_cfg.set_voice("custom_voice");
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSessionFactory> base_factory,
                         OmniRegistry::Get().CreateSessionFactory(
                             "qwen2.5-omni-3b", settings, &custom_cfg));
    auto* factory = dynamic_cast<Any2AnySessionFactory*>(base_factory.get());
    ASSERT_NE(factory, nullptr);
    EXPECT_EQ(factory->GetDefaultVoice("en-US"), "custom_voice");
    EXPECT_THAT(factory->available_voices(),
                ElementsAre("custom_voice", "Chelsie", "Ethan"));
  }
}

TEST_F(Any2AnySessionTest, ResolveAny2AnyMetadataDefaultsAndCopiesMetadata) {
  // Null metadata populates default languages.
  lm::proto::Any2AnyMetadata defaults =
      ResolveAny2AnyMetadata(/*metadata=*/nullptr);
  EXPECT_THAT(defaults.supported_languages(), ElementsAre("en-US", "zh-CN"));

  // Metadata without supported_languages preserves scalar fields and populates
  // defaults.
  lm::proto::Any2AnyMetadata no_lang_meta;
  no_lang_meta.set_input_sample_rate_hz(24000);
  no_lang_meta.set_output_sample_rate_hz(24000);
  no_lang_meta.set_system_prompt("Embedded prompt");

  lm::proto::Any2AnyMetadata resolved_defaults =
      ResolveAny2AnyMetadata(&no_lang_meta);
  EXPECT_EQ(resolved_defaults.input_sample_rate_hz(), 24000);
  EXPECT_EQ(resolved_defaults.output_sample_rate_hz(), 24000);
  EXPECT_EQ(resolved_defaults.system_prompt(), "Embedded prompt");
  EXPECT_THAT(resolved_defaults.supported_languages(),
              ElementsAre("en-US", "zh-CN"));

  // Explicit supported_languages are preserved without appending defaults.
  lm::proto::Any2AnyMetadata explicit_lang_meta;
  explicit_lang_meta.set_input_sample_rate_hz(16000);
  explicit_lang_meta.add_supported_languages("ja-JP");

  lm::proto::Any2AnyMetadata resolved =
      ResolveAny2AnyMetadata(&explicit_lang_meta);
  EXPECT_EQ(resolved.input_sample_rate_hz(), 16000);
  EXPECT_THAT(resolved.supported_languages(), ElementsAre("ja-JP"));
}

TEST_F(Any2AnySessionTest,
       EmbeddedSupportedLanguagesNotDuplicatedThroughOmniRegistry) {
  ASSERT_OK_AND_ASSIGN(std::string qwen_bundle_path,
                       WriteQwenOmniBundleWithEmbeddedMetadata(
                           "qwen_omni_embedded_langs.litertlm",
                           /*llm_vocab_size=*/32, {"en-US"}));
  ASSERT_OK_AND_ASSIGN(std::string minicpm_bundle_path,
                       WriteMiniCpmOBundleWithEmbeddedMetadata(
                           "minicpm_o_embedded_langs.litertlm", {"en-US"}));

  lm::proto::Any2AnyMetadata caller_meta;
  caller_meta.set_input_sample_rate_hz(16000);

  for (const auto& [model_name, bundle_path] :
       std::vector<std::pair<absl::string_view, std::string>>{
           {"qwen2.5-omni-3b", qwen_bundle_path},
           {"openbmb/MiniCPM-o-4_5", minicpm_bundle_path},
       }) {
    OmniEngine::Settings settings;
    settings.model_path = bundle_path;
    settings.cache_dir = ":nocache";
    settings.backend = OmniEngine::Settings::Backend::kCpu;
    settings.num_threads = 1;

    // Without caller metadata override: embedded languages must not duplicate.
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<OmniSessionFactory> base_factory_no_override,
        OmniRegistry::Get().CreateSessionFactory(model_name, settings,
                                                 /*session_config=*/nullptr,
                                                 /*metadata=*/nullptr));
    auto* factory_no_override =
        dynamic_cast<Any2AnySessionFactory*>(base_factory_no_override.get());
    ASSERT_NE(factory_no_override, nullptr);
    EXPECT_THAT(factory_no_override->GetAvailableLanguages(),
                ElementsAre("en-US"));

    // With caller metadata override: embedded languages must still not
    // duplicate while scalar overrides apply.
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<OmniSessionFactory> base_factory_with_override,
        OmniRegistry::Get().CreateSessionFactory(model_name, settings,
                                                 /*session_config=*/nullptr,
                                                 &caller_meta));
    auto* factory_with_override =
        dynamic_cast<Any2AnySessionFactory*>(base_factory_with_override.get());
    ASSERT_NE(factory_with_override, nullptr);
    EXPECT_THAT(factory_with_override->GetAvailableLanguages(),
                ElementsAre("en-US"));
    EXPECT_EQ(factory_with_override->GetInputSampleRateHz(), 16000);
  }
}

TEST_F(Any2AnySessionTest,
       LeasedAny2AnySessionFactoryEnforcesLeaseAndValidatesBuilder) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<Environment>(std::move(*env));
  auto resources = std::make_shared<ModelResources>(shared_env);

  lm::proto::Any2AnyMetadata metadata;
  lm::proto::Any2AnySessionConfig base_cfg;
  base_cfg.set_language("en-US");
  base_cfg.set_generate_audio(false);

  // Null model_resources is rejected.
  {
    LeasedAny2AnySessionFactory factory(
        OmniEngine::Settings{}, &base_cfg, metadata, /*available_voices=*/{},
        /*model_resources=*/nullptr,
        [](const lm::proto::Any2AnySessionConfig&,
           std::unique_ptr<OmniSession::InputSource>,
           std::shared_ptr<ModelResources>,
           std::vector<std::unique_ptr<internal::StageBase>>&)
            -> absl::StatusOr<Stage<Output>* absl_nonnull> {
          return absl::UnimplementedError("should not be called");
        },
        LeasedAny2AnySessionOptions{.fallback_default_voice = "fallback_v"});
    EXPECT_EQ(factory.GetDefaultVoice("en-US"), "fallback_v");
    EXPECT_THAT(factory.Create(std::make_unique<PushInputSource>()),
                StatusIs(absl::StatusCode::kFailedPrecondition));
  }

  // Null component_builder is rejected.
  {
    LeasedAny2AnySessionFactory factory(OmniEngine::Settings{}, &base_cfg,
                                        metadata, {"primary_v"}, resources,
                                        /*component_builder=*/nullptr);
    EXPECT_EQ(factory.GetDefaultVoice("en-US"), "primary_v");
    EXPECT_THAT(factory.Create(std::make_unique<PushInputSource>()),
                StatusIs(absl::StatusCode::kFailedPrecondition));
  }

  // Exercise builder failure modes, lease release on error, lease enforcement
  // while active, default display name vs custom display name, null input
  // source defaulting, and session config merging.
  enum class BuilderMode {
    kNormal,
    kReturnError,
    kReturnNullOutput,
    kReturnMismatchedOutput,
    kReturnNullStage,
  };
  BuilderMode mode = BuilderMode::kNormal;
  lm::proto::Any2AnySessionConfig observed_config;
  bool observed_non_null_input = false;
  PushInputSource extra_input;
  auto extra_stage = std::make_unique<FakeAny2AnyStage>(&extra_input,
                                                        /*emit_audio=*/false);
  Stage<Output>* dangling_holder = extra_stage.get();

  LeasedAny2AnySessionFactory factory(
      OmniEngine::Settings{}, &base_cfg, metadata, {"voice_a"}, resources,
      [&](const lm::proto::Any2AnySessionConfig& effective_config,
          std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
          std::shared_ptr<ModelResources> absl_nonnull /*model_resources*/,
          std::vector<std::unique_ptr<internal::StageBase>>& stages)
          -> absl::StatusOr<Stage<Output>* absl_nonnull> {
        observed_config = effective_config;
        observed_non_null_input = (input_source != nullptr);
        switch (mode) {
          case BuilderMode::kReturnError:
            return absl::NotFoundError("simulated builder failure");
          case BuilderMode::kReturnNullOutput: {
            stages.push_back(std::move(input_source));
            Stage<Output>* null_out = nullptr;
            return null_out;
          }
          case BuilderMode::kReturnMismatchedOutput: {
            stages.push_back(std::move(input_source));
            return dangling_holder;
          }
          case BuilderMode::kReturnNullStage: {
            auto stage = std::make_unique<FakeAny2AnyStage>(
                input_source.get(), effective_config.generate_audio());
            Stage<Output>* raw_out = stage.get();
            stages.push_back(std::move(input_source));
            stages.push_back(nullptr);
            stages.push_back(std::move(stage));
            return raw_out;
          }
          case BuilderMode::kNormal: {
            auto stage = std::make_unique<FakeAny2AnyStage>(
                input_source.get(), effective_config.generate_audio());
            Stage<Output>* raw_out = stage.get();
            stages.push_back(std::move(input_source));
            stages.push_back(std::move(stage));
            return raw_out;
          }
        }
        return absl::InternalError("unreachable");
      },
      LeasedAny2AnySessionOptions{.model_display_name = "TestModel"});

  // Failed builder calls must release the lease.
  mode = BuilderMode::kReturnError;
  EXPECT_THAT(factory.Create(std::make_unique<PushInputSource>()),
              StatusIs(absl::StatusCode::kNotFound));

  mode = BuilderMode::kReturnNullOutput;
  EXPECT_THAT(factory.Create(std::make_unique<PushInputSource>()),
              StatusIs(absl::StatusCode::kInternal));

  mode = BuilderMode::kReturnMismatchedOutput;
  EXPECT_THAT(factory.Create(std::make_unique<PushInputSource>()),
              StatusIs(absl::StatusCode::kInternal));

  mode = BuilderMode::kReturnNullStage;
  EXPECT_THAT(factory.Create(std::make_unique<PushInputSource>()),
              StatusIs(absl::StatusCode::kInternal));

  // Normal creation succeeds after prior failures, merges session config, and
  // defaults null input_source to PushInputSource.
  mode = BuilderMode::kNormal;
  lm::proto::Any2AnySessionConfig override_cfg;
  override_cfg.set_generate_audio(true);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<MultiStagedSession> session1,
                       CallCreateWithInputSource(factory, override_cfg,
                                                 /*input_source=*/nullptr));
  EXPECT_TRUE(observed_non_null_input);
  EXPECT_EQ(observed_config.language(), "en-US");
  EXPECT_TRUE(observed_config.generate_audio());

  // Concurrent creation fails with custom model_display_name in error message.
  EXPECT_THAT(
      factory.Create(std::make_unique<PushInputSource>()),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Another TestModel session is already active")));

  // Releasing session1 allows session2 to be created and run.
  session1.reset();
  auto input_source2 = std::make_unique<PushInputSource>();
  PushInputSource* raw_input2 = input_source2.get();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session2,
                       factory.Create(std::move(input_source2)));
  ASSERT_OK(raw_input2->PushInput(OmniSession::TextInput{.text = "leased"}));
  ASSERT_OK(raw_input2->PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output out, session2->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<TextOutput>(out));
  EXPECT_EQ(std::get<TextOutput>(out).confirmed_text, "text:leased");

  // Empty model_display_name falls back to "Any2Any" in concurrent error.
  {
    LeasedAny2AnySessionFactory default_name_factory(
        OmniEngine::Settings{}, &base_cfg, metadata, {}, resources,
        [](const lm::proto::Any2AnySessionConfig&,
           std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
           std::shared_ptr<ModelResources> absl_nonnull /*model_resources*/,
           std::vector<std::unique_ptr<internal::StageBase>>& stages)
            -> absl::StatusOr<Stage<Output>* absl_nonnull> {
          auto stage = std::make_unique<FakeAny2AnyStage>(input_source.get(),
                                                          /*emit_audio=*/false);
          Stage<Output>* raw_out = stage.get();
          stages.push_back(std::move(input_source));
          stages.push_back(std::move(stage));
          return raw_out;
        },
        LeasedAny2AnySessionOptions{.model_display_name = ""});
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<OmniSession> active,
        default_name_factory.Create(std::make_unique<PushInputSource>()));
    EXPECT_THAT(
        default_name_factory.Create(std::make_unique<PushInputSource>()),
        StatusIs(absl::StatusCode::kFailedPrecondition,
                 HasSubstr("Another Any2Any session is already active")));
  }
}

}  // namespace
}  // namespace litert::omni::any2any
