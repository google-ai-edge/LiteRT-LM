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

#include "omni/text2image/text2image_session.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/text2image/image_decoder.h"
#include "omni/text2image/prompt_source.h"
#include "runtime/components/model_resources.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "runtime/proto/text2image_session_config.pb.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::text2image {

class Text2ImageSessionTest : public ::testing::Test {
 protected:
  static std::unique_ptr<PromptSource> CreatePromptInputSource(
      std::unique_ptr<OmniSession::InputSource> input_source,
      ImageGenInputMetadata default_params = {}) {
    return Text2ImageSessionFactory::CreatePromptInputSource(
        std::move(input_source), std::move(default_params));
  }
};

namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::IsSupersetOf;

class FakeImageDecoder : public ImageDecoder {
 public:
  explicit FakeImageDecoder(PromptSource* prompt_source)
      : prompt_source_(prompt_source) {}

 protected:
  bool NeedScheduleInternal() const override {
    return prompt_source_->HasOutput();
  }

  absl::Status ScheduleInternal() override {
    SetState(State::kIdle);
    auto prompt = prompt_source_->GetOutput();
    if (absl::IsNotFound(prompt.status())) {
      return absl::OkStatus();
    } else if (!prompt.ok()) {
      return prompt.status();
    }
    ImageOutput out;
    out.width = prompt->metadata.width;
    out.height = prompt->metadata.height;
    out.channels = 3;
    out.rgb_data.assign(out.width * out.height * out.channels,
                        static_cast<uint8_t>(prompt->metadata.seed & 0xFF));
    PushOutput(std::move(out));
    return absl::OkStatus();
  }

 private:
  PromptSource* prompt_source_;
};

class FakeText2ImageSessionFactory : public Text2ImageSessionFactory {
 public:
  FakeText2ImageSessionFactory(const OmniEngine::Settings& settings,
                               const MessageLite* absl_nullable session_config,
                               const lm::proto::Text2ImageMetadata& metadata)
      : Text2ImageSessionFactory(settings, session_config, metadata,
                                 /*model_resources=*/nullptr) {}

  ImageGenInputMetadata ResolveDefaultPromptParams(
      const lm::proto::Text2ImageSessionConfig& session_config) const override {
    ImageGenInputMetadata params{
        .width = session_config.has_width() ? session_config.width()
                                            : this->session_config().width(),
        .height = session_config.has_height() ? session_config.height()
                                              : this->session_config().height(),
        .num_inference_steps =
            session_config.has_num_inference_steps()
                ? session_config.num_inference_steps()
                : this->session_config().num_inference_steps(),
        .seed =
            session_config.has_seed()
                ? static_cast<uint64_t>(session_config.seed())
                : (this->session_config().has_seed()
                       ? static_cast<uint64_t>(this->session_config().seed())
                       : 42),
    };
    if (params.width <= 0) params.width = 256;
    if (params.height <= 0) params.height = 256;
    if (params.num_inference_steps <= 0) params.num_inference_steps = 4;
    return params;
  }

 protected:
  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateWithPromptSource(
      const lm::proto::Text2ImageSessionConfig& session_config,
      std::unique_ptr<PromptSource> absl_nullable prompt_source) override {
    lm::proto::Text2ImageSessionConfig effective_config =
        this->session_config();
    effective_config.MergeFrom(session_config);
    if (effective_config.width() != effective_config.height()) {
      return absl::InvalidArgumentError(
          "FLUX.2 requires square image dimensions (width == height).");
    }
    if (prompt_source == nullptr) {
      prompt_source = std::make_unique<PushPromptSource>(
          ResolveDefaultPromptParams(effective_config));
    }
    auto decoder = std::make_unique<FakeImageDecoder>(prompt_source.get());
    Stage<Output>* raw_output_stage = decoder.get();
    std::vector<std::unique_ptr<internal::StageBase>> stages;
    stages.push_back(std::move(prompt_source));
    stages.push_back(std::move(decoder));
    return MultiStagedSession::Create(std::move(stages), raw_output_stage,
                                      thread_pool());
  }
};

TEST(PushPromptSourceTest, PushScheduleAndFinish) {
  PushPromptSource source(ImageGenInputMetadata{
      .width = 256, .height = 256, .num_inference_steps = 4, .seed = 99});
  EXPECT_FALSE(source.NeedSchedule());
  ASSERT_OK(source.Schedule());

  ASSERT_OK(source.PushPrompt("A futuristic city"));
  EXPECT_FALSE(source.NeedSchedule());
  ASSERT_TRUE(source.HasOutput());

  auto p = source.GetOutput();
  ASSERT_OK(p);
  EXPECT_EQ(p->text, "A futuristic city");
  EXPECT_EQ(p->metadata.width, 256);
  EXPECT_EQ(p->metadata.height, 256);
  EXPECT_EQ(p->metadata.seed, 99);

  source.Finish();
  EXPECT_THAT(source.PushPrompt("After finish"),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(Text2ImageSessionTest, AllDefaultModelsAreRegistered) {
  EXPECT_THAT(OmniRegistry::Get().GetRegisteredModels(),
              IsSupersetOf({"bonsai-flux2", "bonsai_flux2", "flux2",
                            "flux2-klein", "flux2_klein", "flux.2-klein-4b"}));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("bonsai-flux2"));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("flux2-klein"));
  EXPECT_FALSE(
      OmniRegistry::Get().IsModelRegistered("unknown-text2image-model"));

  ASSERT_OK_AND_ASSIGN(const auto bonsai_reg,
                       OmniRegistry::Get().GetRegisteredModel("bonsai-flux2"));
  EXPECT_EQ(bonsai_reg.model_remote_url,
            "https://huggingface.co/litert-community/"
            "Bonsai-Image-ternary-4B/resolve/main/bonsai-flux2-4b.litertlm");
  ASSERT_OK_AND_ASSIGN(const auto klein_reg,
                       OmniRegistry::Get().GetRegisteredModel("flux2-klein"));
  EXPECT_EQ(klein_reg.model_remote_url,
            "https://huggingface.co/litert-community/"
            "FLUX.2-klein-4B-LiteRT/resolve/main/flux2-klein-4b.litertlm");
}

TEST_F(Text2ImageSessionTest, RejectsUnknownModelOnCreate) {
  EXPECT_THAT(OmniEngine::Create("unknown-text2image-model"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(Text2ImageSessionTest,
       CreateFailsWhenModelContainerIsMissingOrIncomplete) {
  OmniEngine::Settings settings;
  settings.model_path = "/nonexistent/flux2_model_dir";
  for (const auto& model_name :
       {"bonsai-flux2", "bonsai_flux2", "flux2", "flux2-klein", "flux2_klein",
        "flux.2-klein-4b"}) {
    EXPECT_THAT(OmniEngine::Create(model_name, settings),
                StatusIs(absl::StatusCode::kNotFound));
  }
}

TEST_F(Text2ImageSessionTest,
       DynamicRegisterResolveDefaultPromptParamsAndCreateSession) {
  ASSERT_OK(
      OmniRegistry::Get().RegisterSessionFactory(OmniRegistry::RegistrySpec{
          .model_names = {"fake-text2image-model"},
          .get_session_factory =
              [](absl::string_view /*model_name*/,
                 const OmniEngine::Settings& settings,
                 const MessageLite* absl_nullable session_config,
                 const MessageLite* absl_nullable metadata,
                 ::litert::Environment& /*env*/, ::litert::Options& /*options*/,
                 std::shared_ptr<::litert::lm::ModelResources> absl_nullable
                 /*lm_resources*/)
              -> absl::StatusOr<std::unique_ptr<OmniSessionFactory>> {
            lm::proto::Text2ImageMetadata resolved_meta;
            if (metadata != nullptr) {
              resolved_meta.CheckTypeAndMergeFrom(*metadata);
            }
            return std::make_unique<FakeText2ImageSessionFactory>(
                settings, session_config, resolved_meta);
          },
          .session_config_type_id = "Text2Image",
      }));
  absl::Cleanup cleanup = [] {
    OmniRegistry::Get().UnregisterModel("fake-text2image-model");
  };

  FakeText2ImageSessionFactory factory(OmniEngine::Settings{},
                                       /*session_config=*/nullptr,
                                       lm::proto::Text2ImageMetadata{});
  ImageGenInputMetadata resolved_defaults =
      factory.ResolveDefaultPromptParams({});
  EXPECT_EQ(resolved_defaults.width, 256);
  EXPECT_EQ(resolved_defaults.height, 256);
  EXPECT_EQ(resolved_defaults.num_inference_steps, 4);
  EXPECT_EQ(resolved_defaults.seed, 42);

  lm::proto::Text2ImageSessionConfig custom_input;
  custom_input.set_width(512);
  custom_input.set_height(512);
  custom_input.set_num_inference_steps(8);
  custom_input.set_seed(77);
  ImageGenInputMetadata resolved_custom =
      factory.ResolveDefaultPromptParams(custom_input);
  EXPECT_EQ(resolved_custom.width, 512);
  EXPECT_EQ(resolved_custom.height, 512);
  EXPECT_EQ(resolved_custom.num_inference_steps, 8);
  EXPECT_EQ(resolved_custom.seed, 77);

  lm::proto::Text2ImageSessionConfig non_square_config;
  non_square_config.set_width(512);
  non_square_config.set_height(256);
  ASSERT_OK_AND_ASSIGN(auto engine, OmniEngine::Create("fake-text2image-model",
                                                       OmniEngine::Settings{},
                                                       &non_square_config));
  EXPECT_THAT(engine->CreateSession(std::make_unique<PushInputSource>()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(Text2ImageSessionTest, ProcessNextFlushAndProcessAsync) {
  auto prompt_source = std::make_unique<PushPromptSource>(ImageGenInputMetadata{
      .width = 8, .height = 8, .num_inference_steps = 4, .seed = 123});
  PushPromptSource* raw_prompt_source = prompt_source.get();
  auto decoder = std::make_unique<FakeImageDecoder>(prompt_source.get());
  Stage<Output>* raw_output_stage = decoder.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(prompt_source));
  stages.push_back(std::move(decoder));

  ::litert::lm::ThreadPool pool("test_text2image_pool", 1);
  auto session =
      MultiStagedSession::Create(std::move(stages), raw_output_stage, &pool);
  ASSERT_OK(session);

  ASSERT_OK(raw_prompt_source->PushPrompt("A red apple"));
  auto out = (*session)->ProcessNext();
  ASSERT_OK(out);
  ASSERT_TRUE(std::holds_alternative<ImageOutput>(*out));
  const auto& img = std::get<ImageOutput>(*out);
  EXPECT_EQ(img.width, 8);
  EXPECT_EQ(img.height, 8);
  EXPECT_EQ(img.channels, 3);
  EXPECT_EQ(img.rgb_data.size(), 8 * 8 * 3);
  EXPECT_EQ(img.rgb_data[0], 123);

  // Test Flush() after Reset()
  (*session)->Reset();
  ASSERT_OK(raw_prompt_source->PushPrompt("A green forest"));
  auto flushed = (*session)->Flush();
  ASSERT_OK(flushed);
  ASSERT_TRUE(std::holds_alternative<ImageOutput>(*flushed));
  EXPECT_EQ(std::get<ImageOutput>(*flushed).rgb_data[0], 123);

  // Test ProcessAsync
  (*session)->Reset();
  ASSERT_OK(raw_prompt_source->PushPrompt("A blue ocean"));
  raw_prompt_source->Finish();

  absl::Notification done;
  std::vector<ImageOutput> async_outputs;
  ASSERT_OK((*session)->ProcessAsync(
      [&](absl::StatusOr<OmniSession::Output> res) -> absl::Status {
        if (absl::IsOutOfRange(res.status())) {
          done.Notify();
          return res.status();
        }
        if (res.ok() && std::holds_alternative<ImageOutput>(*res)) {
          async_outputs.push_back(std::get<ImageOutput>(*res));
        }
        return absl::OkStatus();
      }));
  done.WaitForNotification();
  (*session)->Reset();
  ASSERT_EQ(async_outputs.size(), 1);
  EXPECT_EQ(async_outputs[0].width, 8);
  EXPECT_EQ(async_outputs[0].height, 8);
  EXPECT_EQ(async_outputs[0].rgb_data[0], 123);
}

TEST_F(Text2ImageSessionTest,
       PromptInputSourcePullsMetadataAndTextInputResetAndFlush) {
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  auto prompt_source = CreatePromptInputSource(
      std::move(input_source),
      ImageGenInputMetadata{
          .width = 4, .height = 4, .num_inference_steps = 2, .seed = 11});
  auto decoder = std::make_unique<FakeImageDecoder>(prompt_source.get());
  Stage<Output>* raw_output_stage = decoder.get();

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  stages.push_back(std::move(prompt_source));
  stages.push_back(std::move(decoder));

  ::litert::lm::ThreadPool pool("test_omni_text2image_pool", 1);
  auto session =
      MultiStagedSession::Create(std::move(stages), raw_output_stage, &pool);
  ASSERT_OK(session);
  std::unique_ptr<OmniSession> omni_session = *std::move(session);

  // Rejects AudioInput on a Text2Image session.
  ASSERT_OK(
      raw_input->PushInput(OmniSession::AudioInput{.pcm_samples = {1.0f}}));
  EXPECT_THAT(omni_session->ProcessNext(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Rejects empty TextInput.
  omni_session->Reset();
  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = ""}));
  EXPECT_THAT(omni_session->ProcessNext(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Metadata pushed before TextInput returns NotFoundError until TextInput or
  // Finish() is provided.
  omni_session->Reset();
  ASSERT_OK(raw_input->PushInput(OmniSession::ImageGenInputMetadata{
      .width = 16, .height = 16, .num_inference_steps = 4, .seed = 55}));
  EXPECT_THAT(omni_session->ProcessNext(),
              StatusIs(absl::StatusCode::kNotFound));

  ASSERT_OK(
      raw_input->PushInput(OmniSession::TextInput{.text = "A cat in space"}));
  auto out = omni_session->ProcessNext();
  ASSERT_OK(out);
  ASSERT_TRUE(std::holds_alternative<ImageOutput>(*out));
  const auto& img = std::get<ImageOutput>(*out);
  EXPECT_EQ(img.width, 16);
  EXPECT_EQ(img.height, 16);
  EXPECT_EQ(img.rgb_data[0], 55);

  // Reset() restores default_params_ (4x4, seed=11) and Flush() drains input.
  omni_session->Reset();
  ASSERT_OK(
      raw_input->PushInput(OmniSession::TextInput{.text = "Default params"}));
  raw_input->Finish();
  auto flushed = omni_session->Flush();
  ASSERT_OK(flushed);
  ASSERT_TRUE(std::holds_alternative<ImageOutput>(*flushed));
  const auto& default_img = std::get<ImageOutput>(*flushed);
  EXPECT_EQ(default_img.width, 4);
  EXPECT_EQ(default_img.height, 4);
  EXPECT_EQ(default_img.rgb_data[0], 11);

  EXPECT_THAT(omni_session->ProcessNext(),
              StatusIs(absl::StatusCode::kOutOfRange));
}

TEST(ResolveDefaultPromptParamsTest, AppliesOverrideBaseAndModelDefaults) {
  const ImageGenInputMetadata defaults{
      .width = 512, .height = 768, .num_inference_steps = 8, .seed = 1234};

  // Both configs unset: falls back to model defaults.
  const ImageGenInputMetadata from_defaults =
      ResolveDefaultPromptParams({}, {}, defaults);
  EXPECT_EQ(from_defaults.width, 512);
  EXPECT_EQ(from_defaults.height, 768);
  EXPECT_EQ(from_defaults.num_inference_steps, 8);
  EXPECT_EQ(from_defaults.seed, 1234);

  // Base config populated, including non-positive value that should fall back.
  lm::proto::Text2ImageSessionConfig base;
  base.set_width(256);
  base.set_height(0);  // non-positive -> falls back to model_defaults.height.
  base.set_num_inference_steps(4);
  base.set_seed(99);
  const ImageGenInputMetadata from_base =
      ResolveDefaultPromptParams(base, {}, defaults);
  EXPECT_EQ(from_base.width, 256);
  EXPECT_EQ(from_base.height, 768);
  EXPECT_EQ(from_base.num_inference_steps, 4);
  EXPECT_EQ(from_base.seed, 99);

  // Override config takes precedence over base, including explicit seed = 0.
  lm::proto::Text2ImageSessionConfig override_cfg;
  override_cfg.set_width(1024);
  override_cfg.set_seed(0);
  const ImageGenInputMetadata from_override =
      ResolveDefaultPromptParams(base, override_cfg, defaults);
  EXPECT_EQ(from_override.width, 1024);
  EXPECT_EQ(from_override.height, 768);
  EXPECT_EQ(from_override.num_inference_steps, 4);
  EXPECT_EQ(from_override.seed, 0);
}

TEST_F(Text2ImageSessionTest, LeasedFactoryEnforcesSingleSessionLease) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto resources = std::make_shared<ModelResources>(
      std::make_shared<Environment>(std::move(*env)));

  // Rejects null ModelResources.
  LeasedText2ImageSessionFactory no_resources_factory(
      OmniEngine::Settings{}, /*session_config=*/nullptr,
      lm::proto::Text2ImageMetadata{}, /*model_resources=*/nullptr,
      [](const lm::proto::Text2ImageSessionConfig&,
         std::unique_ptr<PromptSource>, std::shared_ptr<ModelResources>,
         std::vector<std::unique_ptr<internal::StageBase>>&)
          -> absl::StatusOr<Stage<Output>* absl_nonnull> {
        return absl::UnimplementedError("unused");
      });
  EXPECT_THAT(no_resources_factory.Create(std::make_unique<PushInputSource>()),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("ModelResources")));

  // Rejects null ComponentBuilder.
  LeasedText2ImageSessionFactory no_builder_factory(
      OmniEngine::Settings{}, /*session_config=*/nullptr,
      lm::proto::Text2ImageMetadata{}, resources,
      /*component_builder=*/nullptr);
  EXPECT_THAT(no_builder_factory.Create(std::make_unique<PushInputSource>()),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("ComponentBuilder")));

  // Valid builder enforces lease and releases it when the session drops.
  bool should_fail = false;
  LeasedText2ImageSessionOptions options{
      .model_display_name = "TestModel",
      .default_prompt_params =
          ImageGenInputMetadata{
              .width = 4, .height = 4, .num_inference_steps = 2, .seed = 7},
  };
  LeasedText2ImageSessionFactory factory(
      OmniEngine::Settings{}, /*session_config=*/nullptr,
      lm::proto::Text2ImageMetadata{}, resources,
      [&should_fail](const lm::proto::Text2ImageSessionConfig&,
                     std::unique_ptr<PromptSource> prompt_source,
                     std::shared_ptr<ModelResources>,
                     std::vector<std::unique_ptr<internal::StageBase>>& stages)
          -> absl::StatusOr<Stage<Output>* absl_nonnull> {
        if (should_fail) {
          return absl::InvalidArgumentError("builder failed");
        }
        auto decoder = std::make_unique<FakeImageDecoder>(prompt_source.get());
        Stage<Output>* out_stage = decoder.get();
        stages.push_back(std::move(prompt_source));
        stages.push_back(std::move(decoder));
        return out_stage;
      },
      options);

  // `ResolveDefaultPromptParams` falls back to `default_prompt_params`
  // (including `seed == 7`) when engine `session_config` omits `seed`.
  const ImageGenInputMetadata defaults = factory.ResolveDefaultPromptParams({});
  EXPECT_EQ(defaults.width, 4);
  EXPECT_EQ(defaults.height, 4);
  EXPECT_EQ(defaults.num_inference_steps, 2);
  EXPECT_EQ(defaults.seed, 7);

  // Verify builder failure releases lease.
  should_fail = true;
  EXPECT_THAT(factory.Create(std::make_unique<PushInputSource>()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("builder failed")));
  should_fail = false;

  {
    auto input = std::make_unique<PushInputSource>();
    PushInputSource* raw_input = input.get();
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session1,
                         factory.Create(std::move(input)));

    // Second concurrent session rejected with model display name.
    EXPECT_THAT(factory.Create(std::make_unique<PushInputSource>()),
                StatusIs(absl::StatusCode::kFailedPrecondition,
                         HasSubstr("Another TestModel session is already")));

    ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "hello"}));
    ASSERT_OK_AND_ASSIGN(OmniSession::Output out, session1->ProcessNext());
    ASSERT_TRUE(std::holds_alternative<ImageOutput>(out));
    const auto& img = std::get<ImageOutput>(out);
    EXPECT_EQ(img.width, 4);
    EXPECT_FALSE(img.rgb_data.empty());
    EXPECT_EQ(img.rgb_data[0], 7);
  }

  // After `session1` is destroyed, a new session acquires the lease cleanly.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session2,
                       factory.Create(std::make_unique<PushInputSource>()));
  EXPECT_NE(session2, nullptr);
}

TEST_F(Text2ImageSessionTest, LeasedFactorySeedPrecedence) {
  class ExposedLeasedFactory : public LeasedText2ImageSessionFactory {
   public:
    using LeasedText2ImageSessionFactory::CreateWithPromptSource;
    using LeasedText2ImageSessionFactory::LeasedText2ImageSessionFactory;
  };

  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto resources = std::make_shared<ModelResources>(
      std::make_shared<Environment>(std::move(*env)));

  auto make_builder =
      [](lm::proto::Text2ImageSessionConfig& captured_effective) {
        return [&captured_effective](
                   const lm::proto::Text2ImageSessionConfig& effective,
                   std::unique_ptr<PromptSource> prompt_source,
                   std::shared_ptr<ModelResources>,
                   std::vector<std::unique_ptr<internal::StageBase>>& stages)
                   -> absl::StatusOr<Stage<Output>* absl_nonnull> {
          captured_effective = effective;
          auto decoder =
              std::make_unique<FakeImageDecoder>(prompt_source.get());
          Stage<Output>* out_stage = decoder.get();
          stages.push_back(std::move(prompt_source));
          stages.push_back(std::move(decoder));
          return out_stage;
        };
      };

  const LeasedText2ImageSessionOptions custom_default_options{
      .model_display_name = "TestModel",
      .default_prompt_params =
          ImageGenInputMetadata{
              .width = 4, .height = 4, .num_inference_steps = 2, .seed = 7},
  };

  // Default options (untouched `default_prompt_params.seed`) still yield 42.
  lm::proto::Text2ImageSessionConfig captured;
  ExposedLeasedFactory default_factory(
      OmniEngine::Settings{}, /*session_config=*/nullptr,
      lm::proto::Text2ImageMetadata{}, resources, make_builder(captured));
  EXPECT_EQ(default_factory.ResolveDefaultPromptParams({}).seed, 42);

  // (c) Neither engine nor per-session sets `seed`: `default_prompt_params`
  // wins (`7`), even when engine sets non-seed fields.
  lm::proto::Text2ImageSessionConfig engine_no_seed;
  engine_no_seed.set_width(8);
  engine_no_seed.set_height(8);
  ExposedLeasedFactory no_seed_factory(
      OmniEngine::Settings{}, &engine_no_seed, lm::proto::Text2ImageMetadata{},
      resources, make_builder(captured), custom_default_options);
  EXPECT_EQ(no_seed_factory.ResolveDefaultPromptParams({}).seed, 7);
  {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<MultiStagedSession> session,
        no_seed_factory.CreateWithPromptSource({}, /*prompt_source=*/nullptr));
    EXPECT_FALSE(captured.has_seed());
    auto* push_source =
        static_cast<PushPromptSource*>(session->stages()[0].get());
    ASSERT_NE(push_source, nullptr);
    ASSERT_OK(push_source->PushPrompt("prompt"));
    ASSERT_OK_AND_ASSIGN(OmniSession::Output out, session->ProcessNext());
    ASSERT_TRUE(std::holds_alternative<ImageOutput>(out));
    EXPECT_EQ(std::get<ImageOutput>(out).rgb_data[0], 7);
  }

  // (b) Engine explicitly sets `seed = 42` (same numeric value as base
  // default): engine wins over `default_prompt_params.seed = 7`.
  lm::proto::Text2ImageSessionConfig engine_seed_42;
  engine_seed_42.set_seed(42);
  ExposedLeasedFactory seed_42_factory(
      OmniEngine::Settings{}, &engine_seed_42, lm::proto::Text2ImageMetadata{},
      resources, make_builder(captured), custom_default_options);
  EXPECT_EQ(seed_42_factory.ResolveDefaultPromptParams({}).seed, 42);
  {
    auto input = std::make_unique<PushInputSource>();
    PushInputSource* raw_input = input.get();
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session,
                         seed_42_factory.Create(std::move(input)));
    EXPECT_TRUE(captured.has_seed());
    EXPECT_EQ(captured.seed(), 42);
    ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "prompt"}));
    ASSERT_OK_AND_ASSIGN(OmniSession::Output out, session->ProcessNext());
    ASSERT_TRUE(std::holds_alternative<ImageOutput>(out));
    EXPECT_EQ(std::get<ImageOutput>(out).rgb_data[0], 42);
  }

  // Engine explicitly sets `seed = 0`: engine wins over default `7`.
  lm::proto::Text2ImageSessionConfig engine_seed_0;
  engine_seed_0.set_seed(0);
  ExposedLeasedFactory seed_0_factory(
      OmniEngine::Settings{}, &engine_seed_0, lm::proto::Text2ImageMetadata{},
      resources, make_builder(captured), custom_default_options);
  EXPECT_EQ(seed_0_factory.ResolveDefaultPromptParams({}).seed, 0);

  // (a) Per-session explicit `seed` wins over both engine (`42`) and default
  // (`7`).
  lm::proto::Text2ImageSessionConfig per_session_seed;
  per_session_seed.set_seed(99);
  EXPECT_EQ(seed_42_factory.ResolveDefaultPromptParams(per_session_seed).seed,
            99);
  EXPECT_EQ(no_seed_factory.ResolveDefaultPromptParams(per_session_seed).seed,
            99);
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<MultiStagedSession> session,
                         seed_42_factory.CreateWithPromptSource(
                             per_session_seed, /*prompt_source=*/nullptr));
    EXPECT_TRUE(captured.has_seed());
    EXPECT_EQ(captured.seed(), 99);
    auto* push_source =
        static_cast<PushPromptSource*>(session->stages()[0].get());
    ASSERT_NE(push_source, nullptr);
    ASSERT_OK(push_source->PushPrompt("prompt"));
    ASSERT_OK_AND_ASSIGN(OmniSession::Output out, session->ProcessNext());
    ASSERT_TRUE(std::holds_alternative<ImageOutput>(out));
    EXPECT_EQ(std::get<ImageOutput>(out).rgb_data[0], 99);
  }
}

TEST_F(Text2ImageSessionTest, LeasedFactoryReleasesLeaseAfterStageDestruction) {
  class CallbackOnDestroyDecoder : public FakeImageDecoder {
   public:
    CallbackOnDestroyDecoder(PromptSource* prompt_source,
                             absl::AnyInvocable<void() &&> on_destroy)
        : FakeImageDecoder(prompt_source), on_destroy_(std::move(on_destroy)) {}

    ~CallbackOnDestroyDecoder() override {
      if (on_destroy_) {
        std::move(on_destroy_)();
      }
    }

   private:
    absl::AnyInvocable<void() &&> on_destroy_;
  };

  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto resources = std::make_shared<ModelResources>(
      std::make_shared<Environment>(std::move(*env)));

  bool destructor_ran = false;
  bool reentered_create = false;
  absl::Status create_during_dtor_status = absl::OkStatus();
  std::unique_ptr<LeasedText2ImageSessionFactory> factory;
  factory = std::make_unique<LeasedText2ImageSessionFactory>(
      OmniEngine::Settings{}, /*session_config=*/nullptr,
      lm::proto::Text2ImageMetadata{}, resources,
      [&](const lm::proto::Text2ImageSessionConfig&,
          std::unique_ptr<PromptSource> prompt_source,
          std::shared_ptr<ModelResources>,
          std::vector<std::unique_ptr<internal::StageBase>>& stages)
          -> absl::StatusOr<Stage<Output>* absl_nonnull> {
        auto decoder = std::make_unique<CallbackOnDestroyDecoder>(
            prompt_source.get(), [&] {
              destructor_ran = true;
              // Guard against unbounded recursion if lease release ordering
              // regresses and the nested `Create()` succeeds.
              if (reentered_create) {
                return;
              }
              reentered_create = true;
              create_during_dtor_status =
                  factory->Create(std::make_unique<PushInputSource>()).status();
            });
        Stage<Output>* out_stage = decoder.get();
        stages.push_back(std::move(prompt_source));
        stages.push_back(std::move(decoder));
        return out_stage;
      },
      LeasedText2ImageSessionOptions{.model_display_name = "TestModel"});

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session,
                       factory->Create(std::make_unique<PushInputSource>()));
  session.reset();

  // Lease must remain held while stages are being destroyed.
  EXPECT_TRUE(destructor_ran);
  EXPECT_THAT(create_during_dtor_status,
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Another TestModel session is already")));

  // Once the full session (including all stages) is destroyed, the lease is
  // released and a new session can be created.
  destructor_ran = false;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> next_session,
                       factory->Create(std::make_unique<PushInputSource>()));
  EXPECT_NE(next_session, nullptr);
  next_session.reset();
  EXPECT_TRUE(destructor_ran);
}

TEST_F(Text2ImageSessionTest, LeasedFactoryRejectsInvalidStageGraph) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto resources = std::make_shared<ModelResources>(
      std::make_shared<Environment>(std::move(*env)));

  LeasedText2ImageSessionFactory bad_order_factory(
      OmniEngine::Settings{}, /*session_config=*/nullptr,
      lm::proto::Text2ImageMetadata{}, resources,
      [](const lm::proto::Text2ImageSessionConfig&,
         std::unique_ptr<PromptSource> prompt_source,
         std::shared_ptr<ModelResources>,
         std::vector<std::unique_ptr<internal::StageBase>>& stages)
          -> absl::StatusOr<Stage<Output>* absl_nonnull> {
        auto decoder = std::make_unique<FakeImageDecoder>(prompt_source.get());
        Stage<Output>* out_stage = decoder.get();
        // Wrong order: output stage pushed before prompt_source.
        stages.push_back(std::move(decoder));
        stages.push_back(std::move(prompt_source));
        return out_stage;
      });
  EXPECT_THAT(bad_order_factory.Create(std::make_unique<PushInputSource>()),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("final stage")));

  LeasedText2ImageSessionFactory null_stage_factory(
      OmniEngine::Settings{}, /*session_config=*/nullptr,
      lm::proto::Text2ImageMetadata{}, resources,
      [](const lm::proto::Text2ImageSessionConfig&,
         std::unique_ptr<PromptSource> prompt_source,
         std::shared_ptr<ModelResources>,
         std::vector<std::unique_ptr<internal::StageBase>>& stages)
          -> absl::StatusOr<Stage<Output>* absl_nonnull> {
        auto decoder = std::make_unique<FakeImageDecoder>(prompt_source.get());
        Stage<Output>* out_stage = decoder.get();
        stages.push_back(nullptr);
        stages.push_back(std::move(decoder));
        return out_stage;
      });
  EXPECT_THAT(null_stage_factory.Create(std::make_unique<PushInputSource>()),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("null")));
}

}  // namespace
}  // namespace litert::omni::text2image
