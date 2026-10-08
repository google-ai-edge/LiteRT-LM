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

#include "omni/text2image/text2image_engine.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "omni/base/io_types.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_session.h"
#include "omni/text2image/flux2/flux2_model_config.h"
#include "omni/text2image/image_decoder.h"
#include "omni/text2image/prompt_source.h"
#include "omni/text2image/text2image_session.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/image_gen_metadata.pb.h"
#include "runtime/proto/image_gen_model_type.pb.h"
#include "schema/core/litertlm_header_schema_generated.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::text2image {

struct Text2ImageEngineTestingPeer {
  static std::unique_ptr<Text2ImageEngine> CreateWithSettings(
      const Text2ImageEngine::Settings& settings) {
    return std::unique_ptr<Text2ImageEngine>(new Text2ImageEngine(
        settings, /*resources=*/nullptr, /*thread_pool=*/nullptr));
  }
};

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

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

void WriteDummyBonsaiLitertLmFile(const std::filesystem::path& path) {
  lm::proto::ImageGenMetadata meta;
  meta.mutable_image_gen_model_type()
      ->mutable_bonsai_flux2()
      ->mutable_flux2_params()
      ->set_seq_len(128);
  std::string meta_bytes = meta.SerializeAsString();

  flatbuffers::FlatBufferBuilder builder(1024);
  uint64_t meta_begin = 256;
  uint64_t meta_end = meta_begin + meta_bytes.size();
  auto image_gen_meta_section = lm::schema::CreateSectionObject(
      builder, /*items=*/0, meta_begin, meta_end,
      lm::schema::AnySectionDataType_ImageGenMetadataProto);

  std::vector<flatbuffers::Offset<lm::schema::SectionObject>> sections = {
      image_gen_meta_section};
  auto section_metadata = lm::schema::CreateSectionMetadata(
      builder, builder.CreateVector(sections));
  auto metadata =
      lm::schema::CreateLiteRTLMMetaData(builder, 0, section_metadata);
  builder.Finish(metadata);

  std::ofstream file(path, std::ios::binary);
  file.write("LITERTLM", 8);
  uint32_t major = 1, minor = 0, patch = 0, padding = 0;
  file.write(reinterpret_cast<const char*>(&major), sizeof(uint32_t));
  file.write(reinterpret_cast<const char*>(&minor), sizeof(uint32_t));
  file.write(reinterpret_cast<const char*>(&patch), sizeof(uint32_t));
  file.write(reinterpret_cast<const char*>(&padding), sizeof(uint32_t));
  uint64_t header_end_offset = 32 + builder.GetSize();
  file.write(reinterpret_cast<const char*>(&header_end_offset),
             sizeof(uint64_t));
  file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
             builder.GetSize());
  if (header_end_offset < meta_begin) {
    std::string pad(meta_begin - header_end_offset, '\0');
    file.write(pad.data(), pad.size());
  }
  file.write(meta_bytes.data(), meta_bytes.size());
}

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

TEST(DetectModelTypeTest, DetectsBonsaiFlux2FromLitertLmFileAndFolder) {
  std::filesystem::path temp_dir =
      std::filesystem::path(::testing::TempDir()) / "bonsai_litertlm_dir";
  std::filesystem::create_directories(temp_dir);
  std::filesystem::path litertlm_file = temp_dir / "bonsai_flux2.litertlm";
  WriteDummyBonsaiLitertLmFile(litertlm_file);

  EXPECT_THAT(DetectModelType(litertlm_file.string()),
              IsOkAndHolds(ModelType::BONSAI_FLUX2));
  EXPECT_THAT(DetectModelType(temp_dir.string()),
              IsOkAndHolds(ModelType::BONSAI_FLUX2));
  std::filesystem::remove_all(temp_dir);
}

TEST(DetectModelTypeTest, RejectsEmptyOrUnknownFolder) {
  EXPECT_THAT(DetectModelType(""),
              StatusIs(absl::StatusCode::kInvalidArgument));

  std::filesystem::path temp_dir =
      std::filesystem::path(::testing::TempDir()) / "empty_text2image_dir";
  std::filesystem::create_directories(temp_dir);
  EXPECT_THAT(DetectModelType(temp_dir.string()),
              StatusIs(absl::StatusCode::kInvalidArgument));
  std::filesystem::remove_all(temp_dir);
}

TEST(Text2ImageEngineTest, GetModelTypeAndResolveDefaultPromptParams) {
  Text2ImageEngine::Settings settings;
  EXPECT_EQ(settings.GetModelType(), ModelType::UNSPECIFIED);

  Flux2ModelConfig flux2_config;
  flux2_config.img_size = 256;
  flux2_config.steps = 4;
  settings.model_config = flux2_config;
  EXPECT_EQ(settings.GetModelType(), ModelType::BONSAI_FLUX2);

  auto engine = Text2ImageEngineTestingPeer::CreateWithSettings(settings);
  ImageGenInputMetadata resolved_defaults =
      engine->ResolveDefaultPromptParams({});
  EXPECT_EQ(resolved_defaults.width, 256);
  EXPECT_EQ(resolved_defaults.height, 256);
  EXPECT_EQ(resolved_defaults.num_inference_steps, 4);

  Text2ImageEngine::SessionSettings custom_input{
      .width = 512, .height = 512, .num_inference_steps = 8, .seed = 77};
  ImageGenInputMetadata resolved_custom =
      engine->ResolveDefaultPromptParams(custom_input);
  EXPECT_EQ(resolved_custom.width, 512);
  EXPECT_EQ(resolved_custom.height, 512);
  EXPECT_EQ(resolved_custom.num_inference_steps, 8);
  EXPECT_EQ(resolved_custom.seed, 77);

  EXPECT_THAT(engine->CreateSession(Text2ImageEngine::SessionSettings{
                  .width = 512, .height = 256}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(engine->CreateSession(
                  Text2ImageEngine::SessionSettings{.width = 256, .height = 0}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Text2ImageEngineTest, CreateFailsWhenModelContainerIsMissingOrIncomplete) {
  Text2ImageEngine::Settings settings;
  settings.model_folder = "/nonexistent/flux2_model_dir";
  settings.model_config = Flux2ModelConfig{};
  EXPECT_THAT(Text2ImageEngine::Create(settings),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(Text2ImageSessionFactory::CreateFactory(settings),
              StatusIs(absl::StatusCode::kNotFound));
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

}  // namespace
}  // namespace litert::omni::text2image
