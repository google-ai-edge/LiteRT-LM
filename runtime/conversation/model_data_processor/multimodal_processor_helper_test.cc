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

#include "runtime/conversation/model_data_processor/multimodal_processor_helper.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "litert/cc/litert_layout.h"  // from @litert
#include "runtime/conversation/model_data_processor/test_utils.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/test_utils.h"  // NOLINT
#include "support/preprocessor/audio_preprocessor.h"
#include "support/preprocessor/audio_preprocessor_miniaudio.h"
#include "support/preprocessor/image_preprocessor.h"

namespace litert::lm {

using AudioPreprocessorConfig = ::litert::support::AudioPreprocessorConfig;
using AudioPreprocessorMiniAudio =
    ::litert::support::AudioPreprocessorMiniAudio;

namespace {

using json = nlohmann::ordered_json;
using ::testing::ElementsAre;

class MultimodalProcessorHelperTest : public ::testing::Test {
 protected:
  void SetUp() override {
    image_preprocessor_ = ImagePreprocessor::Create();
    ASSERT_NE(image_preprocessor_, nullptr);

    auto audio_prep = AudioPreprocessorMiniAudio::Create(
        AudioPreprocessorConfig::CreateDefaultUsmConfig());
    ASSERT_OK(audio_prep);
    audio_preprocessor_ = std::move(*audio_prep);
  }

  std::unique_ptr<ImagePreprocessor> image_preprocessor_;
  std::unique_ptr<AudioPreprocessor> audio_preprocessor_;
};

TEST_F(MultimodalProcessorHelperTest, ProcessTextOnly) {
  const std::string prompt = "Hello world";
  const json messages = json::array();
  MultimodalPromptProcessingConfig config;
  ImagePreprocessParameter image_params;

  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      ProcessMultimodalPrompt(prompt, messages, image_preprocessor_.get(),
                              audio_preprocessor_.get(), config, image_params));

  InputText expected_text("Hello world");
  EXPECT_THAT(input_data, ElementsAre(HasInputText(&expected_text)));
}

TEST_F(MultimodalProcessorHelperTest, ProcessImage) {
  std::string image_path = GetImageTestdataPath("apple.bmp");
  const std::string prompt = "Here is an image: <img>. What is it?";
  const json messages =
      json::array({{{"role", "user"},
                    {"content",
                     {{{"type", "text"}, {"text", "Here is an image: "}},
                      {{"type", "image"}, {"path", image_path}},
                      {{"type", "text"}, {"text", ". What is it?"}}}}}});

  MultimodalPromptProcessingConfig config = {
      .delimiter_regex = "(<img>)",
      .image_token_regex = "<img>",
      .boi_token = "<boi>",
      .eoi_token = "<eoi>",
      .image_prefix = "image_prefix_",
      .image_suffix = "_image_suffix",
      .add_image_end = true,
  };
  ImagePreprocessParameter image_params;
  image_params.SetTargetDimensions(Dimensions({1, 224, 128, 3}));

  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      ProcessMultimodalPrompt(prompt, messages, image_preprocessor_.get(),
                              audio_preprocessor_.get(), config, image_params));

  InputText expected_text1("Here is an image: image_prefix_<boi>");
  ImagePreprocessParameter img_params;
  img_params.SetTargetDimensions(Dimensions({1, 224, 128, 3}));
  ASSERT_OK_AND_ASSIGN(InputImage expected_image,
                       image_preprocessor_->Preprocess(
                           InputImage(ReadFile(image_path)), img_params));
  InputText expected_suffix("_image_suffix");
  InputText expected_rest(". What is it?");

  EXPECT_THAT(
      input_data,
      ElementsAre(HasInputText(&expected_text1), HasInputImage(&expected_image),
                  HasInputImageEnd(), HasInputText(&expected_suffix),
                  HasInputText(&expected_rest)));
}

TEST_F(MultimodalProcessorHelperTest, ProcessAudio) {
  std::string audio_path = GetImageTestdataPath("audio_sample.wav");
  const std::string prompt = "Listen to this: <audio>. Tell me what you heard.";
  const json messages = json::array(
      {{{"role", "user"},
        {"content",
         {{{"type", "text"}, {"text", "Listen to this: "}},
          {{"type", "audio"}, {"path", audio_path}},
          {{"type", "text"}, {"text", ". Tell me what you heard."}}}}}});

  MultimodalPromptProcessingConfig config = {
      .delimiter_regex = "(<audio>)",
      .audio_token_regex = "<audio>",
      .boa_token = "<boa>",
      .eoa_token = "<eoa>",
      .audio_prefix = "audio_prefix_",
      .audio_suffix = "_audio_suffix",
      .add_audio_end = true,
  };
  ImagePreprocessParameter image_params;

  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      ProcessMultimodalPrompt(prompt, messages, image_preprocessor_.get(),
                              audio_preprocessor_.get(), config, image_params));

  InputText expected_text1("Listen to this: audio_prefix_<boa>");
  ASSERT_OK_AND_ASSIGN(
      InputAudio expected_audio,
      audio_preprocessor_->Preprocess(InputAudio(ReadFile(audio_path))));
  InputText expected_suffix("_audio_suffix");
  InputText expected_rest(". Tell me what you heard.");

  EXPECT_THAT(
      input_data,
      ElementsAre(HasInputText(&expected_text1), HasInputAudio(&expected_audio),
                  HasInputAudioEnd(), HasInputText(&expected_suffix),
                  HasInputText(&expected_rest)));
}

TEST_F(MultimodalProcessorHelperTest, ProcessMultimodal) {
  std::string image_path = GetImageTestdataPath("apple.bmp");
  std::string audio_path = GetImageTestdataPath("audio_sample.wav");
  const std::string prompt = "Show: <img>, Listen: <audio>. Finish.";
  const json messages =
      json::array({{{"role", "user"},
                    {"content",
                     {{{"type", "text"}, {"text", "Show: "}},
                      {{"type", "image"}, {"path", image_path}},
                      {{"type", "text"}, {"text", ", Listen: "}},
                      {{"type", "audio"}, {"path", audio_path}},
                      {{"type", "text"}, {"text", ". Finish."}}}}}});

  MultimodalPromptProcessingConfig config = {
      .delimiter_regex = "(<img>|<audio>)",
      .image_token_regex = "<img>",
      .audio_token_regex = "<audio>",
      .boi_token = "<boi>",
      .eoi_token = "<eoi>",
      .image_prefix = "img_",
      .image_suffix = "_img",
      .add_image_end = false,
      .boa_token = "<boa>",
      .eoa_token = "<eoa>",
      .audio_prefix = "aud_",
      .audio_suffix = "_aud",
      .add_audio_end = true,
  };
  ImagePreprocessParameter image_params;
  image_params.SetTargetDimensions(Dimensions({1, 224, 128, 3}));

  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      ProcessMultimodalPrompt(prompt, messages, image_preprocessor_.get(),
                              audio_preprocessor_.get(), config, image_params));

  InputText expected_text1("Show: img_<boi>");
  ImagePreprocessParameter img_params;
  img_params.SetTargetDimensions(Dimensions({1, 224, 128, 3}));
  ASSERT_OK_AND_ASSIGN(InputImage expected_image,
                       image_preprocessor_->Preprocess(
                           InputImage(ReadFile(image_path)), img_params));
  InputText expected_img_suffix("_img");
  InputText expected_text2(", Listen: aud_<boa>");
  ASSERT_OK_AND_ASSIGN(
      InputAudio expected_audio,
      audio_preprocessor_->Preprocess(InputAudio(ReadFile(audio_path))));
  InputText expected_aud_suffix("_aud");
  InputText expected_rest(". Finish.");

  EXPECT_THAT(
      input_data,
      ElementsAre(HasInputText(&expected_text1), HasInputImage(&expected_image),
                  HasInputText(&expected_img_suffix),
                  HasInputText(&expected_text2), HasInputAudio(&expected_audio),
                  HasInputAudioEnd(), HasInputText(&expected_aud_suffix),
                  HasInputText(&expected_rest)));
}

TEST_F(MultimodalProcessorHelperTest, ProcessImagePatchify) {
  std::string image_path = GetImageTestdataPath("apple.bmp");
  const std::string prompt = "Here is <img>";
  const json messages =
      json::array({{{"role", "user"},
                    {"content",
                     {{{"type", "text"}, {"text", "Here is "}},
                      {{"type", "image"}, {"path", image_path}}}}}});

  MultimodalPromptProcessingConfig config = {
      .delimiter_regex = "(<img>)",
      .image_token_regex = "<img>",
  };
  ImagePreprocessParameter image_params;
  image_params.SetPatchifyConfig(ImagePreprocessParameter::PatchifyConfig{
      .patch_width = 16,
      .patch_height = 16,
      .max_num_patches = 4,
      .pooling_kernel_size = 2,
  });

  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      ProcessMultimodalPrompt(prompt, messages, image_preprocessor_.get(),
                              audio_preprocessor_.get(), config, image_params));

  InputText expected_text1("Here is ");
  ImagePreprocessParameter img_params;
  img_params.SetPatchifyConfig(ImagePreprocessParameter::PatchifyConfig{
      .patch_width = 16,
      .patch_height = 16,
      .max_num_patches = 4,
      .pooling_kernel_size = 2,
  });
  ASSERT_OK_AND_ASSIGN(InputImage expected_image,
                       image_preprocessor_->Preprocess(
                           InputImage(ReadFile(image_path)), img_params));

  EXPECT_THAT(input_data, ElementsAre(HasInputText(&expected_text1),
                                      HasInputImage(&expected_image)));
}

TEST_F(MultimodalProcessorHelperTest, ProcessImageVisualTokenBudget) {
  std::string image_path = GetImageTestdataPath("apple.bmp");
  const std::string prompt = "Here is <img>";
  const json messages =
      json::array({{{"role", "user"},
                    {"content",
                     {{{"type", "text"}, {"text", "Here is "}},
                      {{"type", "image"}, {"path", image_path}}}}}});

  MultimodalPromptProcessingConfig config = {
      .delimiter_regex = "(<img>)",
      .image_token_regex = "<img>",
  };
  ImagePreprocessParameter image_params;
  image_params.SetPatchifyConfig(ImagePreprocessParameter::PatchifyConfig{
      .patch_width = 16,
      .patch_height = 16,
      .max_num_patches = 100,
      .pooling_kernel_size = 3,
  });
  std::optional<int> visual_token_budget = 4;

  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      ProcessMultimodalPrompt(prompt, messages, image_preprocessor_.get(),
                              audio_preprocessor_.get(), config, image_params,
                              visual_token_budget));

  InputText expected_text1("Here is ");
  ImagePreprocessParameter img_params;
  img_params.SetPatchifyConfig(ImagePreprocessParameter::PatchifyConfig{
      .patch_width = 16,
      .patch_height = 16,
      .max_num_patches = 36,  // budget * 9 = 4 * 9 = 36
      .pooling_kernel_size = 2,
  });
  ASSERT_OK_AND_ASSIGN(InputImage expected_image,
                       image_preprocessor_->Preprocess(
                           InputImage(ReadFile(image_path)), img_params));

  EXPECT_THAT(input_data, ElementsAre(HasInputText(&expected_text1),
                                      HasInputImage(&expected_image)));
}

TEST_F(MultimodalProcessorHelperTest, ProcessImageVisualTokenBudgetInvalid) {
  std::string image_path = GetImageTestdataPath("apple.bmp");
  const std::string prompt = "Here is <img>";
  const json messages =
      json::array({{{"role", "user"},
                    {"content",
                     {{{"type", "text"}, {"text", "Here is "}},
                      {{"type", "image"}, {"path", image_path}}}}}});

  MultimodalPromptProcessingConfig config = {
      .delimiter_regex = "(<img>)",
      .image_token_regex = "<img>",
  };
  ImagePreprocessParameter image_params;
  image_params.SetPatchifyConfig(ImagePreprocessParameter::PatchifyConfig{
      .patch_width = 16,
      .patch_height = 16,
      .max_num_patches = 100,
      .pooling_kernel_size = 2,
  });
  std::optional<int> visual_token_budget = 0;

  EXPECT_THAT(
      ProcessMultimodalPrompt(prompt, messages, image_preprocessor_.get(),
                              audio_preprocessor_.get(), config, image_params,
                              visual_token_budget),
      testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(MultimodalProcessorHelperTest, MissingPreprocessorForImage) {
  std::string image_path = GetImageTestdataPath("apple.bmp");
  const std::string prompt = "Here is <img>";
  const json messages =
      json::array({{{"role", "user"},
                    {"content",
                     {{{"type", "text"}, {"text", "Here is "}},
                      {{"type", "image"}, {"path", image_path}}}}}});

  MultimodalPromptProcessingConfig config = {
      .delimiter_regex = "(<img>)",
      .image_token_regex = "<img>",
  };
  ImagePreprocessParameter image_params;

  EXPECT_THAT(
      ProcessMultimodalPrompt(prompt, messages, /*image_preprocessor=*/nullptr,
                              audio_preprocessor_.get(), config, image_params),
      testing::status::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(MultimodalProcessorHelperTest, MissingPreprocessorForAudio) {
  std::string audio_path = GetImageTestdataPath("audio_sample.wav");
  const std::string prompt = "Here is <audio>";
  const json messages =
      json::array({{{"role", "user"},
                    {"content",
                     {{{"type", "text"}, {"text", "Here is "}},
                      {{"type", "audio"}, {"path", audio_path}}}}}});

  MultimodalPromptProcessingConfig config = {
      .delimiter_regex = "(<audio>)",
      .audio_token_regex = "<audio>",
  };
  ImagePreprocessParameter image_params;

  EXPECT_THAT(ProcessMultimodalPrompt(
                  prompt, messages, image_preprocessor_.get(),
                  /*audio_preprocessor=*/nullptr, config, image_params),
              testing::status::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(MultimodalProcessorHelperTest, MismatchImageCountLess) {
  const std::string prompt = "Here is <img>";
  const json messages = json::array(
      {{{"role", "user"},
        {"content", {{{"type", "text"}, {"text", "Here is no image"}}}}}});

  MultimodalPromptProcessingConfig config = {
      .delimiter_regex = "(<img>)",
      .image_token_regex = "<img>",
  };
  ImagePreprocessParameter image_params;

  EXPECT_THAT(
      ProcessMultimodalPrompt(prompt, messages, image_preprocessor_.get(),
                              audio_preprocessor_.get(), config, image_params),
      testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(MultimodalProcessorHelperTest, MismatchImageCountMore) {
  std::string image_path = GetImageTestdataPath("apple.bmp");
  const std::string prompt = "Here is text only";
  const json messages =
      json::array({{{"role", "user"},
                    {"content",
                     {{{"type", "text"}, {"text", "Here is "}},
                      {{"type", "image"}, {"path", image_path}}}}}});

  MultimodalPromptProcessingConfig config = {
      .delimiter_regex = "(<img>)",
      .image_token_regex = "<img>",
  };
  ImagePreprocessParameter image_params;

  EXPECT_THAT(
      ProcessMultimodalPrompt(prompt, messages, image_preprocessor_.get(),
                              audio_preprocessor_.get(), config, image_params),
      testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(MultimodalProcessorHelperTest, MismatchAudioCountLess) {
  const std::string prompt = "Here is <audio>";
  const json messages = json::array(
      {{{"role", "user"},
        {"content", {{{"type", "text"}, {"text", "Here is no audio"}}}}}});

  MultimodalPromptProcessingConfig config = {
      .delimiter_regex = "(<audio>)",
      .audio_token_regex = "<audio>",
  };
  ImagePreprocessParameter image_params;

  EXPECT_THAT(
      ProcessMultimodalPrompt(prompt, messages, image_preprocessor_.get(),
                              audio_preprocessor_.get(), config, image_params),
      testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(MultimodalProcessorHelperTest, MismatchAudioCountMore) {
  std::string audio_path = GetImageTestdataPath("audio_sample.wav");
  const std::string prompt = "Here is text only";
  const json messages =
      json::array({{{"role", "user"},
                    {"content",
                     {{{"type", "text"}, {"text", "Here is "}},
                      {{"type", "audio"}, {"path", audio_path}}}}}});

  MultimodalPromptProcessingConfig config = {
      .delimiter_regex = "(<audio>)",
      .audio_token_regex = "<audio>",
  };
  ImagePreprocessParameter image_params;

  EXPECT_THAT(
      ProcessMultimodalPrompt(prompt, messages, image_preprocessor_.get(),
                              audio_preprocessor_.get(), config, image_params),
      testing::status::StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::lm
