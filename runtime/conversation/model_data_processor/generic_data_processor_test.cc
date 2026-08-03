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

#include "runtime/conversation/model_data_processor/generic_data_processor.h"

#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "litert/cc/litert_layout.h"  // from @litert
#include "support/preprocessor/audio_preprocessor.h"  // from @litert
#include "support/preprocessor/audio_preprocessor_miniaudio.h"  // from @litert
#include "support/preprocessor/image_preprocessor.h"  // from @litert
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/generic_data_processor_config.h"
#include "runtime/conversation/model_data_processor/multimodal_processor_helper.h"
#include "runtime/conversation/model_data_processor/test_utils.h"
#include "runtime/engine/io_types.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

using json = nlohmann::ordered_json;
using ::testing::ElementsAre;


TEST(GenericDataProcessorTest, ToInputDataVector) {
  ASSERT_OK_AND_ASSIGN(auto processor, GenericDataProcessor::Create());
  const std::string rendered_template_prompt =
      "<start_of_turn>user\ntest "
      "prompt\n<end_of_turn>\n<start_of_turn>assistant\ntest "
      "response\n<end_of_turn>";
  const nlohmann::ordered_json messages = {
      {"role", "user"},
      {"content", "test prompt"},
      {"role", "assistant"},
      {"content", "test response"},
  };
  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(rendered_template_prompt, messages, {}));

  InputText expected_text(
      "<start_of_turn>user\ntest "
      "prompt\n<end_of_turn>\n<start_of_turn>assistant\ntest "
      "response\n<end_of_turn>");
  EXPECT_THAT(input_data, ElementsAre(HasInputText(&expected_text)));
}

TEST(GenericDataProcessorTest, ToMessageDefault) {
  ASSERT_OK_AND_ASSIGN(auto processor, GenericDataProcessor::Create());

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(Responses(TaskState::kProcessing, {"test response"}),
                           std::monostate{}));

  EXPECT_EQ(
      message,
      json({{"role", "assistant"},
            {"content", {{{"type", "text"}, {"text", "test response"}}}}}));
}

TEST(GenericDataProcessorTest, ToMessageModelRole) {
  ASSERT_OK_AND_ASSIGN(auto processor,
                       GenericDataProcessor::Create(
                           GenericDataProcessorConfig{.model_role = "model"}));

  ASSERT_OK_AND_ASSIGN(
      const Message message,
      processor->ToMessage(Responses(TaskState::kProcessing, {"test response"}),
                           std::monostate{}));

  EXPECT_EQ(
      message,
      json({{"role", "model"},
            {"content", {{{"type", "text"}, {"text", "test response"}}}}}));
}

TEST(GenericDataProcessorTest, ToTemplateInputNoTypedContent) {
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      GenericDataProcessor::Create(
          GenericDataProcessorConfig{.model_role = "model"},
          PromptTemplateCapabilities{.requires_typed_content = false}));
  ASSERT_OK_AND_ASSIGN(const json template_input_1,
                       processor->MessageToTemplateInput(json(
                           {{"role", "user"}, {"content", "test prompt"}})));
  EXPECT_EQ(template_input_1,
            json({{"role", "user"}, {"content", "test prompt"}}));
  ASSERT_OK_AND_ASSIGN(
      const json template_input_2,
      processor->MessageToTemplateInput(
          json({{"role", "user"},
                {"content", {{{"type", "text"}, {"text", "test prompt"}}}}})));
  EXPECT_EQ(template_input_2,
            json({{"role", "user"}, {"content", "test prompt"}}));
}

TEST(GenericDataProcessorTest, ToTemplateInputTypedContent) {
  ASSERT_OK_AND_ASSIGN(
      auto processor,
      GenericDataProcessor::Create(
          GenericDataProcessorConfig{.model_role = "model"},
          PromptTemplateCapabilities{.requires_typed_content = true}));
  ASSERT_OK_AND_ASSIGN(const json template_input_1,
                       processor->MessageToTemplateInput(json(
                           {{"role", "user"}, {"content", "test prompt"}})));
  EXPECT_EQ(template_input_1,
            json({{"role", "user"},
                  {"content", {{{"type", "text"}, {"text", "test prompt"}}}}}));
  ASSERT_OK_AND_ASSIGN(
      const json template_input_2,
      processor->MessageToTemplateInput(
          json({{"role", "user"},
                {"content", {{{"type", "text"}, {"text", "test prompt"}}}}})));
  EXPECT_EQ(template_input_2,
            json({{"role", "user"},
                  {"content", {{{"type", "text"}, {"text", "test prompt"}}}}}));
}

TEST(GenericDataProcessorTest, ToInputDataVectorMultimodal) {
  std::string image_path = GetImageTestdataPath("apple.bmp");
  std::string audio_path = GetImageTestdataPath("audio_sample.wav");

  ImagePreprocessParameter image_preprocess_parameter;
  image_preprocess_parameter.SetTargetDimensions(Dimensions({1, 224, 128, 3}));

  auto audio_config = AudioPreprocessorConfig::Create(
      /*sample_rate_hz=*/16000,
      /*num_channels=*/1,
      /*frame_length=*/320,
      /*hop_length=*/160,
      /*fft_length=*/512,
      /*input_scale=*/1.0,
      /*pre_emphasis_factor=*/0.0,
      /*num_mel_bins=*/128,
      /*mel_low_hz=*/0.0,
      /*mel_high_hz=*/8000.0,
      /*mel_floor=*/1e-3,
      /*normalize_mel=*/false,
      /*add_floor_to_mel_before_log=*/true,
      /*semicausal_padding=*/true,
      /*non_zero_hanning=*/false,
      /*periodic_hanning=*/true,
      /*fft_padding_type=*/AudioPreprocessorConfig::FftPaddingType::kRight,
      /*skip_mel_spectrogram_extraction=*/false
  );

  GenericDataProcessorConfig config = {
      .multimodal = MultimodalConfig{
          .image_enabled = true,
          .image_preprocess_parameter = std::move(image_preprocess_parameter),
          .audio_enabled = true,
          .audio_preprocessor_config = std::move(audio_config),
          .processing_config = MultimodalPromptProcessingConfig{
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
          }}};

  ASSERT_OK_AND_ASSIGN(auto processor, GenericDataProcessor::Create(config));

  const std::string prompt = "Show: <img>, Listen: <audio>. Finish.";
  const json messages = json::array({
      {{"role", "user"},
       {"content",
        {{{"type", "text"}, {"text", "Show: "}},
         {{"type", "image"}, {"path", image_path}},
         {{"type", "text"}, {"text", ", Listen: "}},
         {{"type", "audio"}, {"path", audio_path}},
         {{"type", "text"}, {"text", ". Finish."}}}}}
  });

  ASSERT_OK_AND_ASSIGN(
      const std::vector<InputData> input_data,
      processor->ToInputDataVector(prompt, messages, {}));

  InputText expected_text1("Show: img_<boi>");
  auto image_preprocessor = ImagePreprocessor::Create();
  ImagePreprocessParameter img_params;
  img_params.SetTargetDimensions(Dimensions({1, 224, 128, 3}));
  ASSERT_OK_AND_ASSIGN(InputImage expected_image,
                       image_preprocessor->Preprocess(
                           InputImage(ReadFile(image_path)), img_params));
  InputText expected_img_suffix("_img");
  InputText expected_text2(", Listen: aud_<boa>");
  auto audio_preprocessor = AudioPreprocessorMiniAudio::Create(
      config.multimodal->audio_preprocessor_config);
  ASSERT_OK(audio_preprocessor);
  ASSERT_OK_AND_ASSIGN(
      InputAudio expected_audio,
      (*audio_preprocessor)->Preprocess(InputAudio(ReadFile(audio_path))));
  InputText expected_aud_suffix("_aud");
  InputText expected_rest(". Finish.");

  EXPECT_THAT(input_data, ElementsAre(HasInputText(&expected_text1),
                                      HasInputImage(&expected_image),
                                      HasInputText(&expected_img_suffix),
                                      HasInputText(&expected_text2),
                                      HasInputAudio(&expected_audio),
                                      HasInputAudioEnd(),
                                      HasInputText(&expected_aud_suffix),
                                      HasInputText(&expected_rest)));
}

TEST(GenericDataProcessorTest, CloneStateMultimodal) {
  auto audio_config = AudioPreprocessorConfig::Create(
      /*sample_rate_hz=*/16000,
      /*num_channels=*/1,
      /*frame_length=*/320,
      /*hop_length=*/160,
      /*fft_length=*/512,
      /*input_scale=*/1.0,
      /*pre_emphasis_factor=*/0.0,
      /*num_mel_bins=*/128,
      /*mel_low_hz=*/0.0,
      /*mel_high_hz=*/8000.0,
      /*mel_floor=*/1e-3,
      /*normalize_mel=*/false,
      /*add_floor_to_mel_before_log=*/true,
      /*semicausal_padding=*/true,
      /*non_zero_hanning=*/false,
      /*periodic_hanning=*/true,
      /*fft_padding_type=*/AudioPreprocessorConfig::FftPaddingType::kRight,
      /*skip_mel_spectrogram_extraction=*/false
  );
  GenericDataProcessorConfig config = {
      .multimodal = MultimodalConfig{
          .audio_enabled = true,
          .audio_preprocessor_config = std::move(audio_config),
      }
  };

  ASSERT_OK_AND_ASSIGN(auto processor1, GenericDataProcessor::Create(config));
  ASSERT_OK_AND_ASSIGN(auto processor2, GenericDataProcessor::Create(config));

  // Verify that CloneState succeeds.
  EXPECT_OK(processor2->CloneState(*processor1));
}

}  // namespace
}  // namespace litert::lm

