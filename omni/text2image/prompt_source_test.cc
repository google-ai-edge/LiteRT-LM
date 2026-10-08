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

#include "omni/text2image/prompt_source.h"

#include <cstdint>
#include <utility>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "omni/base/io_types.h"
#include "omni/text2image/image_decoder.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::text2image {
namespace {

using ::absl_testing::StatusIs;

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
  EXPECT_EQ(p->metadata.num_inference_steps, 4);
  EXPECT_EQ(p->metadata.seed, 99);

  source.Finish();
  EXPECT_THAT(source.PushPrompt("After finish"),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(PushPromptSourceTest, RejectsEmptyPromptAndSupportsReset) {
  PushPromptSource source;
  EXPECT_THAT(source.PushPrompt(""),
              StatusIs(absl::StatusCode::kInvalidArgument));

  ASSERT_OK(source.PushPrompt("Initial prompt"));
  source.Finish();
  ASSERT_TRUE(source.HasOutput());
  source.Reset();
  EXPECT_FALSE(source.HasOutput());

  ASSERT_OK(source.PushPrompt("Post-reset prompt"));
  ASSERT_TRUE(source.HasOutput());
  auto p = source.GetOutput();
  ASSERT_OK(p);
  EXPECT_EQ(p->text, "Post-reset prompt");
}

TEST(ImageDecoderTest, DecodesFromPromptSourceAndFlushes) {
  PushPromptSource source(
      ImageGenInputMetadata{.width = 8, .height = 8, .seed = 123});
  FakeImageDecoder decoder(&source);

  ASSERT_OK(source.PushPrompt("A red apple"));
  EXPECT_TRUE(decoder.NeedSchedule());
  ASSERT_OK(decoder.Schedule());
  ASSERT_TRUE(decoder.HasOutput());

  auto out = decoder.GetOutput();
  ASSERT_OK(out);
  ASSERT_TRUE(std::holds_alternative<ImageOutput>(*out));
  const auto& img = std::get<ImageOutput>(*out);
  EXPECT_EQ(img.width, 8);
  EXPECT_EQ(img.height, 8);
  EXPECT_EQ(img.channels, 3);
  EXPECT_EQ(img.rgb_data.size(), 8 * 8 * 3);
  EXPECT_EQ(img.rgb_data[0], 123);

  ASSERT_OK(decoder.Flush());
}

}  // namespace
}  // namespace litert::omni::text2image
