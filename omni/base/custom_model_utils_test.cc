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

#include "omni/base/custom_model_utils.h"

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "runtime/proto/any2any_model_type.pb.h"
#include "runtime/proto/image_gen_model_type.pb.h"
#include "runtime/proto/speech2text_model_type.pb.h"
#include "runtime/proto/text2speech_model_type.pb.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Eq;
using ::testing::HasSubstr;

TEST(CustomModelUtilsTest, EnsureDefaultCustomModelPacksWhenUnset) {
  lm::proto::Any2AnyModelType model_type;
  ASSERT_OK(EnsureDefaultCustomModel<lm::proto::MiniCpmO>(model_type));
  ASSERT_TRUE(model_type.has_custom_model());

  ASSERT_OK_AND_ASSIGN(std::optional<lm::proto::MiniCpmO> unpacked,
                       UnpackCustomModel<lm::proto::MiniCpmO>(model_type));
  ASSERT_TRUE(unpacked.has_value());
  EXPECT_FALSE(unpacked->has_llm_hidden_size());
}

TEST(CustomModelUtilsTest, EnsureDefaultCustomModelUsesExplicitDefault) {
  lm::proto::ImageGenModelType model_type;
  lm::proto::Flux2Params default_params;
  default_params.set_img_size(512);
  default_params.set_default_steps(4);

  ASSERT_OK(EnsureDefaultCustomModel(model_type, default_params));
  ASSERT_TRUE(model_type.has_custom_model());

  ASSERT_OK_AND_ASSIGN(std::optional<lm::proto::Flux2Params> unpacked,
                       UnpackCustomModel<lm::proto::Flux2Params>(model_type));
  ASSERT_TRUE(unpacked.has_value());
  EXPECT_EQ(unpacked->img_size(), 512);
  EXPECT_EQ(unpacked->default_steps(), 4);
}

TEST(CustomModelUtilsTest,
     EnsureDefaultCustomModelLeavesExistingCustomModelUntouched) {
  lm::proto::Speech2TextModelType model_type;
  lm::proto::Kokoro existing;
  existing.set_voice_name("af_heart");
  ASSERT_TRUE(model_type.mutable_custom_model()->PackFrom(existing));

  lm::proto::Kokoro replacement;
  replacement.set_voice_name("other");
  ASSERT_OK(EnsureDefaultCustomModel(model_type, replacement));

  ASSERT_OK_AND_ASSIGN(std::optional<lm::proto::Kokoro> unpacked,
                       UnpackCustomModel<lm::proto::Kokoro>(model_type));
  ASSERT_TRUE(unpacked.has_value());
  EXPECT_EQ(unpacked->voice_name(), "af_heart");
}

TEST(CustomModelUtilsTest,
     EnsureDefaultCustomModelLeavesBuiltinOneofArmUntouched) {
  lm::proto::Text2SpeechModelType model_type;
  model_type.mutable_kokoro()->set_voice_name("af_bella");

  ASSERT_OK(EnsureDefaultCustomModel<lm::proto::Qwen3Tts>(model_type));
  EXPECT_TRUE(model_type.has_kokoro());
  EXPECT_FALSE(model_type.has_custom_model());
  EXPECT_EQ(model_type.kokoro().voice_name(), "af_bella");
}

TEST(CustomModelUtilsTest, UnpackCustomModelReturnsNulloptWhenUnset) {
  lm::proto::Any2AnyModelType model_type;
  EXPECT_THAT(UnpackCustomModel<lm::proto::MiniCpmO>(model_type),
              IsOkAndHolds(Eq(std::nullopt)));
}

TEST(CustomModelUtilsTest, UnpackCustomModelRejectsBuiltinOneofArm) {
  lm::proto::Any2AnyModelType model_type;
  model_type.mutable_minicpm_o()->set_llm_hidden_size(128);

  EXPECT_THAT(
      UnpackCustomModel<lm::proto::QwenOmni>(model_type),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("custom_model")));
}

TEST(CustomModelUtilsTest, UnpackCustomModelRejectsTypeMismatch) {
  lm::proto::ImageGenModelType model_type;
  lm::proto::Kokoro wrong_payload;
  wrong_payload.set_voice_name("af_heart");
  ASSERT_TRUE(model_type.mutable_custom_model()->PackFrom(wrong_payload));

  EXPECT_THAT(UnpackCustomModel<lm::proto::Flux2Params>(model_type),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("litert.lm.proto.Kokoro")));
}

TEST(CustomModelUtilsTest, UnpackCustomModelRejectsCorruptPayload) {
  lm::proto::ImageGenModelType model_type;
  lm::proto::Flux2Params dummy;
  ASSERT_TRUE(model_type.mutable_custom_model()->PackFrom(dummy));
  // Truncated varint inside the serialized `Any.value` forces parse failure.
  model_type.mutable_custom_model()->set_value("\x08\x80");

  EXPECT_THAT(UnpackCustomModel<lm::proto::Flux2Params>(model_type),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Failed to unpack")));
}

}  // namespace
}  // namespace litert::omni
