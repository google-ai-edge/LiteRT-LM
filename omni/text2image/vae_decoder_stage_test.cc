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

#include "omni/text2image/vae_decoder_stage.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/stage.h"
#include "omni/text2image/io_types.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::text2image {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

class FakeDenoiserStage : public SingleThreadedStageWithDeque<DenoisedLatents> {
 public:
  void Push(DenoisedLatents latents) { PushOutput(std::move(latents)); }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

TensorBuffer MakeHostFloatBuffer(Environment& env,
                                 std::vector<int32_t> dimensions) {
  size_t num_elems = 1;
  for (int32_t dim : dimensions) {
    num_elems *= static_cast<size_t>(dim);
  }
  RankedTensorType type(
      ElementType::Float32,
      Layout(Dimensions(dimensions.begin(), dimensions.end())));
  auto buffer =
      TensorBuffer::CreateManaged(env, TensorBufferType::kHostMemory,
                                  std::move(type), num_elems * sizeof(float));
  EXPECT_TRUE(buffer.HasValue());
  return std::move(*buffer);
}

TEST(VaeDecoderStageTest, DecodesLatentsToRgb888ImageOutput) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());

  FakeDenoiserStage denoiser;
  auto runner = std::make_unique<MockLiteRtRunner>();
  EXPECT_CALL(*runner, CreateInputBuffers(absl::string_view("")))
      .WillOnce([&](absl::string_view) {
        std::vector<TensorBuffer> bufs;
        bufs.push_back(MakeHostFloatBuffer(*env, {1, 2, 1, 2}));
        return bufs;
      });
  EXPECT_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
      .WillOnce([&](absl::string_view) {
        std::vector<TensorBuffer> bufs;
        bufs.push_back(MakeHostFloatBuffer(*env, {1, 3, 1, 2}));
        return bufs;
      });

  std::vector<float> captured_latents;
  EXPECT_CALL(*runner, Run(absl::string_view(""), _, _))
      .WillOnce([&](absl::string_view, absl::Span<const TensorBuffer> inputs,
                    absl::Span<const TensorBuffer> outputs) {
        captured_latents.resize(4);
        EXPECT_TRUE(const_cast<TensorBuffer&>(inputs[0])
                        .Read<float>(absl::MakeSpan(captured_latents))
                        .HasValue());
        // Planar NCHW for height=1, width=2: pixel 0=(-1, 0, 1), pixel 1=(1,
        // -1, 0).
        const std::vector<float> nchw = {-1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f};
        EXPECT_TRUE(const_cast<TensorBuffer&>(outputs[0])
                        .Write<float>(absl::MakeConstSpan(nchw))
                        .HasValue());
        return absl::OkStatus();
      })
      .WillOnce([&](absl::string_view, absl::Span<const TensorBuffer>,
                    absl::Span<const TensorBuffer> outputs) {
        // Second write verifies the output buffer's read lock was cleanly
        // released after the first decode.
        const std::vector<float> nchw = {1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f};
        EXPECT_TRUE(const_cast<TensorBuffer&>(outputs[0])
                        .Write<float>(absl::MakeConstSpan(nchw))
                        .HasValue());
        return absl::OkStatus();
      });

  VaeDecoderStage::Config config{
      .image_width = 2,
      .image_height = 1,
      .expected_latent_elements = 4,
      .stage_name = "TestVaeStage",
  };
  auto unpack = [](const DenoisedLatents& denoised)
      -> absl::StatusOr<std::vector<float>> {
    std::vector<float> scaled;
    scaled.reserve(denoised.packed_latents.size());
    for (float v : denoised.packed_latents) {
      scaled.push_back(v * 2.0f);
    }
    return scaled;
  };

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VaeDecoderStage> stage,
      VaeDecoderStage::Create(&denoiser, config, std::move(runner), unpack));

  // Safe no-op when upstream has no output.
  EXPECT_FALSE(stage->NeedSchedule());
  ASSERT_OK(stage->Schedule());
  EXPECT_FALSE(stage->HasOutput());

  denoiser.Push(DenoisedLatents{.packed_latents = {0.5f, 1.0f, -1.5f, 2.0f}});
  EXPECT_TRUE(stage->NeedSchedule());
  ASSERT_OK(stage->Schedule());
  EXPECT_THAT(captured_latents, ElementsAre(1.0f, 2.0f, -3.0f, 4.0f));

  ASSERT_TRUE(stage->HasOutput());
  ASSERT_OK_AND_ASSIGN(Output out, stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<ImageOutput>(out));
  const ImageOutput& img = std::get<ImageOutput>(out);
  EXPECT_EQ(img.width, 2);
  EXPECT_EQ(img.height, 1);
  EXPECT_EQ(img.channels, 3);
  EXPECT_THAT(img.rgb_data, ElementsAre(0, 128, 255, 255, 0, 128));

  // Second consecutive decode reuses `output_buffers_[0]` cleanly.
  denoiser.Push(DenoisedLatents{.packed_latents = {0.0f, 0.0f, 0.0f, 0.0f}});
  ASSERT_OK(stage->Schedule());
  ASSERT_OK_AND_ASSIGN(Output out2, stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<ImageOutput>(out2));
  EXPECT_THAT(std::get<ImageOutput>(out2).rgb_data,
              ElementsAre(255, 0, 128, 0, 255, 128));
  ASSERT_OK(stage->Flush());
}

TEST(VaeDecoderStageTest, CreateValidatesArgumentsAndBuffers) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());

  FakeDenoiserStage denoiser;
  auto valid_unpack =
      [](const DenoisedLatents&) -> absl::StatusOr<std::vector<float>> {
    return std::vector<float>(4);
  };
  const VaeDecoderStage::Config valid_config{
      .image_width = 2,
      .image_height = 2,
      .expected_latent_elements = 4,
      .stage_name = "TestVae",
  };

  // Null denoiser (checked through `ValidateAndCreateBuffers`, whose parameter
  // is `absl_nullable`, avoiding UBSan `invalid-null-argument` on `Create`),
  // null runner, or null callback.
  struct ExposedVaeDecoderStage : public VaeDecoderStage {
    using VaeDecoderStage::ValidateAndCreateBuffers;
  };
  MockLiteRtRunner unused_runner;
  EXPECT_THAT(
      ExposedVaeDecoderStage::ValidateAndCreateBuffers(
          /*denoiser=*/nullptr, valid_config, &unused_runner, valid_unpack),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("denoiser")));
  std::unique_ptr<LiteRtRunner> null_runner;
  EXPECT_THAT(
      VaeDecoderStage::Create(&denoiser, valid_config, std::move(null_runner),
                              valid_unpack),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("runner")));
  VaeDecoderStage::UnpackLatentsFn null_unpack;
  EXPECT_THAT(VaeDecoderStage::Create(&denoiser, valid_config,
                                      std::make_unique<MockLiteRtRunner>(),
                                      std::move(null_unpack)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unpack_latents")));

  // Invalid config dimensions.
  VaeDecoderStage::Config bad_dim = valid_config;
  bad_dim.image_width = 0;
  EXPECT_THAT(VaeDecoderStage::Create(&denoiser, bad_dim,
                                      std::make_unique<MockLiteRtRunner>(),
                                      valid_unpack),
              StatusIs(absl::StatusCode::kInvalidArgument));

  VaeDecoderStage::Config zero_latents = valid_config;
  zero_latents.expected_latent_elements = 0;
  EXPECT_THAT(VaeDecoderStage::Create(&denoiser, zero_latents,
                                      std::make_unique<MockLiteRtRunner>(),
                                      valid_unpack),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Byte-size overflow in image dimensions: `3 * height * width` fits in
  // 64-bit `size_t` at `INT_MAX x INT_MAX`, while `* sizeof(float)` wraps
  // (and already wraps element count on 32-bit `size_t`).
  VaeDecoderStage::Config overflow_dim = valid_config;
  overflow_dim.image_width = std::numeric_limits<int>::max();
  overflow_dim.image_height = std::numeric_limits<int>::max();
  EXPECT_THAT(
      VaeDecoderStage::Create(&denoiser, overflow_dim,
                              std::make_unique<MockLiteRtRunner>(),
                              valid_unpack),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("overflow")));

  // Byte-size overflow in `expected_latent_elements * sizeof(float)`.
  VaeDecoderStage::Config overflow_latents = valid_config;
  overflow_latents.expected_latent_elements =
      std::numeric_limits<size_t>::max() / sizeof(float) + 1;
  EXPECT_THAT(
      VaeDecoderStage::Create(&denoiser, overflow_latents,
                              std::make_unique<MockLiteRtRunner>(),
                              valid_unpack),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("overflow")));

  // Wrong buffer count and capacity.
  auto bad_count_runner = std::make_unique<MockLiteRtRunner>();
  EXPECT_CALL(*bad_count_runner, CreateInputBuffers(absl::string_view("")))
      .WillOnce([](absl::string_view) { return std::vector<TensorBuffer>{}; });
  EXPECT_CALL(*bad_count_runner, CreateOutputBuffers(absl::string_view("")))
      .WillOnce([&](absl::string_view) {
        std::vector<TensorBuffer> bufs;
        bufs.push_back(MakeHostFloatBuffer(*env, {1, 3, 2, 2}));
        return bufs;
      });
  EXPECT_THAT(
      VaeDecoderStage::Create(&denoiser, valid_config,
                              std::move(bad_count_runner), valid_unpack),
      StatusIs(absl::StatusCode::kInvalidArgument));

  auto bad_out_cap_runner = std::make_unique<MockLiteRtRunner>();
  EXPECT_CALL(*bad_out_cap_runner, CreateInputBuffers(absl::string_view("")))
      .WillOnce([&](absl::string_view) {
        std::vector<TensorBuffer> bufs;
        bufs.push_back(MakeHostFloatBuffer(*env, {1, 4}));
        return bufs;
      });
  EXPECT_CALL(*bad_out_cap_runner, CreateOutputBuffers(absl::string_view("")))
      .WillOnce([&](absl::string_view) {
        std::vector<TensorBuffer> bufs;
        bufs.push_back(MakeHostFloatBuffer(*env, {1, 3, 1, 1}));
        return bufs;
      });
  EXPECT_THAT(
      VaeDecoderStage::Create(&denoiser, valid_config,
                              std::move(bad_out_cap_runner), valid_unpack),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(VaeDecoderStageTest, RejectsUnpackLatentSizeMismatch) {
  auto env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());

  FakeDenoiserStage denoiser;
  auto runner = std::make_unique<MockLiteRtRunner>();
  EXPECT_CALL(*runner, CreateInputBuffers(absl::string_view("")))
      .WillOnce([&](absl::string_view) {
        std::vector<TensorBuffer> bufs;
        bufs.push_back(MakeHostFloatBuffer(*env, {1, 4}));
        return bufs;
      });
  EXPECT_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
      .WillOnce([&](absl::string_view) {
        std::vector<TensorBuffer> bufs;
        bufs.push_back(MakeHostFloatBuffer(*env, {1, 3, 1, 1}));
        return bufs;
      });

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VaeDecoderStage> stage,
      VaeDecoderStage::Create(
          &denoiser,
          VaeDecoderStage::Config{
              .image_width = 1,
              .image_height = 1,
              .expected_latent_elements = 4,
              .stage_name = "TestVae",
          },
          std::move(runner),
          [](const DenoisedLatents&) -> absl::StatusOr<std::vector<float>> {
            return std::vector<float>(2, 0.0f);  // Wrong length (2 != 4).
          }));

  denoiser.Push(DenoisedLatents{});
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument,
                                          HasSubstr("expected 4")));
}

}  // namespace
}  // namespace litert::omni::text2image
