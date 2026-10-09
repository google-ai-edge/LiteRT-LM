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

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/text2image/flux2/flux2_denoiser_stage.h"
#include "omni/text2image/flux2/flux2_factory.h"
#include "omni/text2image/flux2/flux2_math.h"
#include "omni/text2image/flux2/flux2_model_config.h"
#include "omni/text2image/flux2/flux2_vae_decoder_stage.h"
#include "omni/text2image/flux2/klein_denoiser_stage.h"
#include "omni/text2image/flux2/klein_text_encoder_stage.h"
#include "omni/text2image/prompt_source.h"
#include "omni/text2image/text_encoder_stage.h"
#include "runtime/executor/executor_settings_base.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::text2image {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;
using ::testing::FloatNear;

class FakeTokenizer : public support::Tokenizer {
 public:
  explicit FakeTokenizer(support::TokenIds token_ids)
      : token_ids_(std::move(token_ids)) {}

  support::TokenizerType GetTokenizerType() const override {
    return support::TokenizerType::kHuggingFace;
  }

  absl::StatusOr<support::TokenIds> TextToTokenIds(
      absl::string_view text) override {
    return token_ids_;
  }

  absl::StatusOr<int> TokenToId(absl::string_view token) override {
    return absl::NotFoundError("not implemented");
  }

  absl::StatusOr<std::string> TokenIdsToText(
      absl::Span<const int> token_ids, bool skip_special_tokens) override {
    return "";
  }

  std::vector<std::string> GetTokens() const override { return {}; }

  int GetVocabSize() const override { return 152000; }

 private:
  support::TokenIds token_ids_;
};

TEST(Flux2MathTest, ComputeFlowMatchSigmasMatchesReference) {
  auto sigmas = ComputeFlowMatchSigmas(/*steps=*/4, /*image_tokens=*/1024);
  ASSERT_OK(sigmas);
  ASSERT_EQ(sigmas->size(), 5);
  EXPECT_FLOAT_EQ((*sigmas)[0], 1.0f);
  EXPECT_FLOAT_EQ((*sigmas)[4], 0.0f);
  for (size_t i = 0; i + 1 < sigmas->size(); ++i) {
    EXPECT_GT((*sigmas)[i], (*sigmas)[i + 1]);
  }
  EXPECT_THAT((*sigmas)[1], FloatNear(0.958085f, 1e-4f));

  // Single-step schedule: [1.0, 0.0].
  auto single_step = ComputeFlowMatchSigmas(/*steps=*/1, /*image_tokens=*/256);
  ASSERT_OK(single_step);
  EXPECT_THAT(*single_step, ElementsAre(1.0f, 0.0f));

  // Above 4300 image tokens uses m200 directly.
  auto high_res = ComputeFlowMatchSigmas(/*steps=*/4, /*image_tokens=*/4301);
  ASSERT_OK(high_res);
  ASSERT_EQ(high_res->size(), 5);
  EXPECT_FLOAT_EQ((*high_res)[0], 1.0f);
  EXPECT_FLOAT_EQ((*high_res)[4], 0.0f);

  EXPECT_THAT(ComputeFlowMatchSigmas(/*steps=*/0, /*image_tokens=*/1024),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ComputeFlowMatchSigmas(/*steps=*/-2, /*image_tokens=*/1024),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ComputeFlowMatchSigmas(/*steps=*/4, /*image_tokens=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ComputeFlowMatchSigmas(/*steps=*/4, /*image_tokens=*/-10),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Flux2MathTest, SampleGaussianLatentsIsDeterministicAndSeedDependent) {
  auto empty = SampleGaussianLatents(/*count=*/0, /*seed=*/42);
  EXPECT_TRUE(empty.empty());

  auto sample_a = SampleGaussianLatents(/*count=*/64, /*seed=*/42);
  auto sample_b = SampleGaussianLatents(/*count=*/64, /*seed=*/42);
  auto sample_c = SampleGaussianLatents(/*count=*/64, /*seed=*/43);
  ASSERT_EQ(sample_a.size(), 64);
  EXPECT_EQ(sample_a, sample_b);
  EXPECT_NE(sample_a, sample_c);
}

TEST(Flux2MathTest, BuildPositionIds) {
  auto img_ids = BuildImagePositionIds(/*grid_h=*/2, /*grid_w=*/3);
  ASSERT_EQ(img_ids.size(), 2 * 3 * 4);
  // Row 1, Col 2 -> index (1*3 + 2)*4 = 20
  EXPECT_FLOAT_EQ(img_ids[20], 0.0f);
  EXPECT_FLOAT_EQ(img_ids[21], 1.0f);
  EXPECT_FLOAT_EQ(img_ids[22], 2.0f);
  EXPECT_FLOAT_EQ(img_ids[23], 0.0f);

  auto txt_ids = BuildTextPositionIds(/*seq_len=*/4);
  ASSERT_EQ(txt_ids.size(), 4 * 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(txt_ids[i * 4 + 0], 0.0f);
    EXPECT_FLOAT_EQ(txt_ids[i * 4 + 1], 0.0f);
    EXPECT_FLOAT_EQ(txt_ids[i * 4 + 2], 0.0f);
    EXPECT_FLOAT_EQ(txt_ids[i * 4 + 3], static_cast<float>(i));
  }
}

TEST(Flux2MathTest, UnpatchifyAndConvertNchwToRgb888) {
  // 1x1 patch (grid_h=1, grid_w=1), packed_ch=128 -> unpacked NCHW [1, 32, 2,
  // 2].
  std::vector<float> lat(128, 1.0f);
  std::vector<float> scale(128, 2.0f);
  std::vector<float> shift(128, 0.5f);
  auto unpacked =
      UnpatchifyLatents(lat, /*grid_h=*/1, /*grid_w=*/1, scale, shift);
  ASSERT_OK(unpacked);
  ASSERT_EQ(unpacked->size(), 128);
  for (float v : *unpacked) {
    EXPECT_FLOAT_EQ(v, 2.5f);
  }

  // Verify exact 2x2 spatial unpatchify ordering on a 1x2 patch grid
  // (out_h = 2, out_w = 4).
  std::vector<float> unit_scale(128, 1.0f);
  std::vector<float> zero_shift(128, 0.0f);
  std::vector<float> patterned_lat(2 * 128, 0.0f);
  for (int w = 0; w < 2; ++w) {
    for (int c = 0; c < 32; ++c) {
      for (int ph = 0; ph < 2; ++ph) {
        for (int pw = 0; pw < 2; ++pw) {
          const int p = (c * 2 + ph) * 2 + pw;
          const int y = ph;
          const int x = w * 2 + pw;
          patterned_lat[w * 128 + p] = static_cast<float>(c * 100 + y * 10 + x);
        }
      }
    }
  }
  auto unpacked_grid = UnpatchifyLatents(patterned_lat, /*grid_h=*/1,
                                         /*grid_w=*/2, unit_scale, zero_shift);
  ASSERT_OK(unpacked_grid);
  ASSERT_EQ(unpacked_grid->size(), 32 * 2 * 4);
  for (int c = 0; c < 32; ++c) {
    for (int y = 0; y < 2; ++y) {
      for (int x = 0; x < 4; ++x) {
        EXPECT_FLOAT_EQ((*unpacked_grid)[(c * 2 + y) * 4 + x],
                        static_cast<float>(c * 100 + y * 10 + x));
      }
    }
  }

  // Invalid UnpatchifyLatents arguments.
  EXPECT_THAT(UnpatchifyLatents(lat, /*grid_h=*/0, /*grid_w=*/1, scale, shift),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(UnpatchifyLatents(lat, /*grid_h=*/1, /*grid_w=*/0, scale, shift),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      UnpatchifyLatents(lat, /*grid_h=*/1, /*grid_w=*/1,
                        /*bn_scale=*/std::vector<float>(64, 1.0f), shift),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(UnpatchifyLatents(lat, /*grid_h=*/1, /*grid_w=*/1, scale,
                                /*bn_shift=*/std::vector<float>(64, 0.0f)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(UnpatchifyLatents(lat, /*grid_h=*/2, /*grid_w=*/2, scale, shift),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // NCHW [1, 3, 1, 2] -> 2 pixels: pixel 0 = (-1, 0, 1), pixel 1 = (1, 0, -1),
  // plus out-of-range clamping (-2.5 -> 0, 3.0 -> 255).
  std::vector<float> nchw = {
      -1.0f, 3.0f,  // R
      0.0f,  0.0f,  // G
      1.0f,  -2.5f  // B
  };
  auto rgb = ConvertNchwToRgb888(nchw, /*height=*/1, /*width=*/2);
  ASSERT_OK(rgb);
  EXPECT_THAT(*rgb, ElementsAre(0, 128, 255, 255, 128, 0));

  // Non-finite (NaN / Inf) handling in ConvertNchwToRgb888: NaN maps safely to
  // 0, +Inf clamps to 255, -Inf clamps to 0.
  const float nan_val = std::numeric_limits<float>::quiet_NaN();
  const float inf_val = std::numeric_limits<float>::infinity();
  std::vector<float> nonfinite_nchw = {nan_val, inf_val, -inf_val};
  auto nonfinite_rgb =
      ConvertNchwToRgb888(nonfinite_nchw, /*height=*/1, /*width=*/1);
  ASSERT_OK(nonfinite_rgb);
  EXPECT_THAT(*nonfinite_rgb, ElementsAre(0, 255, 0));

  // Invalid ConvertNchwToRgb888 arguments.
  EXPECT_THAT(ConvertNchwToRgb888(nchw, /*height=*/0, /*width=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ConvertNchwToRgb888(nchw, /*height=*/1, /*width=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ConvertNchwToRgb888(nchw, /*height=*/2, /*width=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Flux2ModelConfigTest, DefaultBnVectorsHave128Elements) {
  EXPECT_EQ(DefaultFlux2LatentBnScale().size(), 128);
  EXPECT_EQ(DefaultFlux2LatentBnShift().size(), 128);

  Flux2ModelConfig default_config;
  EXPECT_EQ(default_config.seq_len, 256);
  EXPECT_EQ(default_config.img_size, 512);
  EXPECT_EQ(default_config.packed_ch, 128);
  EXPECT_EQ(default_config.prompt_dim, 7680);
  EXPECT_EQ(default_config.steps, 4);
  EXPECT_EQ(default_config.latent_bn_scale.size(), 128);
  EXPECT_EQ(default_config.latent_bn_shift.size(), 128);
}

TEST(Flux2ModelConfigTest, PopulateFlux2ConfigFromProtoPopulatesFields) {
  lm::proto::BonsaiFlux2 bonsai_proto;
  auto* params = bonsai_proto.mutable_flux2_params();
  params->set_seq_len(128);
  params->set_img_size(256);
  params->set_packed_ch(128);
  params->set_prompt_dim(7680);
  params->set_default_steps(6);
  params->add_latent_bn_scale(1.25f);
  params->add_latent_bn_scale(2.25f);
  params->add_latent_bn_shift(-0.25f);
  params->add_latent_bn_shift(0.75f);

  Flux2ModelConfig config;
  PopulateFlux2ConfigFromProto(bonsai_proto, config);
  EXPECT_EQ(config.seq_len, 128);
  EXPECT_EQ(config.img_size, 256);
  EXPECT_EQ(config.packed_ch, 128);
  EXPECT_EQ(config.prompt_dim, 7680);
  EXPECT_EQ(config.steps, 6);
  EXPECT_THAT(config.latent_bn_scale, ElementsAre(1.25f, 2.25f));
  EXPECT_THAT(config.latent_bn_shift, ElementsAre(-0.25f, 0.75f));

  lm::proto::Flux2Klein klein_proto;
  *klein_proto.mutable_flux2_params() = *params;
  klein_proto.mutable_flux2_params()->set_default_steps(8);
  Flux2ModelConfig klein_config;
  PopulateFlux2ConfigFromProto(klein_proto, klein_config);
  EXPECT_EQ(klein_config.steps, 8);
  EXPECT_THAT(klein_config.latent_bn_scale, ElementsAre(1.25f, 2.25f));

  // Empty proto leaves existing fields unchanged.
  lm::proto::BonsaiFlux2 empty_proto;
  PopulateFlux2ConfigFromProto(empty_proto, klein_config);
  EXPECT_EQ(klein_config.seq_len, 128);
  EXPECT_EQ(klein_config.img_size, 256);
  EXPECT_EQ(klein_config.steps, 8);
  EXPECT_THAT(klein_config.latent_bn_scale, ElementsAre(1.25f, 2.25f));

  // Out-of-bounds proto values are ignored, preserving valid config values.
  lm::proto::BonsaiFlux2 oversized_proto;
  auto* bad_params = oversized_proto.mutable_flux2_params();
  bad_params->set_seq_len(8192);
  bad_params->set_img_size(8192);
  bad_params->set_packed_ch(2048);
  bad_params->set_prompt_dim(100000);
  bad_params->set_default_steps(2000);
  PopulateFlux2ConfigFromProto(oversized_proto, klein_config);
  EXPECT_EQ(klein_config.seq_len, 128);
  EXPECT_EQ(klein_config.img_size, 256);
  EXPECT_EQ(klein_config.packed_ch, 128);
  EXPECT_EQ(klein_config.prompt_dim, 7680);
  EXPECT_EQ(klein_config.steps, 8);
}

class Flux2StagesTest : public ::testing::Test {
 protected:
  struct BufferSpec {
    ElementType element_type;
    std::vector<int> dims;
    size_t elem_size;
  };

  void SetUp() override {
    auto env = Environment::Create({});
    ASSERT_TRUE(env.HasValue());
    env_ = std::make_unique<Environment>(std::move(*env));
  }

  absl::StatusOr<TensorBuffer> CreateBuffer(ElementType element_type,
                                            const std::vector<int>& dims,
                                            size_t elem_size) {
    RankedTensorType type(element_type,
                          Layout(Dimensions(dims.begin(), dims.end())));
    size_t num_elements = 1;
    for (int d : dims) num_elements *= d;
    LITERT_ASSIGN_OR_RETURN(
        auto buf,
        TensorBuffer::CreateManaged(*env_, TensorBufferType::kHostMemory,
                                    std::move(type), num_elements * elem_size));
    return buf;
  }

  void ExpectBuffers(MockLiteRtRunner& runner,
                     std::vector<BufferSpec> input_specs,
                     std::vector<BufferSpec> output_specs) {
    EXPECT_CALL(runner, CreateInputBuffers(absl::string_view("")))
        .WillOnce(
            [this, input_specs = std::move(input_specs)](absl::string_view)
                -> absl::StatusOr<std::vector<TensorBuffer>> {
              std::vector<TensorBuffer> bufs;
              bufs.reserve(input_specs.size());
              for (const auto& spec : input_specs) {
                ABSL_ASSIGN_OR_RETURN(
                    TensorBuffer buf,
                    CreateBuffer(spec.element_type, spec.dims, spec.elem_size));
                bufs.push_back(std::move(buf));
              }
              return bufs;
            });
    EXPECT_CALL(runner, CreateOutputBuffers(absl::string_view("")))
        .WillOnce(
            [this, output_specs = std::move(output_specs)](absl::string_view)
                -> absl::StatusOr<std::vector<TensorBuffer>> {
              std::vector<TensorBuffer> bufs;
              bufs.reserve(output_specs.size());
              for (const auto& spec : output_specs) {
                ABSL_ASSIGN_OR_RETURN(
                    TensorBuffer buf,
                    CreateBuffer(spec.element_type, spec.dims, spec.elem_size));
                bufs.push_back(std::move(buf));
              }
              return bufs;
            });
  }

  std::unique_ptr<Environment> env_;
};

TEST_F(Flux2StagesTest, EndToEndStagesWithMockRunners) {
  // Use scaled-down dimensions for fast unit testing:
  // seq_len = 16, prompt_dim = 8, img_size = 16 (tokens = 1, grid_h = 1,
  // grid_w = 1), steps = 2.
  Flux2ModelConfig config;
  config.seq_len = 16;
  config.prompt_dim = 8;
  config.img_size = 16;
  config.packed_ch = 128;
  config.steps = 2;

  PushPromptSource prompt_source(ImageGenInputMetadata{
      .width = 16, .height = 16, .num_inference_steps = 2, .seed = 7});

  // 1. Text Encoder Stage
  auto textenc_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *textenc_runner,
      {{ElementType::Int32, {1, config.seq_len}, 4},
       {ElementType::Int32, {1, config.seq_len}, 4}},
      {{ElementType::Float32, {1, config.seq_len, config.prompt_dim}, 4}});
  EXPECT_CALL(*textenc_runner, Run(absl::string_view(""), _, _))
      .WillOnce([&config](absl::string_view,
                          absl::Span<const TensorBuffer> inputs,
                          absl::Span<const TensorBuffer> outputs) {
        EXPECT_EQ(inputs.size(), 2);
        EXPECT_EQ(outputs.size(), 1);
        std::vector<float> fake_embeds(config.seq_len * config.prompt_dim,
                                       0.25f);
        auto& out = const_cast<TensorBuffer&>(outputs[0]);
        auto status = out.Write<float>(fake_embeds);
        EXPECT_TRUE(status.HasValue());
        return absl::OkStatus();
      });

  TextEncoderStage::Config textenc_config;
  textenc_config.seq_len = config.seq_len;
  auto textenc_stage = TextEncoderStage::Create(
      &prompt_source,
      std::make_unique<FakeTokenizer>(support::TokenIds{872, 198, 11, 22}),
      std::move(textenc_runner), textenc_config);
  ASSERT_OK(textenc_stage);

  // 2. Denoiser Stage
  auto dit_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *dit_runner,
      {{ElementType::Float32, {1, 1, 128}, 4},
       {ElementType::Float32, {1, config.seq_len, config.prompt_dim}, 4},
       {ElementType::Float32, {1}, 4},
       {ElementType::Float32, {1, 4}, 4},
       {ElementType::Float32, {config.seq_len, 4}, 4}},
      {{ElementType::Float32, {1, 1, 128}, 4}});
  int dit_calls = 0;
  EXPECT_CALL(*dit_runner, Run(absl::string_view(""), _, _))
      .WillRepeatedly([&dit_calls](absl::string_view,
                                   absl::Span<const TensorBuffer> inputs,
                                   absl::Span<const TensorBuffer> outputs) {
        ++dit_calls;
        std::vector<float> velocity(128, 0.1f);
        auto& out = const_cast<TensorBuffer&>(outputs[0]);
        auto status = out.Write<float>(velocity);
        EXPECT_TRUE(status.HasValue());
        return absl::OkStatus();
      });

  auto denoiser_stage = Flux2DenoiserStage::Create(textenc_stage->get(), config,
                                                   std::move(dit_runner));
  ASSERT_OK(denoiser_stage);

  // 3. VAE Decoder Stage
  auto vae_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(*vae_runner, {{ElementType::Float32, {1, 32, 2, 2}, 4}},
                {{ElementType::Float32, {1, 3, 16, 16}, 4}});
  EXPECT_CALL(*vae_runner, Run(absl::string_view(""), _, _))
      .WillOnce([](absl::string_view, absl::Span<const TensorBuffer> inputs,
                   absl::Span<const TensorBuffer> outputs) {
        std::vector<float> decoded(3 * 16 * 16, 0.0f);
        auto& out = const_cast<TensorBuffer&>(outputs[0]);
        auto status = out.Write<float>(decoded);
        EXPECT_TRUE(status.HasValue());
        return absl::OkStatus();
      });

  auto vae_stage = Flux2VaeDecoderStage::Create(denoiser_stage->get(), config,
                                                std::move(vae_runner));
  ASSERT_OK(vae_stage);

  // Before any prompt is pushed, downstream stages do not need scheduling and
  // Schedule() is a safe no-op.
  EXPECT_FALSE((*denoiser_stage)->NeedSchedule());
  ASSERT_OK((*denoiser_stage)->Schedule());
  EXPECT_FALSE((*denoiser_stage)->HasOutput());

  EXPECT_FALSE((*vae_stage)->NeedSchedule());
  ASSERT_OK((*vae_stage)->Schedule());
  EXPECT_FALSE((*vae_stage)->HasOutput());

  // Push prompt and drive the 3 stages sequentially.
  ASSERT_OK(prompt_source.PushPrompt("a sunset over mountains"));
  ASSERT_TRUE(prompt_source.HasOutput());

  ASSERT_TRUE((*textenc_stage)->NeedSchedule());
  ASSERT_OK((*textenc_stage)->Schedule());

  ASSERT_TRUE((*denoiser_stage)->NeedSchedule());
  ASSERT_OK((*denoiser_stage)->Schedule());
  EXPECT_EQ(dit_calls, 2);

  ASSERT_TRUE((*vae_stage)->NeedSchedule());
  ASSERT_OK((*vae_stage)->Schedule());

  ASSERT_TRUE((*vae_stage)->HasOutput());
  auto out = (*vae_stage)->GetOutput();
  ASSERT_OK(out);
  ASSERT_TRUE(std::holds_alternative<ImageOutput>(*out));
  const auto& img = std::get<ImageOutput>(*out);
  EXPECT_EQ(img.width, 16);
  EXPECT_EQ(img.height, 16);
  EXPECT_EQ(img.channels, 3);
  EXPECT_EQ(img.rgb_data.size(), 16 * 16 * 3);
  EXPECT_EQ(img.rgb_data[0], 128);
}

TEST_F(Flux2StagesTest, DenoiserVerifiesCustomInputIndicesAndEulerUpdate) {
  Flux2ModelConfig config;
  config.seq_len = 16;
  config.prompt_dim = 8;
  config.img_size = 16;
  config.packed_ch = 128;
  config.steps = 2;
  config.default_seed = 99;

  // Test fallback to config_.steps and config_.default_seed when metadata uses
  // unset/zero values.
  PushPromptSource prompt_source(ImageGenInputMetadata{
      .width = 0, .height = 0, .num_inference_steps = 0, .seed = 0});
  auto textenc_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *textenc_runner,
      {{ElementType::Int32, {1, config.seq_len}, 4},
       {ElementType::Int32, {1, config.seq_len}, 4}},
      {{ElementType::Float32, {1, config.seq_len, config.prompt_dim}, 4}});
  EXPECT_CALL(*textenc_runner, Run(absl::string_view(""), _, _))
      .WillOnce([&config](absl::string_view, absl::Span<const TensorBuffer>,
                          absl::Span<const TensorBuffer> outputs) {
        std::vector<float> embeds(config.seq_len * config.prompt_dim, 0.5f);
        auto& out = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(out.Write<float>(embeds).HasValue());
        return absl::OkStatus();
      });

  TextEncoderStage::Config textenc_config;
  textenc_config.seq_len = config.seq_len;
  auto textenc_stage = TextEncoderStage::Create(
      &prompt_source, std::make_unique<FakeTokenizer>(support::TokenIds{1}),
      std::move(textenc_runner), textenc_config);
  ASSERT_OK(textenc_stage);

  // Permuted buffer order: [txt_ids, img_ids, t, enc, hidden]
  Flux2DenoiserStage::InputIndices custom_indices{
      .hidden = 4, .enc = 3, .t = 2, .img_ids = 1, .txt_ids = 0};
  auto dit_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *dit_runner,
      {{ElementType::Float32, {config.seq_len, 4}, 4},
       {ElementType::Float32, {1, 4}, 4},
       {ElementType::Float32, {1}, 4},
       {ElementType::Float32, {1, config.seq_len, config.prompt_dim}, 4},
       {ElementType::Float32, {1, 1, 128}, 4}},
      {{ElementType::Float32, {1, 1, 128}, 4}});

  std::vector<float> observed_sigmas;
  constexpr float kConstantVelocity = 2.0f;
  EXPECT_CALL(*dit_runner, Run(absl::string_view(""), _, _))
      .WillRepeatedly([&](absl::string_view,
                          absl::Span<const TensorBuffer> inputs,
                          absl::Span<const TensorBuffer> outputs) {
        float sigma = -1.0f;
        auto& t_buf = const_cast<TensorBuffer&>(inputs[custom_indices.t]);
        EXPECT_TRUE(t_buf.Read<float>(absl::MakeSpan(&sigma, 1)).HasValue());
        observed_sigmas.push_back(sigma);

        std::vector<float> enc_read(config.seq_len * config.prompt_dim, 0.0f);
        auto& enc_buf = const_cast<TensorBuffer&>(inputs[custom_indices.enc]);
        EXPECT_TRUE(enc_buf.Read<float>(absl::MakeSpan(enc_read)).HasValue());
        EXPECT_FLOAT_EQ(enc_read[0], 0.5f);

        std::vector<float> velocity(128, kConstantVelocity);
        auto& out = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(out.Write<float>(velocity).HasValue());
        return absl::OkStatus();
      });

  auto denoiser_stage = Flux2DenoiserStage::Create(
      textenc_stage->get(), config, std::move(dit_runner), custom_indices);
  ASSERT_OK(denoiser_stage);

  ASSERT_OK(prompt_source.PushPrompt("euler check"));
  ASSERT_OK((*textenc_stage)->Schedule());
  ASSERT_OK((*denoiser_stage)->Schedule());

  auto expected_sigmas =
      ComputeFlowMatchSigmas(/*steps=*/2, /*image_tokens=*/1);
  ASSERT_OK(expected_sigmas);
  ASSERT_EQ(observed_sigmas.size(), 2);
  EXPECT_FLOAT_EQ(observed_sigmas[0], (*expected_sigmas)[0]);
  EXPECT_FLOAT_EQ(observed_sigmas[1], (*expected_sigmas)[1]);

  // With constant velocity v across steps from sigma_0=1.0 to sigma_N=0.0,
  // sum(dt * v) = (0.0 - 1.0) * v = -v.
  std::vector<float> init_lat =
      SampleGaussianLatents(/*count=*/128, config.default_seed);
  ASSERT_TRUE((*denoiser_stage)->HasOutput());
  auto out = (*denoiser_stage)->GetOutput();
  ASSERT_OK(out);
  EXPECT_EQ(out->metadata.width, 16);
  EXPECT_EQ(out->metadata.height, 16);
  EXPECT_EQ(out->metadata.num_inference_steps, 2);
  EXPECT_EQ(out->metadata.seed, 99);
  ASSERT_EQ(out->packed_latents.size(), 128);
  for (size_t i = 0; i < 128; ++i) {
    EXPECT_THAT(out->packed_latents[i],
                FloatNear(init_lat[i] - kConstantVelocity, 1e-5f));
  }
}

TEST_F(Flux2StagesTest, DenoiserRespectsPerRequestMetadataOverrides) {
  Flux2ModelConfig config;
  config.seq_len = 16;
  config.prompt_dim = 8;
  config.img_size = 16;
  config.packed_ch = 128;
  config.steps = 1;
  config.default_seed = 10;

  // Request metadata overrides steps=3 and seed=777 (differing from config
  // defaults steps=1 and default_seed=10).
  PushPromptSource prompt_source(ImageGenInputMetadata{
      .width = 16, .height = 16, .num_inference_steps = 3, .seed = 777});
  auto textenc_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *textenc_runner,
      {{ElementType::Int32, {1, config.seq_len}, 4},
       {ElementType::Int32, {1, config.seq_len}, 4}},
      {{ElementType::Float32, {1, config.seq_len, config.prompt_dim}, 4}});
  EXPECT_CALL(*textenc_runner, Run(absl::string_view(""), _, _))
      .WillOnce([&config](absl::string_view, absl::Span<const TensorBuffer>,
                          absl::Span<const TensorBuffer> outputs) {
        std::vector<float> embeds(config.seq_len * config.prompt_dim, 0.25f);
        auto& out = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(out.Write<float>(embeds).HasValue());
        return absl::OkStatus();
      });

  TextEncoderStage::Config textenc_config;
  textenc_config.seq_len = config.seq_len;
  auto textenc_stage = TextEncoderStage::Create(
      &prompt_source, std::make_unique<FakeTokenizer>(support::TokenIds{1}),
      std::move(textenc_runner), textenc_config);
  ASSERT_OK(textenc_stage);

  auto dit_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *dit_runner,
      {{ElementType::Float32, {1, 1, 128}, 4},
       {ElementType::Float32, {1, config.seq_len, config.prompt_dim}, 4},
       {ElementType::Float32, {1}, 4},
       {ElementType::Float32, {1, 4}, 4},
       {ElementType::Float32, {config.seq_len, 4}, 4}},
      {{ElementType::Float32, {1, 1, 128}, 4}});
  int dit_calls = 0;
  EXPECT_CALL(*dit_runner, Run(absl::string_view(""), _, _))
      .WillRepeatedly([&](absl::string_view, absl::Span<const TensorBuffer>,
                          absl::Span<const TensorBuffer> outputs) {
        ++dit_calls;
        std::vector<float> velocity(128, 0.0f);
        auto& out = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(out.Write<float>(velocity).HasValue());
        return absl::OkStatus();
      });

  auto denoiser_stage = Flux2DenoiserStage::Create(textenc_stage->get(), config,
                                                   std::move(dit_runner));
  ASSERT_OK(denoiser_stage);

  ASSERT_OK(prompt_source.PushPrompt("override check"));
  ASSERT_OK((*textenc_stage)->Schedule());
  ASSERT_OK((*denoiser_stage)->Schedule());
  EXPECT_EQ(dit_calls, 3);

  auto out = (*denoiser_stage)->GetOutput();
  ASSERT_OK(out);
  EXPECT_EQ(out->metadata.num_inference_steps, 3);
  EXPECT_EQ(out->metadata.seed, 777);
  std::vector<float> expected_lat =
      SampleGaussianLatents(/*count=*/128, /*seed=*/777);
  EXPECT_EQ(out->packed_latents, expected_lat);
}

TEST_F(Flux2StagesTest, PropagatesRunnerFailureAndRecoversOnNextRun) {
  Flux2ModelConfig config;
  config.seq_len = 16;
  config.prompt_dim = 8;
  config.img_size = 16;
  config.packed_ch = 128;
  config.steps = 1;

  PushPromptSource prompt_source(ImageGenInputMetadata{
      .width = 16, .height = 16, .num_inference_steps = 1, .seed = 42});
  auto textenc_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *textenc_runner,
      {{ElementType::Int32, {1, config.seq_len}, 4},
       {ElementType::Int32, {1, config.seq_len}, 4}},
      {{ElementType::Float32, {1, config.seq_len, config.prompt_dim}, 4}});
  EXPECT_CALL(*textenc_runner, Run(absl::string_view(""), _, _))
      .WillRepeatedly([&config](absl::string_view,
                                absl::Span<const TensorBuffer>,
                                absl::Span<const TensorBuffer> outputs) {
        std::vector<float> embeds(config.seq_len * config.prompt_dim, 0.25f);
        auto& out = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(out.Write<float>(embeds).HasValue());
        return absl::OkStatus();
      });

  TextEncoderStage::Config textenc_config;
  textenc_config.seq_len = config.seq_len;
  auto textenc_stage = TextEncoderStage::Create(
      &prompt_source, std::make_unique<FakeTokenizer>(support::TokenIds{1}),
      std::move(textenc_runner), textenc_config);
  ASSERT_OK(textenc_stage);

  auto dit_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *dit_runner,
      {{ElementType::Float32, {1, 1, 128}, 4},
       {ElementType::Float32, {1, config.seq_len, config.prompt_dim}, 4},
       {ElementType::Float32, {1}, 4},
       {ElementType::Float32, {1, 4}, 4},
       {ElementType::Float32, {config.seq_len, 4}, 4}},
      {{ElementType::Float32, {1, 1, 128}, 4}});
  EXPECT_CALL(*dit_runner, Run(absl::string_view(""), _, _))
      .WillOnce([](absl::string_view, absl::Span<const TensorBuffer>,
                   absl::Span<const TensorBuffer>) {
        return absl::InternalError("simulated DiT failure");
      })
      .WillRepeatedly([](absl::string_view, absl::Span<const TensorBuffer>,
                         absl::Span<const TensorBuffer> outputs) {
        std::vector<float> velocity(128, 0.0f);
        auto& out = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(out.Write<float>(velocity).HasValue());
        return absl::OkStatus();
      });

  auto denoiser_stage = Flux2DenoiserStage::Create(textenc_stage->get(), config,
                                                   std::move(dit_runner));
  ASSERT_OK(denoiser_stage);

  auto vae_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(*vae_runner, {{ElementType::Float32, {1, 32, 2, 2}, 4}},
                {{ElementType::Float32, {1, 3, 16, 16}, 4}});
  EXPECT_CALL(*vae_runner, Run(absl::string_view(""), _, _))
      .WillOnce([](absl::string_view, absl::Span<const TensorBuffer>,
                   absl::Span<const TensorBuffer>) {
        return absl::InternalError("simulated VAE failure");
      })
      .WillOnce([](absl::string_view, absl::Span<const TensorBuffer>,
                   absl::Span<const TensorBuffer> outputs) {
        std::vector<float> decoded(3 * 16 * 16, 0.0f);
        auto& out = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(out.Write<float>(decoded).HasValue());
        return absl::OkStatus();
      });

  auto vae_stage = Flux2VaeDecoderStage::Create(denoiser_stage->get(), config,
                                                std::move(vae_runner));
  ASSERT_OK(vae_stage);

  // 1. First prompt: DiT fails, denoiser resets to idle.
  ASSERT_OK(prompt_source.PushPrompt("first"));
  ASSERT_OK((*textenc_stage)->Schedule());
  EXPECT_THAT((*denoiser_stage)->Schedule(),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_TRUE((*denoiser_stage)->IsIdle());
  EXPECT_FALSE((*denoiser_stage)->HasOutput());

  // 2. Second prompt: DiT succeeds, VAE fails and resets to idle.
  ASSERT_OK(prompt_source.PushPrompt("second"));
  ASSERT_OK((*textenc_stage)->Schedule());
  ASSERT_OK((*denoiser_stage)->Schedule());
  EXPECT_THAT((*vae_stage)->Schedule(), StatusIs(absl::StatusCode::kInternal));
  EXPECT_TRUE((*vae_stage)->IsIdle());
  EXPECT_FALSE((*vae_stage)->HasOutput());

  // 3. Third prompt: both DiT and VAE succeed.
  ASSERT_OK(prompt_source.PushPrompt("third"));
  ASSERT_OK((*textenc_stage)->Schedule());
  ASSERT_OK((*denoiser_stage)->Schedule());
  ASSERT_OK((*vae_stage)->Schedule());
  EXPECT_TRUE((*vae_stage)->HasOutput());
}

TEST_F(Flux2StagesTest, RejectsInvalidDenoiserAndVaeConfigs) {
  Flux2ModelConfig config;
  config.seq_len = 16;
  config.prompt_dim = 8;
  config.img_size = 16;
  config.packed_ch = 128;
  config.steps = 2;

  PushPromptSource prompt_source;
  auto textenc_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *textenc_runner,
      {{ElementType::Int32, {1, config.seq_len}, 4},
       {ElementType::Int32, {1, config.seq_len}, 4}},
      {{ElementType::Float32, {1, config.seq_len, config.prompt_dim}, 4}});
  TextEncoderStage::Config textenc_config;
  textenc_config.seq_len = config.seq_len;
  auto textenc_stage = TextEncoderStage::Create(
      &prompt_source, std::make_unique<FakeTokenizer>(support::TokenIds{1}),
      std::move(textenc_runner), textenc_config);
  ASSERT_OK(textenc_stage);

  // Valid denoiser stage for VAE decoder tests below.
  auto valid_dit_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *valid_dit_runner,
      {{ElementType::Float32, {1, 1, 128}, 4},
       {ElementType::Float32, {1, config.seq_len, config.prompt_dim}, 4},
       {ElementType::Float32, {1}, 4},
       {ElementType::Float32, {1, 4}, 4},
       {ElementType::Float32, {config.seq_len, 4}, 4}},
      {{ElementType::Float32, {1, 1, 128}, 4}});
  auto denoiser_stage = Flux2DenoiserStage::Create(textenc_stage->get(), config,
                                                   std::move(valid_dit_runner));
  ASSERT_OK(denoiser_stage);

  // Null runner on Denoiser and VAE Decoder.
  {
    std::unique_ptr<LiteRtRunner> null_runner;
    EXPECT_THAT(Flux2DenoiserStage::Create(textenc_stage->get(), config,
                                           std::move(null_runner)),
                StatusIs(absl::StatusCode::kInvalidArgument));
    std::unique_ptr<LiteRtRunner> null_vae_runner;
    EXPECT_THAT(Flux2VaeDecoderStage::Create(denoiser_stage->get(), config,
                                             std::move(null_vae_runner)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Invalid img_size (not multiple of 16 or above max) on Denoiser and VAE
  // Decoder.
  {
    Flux2ModelConfig bad_config = config;
    bad_config.img_size = 15;
    EXPECT_THAT(
        Flux2DenoiserStage::Create(textenc_stage->get(), bad_config,
                                   std::make_unique<MockLiteRtRunner>()),
        StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_THAT(
        Flux2VaeDecoderStage::Create(denoiser_stage->get(), bad_config,
                                     std::make_unique<MockLiteRtRunner>()),
        StatusIs(absl::StatusCode::kInvalidArgument));

    bad_config.img_size = 8192;
    EXPECT_THAT(
        Flux2DenoiserStage::Create(textenc_stage->get(), bad_config,
                                   std::make_unique<MockLiteRtRunner>()),
        StatusIs(absl::StatusCode::kInvalidArgument));
    EXPECT_THAT(
        Flux2VaeDecoderStage::Create(denoiser_stage->get(), bad_config,
                                     std::make_unique<MockLiteRtRunner>()),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Non-positive or out-of-bounds steps on Denoiser.
  {
    Flux2ModelConfig bad_config = config;
    bad_config.steps = 0;
    EXPECT_THAT(
        Flux2DenoiserStage::Create(textenc_stage->get(), bad_config,
                                   std::make_unique<MockLiteRtRunner>()),
        StatusIs(absl::StatusCode::kInvalidArgument));
    bad_config.steps = 2000;
    EXPECT_THAT(
        Flux2DenoiserStage::Create(textenc_stage->get(), bad_config,
                                   std::make_unique<MockLiteRtRunner>()),
        StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Fewer than 5 input buffers on DiT runner.
  {
    auto dit_runner = std::make_unique<MockLiteRtRunner>();
    ExpectBuffers(*dit_runner, {{ElementType::Float32, {1, 1, 128}, 4}},
                  {{ElementType::Float32, {1, 1, 128}, 4}});
    EXPECT_THAT(Flux2DenoiserStage::Create(textenc_stage->get(), config,
                                           std::move(dit_runner)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Out-of-bounds input index on DiT runner.
  {
    auto dit_runner = std::make_unique<MockLiteRtRunner>();
    ExpectBuffers(
        *dit_runner,
        {{ElementType::Float32, {1, 1, 128}, 4},
         {ElementType::Float32, {1, config.seq_len, config.prompt_dim}, 4},
         {ElementType::Float32, {1}, 4},
         {ElementType::Float32, {1, 4}, 4},
         {ElementType::Float32, {config.seq_len, 4}, 4}},
        {{ElementType::Float32, {1, 1, 128}, 4}});
    Flux2DenoiserStage::InputIndices oob_indices;
    oob_indices.txt_ids = 5;
    EXPECT_THAT(Flux2DenoiserStage::Create(textenc_stage->get(), config,
                                           std::move(dit_runner), oob_indices),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Duplicate input indices on DiT runner.
  {
    auto dit_runner = std::make_unique<MockLiteRtRunner>();
    ExpectBuffers(
        *dit_runner,
        {{ElementType::Float32, {1, 1, 128}, 4},
         {ElementType::Float32, {1, config.seq_len, config.prompt_dim}, 4},
         {ElementType::Float32, {1}, 4},
         {ElementType::Float32, {1, 4}, 4},
         {ElementType::Float32, {config.seq_len, 4}, 4}},
        {{ElementType::Float32, {1, 1, 128}, 4}});
    Flux2DenoiserStage::InputIndices bad_indices;
    bad_indices.hidden = 0;
    bad_indices.enc = 0;
    EXPECT_THAT(Flux2DenoiserStage::Create(textenc_stage->get(), config,
                                           std::move(dit_runner), bad_indices),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Wrong output buffer count on DiT runner.
  {
    auto dit_runner = std::make_unique<MockLiteRtRunner>();
    ExpectBuffers(
        *dit_runner,
        {{ElementType::Float32, {1, 1, 128}, 4},
         {ElementType::Float32, {1, config.seq_len, config.prompt_dim}, 4},
         {ElementType::Float32, {1}, 4},
         {ElementType::Float32, {1, 4}, 4},
         {ElementType::Float32, {config.seq_len, 4}, 4}},
        {});
    EXPECT_THAT(Flux2DenoiserStage::Create(textenc_stage->get(), config,
                                           std::move(dit_runner)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Mismatched buffer capacity or element type on DiT runner.
  {
    auto dit_runner_bad_size = std::make_unique<MockLiteRtRunner>();
    ExpectBuffers(
        *dit_runner_bad_size,
        {{ElementType::Float32,
          {1, 1, 64},
          4},  // wrong hidden size (64 != 128)
         {ElementType::Float32, {1, config.seq_len, config.prompt_dim}, 4},
         {ElementType::Float32, {1}, 4},
         {ElementType::Float32, {1, 4}, 4},
         {ElementType::Float32, {config.seq_len, 4}, 4}},
        {{ElementType::Float32, {1, 1, 128}, 4}});
    EXPECT_THAT(Flux2DenoiserStage::Create(textenc_stage->get(), config,
                                           std::move(dit_runner_bad_size)),
                StatusIs(absl::StatusCode::kInvalidArgument));

    auto dit_runner_bad_type = std::make_unique<MockLiteRtRunner>();
    ExpectBuffers(
        *dit_runner_bad_type,
        {{ElementType::Int32, {1, 1, 128}, 4},  // wrong element type
         {ElementType::Float32, {1, config.seq_len, config.prompt_dim}, 4},
         {ElementType::Float32, {1}, 4},
         {ElementType::Float32, {1, 4}, 4},
         {ElementType::Float32, {config.seq_len, 4}, 4}},
        {{ElementType::Float32, {1, 1, 128}, 4}});
    EXPECT_THAT(Flux2DenoiserStage::Create(textenc_stage->get(), config,
                                           std::move(dit_runner_bad_type)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // Wrong input/output buffer counts and capacities on VAE Decoder runner.
  {
    auto vae_runner_bad_in = std::make_unique<MockLiteRtRunner>();
    ExpectBuffers(*vae_runner_bad_in, {},
                  {{ElementType::Float32, {1, 3, 16, 16}, 4}});
    EXPECT_THAT(Flux2VaeDecoderStage::Create(denoiser_stage->get(), config,
                                             std::move(vae_runner_bad_in)),
                StatusIs(absl::StatusCode::kInvalidArgument));

    auto vae_runner_bad_out = std::make_unique<MockLiteRtRunner>();
    ExpectBuffers(*vae_runner_bad_out,
                  {{ElementType::Float32, {1, 32, 2, 2}, 4}}, {});
    EXPECT_THAT(Flux2VaeDecoderStage::Create(denoiser_stage->get(), config,
                                             std::move(vae_runner_bad_out)),
                StatusIs(absl::StatusCode::kInvalidArgument));

    auto vae_runner_bad_cap = std::make_unique<MockLiteRtRunner>();
    ExpectBuffers(*vae_runner_bad_cap,
                  {{ElementType::Float32, {1, 32, 2, 2}, 4}},
                  {{ElementType::Float32, {1, 3, 8, 8}, 4}});  // 8x8 != 16x16
    EXPECT_THAT(Flux2VaeDecoderStage::Create(denoiser_stage->get(), config,
                                             std::move(vae_runner_bad_cap)),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST_F(Flux2StagesTest, FactoryRejectsMissingLmModelResources) {
  auto shared_env = std::make_shared<Environment>(std::move(*env_));
  auto resources = std::make_shared<ModelResources>(shared_env);
  Flux2ModelConfig config;

  EXPECT_THAT(
      InitFlux2Resources(config, "/nonexistent/folder", "", lm::Backend::CPU,
                         /*num_threads=*/2, *shared_env, *resources),
      StatusIs(absl::StatusCode::kNotFound));

  std::vector<std::unique_ptr<internal::StageBase>> stages;
  Stage<Output>* output_stage = nullptr;
  EXPECT_THAT(CreateFlux2Components(config, "/nonexistent/folder",
                                    std::make_unique<PushPromptSource>(),
                                    resources, stages, &output_stage),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Flux2MathTest, Fp16ToFp32AndLookupFp16TokenEmbeddings) {
  EXPECT_FLOAT_EQ(Fp16ToFp32(0x0000), 0.0f);
  EXPECT_FLOAT_EQ(Fp16ToFp32(0x8000), -0.0f);
  EXPECT_TRUE(std::signbit(Fp16ToFp32(0x8000)));
  EXPECT_FLOAT_EQ(Fp16ToFp32(0x3c00), 1.0f);
  EXPECT_FLOAT_EQ(Fp16ToFp32(0xc100), -2.5f);
  // Subnormal: 2^-24 = 0x0001
  EXPECT_FLOAT_EQ(Fp16ToFp32(0x0001), std::ldexp(1.0f, -24));
  // Infinity and NaN
  EXPECT_TRUE(std::isinf(Fp16ToFp32(0x7c00)));
  EXPECT_GT(Fp16ToFp32(0x7c00), 0.0f);
  EXPECT_TRUE(std::isinf(Fp16ToFp32(0xfc00)));
  EXPECT_LT(Fp16ToFp32(0xfc00), 0.0f);
  EXPECT_TRUE(std::isnan(Fp16ToFp32(0x7e00)));

  // Build a 3-row x 2-col FP16 embedding table:
  // row 0: [1.0, -2.5], row 1: [0.5, 2.0], row 2: [-1.0, 0.0]
  const std::vector<uint16_t> table_u16 = {
      0x3c00, 0xc100,  // token 0
      0x3800, 0x4000,  // token 1
      0xbc00, 0x0000,  // token 2
  };
  absl::string_view table_bytes(reinterpret_cast<const char*>(table_u16.data()),
                                table_u16.size() * sizeof(uint16_t));
  const std::vector<int32_t> token_ids = {2, 0, 1};
  auto looked_up =
      LookupFp16TokenEmbeddings(token_ids, table_bytes, /*hidden_dim=*/2);
  ASSERT_OK(looked_up);
  EXPECT_THAT(*looked_up, ElementsAre(-1.0f, 0.0f, 1.0f, -2.5f, 0.5f, 2.0f));

  // Invalid arguments / out of bounds
  EXPECT_THAT(LookupFp16TokenEmbeddings(token_ids, table_bytes,
                                        /*hidden_dim=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(LookupFp16TokenEmbeddings(token_ids, table_bytes.substr(0, 3),
                                        /*hidden_dim=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(LookupFp16TokenEmbeddings({-1}, table_bytes, /*hidden_dim=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(LookupFp16TokenEmbeddings({3}, table_bytes, /*hidden_dim=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(Flux2MathTest, BuildCausalPaddingAttentionMaskAndRotaryEmbeds) {
  // 2 heads, seq_len=3, active_len=2 (token 2 is padding).
  constexpr float kNegInf = -1e9f;
  auto mask_4d = BuildCausalPaddingAttentionMask(
      /*seq_len=*/3, /*active_len=*/2, /*num_heads=*/2, kNegInf);
  ASSERT_OK(mask_4d);
  ASSERT_EQ(mask_4d->size(), 2 * 3 * 3);
  for (int h = 0; h < 2; ++h) {
    const size_t base = static_cast<size_t>(h) * 9;
    // Query 0 can attend only to Key 0: [0, -inf, -2*inf]
    EXPECT_FLOAT_EQ((*mask_4d)[base + 0], 0.0f);
    EXPECT_FLOAT_EQ((*mask_4d)[base + 1], kNegInf);
    EXPECT_FLOAT_EQ((*mask_4d)[base + 2], 2.0f * kNegInf);
    // Query 1 can attend to Keys 0 and 1, Key 2 is future/pad: [0, 0, -2*inf]
    EXPECT_FLOAT_EQ((*mask_4d)[base + 3], 0.0f);
    EXPECT_FLOAT_EQ((*mask_4d)[base + 4], 0.0f);
    EXPECT_FLOAT_EQ((*mask_4d)[base + 5], 2.0f * kNegInf);
    // Query 2 can attend to Keys 0 and 1, Key 2 is pad: [0, 0, -inf]
    EXPECT_FLOAT_EQ((*mask_4d)[base + 6], 0.0f);
    EXPECT_FLOAT_EQ((*mask_4d)[base + 7], 0.0f);
    EXPECT_FLOAT_EQ((*mask_4d)[base + 8], kNegInf);
  }
  EXPECT_THAT(BuildCausalPaddingAttentionMask(/*seq_len=*/0, /*active_len=*/1,
                                              /*num_heads=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(BuildCausalPaddingAttentionMask(/*seq_len=*/3, /*active_len=*/2,
                                              /*num_heads=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Qwen3 RoPE
  auto qwen_rope = BuildQwen3RotaryPosEmbed(/*seq_len=*/2, /*head_dim=*/4,
                                            /*theta=*/10000.0f);
  ASSERT_OK(qwen_rope);
  ASSERT_EQ(qwen_rope->first.size(), 8);
  ASSERT_EQ(qwen_rope->second.size(), 8);
  // Position 0: cos=1, sin=0
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(qwen_rope->first[i], 1.0f);
    EXPECT_FLOAT_EQ(qwen_rope->second[i], 0.0f);
  }
  // Position 1: freq_0 = 1.0, freq_1 = 10000^(-2/4) = 0.01, duplicated across
  // half_dim=2
  EXPECT_THAT(qwen_rope->first[4], FloatNear(std::cos(1.0f), 1e-6f));
  EXPECT_THAT(qwen_rope->first[5], FloatNear(std::cos(0.01f), 1e-6f));
  EXPECT_THAT(qwen_rope->first[6], FloatNear(std::cos(1.0f), 1e-6f));
  EXPECT_THAT(qwen_rope->first[7], FloatNear(std::cos(0.01f), 1e-6f));
  EXPECT_THAT(BuildQwen3RotaryPosEmbed(/*seq_len=*/0, /*head_dim=*/4),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(BuildQwen3RotaryPosEmbed(/*seq_len=*/2, /*head_dim=*/3),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // InterleaveThreeEncoderTaps: seq_len=2, hidden_dim=2
  const std::vector<float> tap0 = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> tap1 = {10.0f, 20.0f, 30.0f, 40.0f};
  const std::vector<float> tap2 = {100.0f, 200.0f, 300.0f, 400.0f};
  auto interleaved = InterleaveThreeEncoderTaps(tap0, tap1, tap2, /*seq_len=*/2,
                                                /*hidden_dim=*/2);
  ASSERT_OK(interleaved);
  EXPECT_THAT(*interleaved,
              ElementsAre(1.0f, 2.0f, 10.0f, 20.0f, 100.0f, 200.0f, 3.0f, 4.0f,
                          30.0f, 40.0f, 300.0f, 400.0f));
  EXPECT_THAT(InterleaveThreeEncoderTaps(tap0, tap1, {1.0f}, /*seq_len=*/2,
                                         /*hidden_dim=*/2),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // BuildFlux2RotaryPosEmbed: seq_len=2, grid_h=1, grid_w=2 -> total_tokens=4
  std::vector<float> joint_ids = BuildTextPositionIds(/*seq_len=*/2);
  const std::vector<float> img_ids =
      BuildImagePositionIds(/*grid_h=*/1, /*grid_w=*/2);
  joint_ids.insert(joint_ids.end(), img_ids.begin(), img_ids.end());
  const std::vector<int> axes_dim = {4, 4, 4, 4};
  auto flux_rope = BuildFlux2RotaryPosEmbed(joint_ids, /*num_tokens=*/4,
                                            axes_dim, /*theta=*/2000.0f);
  ASSERT_OK(flux_rope);
  ASSERT_EQ(flux_rope->first.size(), 4 * 16);
  ASSERT_EQ(flux_rope->second.size(), 4 * 16);
  // Token 0 is text token at pos (0, 0, 0, 0) -> all cos=1, sin=0
  for (int d = 0; d < 16; ++d) {
    EXPECT_FLOAT_EQ(flux_rope->first[d], 1.0f);
    EXPECT_FLOAT_EQ(flux_rope->second[d], 0.0f);
  }
  EXPECT_THAT(
      BuildFlux2RotaryPosEmbed(joint_ids, /*num_tokens=*/0, axes_dim),
      StatusIs(absl::StatusCode::kInvalidArgument));
  const std::vector<int> bad_axes = {4, 4, 4};
  EXPECT_THAT(BuildFlux2RotaryPosEmbed(joint_ids, /*num_tokens=*/4, bad_axes),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(Flux2StagesTest, EndToEndKleinStagesWithMockRunners) {
  // Scaled-down dimensions for fast unit testing:
  // seq_len = 8, hidden_dim = 4 -> prompt_dim = 12, num_heads = 2, head_dim = 4
  // img_size = 16 -> tokens = 1, packed_ch = 128, dit_hidden_dim = 8, steps = 2
  Flux2ModelConfig config;
  config.is_klein = true;
  config.seq_len = 8;
  config.prompt_dim = 12;
  config.img_size = 16;
  config.packed_ch = 128;
  config.steps = 2;

  const int kHiddenDim = 4;
  const int kNumHeads = 2;
  const int kQwenHeadDim = 4;
  const int kDitHiddenDim = 8;
  const int kDitHeadDim = 128;
  const int kJointTokens = config.seq_len + 1;

  // Build a dummy FP16 embedding table with 1024 vocab entries x 4 hidden_dim
  // filled with FP16 0.5 (0x3800).
  std::vector<uint16_t> embed_u16(1024 * kHiddenDim, 0x3800);
  absl::string_view embed_table_bytes(
      reinterpret_cast<const char*>(embed_u16.data()),
      embed_u16.size() * sizeof(uint16_t));

  PushPromptSource prompt_source(ImageGenInputMetadata{
      .width = 16, .height = 16, .num_inference_steps = 2, .seed = 11});

  // 1. KleinTextEncoderStage (3 shards)
  std::array<std::unique_ptr<LiteRtRunner>, 3> textenc_runners;
  for (int i = 0; i < 3; ++i) {
    auto runner = std::make_unique<MockLiteRtRunner>();
    ExpectBuffers(
        *runner,
        {{ElementType::Float32, {1, config.seq_len, kHiddenDim}, 4},
         {ElementType::Float32,
          {1, kNumHeads, config.seq_len, config.seq_len},
          4},
         {ElementType::Float32, {1, config.seq_len, kQwenHeadDim}, 4},
         {ElementType::Float32, {1, config.seq_len, kQwenHeadDim}, 4}},
        {{ElementType::Float32, {1, config.seq_len, kHiddenDim}, 4}});
    EXPECT_CALL(*runner, Run(absl::string_view(""), _, _))
        .WillOnce([i, &config](
                      absl::string_view, absl::Span<const TensorBuffer> inputs,
                      absl::Span<const TensorBuffer> outputs) {
          EXPECT_EQ(inputs.size(), 4);
          EXPECT_EQ(outputs.size(), 1);
          std::vector<float> tap_vals(config.seq_len * kHiddenDim,
                                      static_cast<float>(i + 1));
          auto& tap_out = const_cast<TensorBuffer&>(outputs[0]);
          EXPECT_TRUE(tap_out.Write<float>(tap_vals).HasValue());
          return absl::OkStatus();
        });
    textenc_runners[i] = std::move(runner);
  }

  KleinTextEncoderStage::Config textenc_config;
  textenc_config.framing.seq_len = config.seq_len;
  textenc_config.framing.prefix_token_ids = {1, 2};
  textenc_config.framing.suffix_token_ids = {3};
  textenc_config.framing.pad_token_id = 0;
  textenc_config.prompt_dim = config.prompt_dim;

  auto textenc_stage = KleinTextEncoderStage::Create(
      &prompt_source,
      std::make_unique<FakeTokenizer>(support::TokenIds{10, 20}),
      std::move(textenc_runners), embed_table_bytes, textenc_config);
  ASSERT_OK(textenc_stage);

  // 2. KleinDenoiserStage (8 shards: initial, 2 double, 4 single, final)
  KleinDenoiserStage::Runners dit_runners;

  auto init_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *init_runner,
      {{ElementType::Float32, {1, 1, config.packed_ch}, 4},
       {ElementType::Float32, {1, config.seq_len, config.prompt_dim}, 4},
       {ElementType::Float32, {1}, 4}},
      {{ElementType::Float32, {1, 1, kDitHiddenDim}, 4},
       {ElementType::Float32, {1, config.seq_len, kDitHiddenDim}, 4},
       {ElementType::Float32, {1, 6 * kDitHiddenDim}, 4},
       {ElementType::Float32, {1, 6 * kDitHiddenDim}, 4},
       {ElementType::Float32, {1, 3 * kDitHiddenDim}, 4},
       {ElementType::Float32, {1, kDitHiddenDim}, 4}});
  EXPECT_CALL(*init_runner, Run(absl::string_view(""), _, _))
      .WillRepeatedly([&](absl::string_view, absl::Span<const TensorBuffer>,
                          absl::Span<const TensorBuffer> outputs) {
        for (size_t idx = 0; idx < outputs.size(); ++idx) {
          auto& out = const_cast<TensorBuffer&>(outputs[idx]);
          auto sz = out.PackedSize();
          EXPECT_TRUE(sz.HasValue());
          std::vector<float> zeros(*sz / sizeof(float), 0.1f);
          EXPECT_TRUE(out.Write<float>(zeros).HasValue());
        }
        return absl::OkStatus();
      });
  dit_runners.initial = std::move(init_runner);

  for (int i = 0; i < 2; ++i) {
    auto dbl_runner = std::make_unique<MockLiteRtRunner>();
    ExpectBuffers(
        *dbl_runner,
        {{ElementType::Float32, {1, 1, kDitHiddenDim}, 4},
         {ElementType::Float32, {1, config.seq_len, kDitHiddenDim}, 4},
         {ElementType::Float32, {1, kJointTokens, 1, kDitHeadDim}, 4},
         {ElementType::Float32, {1, kJointTokens, 1, kDitHeadDim}, 4},
         {ElementType::Float32, {1, 6 * kDitHiddenDim}, 4},
         {ElementType::Float32, {1, 6 * kDitHiddenDim}, 4}},
        {{ElementType::Float32, {1, 1, kDitHiddenDim}, 4},
         {ElementType::Float32, {1, config.seq_len, kDitHiddenDim}, 4}});
    EXPECT_CALL(*dbl_runner, Run(absl::string_view(""), _, _))
        .WillRepeatedly([&](absl::string_view, absl::Span<const TensorBuffer>,
                            absl::Span<const TensorBuffer> outputs) {
          std::vector<float> img_out(1 * kDitHiddenDim, 0.2f);
          std::vector<float> txt_out(config.seq_len * kDitHiddenDim, 0.3f);
          EXPECT_TRUE(const_cast<TensorBuffer&>(outputs[0])
                          .Write<float>(img_out)
                          .HasValue());
          EXPECT_TRUE(const_cast<TensorBuffer&>(outputs[1])
                          .Write<float>(txt_out)
                          .HasValue());
          return absl::OkStatus();
        });
    dit_runners.double_blocks[i] = std::move(dbl_runner);
  }

  for (int i = 0; i < 4; ++i) {
    auto sgl_runner = std::make_unique<MockLiteRtRunner>();
    ExpectBuffers(
        *sgl_runner,
        {{ElementType::Float32, {1, kJointTokens, kDitHiddenDim}, 4},
         {ElementType::Float32, {1, kJointTokens, 1, kDitHeadDim}, 4},
         {ElementType::Float32, {1, kJointTokens, 1, kDitHeadDim}, 4},
         {ElementType::Float32, {1, 3 * kDitHiddenDim}, 4}},
        {{ElementType::Float32, {1, kJointTokens, kDitHiddenDim}, 4}});
    EXPECT_CALL(*sgl_runner, Run(absl::string_view(""), _, _))
        .WillRepeatedly([&](absl::string_view, absl::Span<const TensorBuffer>,
                            absl::Span<const TensorBuffer> outputs) {
          std::vector<float> joint_out(kJointTokens * kDitHiddenDim, 0.4f);
          EXPECT_TRUE(const_cast<TensorBuffer&>(outputs[0])
                          .Write<float>(joint_out)
                          .HasValue());
          return absl::OkStatus();
        });
    dit_runners.single_blocks[i] = std::move(sgl_runner);
  }

  int final_calls = 0;
  auto final_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(
      *final_runner,
      {{ElementType::Float32, {1, kJointTokens, kDitHiddenDim}, 4},
       {ElementType::Float32, {1, kDitHiddenDim}, 4}},
      {{ElementType::Float32, {1, 1, config.packed_ch}, 4}});
  EXPECT_CALL(*final_runner, Run(absl::string_view(""), _, _))
      .WillRepeatedly([&](absl::string_view, absl::Span<const TensorBuffer>,
                          absl::Span<const TensorBuffer> outputs) {
        ++final_calls;
        std::vector<float> vel(config.packed_ch, 0.5f);
        EXPECT_TRUE(
            const_cast<TensorBuffer&>(outputs[0]).Write<float>(vel).HasValue());
        return absl::OkStatus();
      });
  dit_runners.final_stage = std::move(final_runner);

  auto denoiser_stage = KleinDenoiserStage::Create(
      textenc_stage->get(), config, std::move(dit_runners));
  ASSERT_OK(denoiser_stage);

  // 3. Flux2VaeDecoderStage
  auto vae_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(*vae_runner, {{ElementType::Float32, {1, 32, 2, 2}, 4}},
                {{ElementType::Float32, {1, 3, 16, 16}, 4}});
  EXPECT_CALL(*vae_runner, Run(absl::string_view(""), _, _))
      .WillOnce([](absl::string_view, absl::Span<const TensorBuffer>,
                   absl::Span<const TensorBuffer> outputs) {
        std::vector<float> decoded(3 * 16 * 16, 0.0f);
        EXPECT_TRUE(const_cast<TensorBuffer&>(outputs[0])
                        .Write<float>(decoded)
                        .HasValue());
        return absl::OkStatus();
      });
  auto vae_stage = Flux2VaeDecoderStage::Create(denoiser_stage->get(), config,
                                                std::move(vae_runner));
  ASSERT_OK(vae_stage);

  ASSERT_OK(prompt_source.PushPrompt("sharded klein test"));
  ASSERT_OK((*textenc_stage)->Schedule());
  ASSERT_OK((*denoiser_stage)->Schedule());
  EXPECT_EQ(final_calls, 2);
  ASSERT_OK((*vae_stage)->Schedule());

  ASSERT_TRUE((*vae_stage)->HasOutput());
  auto out = (*vae_stage)->GetOutput();
  ASSERT_OK(out);
  const auto& img = std::get<ImageOutput>(*out);
  EXPECT_EQ(img.width, 16);
  EXPECT_EQ(img.height, 16);
  EXPECT_EQ(img.channels, 3);
  EXPECT_EQ(img.rgb_data[0], 128);
}

}  // namespace
}  // namespace litert::omni::text2image
