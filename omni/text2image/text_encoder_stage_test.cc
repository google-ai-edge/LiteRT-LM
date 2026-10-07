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

#include "omni/text2image/text_encoder_stage.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
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
#include "omni/text2image/prompt_source.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::text2image {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ElementsAre;

class FakeTokenizer : public support::Tokenizer {
 public:
  explicit FakeTokenizer(support::TokenIds token_ids,
                         std::string* last_input_text = nullptr)
      : token_ids_(std::move(token_ids)), last_input_text_(last_input_text) {}

  void SetNextError(absl::Status status) { next_error_ = std::move(status); }

  support::TokenizerType GetTokenizerType() const override {
    return support::TokenizerType::kHuggingFace;
  }

  absl::StatusOr<support::TokenIds> TextToTokenIds(
      absl::string_view text) override {
    if (!next_error_.ok()) {
      absl::Status err = std::move(next_error_);
      next_error_ = absl::OkStatus();
      return err;
    }
    if (last_input_text_ != nullptr) {
      *last_input_text_ = std::string(text);
    }
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
  std::string* last_input_text_;
  absl::Status next_error_;
};

class TextEncoderStageTest : public ::testing::Test {
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

TEST_F(TextEncoderStageTest, EncodesAndPadsPromptWithDefaultConfig) {
  constexpr int kSeqLen = 20;
  constexpr int kPromptDim = 8;

  PushPromptSource prompt_source(ImageGenInputMetadata{
      .width = 256, .height = 512, .num_inference_steps = 8, .seed = 777});
  std::string tokenized_text;

  auto textenc_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(*textenc_runner,
                {{ElementType::Int32, {1, kSeqLen}, 4},
                 {ElementType::Int32, {1, kSeqLen}, 4}},
                {{ElementType::Float32, {1, kSeqLen, kPromptDim}, 4}});
  EXPECT_CALL(*textenc_runner, Run(absl::string_view(""), _, _))
      .WillOnce([](absl::string_view, absl::Span<const TensorBuffer> inputs,
                   absl::Span<const TensorBuffer> outputs) {
        EXPECT_EQ(inputs.size(), 2);
        EXPECT_EQ(outputs.size(), 1);

        std::vector<int32_t> input_ids(kSeqLen, 0);
        std::vector<int32_t> attention_mask(kSeqLen, 0);
        auto& in0 = const_cast<TensorBuffer&>(inputs[0]);
        auto& in1 = const_cast<TensorBuffer&>(inputs[1]);
        EXPECT_TRUE(in0.Read<int32_t>(absl::MakeSpan(input_ids)).HasValue());
        EXPECT_TRUE(
            in1.Read<int32_t>(absl::MakeSpan(attention_mask)).HasValue());

        // Default prefix: {151644}, user_ids: {872, 198, 101, 102, 103},
        // Default suffix: {151645, 198, 151644, 77091, 198, 151667, 271,
        //                  151668, 271}
        // => 15 active tokens, 5 pad tokens (151643).
        EXPECT_THAT(absl::MakeConstSpan(input_ids).subspan(0, 15),
                    ElementsAre(151644, 872, 198, 101, 102, 103, 151645, 198,
                                151644, 77091, 198, 151667, 271, 151668, 271));
        for (int i = 0; i < 15; ++i) {
          EXPECT_EQ(attention_mask[i], 1);
        }
        for (int i = 15; i < kSeqLen; ++i) {
          EXPECT_EQ(input_ids[i], 151643);
          EXPECT_EQ(attention_mask[i], 0);
        }

        std::vector<float> fake_embeds(kSeqLen * kPromptDim, 0.25f);
        auto& out = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(out.Write<float>(fake_embeds).HasValue());
        return absl::OkStatus();
      });

  TextEncoderStage::Config config;
  config.seq_len = kSeqLen;
  auto textenc_stage = TextEncoderStage::Create(
      &prompt_source,
      std::make_unique<FakeTokenizer>(
          support::TokenIds{872, 198, 101, 102, 103}, &tokenized_text),
      std::move(textenc_runner), config);
  ASSERT_OK(textenc_stage);

  ASSERT_OK(prompt_source.PushPrompt("a sunset over mountains"));
  ASSERT_TRUE(prompt_source.HasOutput());

  ASSERT_TRUE((*textenc_stage)->NeedSchedule());
  ASSERT_OK((*textenc_stage)->Schedule());

  EXPECT_EQ(tokenized_text, "user\na sunset over mountains");
  ASSERT_TRUE((*textenc_stage)->HasOutput());
  auto out = (*textenc_stage)->GetOutput();
  ASSERT_OK(out);
  EXPECT_EQ(out->metadata.width, 256);
  EXPECT_EQ(out->metadata.height, 512);
  EXPECT_EQ(out->metadata.num_inference_steps, 8);
  EXPECT_EQ(out->metadata.seed, 777);
  EXPECT_EQ(out->prompt_embeds.size(), kSeqLen * kPromptDim);
  EXPECT_FLOAT_EQ(out->prompt_embeds[0], 0.25f);
}

TEST_F(TextEncoderStageTest, EncodesAndPadsPromptWithInt64InputBuffers) {
  constexpr int kSeqLen = 20;
  constexpr int kPromptDim = 8;

  PushPromptSource prompt_source;
  auto textenc_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(*textenc_runner,
                {{ElementType::Int64, {1, kSeqLen}, 8},
                 {ElementType::Int64, {1, kSeqLen}, 8}},
                {{ElementType::Float32, {1, kSeqLen, kPromptDim}, 4}});
  EXPECT_CALL(*textenc_runner, Run(absl::string_view(""), _, _))
      .WillOnce([](absl::string_view, absl::Span<const TensorBuffer> inputs,
                   absl::Span<const TensorBuffer> outputs) {
        std::vector<int64_t> input_ids(kSeqLen, 0);
        std::vector<int64_t> attention_mask(kSeqLen, 0);
        auto& in0 = const_cast<TensorBuffer&>(inputs[0]);
        auto& in1 = const_cast<TensorBuffer&>(inputs[1]);
        EXPECT_TRUE(in0.Read<int64_t>(absl::MakeSpan(input_ids)).HasValue());
        EXPECT_TRUE(
            in1.Read<int64_t>(absl::MakeSpan(attention_mask)).HasValue());

        EXPECT_THAT(absl::MakeConstSpan(input_ids).subspan(0, 15),
                    ElementsAre(151644, 872, 198, 101, 102, 103, 151645, 198,
                                151644, 77091, 198, 151667, 271, 151668, 271));
        for (int i = 0; i < 15; ++i) {
          EXPECT_EQ(attention_mask[i], 1);
        }
        for (int i = 15; i < kSeqLen; ++i) {
          EXPECT_EQ(input_ids[i], 151643);
          EXPECT_EQ(attention_mask[i], 0);
        }

        std::vector<float> fake_embeds(kSeqLen * kPromptDim, 0.75f);
        auto& out = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(out.Write<float>(fake_embeds).HasValue());
        return absl::OkStatus();
      });

  TextEncoderStage::Config config;
  config.seq_len = kSeqLen;
  auto textenc_stage =
      TextEncoderStage::Create(&prompt_source,
                               std::make_unique<FakeTokenizer>(
                                   support::TokenIds{872, 198, 101, 102, 103}),
                               std::move(textenc_runner), config);
  ASSERT_OK(textenc_stage);

  ASSERT_OK(prompt_source.PushPrompt("int64 prompt"));
  ASSERT_OK((*textenc_stage)->Schedule());
  ASSERT_TRUE((*textenc_stage)->HasOutput());
  auto out = (*textenc_stage)->GetOutput();
  ASSERT_OK(out);
  EXPECT_EQ(out->prompt_embeds.size(), kSeqLen * kPromptDim);
  EXPECT_FLOAT_EQ(out->prompt_embeds[0], 0.75f);
}

TEST_F(TextEncoderStageTest, TruncatesLongPromptWithCustomConfig) {
  constexpr int kSeqLen = 6;
  constexpr int kPromptDim = 4;

  PushPromptSource prompt_source;
  std::string tokenized_text;

  auto textenc_runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(*textenc_runner,
                {{ElementType::Int32, {1, kSeqLen}, 4},
                 {ElementType::Int32, {1, kSeqLen}, 4}},
                {{ElementType::Float32, {1, kSeqLen, kPromptDim}, 4}});
  EXPECT_CALL(*textenc_runner, Run(absl::string_view(""), _, _))
      .WillOnce([](absl::string_view, absl::Span<const TensorBuffer> inputs,
                   absl::Span<const TensorBuffer> outputs) {
        std::vector<int32_t> input_ids(kSeqLen, 0);
        std::vector<int32_t> attention_mask(kSeqLen, 0);
        auto& in0 = const_cast<TensorBuffer&>(inputs[0]);
        auto& in1 = const_cast<TensorBuffer&>(inputs[1]);
        EXPECT_TRUE(in0.Read<int32_t>(absl::MakeSpan(input_ids)).HasValue());
        EXPECT_TRUE(
            in1.Read<int32_t>(absl::MakeSpan(attention_mask)).HasValue());

        // prefix: {10}, suffix: {20, 30} => max_user = 6 - 3 = 3 tokens.
        EXPECT_THAT(input_ids, ElementsAre(10, 1, 2, 3, 20, 30));
        EXPECT_THAT(attention_mask, ElementsAre(1, 1, 1, 1, 1, 1));

        std::vector<float> fake_embeds(kSeqLen * kPromptDim, 0.5f);
        auto& out = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(out.Write<float>(fake_embeds).HasValue());
        return absl::OkStatus();
      });

  TextEncoderStage::Config config;
  config.seq_len = kSeqLen;
  config.pad_token_id = 0;
  config.prompt_prefix = "";
  config.prefix_token_ids = {10};
  config.suffix_token_ids = {20, 30};

  auto textenc_stage = TextEncoderStage::Create(
      &prompt_source,
      std::make_unique<FakeTokenizer>(support::TokenIds{1, 2, 3, 4, 5},
                                      &tokenized_text),
      std::move(textenc_runner), config);
  ASSERT_OK(textenc_stage);

  ASSERT_OK(prompt_source.PushPrompt("hello"));
  ASSERT_OK((*textenc_stage)->Schedule());
  EXPECT_EQ(tokenized_text, "hello");
  ASSERT_TRUE((*textenc_stage)->HasOutput());
  auto out = (*textenc_stage)->GetOutput();
  ASSERT_OK(out);
  EXPECT_EQ(out->prompt_embeds.size(), kSeqLen * kPromptDim);
  EXPECT_FLOAT_EQ(out->prompt_embeds[0], 0.5f);
}

TEST_F(TextEncoderStageTest, ScheduleWithEmptySourceIsNoOp) {
  constexpr int kSeqLen = 16;
  constexpr int kPromptDim = 8;

  PushPromptSource prompt_source;
  auto runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(*runner,
                {{ElementType::Int32, {1, kSeqLen}, 4},
                 {ElementType::Int32, {1, kSeqLen}, 4}},
                {{ElementType::Float32, {1, kSeqLen, kPromptDim}, 4}});
  EXPECT_CALL(*runner, Run(absl::string_view(""), _, _)).Times(0);

  TextEncoderStage::Config config;
  config.seq_len = kSeqLen;
  auto textenc_stage = TextEncoderStage::Create(
      &prompt_source, std::make_unique<FakeTokenizer>(support::TokenIds{1}),
      std::move(runner), config);
  ASSERT_OK(textenc_stage);

  EXPECT_FALSE((*textenc_stage)->NeedSchedule());
  ASSERT_OK((*textenc_stage)->Schedule());
  EXPECT_TRUE((*textenc_stage)->IsIdle());
  EXPECT_FALSE((*textenc_stage)->HasOutput());
}

TEST_F(TextEncoderStageTest, PropagatesRuntimeErrorsAndRecoversToIdle) {
  constexpr int kSeqLen = 16;
  constexpr int kPromptDim = 8;

  PushPromptSource prompt_source;
  auto tokenizer = std::make_unique<FakeTokenizer>(support::TokenIds{1, 2});
  FakeTokenizer* raw_tokenizer = tokenizer.get();

  auto runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(*runner,
                {{ElementType::Int32, {1, kSeqLen}, 4},
                 {ElementType::Int32, {1, kSeqLen}, 4}},
                {{ElementType::Float32, {1, kSeqLen, kPromptDim}, 4}});
  EXPECT_CALL(*runner, Run(absl::string_view(""), _, _))
      .WillOnce([](absl::string_view, absl::Span<const TensorBuffer>,
                   absl::Span<const TensorBuffer>) {
        return absl::UnavailableError("runner transient error");
      })
      .WillOnce([](absl::string_view, absl::Span<const TensorBuffer>,
                   absl::Span<const TensorBuffer> outputs) {
        std::vector<float> fake_embeds(kSeqLen * kPromptDim, 1.25f);
        auto& out = const_cast<TensorBuffer&>(outputs[0]);
        EXPECT_TRUE(out.Write<float>(fake_embeds).HasValue());
        return absl::OkStatus();
      });

  TextEncoderStage::Config config;
  config.seq_len = kSeqLen;
  auto textenc_stage = TextEncoderStage::Create(
      &prompt_source, std::move(tokenizer), std::move(runner), config);
  ASSERT_OK(textenc_stage);

  // 1. Tokenizer error -> returns error, no output, returns to idle.
  raw_tokenizer->SetNextError(absl::InternalError("tokenizer failure"));
  ASSERT_OK(prompt_source.PushPrompt("first prompt"));
  EXPECT_THAT((*textenc_stage)->Schedule(),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_TRUE((*textenc_stage)->IsIdle());
  EXPECT_FALSE((*textenc_stage)->HasOutput());

  // 2. Runner error -> returns error, no output, returns to idle.
  ASSERT_OK(prompt_source.PushPrompt("second prompt"));
  EXPECT_THAT((*textenc_stage)->Schedule(),
              StatusIs(absl::StatusCode::kUnavailable));
  EXPECT_TRUE((*textenc_stage)->IsIdle());
  EXPECT_FALSE((*textenc_stage)->HasOutput());

  // 3. Subsequent prompt succeeds after recovery.
  ASSERT_OK(prompt_source.PushPrompt("third prompt"));
  ASSERT_OK((*textenc_stage)->Schedule());
  EXPECT_TRUE((*textenc_stage)->IsIdle());
  ASSERT_TRUE((*textenc_stage)->HasOutput());
  auto out = (*textenc_stage)->GetOutput();
  ASSERT_OK(out);
  EXPECT_EQ(out->prompt_embeds.size(), kSeqLen * kPromptDim);
  EXPECT_FLOAT_EQ(out->prompt_embeds[0], 1.25f);
}

TEST_F(TextEncoderStageTest, RejectsInvalidCreateArguments) {
  constexpr int kSeqLen = 16;
  constexpr int kPromptDim = 8;

  PushPromptSource prompt_source;

  // 1. seq_len <= prefix + suffix (default prefix + suffix is 10 tokens).
  {
    TextEncoderStage::Config config;
    config.seq_len = 10;
    EXPECT_THAT(TextEncoderStage::Create(
                    &prompt_source,
                    std::make_unique<FakeTokenizer>(support::TokenIds{1}),
                    std::make_unique<MockLiteRtRunner>(), config),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 2. Null runner.
  {
    std::unique_ptr<LiteRtRunner> null_runner;
    TextEncoderStage::Config config;
    config.seq_len = kSeqLen;
    EXPECT_THAT(TextEncoderStage::Create(
                    &prompt_source,
                    std::make_unique<FakeTokenizer>(support::TokenIds{1}),
                    std::move(null_runner), config),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 3. Fewer than 2 input buffers on runner.
  {
    auto runner = std::make_unique<MockLiteRtRunner>();
    ExpectBuffers(*runner, {{ElementType::Int32, {1, kSeqLen}, 4}},
                  {{ElementType::Float32, {1, kSeqLen, kPromptDim}, 4}});
    TextEncoderStage::Config config;
    config.seq_len = kSeqLen;
    EXPECT_THAT(TextEncoderStage::Create(
                    &prompt_source,
                    std::make_unique<FakeTokenizer>(support::TokenIds{1}),
                    std::move(runner), config),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 4. Runner with 0 output buffers.
  {
    auto runner = std::make_unique<MockLiteRtRunner>();
    ExpectBuffers(*runner,
                  {{ElementType::Int32, {1, kSeqLen}, 4},
                   {ElementType::Int32, {1, kSeqLen}, 4}},
                  {});
    TextEncoderStage::Config config;
    config.seq_len = kSeqLen;
    EXPECT_THAT(TextEncoderStage::Create(
                    &prompt_source,
                    std::make_unique<FakeTokenizer>(support::TokenIds{1}),
                    std::move(runner), config),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }

  // 5. Runner with >1 output buffers.
  {
    auto runner = std::make_unique<MockLiteRtRunner>();
    ExpectBuffers(*runner,
                  {{ElementType::Int32, {1, kSeqLen}, 4},
                   {ElementType::Int32, {1, kSeqLen}, 4}},
                  {{ElementType::Float32, {1, kSeqLen, kPromptDim}, 4},
                   {ElementType::Float32, {1, kSeqLen, kPromptDim}, 4}});
    TextEncoderStage::Config config;
    config.seq_len = kSeqLen;
    EXPECT_THAT(TextEncoderStage::Create(
                    &prompt_source,
                    std::make_unique<FakeTokenizer>(support::TokenIds{1}),
                    std::move(runner), config),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST_F(TextEncoderStageTest, RejectsIdenticalInputIdsAndAttentionMaskIndices) {
  constexpr int kSeqLen = 16;
  constexpr int kPromptDim = 8;

  PushPromptSource prompt_source;
  auto runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(*runner,
                {{ElementType::Int32, {1, kSeqLen}, 4},
                 {ElementType::Int32, {1, kSeqLen}, 4}},
                {{ElementType::Float32, {1, kSeqLen, kPromptDim}, 4}});

  TextEncoderStage::Config config;
  config.seq_len = kSeqLen;
  config.input_indices = {.input_ids = 0, .attention_mask = 0};
  auto textenc_stage = TextEncoderStage::Create(
      &prompt_source, std::make_unique<FakeTokenizer>(support::TokenIds{1}),
      std::move(runner), config);
  EXPECT_THAT(textenc_stage, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(TextEncoderStageTest, RejectsInvalidInputBufferElementType) {
  constexpr int kSeqLen = 16;
  constexpr int kPromptDim = 8;

  PushPromptSource prompt_source;
  auto runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(*runner,
                {{ElementType::Float32, {1, kSeqLen}, 4},
                 {ElementType::Int32, {1, kSeqLen}, 4}},
                {{ElementType::Float32, {1, kSeqLen, kPromptDim}, 4}});

  TextEncoderStage::Config config;
  config.seq_len = kSeqLen;
  auto textenc_stage = TextEncoderStage::Create(
      &prompt_source, std::make_unique<FakeTokenizer>(support::TokenIds{1}),
      std::move(runner), config);
  EXPECT_THAT(textenc_stage, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(TextEncoderStageTest, RejectsMismatchedInputBufferSeqLen) {
  constexpr int kSeqLen = 16;
  constexpr int kPromptDim = 8;

  PushPromptSource prompt_source;
  auto runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(*runner,
                {{ElementType::Int32, {1, kSeqLen + 4}, 4},
                 {ElementType::Int32, {1, kSeqLen}, 4}},
                {{ElementType::Float32, {1, kSeqLen, kPromptDim}, 4}});

  TextEncoderStage::Config config;
  config.seq_len = kSeqLen;
  auto textenc_stage = TextEncoderStage::Create(
      &prompt_source, std::make_unique<FakeTokenizer>(support::TokenIds{1}),
      std::move(runner), config);
  EXPECT_THAT(textenc_stage, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(TextEncoderStageTest, RejectsNonFloat32OutputBuffer) {
  constexpr int kSeqLen = 16;
  constexpr int kPromptDim = 8;

  PushPromptSource prompt_source;
  auto runner = std::make_unique<MockLiteRtRunner>();
  ExpectBuffers(*runner,
                {{ElementType::Int32, {1, kSeqLen}, 4},
                 {ElementType::Int32, {1, kSeqLen}, 4}},
                {{ElementType::Int32, {1, kSeqLen, kPromptDim}, 4}});

  TextEncoderStage::Config config;
  config.seq_len = kSeqLen;
  auto textenc_stage = TextEncoderStage::Create(
      &prompt_source, std::make_unique<FakeTokenizer>(support::TokenIds{1}),
      std::move(runner), config);
  EXPECT_THAT(textenc_stage, StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::omni::text2image
