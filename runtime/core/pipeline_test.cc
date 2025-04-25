#include "third_party/odml/litert_lm/runtime/core/pipeline.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "third_party/odml/litert_lm/runtime/components/sentencepiece_tokenizer.h"
#include "third_party/odml/litert_lm/runtime/components/tokenizer.h"
#include "third_party/odml/litert_lm/runtime/components/top_p_cpu_sampler.h"
#include "third_party/odml/litert_lm/runtime/engine/io_types.h"
#include "third_party/odml/litert_lm/runtime/executor/fake_llm_executor.h"
#include "third_party/odml/litert_lm/runtime/executor/llm_executor.h"
#include "third_party/odml/litert_lm/runtime/util/convert_tensor_buffer.h"

namespace litert::lm {
namespace {

constexpr char kTestdataDir[] =
    "runtime/components/testdata/";

class PipelineTest : public testing::Test {
 protected:
  void SetUp() override {
    auto tokenizer_or = SentencePieceTokenizer::CreateFromFile(
        (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
         "sentencepiece.model")
            .string());
    ASSERT_OK(tokenizer_or);
    tokenizer_ = std::move(tokenizer_or.value());
    // The prefill tokens are the expected tokens that will be passed in at each
    // time the Prefill function is called. The values are the token ids of the
    // input prompt "Hello World!" prepended with the bos token id (2).
    std::vector<std::vector<int>> prefill_tokens = {
      {2, 90, 547, 58, 735, 210, 466, 2294}};
    // The decode tokens are the expected tokens that will be returned by the
    // Decode function. The values are the token ids of the output response
    // "How's it going?" followed by the stop token id (2294).
    std::vector<std::vector<int>> decode_tokens = {{224}, {24}, {8},    {66},
                                                   {246}, {18}, {2295}, {2294}};
    // Vocab size needs to at least be larger than the largest token id 2294.
    executor_ = std::make_shared<FakeLlmExecutor>(
        /*vocab_size=*/2560, prefill_tokens, decode_tokens);
  }

  std::shared_ptr<Tokenizer> tokenizer_;
  std::shared_ptr<LlmExecutor> executor_;
};

TEST_F(PipelineTest, Prefill) {
  const std::string prompt = "Hello World!";
  auto last_prefill_token_id =
      Prefill(executor_, tokenizer_, prompt, /*bos_token_id=*/2);
  EXPECT_OK(last_prefill_token_id.status());
  EXPECT_EQ(*last_prefill_token_id, 2294);
}

TEST_F(PipelineTest, Decode) {
  auto responses = Decode(executor_, tokenizer_, /*stop_token_ids=*/{2294});
  EXPECT_OK(responses);
  EXPECT_EQ(*(responses->GetResponseTextAt(0)), " How's it going?!");
}

class PipelineCustomSamplingTest : public testing::Test {
 protected:
  void SetUp() override {
    auto tokenizer_or = SentencePieceTokenizer::CreateFromFile(
        (std::filesystem::path(::testing::SrcDir()) / kTestdataDir /
         "sentencepiece.model")
            .string());
    ASSERT_OK(tokenizer_or);
    tokenizer_ = std::move(tokenizer_or.value());
    // The prefill tokens are the expected tokens that will be passed in at each
    // time the Prefill function is called. The values are the token ids of the
    // input prompt "Hello World!" prepended with the bos token id (2).
    std::vector<std::vector<int>> prefill_tokens = {
        {2, 90, 547, 58, 735, 210, 466, 2294}};
    // The decode tokens are the expected tokens that will be returned by the
    // Decode function. The  two values are the token ids of the output
    // responses " How's it going?!" and " Hello World!" followed by the stop
    // token id (0).
    std::vector<std::vector<int>> decode_tokens = {
        {224, 90}, {24, 547},    {8, 58},   {66, 735}, {246, 210},
        {18, 466}, {2295, 2294}, {2294, 0}, {0, 0}};
    // Vocab size needs to at least be larger than the largest token id 2294.
    executor_ = std::make_shared<FakeLlmExecutor>(
        /*vocab_size=*/2560, prefill_tokens, decode_tokens);
  }

  std::shared_ptr<Tokenizer> tokenizer_;
  std::shared_ptr<LlmExecutor> executor_;
};

TEST_F(PipelineCustomSamplingTest, Prefill) {
  const std::string prompt = "Hello World!";
  auto last_prefill_token_id =
      Prefill(executor_, tokenizer_, prompt, /*bos_token_id=*/2);
  EXPECT_OK(last_prefill_token_id.status());
  EXPECT_EQ(*last_prefill_token_id, 2294);
}

TEST_F(PipelineCustomSamplingTest, DecodeCustomSampling) {
  auto sampler_or = TopPSampler::Create(/*k=*/1, /*p=*/0.5, /*temperature=*/1.0,
                                        /*batch_size=*/2, /*seed=*/1);
  EXPECT_TRUE(sampler_or.ok());
  std::unique_ptr<TopPSampler> sampler = std::move(sampler_or.value());

  auto decoded_ids = CreateTensorBuffer<int>({2, 1});
  auto responses =
      DecodeCustomSampling(executor_, tokenizer_,
                           /*stop_token_ids=*/{0},
                           /*num_output_candidates=*/2, *sampler, *decoded_ids);
  EXPECT_OK(responses);
  EXPECT_EQ(responses->GetNumOutputCandidates(), 2);
  // First candidate: " How's it going?!".
  EXPECT_EQ(*(responses->GetResponseTextAt(0)), " How's it going?!");
  // Second candidate: " Hello World!".
  EXPECT_EQ(*(responses->GetResponseTextAt(1)), " Hello World!");

  // The scores are all equal to 0.0f (log(1.0f)).
  EXPECT_EQ(*(responses->GetScoreAt(0)), 0.0f);
  EXPECT_EQ(*(responses->GetScoreAt(1)), 0.0f);
}

}  // namespace
}  // namespace litert::lm
