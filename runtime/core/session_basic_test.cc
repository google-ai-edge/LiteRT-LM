#include "runtime/core/session_basic.h"

#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/util/test_utils.h"  // NOLINT

namespace litert::lm {
namespace {

constexpr char kTestdataDir[] =
    "litert_lm/runtime/components/testdata/";

class SessionBasicTest : public testing::Test {
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
    // input prompt "Hello World!" surrounded by the start/end of turn prompt
    // template and prepended with the bos token id (2).
    std::vector<std::vector<int>> prefill_tokens = {
        {2,   4,  0,   202, 3764, 0,    751, 0,  1972, 0, 555,  179, 90,
         547, 58, 735, 210, 466,  2294, 0,   40, 23,   0, 751,  0,   1972,
         0,   4,  0,   202, 3764, 0,    751, 0,  1972, 0, 3065, 23,  3076}};
    // The decode tokens are the expected tokens that will be returned by the
    // Decode function. The values are the token ids of the output response
    // "How's it going?" followed by the stop token id (2294).
    std::vector<std::vector<int>> decode_tokens = {{224}, {24}, {8},    {66},
                                                   {246}, {18}, {2295}, {2294}};
    executor_ =
        std::make_shared<FakeLlmExecutor>(2560, prefill_tokens, decode_tokens);

    sampler_params_.set_type(proto::SamplerParameters::TYPE_UNSPECIFIED);
  }

  std::shared_ptr<Tokenizer> tokenizer_;
  std::shared_ptr<LlmExecutor> executor_;
  proto::SamplerParameters sampler_params_;
};

TEST_F(SessionBasicTest, RunPrefill) {
  std::vector<int> stop_token_ids = {2294};
  auto session = SessionBasic::Create(executor_, tokenizer_, stop_token_ids,
                                      sampler_params_);
  EXPECT_OK((*session)->RunPrefill("Hello World!"));
}

TEST_F(SessionBasicTest, RunDecode) {
  std::vector<int> stop_token_ids = {2294};
  auto session = SessionBasic::Create(executor_, tokenizer_, stop_token_ids,
                                      sampler_params_);
  EXPECT_OK((*session)->RunPrefill("Hello World!"));
  auto responses = (*session)->RunDecode();
  EXPECT_OK(responses);
  EXPECT_EQ(responses->GetNumOutputCandidates(), 1);
  EXPECT_EQ(*(responses->GetResponseTextAt(0)), " How's it going?!");
}

}  // namespace
}  // namespace litert::lm
