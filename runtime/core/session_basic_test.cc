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

#include "runtime/core/session_basic.h"

#include <array>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/constrained_decoding/fake_constraint.h"
#include "runtime/components/sentencepiece_tokenizer.h"
#include "runtime/components/tokenizer.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/audio_litert_compiled_model_executor.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/fake_llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/framework/threadpool.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

constexpr absl::string_view kTestdataDir =
    "litert_lm/runtime/components/testdata/";
constexpr absl::string_view kTestAudioModelPath =
    "litert_lm/runtime/testdata/dummy_audio_only.litertlm";

constexpr int kSpectrogramFrequencySlots = 8;
constexpr int kSpectrogramSequenceLength = 10;
constexpr int kEmbeddingSequenceLength = 5;
constexpr int kEmbeddingDimensions = 6;

// Audio embedding tensor will have shape [1, kEmbeddingSequenceLength,
// kEmbeddingDimensions].
constexpr std::array<float, kEmbeddingSequenceLength * kEmbeddingDimensions>
    kExpectedAudioEmbedding = {0., 0., 0., 0., 0., 0., 0., 1., 2., 3.,
                               3., 3., 0., 1., 2., 4., 4., 4., 1., 2.,
                               3., 5., 5., 5., 0., 1., 2., 4., 4., 4.};

// Mel spectrogram tensor will have shape [1, kSpectrogramSequenceLength,
// kSpectrogramFrequencySlots].
constexpr std::array<float,
                     kSpectrogramSequenceLength * kSpectrogramFrequencySlots>
    mel_spectrogram_data = {
        0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 0., 0., 0., 0.,
        0., 1., 0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
        0., 1., 0., 1., 0., 0., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0.,
        1., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 1., 0., 1., 0., 0.,
        0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 1., 1., 1., 1.};

absl::StatusOr<std::unique_ptr<FakeLlmExecutor>> CreateFakeLlmExecutor(
    std::vector<std::vector<int>> prefill_tokens,
    std::vector<std::vector<int>> decode_tokens,
    std::optional<std::vector<float>> audio_embedding = std::nullopt) {
  auto batch_size = decode_tokens.empty() ? 1 : decode_tokens[0].size();
  auto fake_executor = std::make_unique<FakeLlmExecutor>(
      2560, prefill_tokens, decode_tokens, batch_size, audio_embedding);
  return std::move(fake_executor);
}

absl::StatusOr<ScoringResponses> RunTextScoring(
    const std::vector<std::vector<int>>& prefill_tokens,
    const std::vector<std::vector<int>>& decode_tokens,
    absl::string_view input_prompt, absl::string_view target_text,
    const proto::SamplerParameters& sampler_params, bool store_token_lengths,
    Tokenizer* tokenizer, ThreadPool* worker_thread_pool) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  ASSIGN_OR_RETURN(auto executor,
                   CreateFakeLlmExecutor(prefill_tokens, decode_tokens));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer, /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config, std::nullopt,
      worker_thread_pool);
  std::vector<InputData> inputs;
  inputs.emplace_back(InputText(std::string(input_prompt)));
  EXPECT_OK((*session)->RunPrefill(inputs));
  std::vector<absl::string_view> target_texts;
  target_texts.push_back(target_text);
  return (*session)->RunTextScoring(target_texts, store_token_lengths);
}

class ExtendedTokenizer : public Tokenizer {
 public:
  static absl::StatusOr<std::unique_ptr<ExtendedTokenizer>> CreateFromFile(
      absl::string_view model_path) {
    ASSIGN_OR_RETURN(auto tokenizer,
                     SentencePieceTokenizer::CreateFromFile(model_path));
    return absl::WrapUnique(new ExtendedTokenizer(std::move(tokenizer)));
  }

  void SetExtendedToken(int token_id, absl::string_view token_str) {
    extended_tokens_to_id_[token_str] = token_id;
    id_to_extended_tokens_[token_id] = token_str;
  }

  absl::StatusOr<std::vector<int>> TextToTokenIds(
      absl::string_view text) override {
    std::vector<int> token_ids;
    bool is_extended_token_found = false;
    do {
      is_extended_token_found = false;
      for (const auto& [extended_token_str, extended_token_id] :
           extended_tokens_to_id_) {
        auto extended_token_pos = text.find(extended_token_str);
        if (extended_token_pos != std::string::npos) {
          // The text before the extended token.
          ASSIGN_OR_RETURN(
              auto text_ids,
              tokenizer_->TextToTokenIds(text.substr(0, extended_token_pos)));
          token_ids.insert(token_ids.end(), text_ids.begin(), text_ids.end());
          token_ids.push_back(extended_token_id);
          text = text.substr(extended_token_pos + extended_token_str.size());
          is_extended_token_found = true;
        }
      }
    } while (is_extended_token_found);
    if (!text.empty()) {
      ASSIGN_OR_RETURN(auto text_ids, tokenizer_->TextToTokenIds(text));
      token_ids.insert(token_ids.end(), text_ids.begin(), text_ids.end());
    }
    return token_ids;
  }

  absl::StatusOr<std::string> TokenIdsToText(
      const std::vector<int>& token_ids) override {
    std::vector<std::string> token_strs;
    for (int token_id : token_ids) {
      if (id_to_extended_tokens_.contains(token_id)) {
        token_strs.push_back(id_to_extended_tokens_[token_id]);
      } else {
        token_strs.push_back(tokenizer_->TokenIdsToText({token_id}).value());
      }
    }
    return absl::StrJoin(token_strs, "");
  }

  absl::StatusOr<int> TokenToId(absl::string_view token) override {
    if (extended_tokens_to_id_.contains(token)) {
      return extended_tokens_to_id_[token];
    }
    return tokenizer_->TokenToId(token);
  }

  TokenizerType GetTokenizerType() const override {
    return tokenizer_->GetTokenizerType();
  }

  std::vector<std::string> GetTokens() const override {
    return tokenizer_->GetTokens();
  }

 private:
  explicit ExtendedTokenizer(std::unique_ptr<SentencePieceTokenizer> tokenizer)
      : tokenizer_(std::move(tokenizer)) {};

  absl::flat_hash_map<int, std::string> id_to_extended_tokens_;
  absl::flat_hash_map<std::string, int> extended_tokens_to_id_;
  std::unique_ptr<SentencePieceTokenizer> tokenizer_;
};

class SessionBasicTest : public testing::Test {
 protected:
  void SetUp() override {
    auto tokenizer = ExtendedTokenizer::CreateFromFile(
        (std::filesystem::path(::testing::SrcDir()) /
         std::string(kTestdataDir) / "sentencepiece.model")
            .string());
    ASSERT_OK(tokenizer);
    tokenizer.value()->SetExtendedToken(256000, "<start_of_audio>");
    tokenizer_ = std::move(*tokenizer);
    sampler_params_.set_type(proto::SamplerParameters::TYPE_UNSPECIFIED);
    // Creating the thread pool of a single thread to execute the works.
    worker_thread_pool_ = std::make_unique<ThreadPool>(/*name_prefix=*/"engine",
                                                       /*max_num_threads=*/1);
  }

  std::unique_ptr<Tokenizer> tokenizer_;
  proto::SamplerParameters sampler_params_;
  std::unique_ptr<ThreadPool> worker_thread_pool_;
};

absl::StatusOr<std::unique_ptr<AudioLiteRtCompiledModelExecutor>>
CreateAudioExecutor(Environment& env, const std::string& model_path,
                    int max_sequence_length, Backend backend) {
  ASSIGN_OR_RETURN(auto model_file, ScopedFile::Open(model_path));
  auto model_file_ptr = std::make_shared<ScopedFile>(std::move(model_file));
  ASSIGN_OR_RETURN(auto model_assets, ModelAssets::Create(model_file_ptr));
  // Create the audio executor settings.
  ASSIGN_OR_RETURN(auto audio_executor_settings,
                   AudioExecutorSettings::CreateDefault(
                       model_assets, max_sequence_length, backend));
  // Create the audio executor.
  return litert::lm::AudioLiteRtCompiledModelExecutor::Create(
      audio_executor_settings, env);
}

absl::AnyInvocable<void(absl::StatusOr<Responses>)> CreateStreamingTestCallback(
    absl::Status& status_ref, std::vector<std::string>& texts_ref,
    absl::Notification& done_ref, bool delay_on_next = false) {
  return [&status_ref, &texts_ref, &done_ref,
          delay_on_next](absl::StatusOr<Responses> responses) mutable {
    if (!responses.ok()) {
      status_ref = std::move(responses.status());
      done_ref.Notify();
      return;
    }
    if (responses->GetTaskState() == TaskState::kDone) {
      done_ref.Notify();
      return;
    }
    if (delay_on_next) {
      absl::SleepFor(absl::Milliseconds(50));
    }
    EXPECT_EQ(responses->GetTexts().size(), 1);
    texts_ref.push_back(responses->GetTexts()[0]);
  };
}

TEST_F(SessionBasicTest, SessionCreation) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // The prefill tokens are the expected tokens that will be passed in
          // at each time the Prefill function is called. The values are the
          // token ids of the input prompt "Hello World!".
          // The decode tokens are the expected tokens that will be returned
          // by the Decode function. The values are the token ids of the
          // output response "How's it going?" followed by the stop token id
          // (2294).
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config, std::nullopt,
      worker_thread_pool_.get());
  EXPECT_OK(session);

  // Second session creation should fail because there is already a session.
  EXPECT_THAT(SessionBasic::Create(executor.get(), tokenizer_.get(),
                                   /*vision_executor=*/nullptr,
                                   /*audio_executor=*/nullptr, session_config,
                                   std::nullopt, worker_thread_pool_.get()),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  // After the first session is destroyed, the second session creation should
  // succeed.
  session->reset();
  EXPECT_OK(SessionBasic::Create(executor.get(), tokenizer_.get(),
                                 /*vision_executor=*/nullptr,
                                 /*audio_executor=*/nullptr, session_config,
                                 std::nullopt, worker_thread_pool_.get()));
}

TEST_F(SessionBasicTest, RunPrefill) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // The prefill tokens are the expected tokens that will be passed in
          // at each time the Prefill function is called. The values are the
          // token ids of the input prompt "Hello World!".
          // The decode tokens are the expected tokens that will be returned
          // by the Decode function. The values are the token ids of the
          // output response "How's it going?" followed by the stop token id
          // (2294).
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config, std::nullopt,
      worker_thread_pool_.get());
  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  EXPECT_OK((*session)->RunPrefill(inputs));
}

TEST_F(SessionBasicTest, EmptyInputTextReturnsError) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetSamplerBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(auto executor, CreateFakeLlmExecutor(
                                          /*prefill_tokens=*/{{}},
                                          /*decode_tokens=*/{{}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config, std::nullopt,
      worker_thread_pool_.get());
  std::vector<InputData> inputs;
  inputs.emplace_back(InputText(""));
  EXPECT_THAT((*session)->RunPrefill(inputs),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "No token IDs found in preprocessed_contents."));
}

TEST_F(SessionBasicTest, RunDecode) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config, std::nullopt,
      worker_thread_pool_.get());
  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  EXPECT_OK((*session)->RunPrefill(inputs));
  auto responses = (*session)->RunDecode();
  EXPECT_OK(responses);
  // Expect a single output candidate.
  EXPECT_EQ(responses->GetTexts().size(), 1);
  // The response is " How's it going?" since "!" is the stop token which is
  // not included in the response.
  EXPECT_EQ(responses->GetTexts()[0], " How's it going?");
}

TEST_F(SessionBasicTest, RunDecodeWithMaxOutputTokens) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config, std::nullopt,
      worker_thread_pool_.get());
  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  EXPECT_OK((*session)->RunPrefill(inputs));

  auto decode_config = DecodeConfig::CreateDefault();
  decode_config.SetMaxOutputTokens(2);
  auto responses = (*session)->RunDecode(decode_config);
  ASSERT_OK(responses);
  // Expect a single output candidate.
  EXPECT_EQ(responses->GetTexts().size(), 1);
  EXPECT_EQ(responses->GetTexts()[0], " How'");
}

TEST_F(SessionBasicTest, RunDecodeWithMultipleOutputCandidates) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetNumOutputCandidates(3);
  session_config.SetSamplerBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?", "Hello World", "How's it going?"
          /*decode_tokens=*/{{224, 90, 224},
                             {24, 547, 24},
                             {8, 58, 8},
                             {66, 735, 66},
                             {246, 210, 246},
                             {18, 466, 18},
                             {2295, 2294, 2295},
                             {2294, 0, 2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config, std::nullopt,
      worker_thread_pool_.get());
  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  EXPECT_OK((*session)->RunPrefill(inputs));
  auto responses = (*session)->RunDecode();
  ASSERT_OK(responses);
  EXPECT_EQ(responses->GetTexts().size(), 3);
  // The response is " How's it going?" since "!" is the stop token which is
  // not included in the response.
  EXPECT_EQ(responses->GetTexts()[0], " How's it going?");
  EXPECT_EQ(responses->GetTexts()[1], " Hello World");
  EXPECT_EQ(responses->GetTexts()[2], " How's it going?");
}

TEST_F(SessionBasicTest, RunDecodeWithSamplerAndConstrainedDecoding) {
  // Fake constraint that expects "'s it".
  std::vector<int> expected_token_ids = {24, 8, 66, 0};
  auto constraint =
      FakeConstraint(expected_token_ids, /*vocabulary_size=*/2560);

  const std::vector<std::vector<int>> stop_token_ids = {{2294}, {0}};
  // Top P sampler.
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_k(1);
  sampler_params.set_temperature(1.0);
  sampler_params.set_p(0.5);
  sampler_params.set_seed(1);
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          /*prefill_tokens=*/{{2, 224},  // The first prefill.
                              {0}},  // The expected prefill tokens that after
                                     // stop tokens are found in decoding with
                                     // sampler. That is, the last
                                     // sampled tokens at stop condition.
                                     // "How's it going?"
          /*decode_tokens=*/{{24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config,
      /*benchmark_info=*/std::nullopt, worker_thread_pool_.get());

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("How"));
  auto decode_config = DecodeConfig::CreateDefault();
  decode_config.SetConstraint(&constraint);
  EXPECT_OK((*session)->RunPrefill(inputs));
  ASSERT_OK_AND_ASSIGN(auto responses, (*session)->RunDecode(decode_config));
  // Expect a single output candidate.
  EXPECT_EQ(responses.GetTexts().size(), 1);
  EXPECT_EQ(responses.GetTexts()[0], "'s it");
}

TEST_F(SessionBasicTest, RunDecodeWithConstrainedDecodingNoSampler) {
  // Fake constraint that expects "'s it".
  std::vector<int> expected_token_ids = {24, 8, 66, 0};
  auto constraint =
      FakeConstraint(expected_token_ids, /*vocabulary_size=*/2560);

  const std::vector<std::vector<int>> stop_token_ids = {{2294}, {0}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::GPU);
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          /*prefill_tokens=*/{{2, 224}},
          // "How's it going?"
          /*decode_tokens=*/{{24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config,
      /*benchmark_info=*/std::nullopt, worker_thread_pool_.get());

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("How"));
  auto decode_config = DecodeConfig::CreateDefault();
  decode_config.SetConstraint(&constraint);
  EXPECT_OK((*session)->RunPrefill(inputs));
  ASSERT_OK_AND_ASSIGN(auto responses, (*session)->RunDecode(decode_config));
  // Expect a single output candidate.
  EXPECT_EQ(responses.GetTexts().size(), 1);
  EXPECT_EQ(responses.GetTexts()[0], "'s it");
}

absl::AnyInvocable<void(absl::StatusOr<Responses>)> CreateTestCallback(
    bool& done_ref) {
  return [&done_ref](absl::StatusOr<Responses> responses) mutable {
    if (responses.ok() && responses->GetTexts().empty()) {
      done_ref = true;
    }
  };
}

TEST_F(SessionBasicTest, RunPrefillAsync) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.SetStartTokenId(2);
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetSamplerBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config, std::nullopt,
      worker_thread_pool_.get());

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  bool done = false;
  auto callback = CreateTestCallback(done);
  EXPECT_OK((*session)->RunPrefillAsync(inputs, std::move(callback)));
  // Wait for the async call to finish.
  EXPECT_OK(worker_thread_pool_->WaitUntilDone(absl::Seconds(100)));
  EXPECT_TRUE(done);
}

TEST_F(SessionBasicTest, RunDecodeAsync) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.SetStartTokenId(2);
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetSamplerBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config, std::nullopt,
      worker_thread_pool_.get());

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  bool done_prefill = false;
  EXPECT_OK(
      (*session)->RunPrefillAsync(inputs, CreateTestCallback(done_prefill)));
  bool done_decode = false;
  EXPECT_OK((*session)->RunDecodeAsync(CreateTestCallback(done_decode)));
  EXPECT_OK(worker_thread_pool_->WaitUntilDone(absl::Seconds(100)));
  EXPECT_TRUE(done_prefill);
  EXPECT_TRUE(done_decode);
}

TEST_F(SessionBasicTest, RunDecodeAsyncWithSamplerAndConstrainedDecoding) {
  // Fake constraint that expects "'s it".
  std::vector<int> expected_token_ids = {24, 8, 66, 0};
  auto constraint =
      FakeConstraint(expected_token_ids, /*vocabulary_size=*/2560);

  const std::vector<std::vector<int>> stop_token_ids = {{2294}, {0}};
  // Top P sampler.
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_k(1);
  sampler_params.set_temperature(1.0);
  sampler_params.set_p(0.5);
  sampler_params.set_seed(1);
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          /*prefill_tokens=*/{{2, 224},  // The first prefill.
                              {0}},  // The expected prefill tokens that after
                                     // stop tokens are found in decoding with
                                     // sampler. That is, the last
                                     // sampled tokens at stop condition.
                                     // "How's it going?"
          /*decode_tokens=*/{{24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config,
      /*benchmark_info=*/std::nullopt, worker_thread_pool_.get());

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("How"));
  bool done_prefill = false;
  EXPECT_OK(
      (*session)->RunPrefillAsync(inputs, CreateTestCallback(done_prefill)));

  absl::Status status;
  std::vector<std::string> texts;
  absl::Notification done_decode = absl::Notification();
  auto decode_config = DecodeConfig::CreateDefault();
  decode_config.SetConstraint(&constraint);
  EXPECT_OK((*session)->RunDecodeAsync(
      CreateStreamingTestCallback(status, texts, done_decode), decode_config));

  done_decode.WaitForNotification();
  EXPECT_OK(status);
  EXPECT_EQ(texts.size(), 3);
  EXPECT_THAT(texts, testing::ElementsAre("'", "s", " it"));
}

TEST_F(SessionBasicTest, RunDecodeAsyncWithConstrainedDecodingNoSampler) {
  // Fake constraint that expects "'s it".
  std::vector<int> expected_token_ids = {24, 8, 66, 0};
  auto constraint =
      FakeConstraint(expected_token_ids, /*vocabulary_size=*/2560);

  const std::vector<std::vector<int>> stop_token_ids = {{2294}, {0}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          /*prefill_tokens=*/{{2, 224}},
          // "How's it going?"
          /*decode_tokens=*/{{24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config,
      /*benchmark_info=*/std::nullopt, worker_thread_pool_.get());

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("How"));
  bool done_prefill = false;
  EXPECT_OK(
      (*session)->RunPrefillAsync(inputs, CreateTestCallback(done_prefill)));

  absl::Status status;
  std::vector<std::string> texts;
  absl::Notification done_decode = absl::Notification();
  auto decode_config = DecodeConfig::CreateDefault();
  decode_config.SetConstraint(&constraint);
  EXPECT_OK((*session)->RunDecodeAsync(
      CreateStreamingTestCallback(status, texts, done_decode), decode_config));

  done_decode.WaitForNotification();
  EXPECT_OK(status);
  EXPECT_EQ(texts.size(), 3);
  EXPECT_THAT(texts, testing::ElementsAre("'", "s", " it"));
}

TEST_F(SessionBasicTest, RunTextScoringEmptyTargetTextFailure) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config, std::nullopt,
      worker_thread_pool_.get());
  std::vector<absl::string_view> target_text;
  EXPECT_THAT((*session)->RunTextScoring(target_text,
                                         /*store_token_lengths=*/false),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Target text size should be 1."));
}

TEST_F(SessionBasicTest, RunTextScoringMultipleTargetTextFailure) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config, std::nullopt,
      worker_thread_pool_.get());
  std::vector<absl::string_view> target_text;
  target_text.push_back("How's it going?");
  target_text.push_back("How are you?");
  EXPECT_THAT(
      (*session)->RunTextScoring(target_text, /*store_token_lengths=*/false),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Target text size should be 1."));
}

TEST_F(SessionBasicTest, RunTextScoringWithoutTokenLengthsSuccess) {
  const auto responses_without_token_lengths = RunTextScoring(
      // "Hello World!"
      /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
      // "How's it going?"
      /*decode_tokens=*/{{224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}},
      /*input_prompt=*/"Hello World!",
      /*target_text=*/"How's it going?", sampler_params_,
      /*store_token_lengths=*/false, tokenizer_.get(),
      worker_thread_pool_.get());
  EXPECT_OK(responses_without_token_lengths);
  // Expect a single output candidate with score 0.0f.
  EXPECT_EQ(responses_without_token_lengths->GetScorerOutputs().size(), 1);
  EXPECT_EQ(responses_without_token_lengths->GetScorerOutputs()[0].score, 0.0f);
  EXPECT_FALSE(responses_without_token_lengths->GetScorerOutputs()[0]
                   .option_text_token_length.has_value());
}

TEST_F(SessionBasicTest, RunTextScoringWithTokenLengthsSuccess) {
  const auto responses_with_token_lengths = RunTextScoring(
      // "Hello World!"
      /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
      // "How's it going?"
      /*decode_tokens=*/{{224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}},
      /*input_prompt=*/"Hello World!",
      /*target_text=*/"How's it going?", sampler_params_,
      /*store_token_lengths=*/true, tokenizer_.get(),
      worker_thread_pool_.get());
  EXPECT_OK(responses_with_token_lengths);
  // Expect a single output candidate with score 0.0f and token length 7.
  EXPECT_EQ(responses_with_token_lengths->GetScorerOutputs().size(), 1);
  EXPECT_EQ(responses_with_token_lengths->GetScorerOutputs()[0].score, 0.0f);
  EXPECT_TRUE(responses_with_token_lengths->GetScorerOutputs()[0]
                  .option_text_token_length.has_value());
  EXPECT_EQ(*responses_with_token_lengths->GetScorerOutputs()[0]
                 .option_text_token_length,
            7);
}

TEST_F(SessionBasicTest, RunTextScoringAsyncWithTokenLengthsSuccess) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config, std::nullopt,
      worker_thread_pool_.get());
  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  EXPECT_OK((*session)->RunPrefill(inputs));

  absl::Notification done;
  absl::StatusOr<ScoringResponses> responses;
  auto callback = [&done, &responses](absl::StatusOr<ScoringResponses> r) {
    responses = std::move(r);
    done.Notify();
  };

  std::vector<absl::string_view> target_texts;
  target_texts.push_back("How's it going?");
  ASSERT_OK((*session)->RunTextScoringAsync(target_texts, std::move(callback),
                                            /*store_token_lengths=*/true));

  done.WaitForNotification();

  EXPECT_OK(responses);
  // Expect a single output candidate with score 0.0f and token length 7.
  EXPECT_EQ(responses->GetScorerOutputs().size(), 1);
  EXPECT_EQ(responses->GetScorerOutputs()[0].score, 0.0f);
  EXPECT_TRUE(
      responses->GetScorerOutputs()[0].option_text_token_length.has_value());
  EXPECT_EQ(*responses->GetScorerOutputs()[0].option_text_token_length, 7);
}

TEST_F(SessionBasicTest, GenerateContentStream) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config,
      /*benchmark_info=*/std::nullopt, worker_thread_pool_.get());

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  absl::Status status;
  std::vector<std::string> texts;
  absl::Notification done = absl::Notification();
  EXPECT_OK((*session)->GenerateContentStream(
      inputs, CreateStreamingTestCallback(status, texts, done)));

  done.WaitForNotification();
  EXPECT_OK(status);
  EXPECT_EQ(texts.size(), 7);
  EXPECT_THAT(texts,
              testing::ElementsAre(" How", "'", "s", " it", " go", "ing", "?"));
}

TEST_F(SessionBasicTest, GenerateContentStreamEmptyInput) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config, std::nullopt,
      worker_thread_pool_.get());

  std::vector<InputData> inputs;
  absl::Status status;
  std::vector<std::string> texts;
  absl::Notification done;
  EXPECT_THAT((*session)->GenerateContentStream(
                  inputs, CreateStreamingTestCallback(status, texts, done)),
              StatusIs(absl::StatusCode::kInvalidArgument, "Input is empty."));
}

TEST_F(SessionBasicTest, GenerateContentStreamPrefillError) {
  // Configure the executor to fail at prefill.
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));

  auto* fake_executor = static_cast<FakeLlmExecutor*>(executor.get());
  fake_executor->SetPrefillStatus(absl::InternalError("Prefill failed"));

  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config, std::nullopt,
      worker_thread_pool_.get());

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  absl::Status status;
  std::vector<std::string> texts;
  absl::Notification done;
  EXPECT_OK((*session)->GenerateContentStream(
      inputs, CreateStreamingTestCallback(status, texts, done)));

  done.WaitForNotification();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal, "Prefill failed"));
}

TEST_F(SessionBasicTest, GenerateContentStreamDecodeError) {
  // Configure the executor to fail at decode.
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto* fake_executor = static_cast<FakeLlmExecutor*>(executor.get());
  fake_executor->SetDecodeStatus(absl::InternalError("Decode failed"));

  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config, std::nullopt,
      worker_thread_pool_.get());

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  absl::Status status;
  std::vector<std::string> texts;
  absl::Notification done;
  EXPECT_OK((*session)->GenerateContentStream(
      inputs, CreateStreamingTestCallback(status, texts, done)));

  done.WaitForNotification();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal, "Decode failed"));
}

TEST_F(SessionBasicTest, ProcessAndCombineContentsSingleText) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  session_config.GetMutablePromptTemplates().mutable_user()->set_prefix(
      "<test>User\n");
  session_config.GetMutablePromptTemplates().mutable_user()->set_suffix(
      "<end>\n");
  session_config.GetMutablePromptTemplates().mutable_model()->set_prefix(
      "<test>Model\n");
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SessionBasic::Create(executor.get(), tokenizer_.get(),
                           /*vision_executor=*/nullptr,
                           /*audio_executor=*/nullptr, session_config,
                           std::nullopt, worker_thread_pool_.get()));

  std::vector<InputData> preprocessed_contents;
  ASSERT_OK_AND_ASSIGN(auto ids_buffer, tokenizer_->TokenIdsToTensorBuffer(
                                            {90, 547, 58, 735, 210, 466}));
  preprocessed_contents.emplace_back(InputText(std::move(ids_buffer)));

  auto executor_input_or =
      session->ProcessAndCombineContents(preprocessed_contents);
  ASSERT_OK(executor_input_or);

  ASSERT_OK_AND_ASSIGN(auto text_data, executor_input_or->GetTextDataPtr());
  ASSERT_NE(text_data, nullptr);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto token_ids_span,
      ReferTensorBufferAsSpan<int>(text_data->GetTokenIds()));
  EXPECT_THAT(std::vector<int>(token_ids_span.begin(), token_ids_span.end()),
              testing::ElementsAre(90, 547, 58, 735, 210, 466));
}

TEST_F(SessionBasicTest, ProcessAndCombineContentsMultiText) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  session_config.GetMutablePromptTemplates().mutable_user()->set_prefix(
      "<test>User");
  session_config.GetMutablePromptTemplates().mutable_user()->set_suffix(
      "<end>\n");
  session_config.GetMutablePromptTemplates().mutable_model()->set_prefix(
      "<test>Model\n");
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SessionBasic::Create(executor.get(), tokenizer_.get(),
                           /*vision_executor=*/nullptr,
                           /*audio_executor=*/nullptr, session_config,
                           std::nullopt, worker_thread_pool_.get()));

  std::vector<InputData> preprocessed_contents;
  LITERT_ASSERT_OK_AND_ASSIGN(auto ids_buffer1,
                              tokenizer_->TokenIdsToTensorBuffer({90, 547}));
  preprocessed_contents.emplace_back(InputText(std::move(ids_buffer1)));
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto ids_buffer2,
      tokenizer_->TokenIdsToTensorBuffer({58, 735, 210, 466}));
  preprocessed_contents.emplace_back(InputText(std::move(ids_buffer2)));

  auto executor_input_or =
      session->ProcessAndCombineContents(preprocessed_contents);
  ASSERT_OK(executor_input_or);

  ASSERT_OK_AND_ASSIGN(auto text_data, executor_input_or->GetTextDataPtr());
  ASSERT_NE(text_data, nullptr);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto token_ids_span,
      ReferTensorBufferAsSpan<int>(text_data->GetTokenIds()));
  EXPECT_THAT(std::vector<int>(token_ids_span.begin(), token_ids_span.end()),
              testing::ElementsAre(90, 547, 58, 735, 210, 466));
}

TEST_F(SessionBasicTest, ProcessAndCombineContentsEmptyFails) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetSamplerBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SessionBasic::Create(executor.get(), tokenizer_.get(),
                           /*vision_executor=*/nullptr,
                           /*audio_executor=*/nullptr, session_config,
                           std::nullopt, worker_thread_pool_.get()));

  std::vector<InputData> preprocessed_contents;
  auto result = session->ProcessAndCombineContents(preprocessed_contents);
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument,
                               "No token IDs found in preprocessed_contents."));
}

TEST_F(SessionBasicTest, RunIncrementalPrefillWithDecode) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.GetMutablePromptTemplates().mutable_user()->set_prefix(
      "User:");
  session_config.GetMutablePromptTemplates().mutable_user()->set_suffix(
      "[END]");
  session_config.GetMutablePromptTemplates().mutable_model()->set_prefix(
      "Model:");
  session_config.GetMutableLlmModelType().mutable_gemma3n();

  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          /*prefill_tokens=*/
          {
              {2, 423, 8, 179, 29, 207, 19, 547, 58},  // prefill chunk 1.1
              {735, 210, 466, 2294},                   // prefill chunk 1.2
              {433, 2172, 1920, 432, 197, 979, 3076,
               29},  // prefill ran before decode with turn change template
              {423, 8, 179, 29, 207, 19, 547, 58, 735, 210, 466,
               2294},  // prefill chunk 2.1
              {433, 2172, 1920, 432, 197, 979, 3076,
               29},  // prefill ran before decode with turn change template
          },
          /*decode_tokens=*/
          {{1}, {2}, {3}, {2294}, {1}, {2}, {3}, {2294}}));
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SessionBasic::Create(executor.get(), tokenizer_.get(),
                           /*vision_executor=*/nullptr,
                           /*audio_executor=*/nullptr, session_config,
                           std::nullopt, worker_thread_pool_.get()));

  {
    std::vector<InputData> inputs;
    inputs.emplace_back(InputText("Hello "));
    EXPECT_OK(session->RunPrefill(inputs));
  }
  {
    std::vector<InputData> inputs;
    inputs.emplace_back(InputText("World!"));
    EXPECT_OK(session->RunPrefill(inputs));
  }
  {
    EXPECT_OK(session->RunDecode());
  }
  {
    std::vector<InputData> inputs;
    inputs.emplace_back(InputText("Hello World!"));
    EXPECT_OK(session->RunPrefill(inputs));
  }
  {
    EXPECT_OK(session->RunDecode());
  }
}

// TODO: b/441514829 - Enable the tests on Windows once the bug is fixed.
#if !defined(WIN32) && !defined(_WIN32) && !defined(__WIN32__) && \
    !defined(__NT__) && !defined(_WIN64)
TEST_F(SessionBasicTest, ProcessAndCombineContentsAudioSuccess) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  session_config.GetMutableLlmModelType().mutable_gemma3n();
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(env,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // "Hello World!<start_of_audio>"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294, 256000}},
          // "How's it going?"
          /*decode_tokens=*/
          {{224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}},
          /*audio_embedding=*/
          std::vector<float>(kExpectedAudioEmbedding.begin(),
                             kExpectedAudioEmbedding.end())));
  ASSERT_OK_AND_ASSIGN(
      auto session, SessionBasic::Create(
                        executor.get(), tokenizer_.get(),
                        /*vision_executor=*/nullptr,
                        /*audio_executor=*/audio_executor.get(), session_config,
                        std::nullopt, worker_thread_pool_.get()));

  std::vector<InputData> preprocessed_contents;
  ASSERT_OK_AND_ASSIGN(
      auto ids_buffer,
      tokenizer_->TokenIdsToTensorBuffer({90, 547, 58, 735, 210, 466, 256000}));
  preprocessed_contents.emplace_back(InputText(std::move(ids_buffer)));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer mel_spectrogram_data,
      CopyToTensorBuffer<float>(
          mel_spectrogram_data,
          {1, kSpectrogramSequenceLength, kSpectrogramFrequencySlots}));
  InputAudio input_audio(std::move(mel_spectrogram_data));
  preprocessed_contents.emplace_back(std::move(input_audio));
  preprocessed_contents.emplace_back(InputAudioEnd());

  ASSERT_OK_AND_ASSIGN(
      auto result, session->ProcessAndCombineContents(preprocessed_contents));

  ASSERT_OK_AND_ASSIGN(auto text_data, result.GetTextDataPtr());
  ASSERT_NE(text_data, nullptr);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto token_ids_span,
      ReferTensorBufferAsSpan<int>(text_data->GetTokenIds()));
  // The input to audio executor has length 10.
  // The processed audio embedding has length 5.
  EXPECT_THAT(std::vector<int>(token_ids_span.begin(), token_ids_span.end()),
              testing::ElementsAre(90, 547, 58, 735, 210, 466, 256000, -2, -2,
                                   -2, -2, -2, -4));
}

TEST_F(SessionBasicTest, ProcessAndCombineContentsTextAndAudioSuccess) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.GetMutablePromptTemplates().mutable_user()->set_prefix(
      "User:");
  session_config.GetMutablePromptTemplates().mutable_user()->set_suffix(
      "[END]");
  session_config.GetMutablePromptTemplates().mutable_model()->set_prefix(
      "Model:");
  session_config.GetMutableLlmModelType().mutable_gemma3n();

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(env,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // "User:Hello World!<start_of_audio>[END]Model:"
          /*prefill_tokens=*/{{2,   423, 8,   179, 29,  207,  19,
                               547, 58,  735, 210, 466, 2294, 256000,
                               -2,  -2,  -2,  -2,  -2,  -4},
                              {433, 2172, 1920, 432, 197, 979, 3076, 29}},
          // "How's it going?"
          /*decode_tokens=*/
          {{224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}},
          /*audio_embedding=*/
          std::vector<float>(kExpectedAudioEmbedding.begin(),
                             kExpectedAudioEmbedding.end())));
  ASSERT_OK_AND_ASSIGN(
      auto session, SessionBasic::Create(
                        executor.get(), tokenizer_.get(),
                        /*vision_executor=*/nullptr,
                        /*audio_executor=*/audio_executor.get(), session_config,
                        std::nullopt, worker_thread_pool_.get()));

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!<start_of_audio>"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer mel_spectrogram_data,
      CopyToTensorBuffer<float>(
          mel_spectrogram_data,
          {1, kSpectrogramSequenceLength, kSpectrogramFrequencySlots}));
  InputAudio input_audio(std::move(mel_spectrogram_data));
  inputs.emplace_back(std::move(input_audio));
  inputs.emplace_back(InputAudioEnd());
  EXPECT_OK(session->RunPrefill(inputs));
}

TEST_F(SessionBasicTest, ProcessAndCombineContentsTextAudioTextSuccess) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.GetMutablePromptTemplates().mutable_user()->set_prefix(
      "User:");
  session_config.GetMutablePromptTemplates().mutable_user()->set_suffix(
      "[END]");
  session_config.GetMutablePromptTemplates().mutable_model()->set_prefix(
      "Model:");
  session_config.GetMutableLlmModelType().mutable_gemma3n();

  LITERT_ASSERT_OK_AND_ASSIGN(
      auto env, Environment::Create(std::vector<Environment::Option>()));
  ASSERT_OK_AND_ASSIGN(
      auto audio_executor,
      CreateAudioExecutor(env,
                          (std::filesystem::path(::testing::SrcDir()) /
                           std::string(kTestAudioModelPath))
                              .string(),
                          /*max_sequence_length=*/0, Backend::CPU));
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // "User:Hello World!<start_of_audio>What does the audio say?"
          // "[END]Model:"
          /*prefill_tokens=*/
          {{2,   423,  8,      179, 29,   207, 19, 547, 58,  735, 210,
            466, 2294, 256000, -2,  -2,   -2,  -2, -2,  -4,  583, 378,
            844, 166,  3,      14,  1252, 54,  58, 626, 2295},
           {3995, 2172, 1920, 432, 197, 979, 3076, 29}},

          // "How's it going?"
          /*decode_tokens=*/
          {{224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}},
          /*audio_embedding=*/
          std::vector<float>(kExpectedAudioEmbedding.begin(),
                             kExpectedAudioEmbedding.end())));

  ASSERT_OK_AND_ASSIGN(
      auto session, SessionBasic::Create(
                        executor.get(), tokenizer_.get(),
                        /*vision_executor=*/nullptr,
                        /*audio_executor=*/audio_executor.get(), session_config,
                        std::nullopt, worker_thread_pool_.get()));

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!<start_of_audio>"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer mel_spectrogram_data,
      CopyToTensorBuffer<float>(
          mel_spectrogram_data,
          {1, kSpectrogramSequenceLength, kSpectrogramFrequencySlots}));
  InputAudio input_audio(std::move(mel_spectrogram_data));
  inputs.emplace_back(std::move(input_audio));
  inputs.emplace_back(InputAudioEnd());
  inputs.emplace_back(InputText("What does the audio say?"));
  EXPECT_OK(session->RunPrefill(inputs));
}
#endif  // !defined(WIN32) && !defined(_WIN32) && !defined(__WIN32__) && \
        // !defined(__NT__) && !defined(_WIN64)

TEST_F(SessionBasicTest, GenerateContentStreamWithCancellation) {
  // Configure the executor to have a delay to simulate a long-running task.
  ASSERT_OK_AND_ASSIGN(
      auto fake_executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  fake_executor->SetDecodeDelay(absl::Milliseconds(200));

  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  auto session =
      SessionBasic::Create(fake_executor.get(), tokenizer_.get(),
                           /*vision_executor=*/nullptr,
                           /*audio_executor=*/nullptr, session_config,
                           std::nullopt, worker_thread_pool_.get());
  ASSERT_OK(session);

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));

  absl::Status status;
  std::vector<std::string> responses;
  absl::Notification done;

  (*session)
      ->GenerateContentStream(
          inputs, CreateStreamingTestCallback(status, responses, done,
                                              /*delay_on_next=*/true))
      .IgnoreError();

  // Wait for a short time to ensure the decoding has started.
  absl::SleepFor(absl::Milliseconds(100));

  // Cancel the process.
  (*session)->CancelProcess();

  // Wait for the callback to be done.
  done.WaitForNotification();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kCancelled));
}

TEST_F(SessionBasicTest, GenerateContentStreamAndDelete) {
  // Configure the executor to have a delay to simulate a long-running task.
  ASSERT_OK_AND_ASSIGN(
      auto fake_executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  fake_executor->SetDecodeDelay(absl::Milliseconds(200));

  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  auto session =
      SessionBasic::Create(fake_executor.get(), tokenizer_.get(),
                           /*vision_executor=*/nullptr,
                           /*audio_executor=*/nullptr, session_config,
                           std::nullopt, worker_thread_pool_.get());
  ASSERT_OK(session);

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));

  absl::Status status;
  std::vector<std::string> responses;
  absl::Notification done;

  (*session)
      ->GenerateContentStream(
          inputs, CreateStreamingTestCallback(status, responses, done,
                                              /*delay_on_next=*/true))
      .IgnoreError();

  // Reset the session to trigger destructor.
  session->reset();

  // Wait for thread pool to finish. If destructor does not wait, this might
  // cause use-after-free.
  worker_thread_pool_->WaitUntilDone(absl::Seconds(10)).IgnoreError();
  if (!done.HasBeenNotified()) {
    done.Notify();
  }
}

class SessionBasicCancellationTest : public testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    auto tokenizer = ExtendedTokenizer::CreateFromFile(
        (std::filesystem::path(::testing::SrcDir()) /
         std::string(kTestdataDir) / "sentencepiece.model")
            .string());
    ASSERT_OK(tokenizer);
    tokenizer.value()->SetExtendedToken(256000, "<start_of_audio>");
    tokenizer_ = std::move(*tokenizer);
    sampler_params_.set_type(proto::SamplerParameters::TYPE_UNSPECIFIED);
    // Creating the thread pool of a single thread to execute the works.
    worker_thread_pool_ = std::make_unique<ThreadPool>(/*name_prefix=*/"engine",
                                                       /*max_num_threads=*/1);
  }
  bool use_benchmark_info_ = GetParam();
  std::unique_ptr<Tokenizer> tokenizer_;
  proto::SamplerParameters sampler_params_;
  std::unique_ptr<ThreadPool> worker_thread_pool_;
};

TEST_P(SessionBasicCancellationTest,
       GenerateContentStreamCancelThenGenerateWithBenchmark) {
  // Configure the executor to have a delay to simulate a long-running task.
  ASSERT_OK_AND_ASSIGN(
      auto fake_executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294},
                              // The second prefill doesn't have bos token.
                              {90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  fake_executor->SetDecodeDelay(absl::Milliseconds(200));

  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);

  std::optional<BenchmarkInfo> benchmark_info;
  if (use_benchmark_info_) {
    proto::BenchmarkParams benchmark_params;
    benchmark_info.emplace(benchmark_params);
  }
  auto session =
      SessionBasic::Create(fake_executor.get(), tokenizer_.get(),
                           /*vision_executor=*/nullptr,
                           /*audio_executor=*/nullptr, session_config,
                           benchmark_info, worker_thread_pool_.get());
  ASSERT_OK(session);

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));

  absl::Status status;
  std::vector<std::string> responses;
  absl::Notification done1;

  (*session)
      ->GenerateContentStream(
          inputs, CreateStreamingTestCallback(status, responses, done1,
                                              /*delay_on_next=*/true))
      .IgnoreError();

  // Cancel the process.
  (*session)->CancelProcess();

  // Wait for the callback to be done.
  done1.WaitForNotification();
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kCancelled));

  // Generate again after cancellation.
  // The second generation should succeed.
  status = absl::OkStatus();
  responses.clear();
  absl::Notification done2;
  (*session)
      ->GenerateContentStream(
          inputs, CreateStreamingTestCallback(status, responses, done2,
                                              /*delay_on_next=*/true))
      .IgnoreError();
  done2.WaitForNotification();
  EXPECT_OK(status);
  // Reset worker thread pool to stop accessing session and fake executor.
  // We need to reset session first because it holds a reference to the
  // thread pool.
  (*session).reset();
  worker_thread_pool_.reset();
}

INSTANTIATE_TEST_SUITE_P(SessionBasicCancellationTest,
                         SessionBasicCancellationTest, testing::Bool(),
                         testing::PrintToStringParamName());

TEST_F(SessionBasicTest, GenerateContentStreamOnCancelledSession) {
  ASSERT_OK_AND_ASSIGN(
      auto fake_executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  auto session =
      SessionBasic::Create(fake_executor.get(), tokenizer_.get(),
                           /*vision_executor=*/nullptr,
                           /*audio_executor=*/nullptr, session_config,
                           std::nullopt, worker_thread_pool_.get());
  ASSERT_OK(session);

  (*session)->CancelProcess();

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  absl::Status status;
  std::vector<std::string> responses;
  absl::Notification done;
  // The session is cancelled, so the call should return with a kCancelled
  // error.
  EXPECT_OK((*session)->GenerateContentStream(
      inputs, CreateStreamingTestCallback(status, responses, done)));
  // Wait for the callback to be done.
  done.WaitForNotification();
  EXPECT_OK(status);
}

TEST_F(SessionBasicTest,
       TestBenchmarkModeWithoutNumPrefillTokensRespectPromptTemplate) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  session_config.GetMutablePromptTemplates().mutable_user()->set_prefix(
      "<test>User\n");
  session_config.GetMutablePromptTemplates().mutable_user()->set_suffix(
      "<end>\n");
  session_config.GetMutablePromptTemplates().mutable_model()->set_prefix(
      "<test>Model\n");

  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // Expected tokens: "</s><test>User\nHello World!" +
          // "<end>\n<test>Model\n"
          /*prefill_tokens=*/{{2, 4, 0, 39, 637, 0, 3328, 8, 179, 90, 547, 58,
                               735, 210, 466, 2294},
                              {0, 40, 23, 0, 4, 0, 39, 637, 0, 197, 979, 3076}},
          /*decode_tokens=*/{{224}}));

  proto::BenchmarkParams benchmark_params;
  BenchmarkInfo benchmark_info(benchmark_params);

  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config, benchmark_info,
      worker_thread_pool_.get());
  ASSERT_OK(session);

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  EXPECT_OK((*session)->RunPrefill(inputs));
  EXPECT_EQ((*session)->GetBenchmarkInfo()->GetTotalPrefillTurns(), 1);
}

TEST_F(SessionBasicTest,
       TestBenchmarkModeWithNumPrefillTokensIgnorePromptTemplate) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  session_config.GetMutablePromptTemplates().mutable_user()->set_prefix(
      "<test>User\n");
  session_config.GetMutablePromptTemplates().mutable_user()->set_suffix(
      "<end>\n");
  session_config.GetMutablePromptTemplates().mutable_model()->set_prefix(
      "<test>Model\n");

  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // Expected tokens: "Hello World!" (No templates)
          /*prefill_tokens=*/{{90, 547, 58, 735, 210, 466, 2294}},
          /*decode_tokens=*/{{224}}));

  proto::BenchmarkParams benchmark_params;
  benchmark_params.set_num_prefill_tokens(7);
  BenchmarkInfo benchmark_info(benchmark_params);

  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config, benchmark_info,
      worker_thread_pool_.get());
  ASSERT_OK(session);

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));
  EXPECT_OK((*session)->RunPrefill(inputs));
  EXPECT_EQ((*session)->GetBenchmarkInfo()->GetTotalPrefillTurns(), 1);
}

TEST_F(SessionBasicTest,
       GenerateContentStreamWithSamplerAndConstrainedDecoding) {
  // Fake constraint that expects "'s it".
  std::vector<int> expected_token_ids = {24, 8, 66, 0};
  auto constraint =
      FakeConstraint(expected_token_ids, /*vocabulary_size=*/2560);

  const std::vector<std::vector<int>> stop_token_ids = {{2294}, {0}};
  // Top P sampler.
  proto::SamplerParameters sampler_params;
  sampler_params.set_type(proto::SamplerParameters::TOP_P);
  sampler_params.set_k(1);
  sampler_params.set_temperature(1.0);
  sampler_params.set_p(0.5);
  sampler_params.set_seed(1);
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          /*prefill_tokens=*/{{2, 224},  // The first prefill.
                              {0}},  // The expected prefill tokens that after
                                     // stop tokens are found in decoding with
                                     // sampler. That is, the last
                                     // sampled tokens at stop condition.
                                     // "How's it going?"
          /*decode_tokens=*/{{24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(),
      /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config,
      /*benchmark_info=*/std::nullopt, worker_thread_pool_.get());

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("How"));

  absl::Status status;
  std::vector<std::string> texts;
  absl::Notification done_decode = absl::Notification();
  auto decode_config = DecodeConfig::CreateDefault();
  decode_config.SetConstraint(&constraint);
  EXPECT_OK((*session)->GenerateContentStream(
      inputs, CreateStreamingTestCallback(status, texts, done_decode),
      decode_config));

  done_decode.WaitForNotification();
  EXPECT_OK(status);
  EXPECT_EQ(texts.size(), 3);
  EXPECT_THAT(texts, testing::ElementsAre("'", "s", " it"));
}

TEST_F(SessionBasicTest, SaveAndRewindCheckpoint) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);
  session_config.SetSamplerBackend(Backend::CPU);

  // Set up a fake executor.
  ASSERT_OK_AND_ASSIGN(
      auto executor,
      CreateFakeLlmExecutor(
          // "Hello World!"
          /*prefill_tokens=*/{{2, 90, 547, 58, 735, 210, 466, 2294},
                              {2, 90, 547, 58, 735, 210, 466, 2294}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));

  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config, std::nullopt,
      worker_thread_pool_.get());
  ASSERT_OK(session);

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("Hello World!"));

  // Save checkpoint before prefill.
  EXPECT_OK((*session)->SaveCheckpoint("start"));
  EXPECT_THAT((*session)->GetCurrentStep(), testing::status::IsOkAndHolds(0));

  EXPECT_OK((*session)->RunPrefill(inputs));

  // Save checkpoint after prefill.
  EXPECT_OK((*session)->SaveCheckpoint("prefill"));
  EXPECT_THAT((*session)->GetCurrentStep(), testing::status::IsOkAndHolds(8));

  // Rewind should succeed and restore the state.
  EXPECT_OK((*session)->RewindToCheckpoint("start"));
  EXPECT_THAT((*session)->GetCurrentStep(), testing::status::IsOkAndHolds(0));

  // The "prefill" checkpoint should no longer exist.
  EXPECT_THAT((*session)->RewindToCheckpoint("prefill"),
              testing::status::StatusIs(absl::StatusCode::kNotFound));

  // Checkpoint that doesn't exist should fail.
  EXPECT_THAT((*session)->RewindToCheckpoint("nonexistent"),
              testing::status::StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace litert::lm
