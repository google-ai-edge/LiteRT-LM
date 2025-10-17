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
#include "runtime/util/status_macros.h"  // NOLINT
#include "runtime/util/tensor_buffer_util.h"
#include "runtime/util/test_utils.h"  // NOLINT

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

class StreamingTestCallbacks : public InferenceCallbacks {
 public:
  StreamingTestCallbacks(absl::Status& status, std::vector<std::string>& texts,
                         absl::Notification& done)
      : status_(status), texts_(texts), done_(done) {}

  void OnNext(const Responses& responses) override {
    ASSERT_EQ(responses.GetNumOutputCandidates(), 1);
    texts_.push_back(std::string(*responses.GetResponseTextAt(0)));
  }

  void OnError(const absl::Status& status) override {
    status_ = status;
    done_.Notify();
  }

  void OnDone() override { done_.Notify(); }

 private:
  absl::Status& status_;
  std::vector<std::string>& texts_;
  absl::Notification& done_;
};

class CancelledTestCallbacks : public StreamingTestCallbacks {
 public:
  CancelledTestCallbacks(absl::Status& status, std::vector<std::string>& texts,
                         absl::Notification& done)
      : StreamingTestCallbacks(status, texts, done) {}

  void OnNext(const Responses& responses) override {
    // Wait for a while to allow cancellation happens in the middle of the
    // decode step.
    absl::SleepFor(absl::Milliseconds(50));
    StreamingTestCallbacks::OnNext(responses);
  }
};

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
  EXPECT_EQ(responses->GetNumOutputCandidates(), 1);
  // The response is " How's it going?" since "!" is the stop token which is
  // not included in the response.
  EXPECT_EQ(*(responses->GetResponseTextAt(0)), " How's it going?");
  EXPECT_THAT(responses->GetResponseTextAt(1),
              StatusIs(absl::StatusCode::kInvalidArgument));
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
  EXPECT_OK(responses);
  EXPECT_EQ(responses->GetNumOutputCandidates(), 3);
  // The response is " How's it going?" since "!" is the stop token which is
  // not included in the response.
  EXPECT_EQ(*(responses->GetResponseTextAt(0)), " How's it going?");
  EXPECT_EQ(*(responses->GetResponseTextAt(1)), " Hello World");
  EXPECT_EQ(*(responses->GetResponseTextAt(2)), " How's it going?");
  EXPECT_THAT(responses->GetResponseTextAt(3),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(SessionBasicTest, RunDecodeWithSamplerAndConstrainedDecoding) {
  // Fake constraint that expects " How's it".
  std::vector<int> expected_token_ids = {2, 224, 24, 8, 66, 0};
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
          /*prefill_tokens=*/{{2, 224}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
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
  EXPECT_EQ(responses.GetNumOutputCandidates(), 1);
  EXPECT_EQ(*responses.GetResponseTextAt(0), " How's it");
}

TEST_F(SessionBasicTest, RunDecodeWithConstrainedDecodingNoSampler) {
  // Fake constraint that expects " How's it".
  std::vector<int> expected_token_ids = {2, 224, 24, 8, 66, 0};
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
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
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
  EXPECT_EQ(responses.GetNumOutputCandidates(), 1);
  EXPECT_EQ(*responses.GetResponseTextAt(0), " How's it");
}

class TestCallbacks : public InferenceCallbacks {
 public:
  explicit TestCallbacks(bool& done) : done_(done) {}

  void OnDone() override { done_ = true; }

 private:
  bool& done_;
};

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
  auto callbacks = std::make_unique<TestCallbacks>(done);
  EXPECT_OK((*session)->RunPrefillAsync(inputs, std::move(callbacks)));
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
  EXPECT_OK((*session)->RunPrefillAsync(
      inputs, std::make_unique<TestCallbacks>(done_prefill)));
  bool done_decode = false;
  EXPECT_OK(
      (*session)->RunDecodeAsync(std::make_unique<TestCallbacks>(done_decode)));
  EXPECT_OK(worker_thread_pool_->WaitUntilDone(absl::Seconds(100)));
  EXPECT_TRUE(done_prefill);
  EXPECT_TRUE(done_decode);
}

TEST_F(SessionBasicTest, RunDecodeAsyncWithSamplerAndConstrainedDecoding) {
  // Fake constraint that expects " How's it".
  std::vector<int> expected_token_ids = {2, 224, 24, 8, 66, 0};
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
          /*prefill_tokens=*/{{2, 224}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config,
      /*benchmark_info=*/std::nullopt, worker_thread_pool_.get());

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("How"));
  bool done_prefill = false;
  EXPECT_OK((*session)->RunPrefillAsync(
      inputs, std::make_unique<TestCallbacks>(done_prefill)));

  absl::Status status;
  std::vector<std::string> texts;
  absl::Notification done_decode = absl::Notification();
  auto decode_config = DecodeConfig::CreateDefault();
  decode_config.SetConstraint(&constraint);
  EXPECT_OK((*session)->RunDecodeAsync(
      std::make_unique<StreamingTestCallbacks>(status, texts, done_decode),
      decode_config));

  done_decode.WaitForNotification();
  EXPECT_OK(status);
  EXPECT_EQ(texts.size(), 4);
  EXPECT_THAT(texts, testing::ElementsAre(" How", "'", "s", " it"));
}

TEST_F(SessionBasicTest, RunDecodeAsyncWithConstrainedDecodingNoSampler) {
  // Fake constraint that expects " How's it".
  std::vector<int> expected_token_ids = {2, 224, 24, 8, 66, 0};
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
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
      /*audio_executor=*/nullptr, session_config,
      /*benchmark_info=*/std::nullopt, worker_thread_pool_.get());

  std::vector<InputData> inputs;
  inputs.emplace_back(InputText("How"));
  bool done_prefill = false;
  EXPECT_OK((*session)->RunPrefillAsync(
      inputs, std::make_unique<TestCallbacks>(done_prefill)));

  absl::Status status;
  std::vector<std::string> texts;
  absl::Notification done_decode = absl::Notification();
  auto decode_config = DecodeConfig::CreateDefault();
  decode_config.SetConstraint(&constraint);
  EXPECT_OK((*session)->RunDecodeAsync(
      std::make_unique<StreamingTestCallbacks>(status, texts, done_decode),
      decode_config));

  done_decode.WaitForNotification();
  EXPECT_OK(status);
  EXPECT_EQ(texts.size(), 4);
  EXPECT_THAT(texts, testing::ElementsAre(" How", "'", "s", " it"));
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
  EXPECT_THAT((*session)->RunTextScoring(target_text),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument,
                                        "Target text is empty."));
}

TEST_F(SessionBasicTest, RunTextScoringSuccess) {
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
  std::vector<absl::string_view> target_text;
  target_text.push_back("How's it going?");
  auto responses = (*session)->RunTextScoring(target_text);
  EXPECT_OK(responses);
  EXPECT_EQ(responses->GetNumOutputCandidates(), 1);
  EXPECT_EQ(*(responses->GetScoreAt(0)), 0.0f);
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
      inputs, std::make_unique<StreamingTestCallbacks>(status, texts, done)));

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
                  inputs, std::make_unique<StreamingTestCallbacks>(
                              status, texts, done)),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument,
                                        "Input is empty."));
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
      inputs, std::make_unique<StreamingTestCallbacks>(status, texts, done)));

  done.WaitForNotification();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status, testing::status::StatusIs(absl::StatusCode::kInternal,
                                                "Prefill failed"));
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
      inputs, std::make_unique<StreamingTestCallbacks>(status, texts, done)));

  done.WaitForNotification();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status, testing::status::StatusIs(absl::StatusCode::kInternal,
                                                "Decode failed"));
}

TEST_F(SessionBasicTest, ApplyPromptTemplatesFails) {
  const std::vector<std::vector<int>> stop_token_ids = {{2294}};
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.GetMutableSamplerParams() = sampler_params_;
  session_config.GetMutableStopTokenIds() = stop_token_ids;
  session_config.SetStartTokenId(2);  // Corresponds to "</s>"
  session_config.SetSamplerBackend(Backend::CPU);
  ASSERT_OK_AND_ASSIGN(auto executor, CreateFakeLlmExecutor(
                                          /*prefill_tokens=*/{{}},
                                          /*decode_tokens=*/{{}}));
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SessionBasic::Create(executor.get(), tokenizer_.get(),
                           /*vision_executor=*/nullptr,
                           /*audio_executor=*/nullptr, session_config,
                           std::nullopt, worker_thread_pool_.get()));

  // Test Case 1: Input text starts with the BOS token string.
  std::vector<InputData> inputs_with_bos;
  inputs_with_bos.emplace_back(InputText("</s>Hello World!"));
  EXPECT_THAT(
      session->ApplyPromptTemplates(inputs_with_bos),
      testing::status::StatusIs(absl::StatusCode::kInvalidArgument,
                                "Input contains bos control token. Control "
                                "token should not be included in the input."));

  // Test Case 2: Empty input. ApplyPromptTemplates returns an empty vector,
  // which is not an error at this stage. The error for empty content is
  // handled in ProcessAndCombineContents.
  std::vector<InputData> empty_inputs;
  ASSERT_OK_AND_ASSIGN(auto templated_empty,
                       session->ApplyPromptTemplates(empty_inputs));
  EXPECT_TRUE(templated_empty.empty());
}

TEST_F(SessionBasicTest, ApplyPromptTemplatesWithSingleTextChunk) {
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

  // Single text chunk. (is_first_chunk=true, is_last_chunk=true)
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SessionBasic::Create(executor.get(), tokenizer_.get(),
                           /*vision_executor=*/nullptr,
                           /*audio_executor=*/nullptr, session_config,
                           std::nullopt, worker_thread_pool_.get()));
  std::vector<InputData> single_chunk;
  single_chunk.emplace_back(InputText("Hello World!"));
  ASSERT_OK_AND_ASSIGN(auto templated_single,
                       session->ApplyPromptTemplates(single_chunk));
  ASSERT_EQ(templated_single.size(), 1);
  EXPECT_THAT(std::get<InputText>(templated_single[0]).GetRawTextString(),
              testing::status::IsOkAndHolds(
                  "</s><test>User\nHello World!<end>\n<test>Model\n"));
}

TEST_F(SessionBasicTest, ApplyPromptTemplatesWithTwoTextChunks) {
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

  // Two text chunks. (First chunk: is_first=true, is_last=false;
  // Second chunk: is_first=false, is_last=true)
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SessionBasic::Create(executor.get(), tokenizer_.get(),
                           /*vision_executor=*/nullptr,
                           /*audio_executor=*/nullptr, session_config,
                           std::nullopt, worker_thread_pool_.get()));
  std::vector<InputData> two_chunks;
  two_chunks.emplace_back(InputText("First"));
  two_chunks.emplace_back(InputText("Second"));
  ASSERT_OK_AND_ASSIGN(auto templated_two,
                       session->ApplyPromptTemplates(two_chunks));
  ASSERT_EQ(templated_two.size(), 2);
  EXPECT_THAT(std::get<InputText>(templated_two[0]).GetRawTextString(),
              testing::status::IsOkAndHolds("</s><test>User\nFirst"));
  EXPECT_THAT(std::get<InputText>(templated_two[1]).GetRawTextString(),
              testing::status::IsOkAndHolds("Second<end>\n<test>Model\n"));
}

TEST_F(SessionBasicTest, ApplyPromptTemplatesWithThreeTextChunks) {
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

  // Three text chunks. (Middle chunk: is_first=false, is_last=false)
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SessionBasic::Create(executor.get(), tokenizer_.get(),
                           /*vision_executor=*/nullptr,
                           /*audio_executor=*/nullptr, session_config,
                           std::nullopt, worker_thread_pool_.get()));
  std::vector<InputData> three_chunks;
  three_chunks.emplace_back(InputText("First"));
  three_chunks.emplace_back(InputText("Middle"));
  three_chunks.emplace_back(InputText("Last"));
  ASSERT_OK_AND_ASSIGN(auto templated_three,
                       session->ApplyPromptTemplates(three_chunks));
  ASSERT_EQ(templated_three.size(), 3);
  EXPECT_THAT(std::get<InputText>(templated_three[0]).GetRawTextString(),
              testing::status::IsOkAndHolds("</s><test>User\nFirst"));
  EXPECT_THAT(std::get<InputText>(templated_three[1]).GetRawTextString(),
              testing::status::IsOkAndHolds("Middle"));
  EXPECT_THAT(std::get<InputText>(templated_three[2]).GetRawTextString(),
              testing::status::IsOkAndHolds("Last<end>\n<test>Model\n"));
}

TEST_F(SessionBasicTest, ApplyPromptTemplatesWithMixedChunksTextAndImage) {
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

  // Mixed chunks - text and image. Non-text inputs are passed through.
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SessionBasic::Create(executor.get(), tokenizer_.get(),
                           /*vision_executor=*/nullptr,
                           /*audio_executor=*/nullptr, session_config,
                           std::nullopt, worker_thread_pool_.get()));
  std::vector<InputData> mixed_chunks;
  mixed_chunks.emplace_back(InputText("Text1"));
  mixed_chunks.emplace_back(InputImage("123"));
  mixed_chunks.emplace_back(InputText("Text2"));
  ASSERT_OK_AND_ASSIGN(auto templated_mixed,
                       session->ApplyPromptTemplates(mixed_chunks));
  ASSERT_EQ(templated_mixed.size(), 3);
  EXPECT_THAT(std::get<InputText>(templated_mixed[0]).GetRawTextString(),
              testing::status::IsOkAndHolds("</s><test>User\nText1"));
  EXPECT_TRUE(std::holds_alternative<InputImage>(templated_mixed[1]));
  EXPECT_THAT(std::get<InputText>(templated_mixed[2]).GetRawTextString(),
              testing::status::IsOkAndHolds("Text2<end>\n<test>Model\n"));
}

TEST_F(SessionBasicTest, ApplyPromptTemplatesWithSubsequentTurn) {
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

  // First turn is false (subsequent call).
  // The first call to ApplyPromptTemplates sets is_first_turn_ to false.
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SessionBasic::Create(executor.get(), tokenizer_.get(),
                           /*vision_executor=*/nullptr,
                           /*audio_executor=*/nullptr, session_config,
                           std::nullopt, worker_thread_pool_.get()));
  std::vector<InputData> single_chunk_again;
  single_chunk_again.emplace_back(InputText("Another turn"));
  ASSERT_OK_AND_ASSIGN(auto templated_first_turn,
                       session->ApplyPromptTemplates(single_chunk_again));
  ASSERT_EQ(templated_first_turn.size(), 1);
  EXPECT_THAT(std::get<InputText>(templated_first_turn[0]).GetRawTextString(),
              testing::status::IsOkAndHolds(
                  "</s><test>User\nAnother turn<end>\n<test>Model\n"));
  ASSERT_OK_AND_ASSIGN(auto templated_again,
                       session->ApplyPromptTemplates(single_chunk_again));
  ASSERT_EQ(templated_again.size(), 1);
  EXPECT_THAT(std::get<InputText>(templated_again[0]).GetRawTextString(),
              testing::status::IsOkAndHolds(
                  "\n<test>User\nAnother turn<end>\n<test>Model\n"));
}

TEST_F(SessionBasicTest, ApplyPromptTemplatesWithSingleImageInput) {
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

  // Single image input. Templates are applied to the first and
  // last chunks. In this case, the image input is both the first and last
  // chunks, and the text chunks (templates) will be added before and after
  // the image.
  ASSERT_OK_AND_ASSIGN(
      auto session,
      SessionBasic::Create(executor.get(), tokenizer_.get(),
                           /*vision_executor=*/nullptr,
                           /*audio_executor=*/nullptr, session_config,
                           std::nullopt, worker_thread_pool_.get()));
  std::vector<InputData> single_image;
  single_image.emplace_back(InputImage("456"));
  ASSERT_OK_AND_ASSIGN(auto templated_image,
                       session->ApplyPromptTemplates(single_image));
  ASSERT_EQ(templated_image.size(), 3);
  EXPECT_THAT(std::get<InputText>(templated_image[0]).GetRawTextString(),
              testing::status::IsOkAndHolds("</s><test>User\n"));
  EXPECT_TRUE(std::holds_alternative<InputImage>(templated_image[1]));
  EXPECT_THAT(std::get<InputText>(templated_image[2]).GetRawTextString(),
              testing::status::IsOkAndHolds("<end>\n<test>Model\n"));
}

TEST_F(SessionBasicTest, PreprocessContents) {
  SessionConfig session_config = SessionConfig::CreateDefault();
  session_config.SetSamplerBackend(Backend::CPU);
  session_config.SetStartTokenId(2);
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
  std::vector<InputData> contents;
  contents.emplace_back(InputText("</s>Hello World!"));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_contents,
                       session->PreprocessContents(contents));
  ASSERT_EQ(preprocessed_contents.size(), 1);
  ASSERT_TRUE(std::holds_alternative<InputText>(preprocessed_contents[0]));
  const auto& text_data = std::get<InputText>(preprocessed_contents[0]);
  ASSERT_TRUE(text_data.IsTensorBuffer());
  ASSERT_OK_AND_ASSIGN(auto text_tensor, text_data.GetPreprocessedTextTensor());
  ASSERT_NE(text_tensor, nullptr);
  LITERT_ASSERT_OK_AND_ASSIGN(auto token_ids_span,
                              ReferTensorBufferAsSpan<int>(*text_tensor));
  EXPECT_THAT(std::vector<int>(token_ids_span.begin(), token_ids_span.end()),
              testing::ElementsAre(2, 90, 547, 58, 735, 210, 466, 2294));
}

TEST_F(SessionBasicTest, CombineExecutorAudioDataEmptyFails) {
  std::vector<ExecutorAudioData> executor_data;
  EXPECT_THAT(SessionBasic::CombineExecutorData(executor_data),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument,
                                        "Executor data is empty."));
}

TEST_F(SessionBasicTest, CombineExecutorAudioDataSingleSuccess) {
  std::vector<ExecutorAudioData> executor_data;
  ExecutorAudioData executor_audio_data;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto audio_buffer,
      CopyToTensorBuffer<float>({4.0, 3.0, 2.0, 1.0}, {1, 2, 2}));
  executor_audio_data.SetEmbeddings(std::move(audio_buffer));
  executor_data.push_back(std::move(executor_audio_data));
  ASSERT_OK_AND_ASSIGN(auto combined_executor_data,
                       SessionBasic::CombineExecutorData(executor_data));
  ASSERT_OK_AND_ASSIGN(auto combined_embeddings_ptr,
                       combined_executor_data.GetEmbeddingsPtr());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto combined_embeddings_span,
      ReferTensorBufferAsSpan<float>(*combined_embeddings_ptr));
  EXPECT_THAT(std::vector<float>(combined_embeddings_span.begin(),
                                 combined_embeddings_span.end()),
              testing::ElementsAre(4.0, 3.0, 2.0, 1.0));
}

TEST_F(SessionBasicTest, CombineExecutorAudioDataMultiSuccess) {
  std::vector<ExecutorAudioData> executor_data;

  ExecutorAudioData executor_audio_data_1;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto audio_buffer_1,
      CopyToTensorBuffer<float>({6.0, 5.0, 4.0, 3.0, 2.0, 1.0}, {1, 3, 2}));
  executor_audio_data_1.SetEmbeddings(std::move(audio_buffer_1));
  executor_audio_data_1.SetValidTokens(3);
  executor_data.push_back(std::move(executor_audio_data_1));

  ExecutorAudioData executor_audio_data_2;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto audio_buffer_2,
      CopyToTensorBuffer<float>({5.0, 6.0, 7.0, 8.0}, {1, 2, 2}));
  executor_audio_data_2.SetEmbeddings(std::move(audio_buffer_2));
  executor_audio_data_2.SetValidTokens(2);
  executor_data.push_back(std::move(executor_audio_data_2));

  ExecutorAudioData executor_audio_data_3;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto audio_buffer_3, CopyToTensorBuffer<float>({11.0, 12.0}, {1, 1, 2}));
  executor_audio_data_3.SetEmbeddings(std::move(audio_buffer_3));
  executor_audio_data_3.SetValidTokens(1);
  executor_data.push_back(std::move(executor_audio_data_3));

  ASSERT_OK_AND_ASSIGN(auto combined_executor_data,
                       SessionBasic::CombineExecutorData(executor_data));
  ASSERT_OK_AND_ASSIGN(auto combined_embeddings_ptr,
                       combined_executor_data.GetEmbeddingsPtr());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto combined_embeddings_span,
      ReferTensorBufferAsSpan<float>(*combined_embeddings_ptr));
  const auto& dimensions = TensorBufferDims(*combined_embeddings_ptr);
  EXPECT_THAT(dimensions, testing::ElementsAre(1, 6, 2));
  EXPECT_THAT(std::vector<float>(combined_embeddings_span.begin(),
                                 combined_embeddings_span.end()),
              testing::ElementsAre(6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 5.0, 6.0, 7.0,
                                   8.0, 11.0, 12.0));
}

TEST_F(SessionBasicTest, CombineExecutorVisionDataEmptyFails) {
  std::vector<ExecutorVisionData> executor_data;
  EXPECT_THAT(SessionBasic::CombineExecutorData(executor_data),
              testing::status::StatusIs(absl::StatusCode::kInvalidArgument,
                                        "Executor data is empty."));
}

TEST_F(SessionBasicTest, CombineExecutorVisionDataSingleSuccess) {
  std::vector<ExecutorVisionData> executor_data;
  ExecutorVisionData executor_vision_data;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto vision_buffer,
      CopyToTensorBuffer<float>({1.0, 2.0, 3.0, 4.0}, {1, 2, 2}));
  executor_vision_data.SetEmbeddings(std::move(vision_buffer));
  executor_data.push_back(std::move(executor_vision_data));
  ASSERT_OK_AND_ASSIGN(auto combined_executor_data,
                       SessionBasic::CombineExecutorData(executor_data));
  ASSERT_OK_AND_ASSIGN(auto combined_embeddings_ptr,
                       combined_executor_data.GetEmbeddingsPtr());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto combined_embeddings_span,
      ReferTensorBufferAsSpan<float>(*combined_embeddings_ptr));
  EXPECT_THAT(std::vector<float>(combined_embeddings_span.begin(),
                                 combined_embeddings_span.end()),
              testing::ElementsAre(1.0, 2.0, 3.0, 4.0));
}

TEST_F(SessionBasicTest, CombineExecutorVisionDataMultiSuccess) {
  std::vector<ExecutorVisionData> executor_data;

  ExecutorVisionData executor_vision_data_1;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto vision_buffer,
      CopyToTensorBuffer<float>({1.0, 2.0, 3.0, 4.0}, {1, 2, 2}));
  executor_vision_data_1.SetEmbeddings(std::move(vision_buffer));
  executor_data.push_back(std::move(executor_vision_data_1));

  ExecutorVisionData executor_vision_data_2;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto vision_buffer_2,
      CopyToTensorBuffer<float>({5.0, 6.0, 7.0, 8.0}, {1, 2, 2}));
  executor_vision_data_2.SetEmbeddings(std::move(vision_buffer_2));
  executor_data.push_back(std::move(executor_vision_data_2));

  ExecutorVisionData executor_vision_data_3;
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto vision_buffer_3,
      CopyToTensorBuffer<float>({9.0, 10.0, 11.0, 12.0}, {1, 2, 2}));
  executor_vision_data_3.SetEmbeddings(std::move(vision_buffer_3));
  executor_data.push_back(std::move(executor_vision_data_3));

  ASSERT_OK_AND_ASSIGN(auto combined_executor_data,
                       SessionBasic::CombineExecutorData(executor_data));
  ASSERT_OK_AND_ASSIGN(auto combined_embeddings_ptr,
                       combined_executor_data.GetEmbeddingsPtr());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto combined_embeddings_span,
      ReferTensorBufferAsSpan<float>(*combined_embeddings_ptr));
  const auto& dimensions = TensorBufferDims(*combined_embeddings_ptr);
  EXPECT_THAT(dimensions, testing::ElementsAre(1, 1, 6, 2));
  EXPECT_THAT(std::vector<float>(combined_embeddings_span.begin(),
                                 combined_embeddings_span.end()),
              testing::ElementsAre(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                                   10.0, 11.0, 12.0));
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
  EXPECT_THAT(result, testing::status::StatusIs(
                          absl::StatusCode::kInvalidArgument,
                          "No token IDs found in preprocessed_contents."));
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
          /*prefill_tokens=*/{{2,    423,  8,   179, 29,  207,  19,
                               547,  58,   735, 210, 466, 2294, 256000,
                               -2,   -2,   -2,  -2,  -2,  -4,   433,
                               2172, 1920, 432, 197, 979, 3076, 29}},
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
          // clang-format off
          // "User:Hello World!<start_of_audio>What does the audio say?[END]Model:" // NOLINT
          // clang-format on
          /*prefill_tokens=*/
          {{2,    423,  8,    179,    29,  207, 19,   547,  58, 735,
            210,  466,  2294, 256000, -2,  -2,  -2,   -2,   -2, -4,
            583,  378,  844,  166,    3,   14,  1252, 54,   58, 626,
            2295, 3995, 2172, 1920,   432, 197, 979,  3076, 29}},

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
      ->GenerateContentStream(inputs, std::make_unique<CancelledTestCallbacks>(
                                          status, responses, done))
      .IgnoreError();

  // Wait for a short time to ensure the decoding has started.
  absl::SleepFor(absl::Milliseconds(100));

  // Cancel the process.
  (*session)->CancelProcess();

  // Wait for the callbacks to be done.
  done.WaitForNotification();
  EXPECT_THAT(status, testing::status::StatusIs(absl::StatusCode::kCancelled));
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
  bool use_benchmark_info = GetParam();
  if (use_benchmark_info) {
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
      ->GenerateContentStream(inputs, std::make_unique<CancelledTestCallbacks>(
                                          status, responses, done1))
      .IgnoreError();

  // Cancel the process.
  (*session)->CancelProcess();

  // Wait for the callbacks to be done.
  done1.WaitForNotification();
  EXPECT_THAT(status, testing::status::StatusIs(absl::StatusCode::kCancelled));

  // Generate again after cancellation.
  // The second generation should succeed.
  status = absl::OkStatus();
  responses.clear();
  absl::Notification done2;
  (*session)
      ->GenerateContentStream(inputs, std::make_unique<CancelledTestCallbacks>(
                                          status, responses, done2))
      .IgnoreError();
  done2.WaitForNotification();
  EXPECT_OK(status);
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
      inputs,
      std::make_unique<StreamingTestCallbacks>(status, responses, done)));
  // Wait for the callbacks to be done.
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
          // Expected tokens: "</s><test>User\nHello World!<end>\n<test>Model\n"
          /*prefill_tokens=*/{{2,   4,  0,   39,  637, 0,    3328, 8,   179, 90,
                               547, 58, 735, 210, 466, 2294, 0,    40,  23,  0,
                               4,   0,  39,  637, 0,   197,  979,  3076}},
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
  // Fake constraint that expects " How's it".
  std::vector<int> expected_token_ids = {2, 224, 24, 8, 66, 0};
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
          /*prefill_tokens=*/{{2, 224}},
          // "How's it going?"
          /*decode_tokens=*/{
              {224}, {24}, {8}, {66}, {246}, {18}, {2295}, {2294}}));
  auto session = SessionBasic::Create(
      executor.get(), tokenizer_.get(), /*vision_executor=*/nullptr,
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
      inputs,
      std::make_unique<StreamingTestCallbacks>(status, texts, done_decode),
      decode_config));

  done_decode.WaitForNotification();
  EXPECT_OK(status);
  EXPECT_EQ(texts.size(), 4);
  EXPECT_THAT(texts, testing::ElementsAre(" How", "'", "s", " it"));
}

}  // namespace
}  // namespace litert::lm
