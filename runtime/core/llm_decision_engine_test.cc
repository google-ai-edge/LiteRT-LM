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

#include "runtime/core/llm_decision_engine.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/strings/strip.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/engine/decision_engine.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/token.pb.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

using ::testing::FloatNear;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SizeIs;
using ::testing::status::StatusIs;

constexpr absl::string_view kTestLlmPath =
    "litert_lm/runtime/testdata/test_lm.litertlm";

constexpr absl::string_view kSimpleChatTemplate =
    "{% for m in messages %}<u>{{ m['content'][0]['text'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<m>{% endif %}";

constexpr float kTolerance = 2e-3f;
constexpr float kMaskedLogit = -1000.0f;

class MockTokenizer : public support::Tokenizer {
 public:
  MOCK_METHOD(absl::StatusOr<std::vector<int>>, TextToTokenIds,
              (absl::string_view text), (override));
  MOCK_METHOD(absl::StatusOr<std::string>, TokenIdsToText,
              (absl::Span<const int> token_ids, bool skip_special_tokens),
              (override));
  MOCK_METHOD(absl::StatusOr<int>, TokenToId, (absl::string_view token),
              (override));
  MOCK_METHOD(support::TokenizerType, GetTokenizerType, (), (const, override));
  MOCK_METHOD(std::vector<std::string>, GetTokens, (), (const, override));
  MOCK_METHOD(int, GetVocabSize, (), (const, override));
};

std::unique_ptr<NiceMock<MockTokenizer>> MakeLetterMockTokenizer(
    int vocab_size = 100) {
  auto tokenizer = std::make_unique<NiceMock<MockTokenizer>>();
  ON_CALL(*tokenizer, GetVocabSize).WillByDefault(Return(vocab_size));
  ON_CALL(*tokenizer, TokenToId)
      .WillByDefault(Return(absl::NotFoundError("not set")));
  // Token 2 is BOS, written "<bos>".
  ON_CALL(*tokenizer, TokenIdsToText)
      .WillByDefault(
          [](absl::Span<const int> ids,
             bool /*skip_special_tokens*/) -> absl::StatusOr<std::string> {
            if (ids.size() == 1 && ids[0] == 2) {
              return std::string("<bos>");
            }
            return absl::NotFoundError("not set");
          });
  // "A".."Z" -> 10..35, " A".." Z" -> 40..65, other characters -> 70..94.
  ON_CALL(*tokenizer, TextToTokenIds)
      .WillByDefault(
          [](absl::string_view text) -> absl::StatusOr<std::vector<int>> {
            if (text.size() == 1 && text[0] >= 'A' && text[0] <= 'Z') {
              return std::vector<int>{10 + (text[0] - 'A')};
            }
            if (text.size() == 2 && text[0] == ' ' && text[1] >= 'A' &&
                text[1] <= 'Z') {
              return std::vector<int>{40 + (text[1] - 'A')};
            }
            std::vector<int> tokens;
            if (absl::ConsumePrefix(&text, "<bos>")) {
              tokens.push_back(2);
            }
            for (unsigned char ch : text) {
              tokens.push_back(70 + (ch % 25));
            }
            return tokens;
          });
  return tokenizer;
}

class ImmediateTaskController : public SessionInterface::TaskController {
 public:
  absl::Status WaitUntilDone(absl::Duration timeout) override {
    return absl::OkStatus();
  }
  absl::Status Cancel() override { return absl::OkStatus(); }
};

struct PrefillRecord {
  int session_id = 0;
  int start_step = 0;
  std::vector<int> tokens;
};

// Returns `EngineSettings` whose `LlmMetadata` describes a fake model with
// `jinja_template`, start token `start_token_id` (none if negative) and the
// Gemma 4 model type if `gemma4` (enables prefix reuse and its calibration),
// or the generic model type otherwise.
EngineSettings MakeFakeEngineSettings(
    absl::string_view jinja_template = kSimpleChatTemplate,
    int start_token_id = 2, bool gemma4 = false) {
  absl::StatusOr<ModelAssets> assets =
      ModelAssets::Create("fake_model.litertlm");
  ABSL_CHECK_OK(assets);
  absl::StatusOr<EngineSettings> settings =
      EngineSettings::CreateDefault(*std::move(assets));
  ABSL_CHECK_OK(settings);
  proto::LlmMetadata& metadata = settings->GetMutableLlmMetadata();
  metadata.set_jinja_prompt_template(jinja_template);
  if (start_token_id >= 0) {
    metadata.mutable_start_token()->mutable_token_ids()->add_ids(
        start_token_id);
  }
  metadata.add_stop_tokens()->mutable_token_ids()->add_ids(1);
  if (gemma4) {
    metadata.mutable_llm_model_type()->mutable_gemma4();
  } else {
    metadata.mutable_llm_model_type()->mutable_generic_model();
  }
  return *std::move(settings);
}

// Fake `Engine` + `SessionInterface` that records every session creation,
// prefill chunk and rewind, and serves `RunTextScoring` from caller-scripted
// next-token logits: one logits vector per distinct scored prompt, in order.
class ScriptedScoringFakeEngine : public Engine {
 public:
  ScriptedScoringFakeEngine(
      std::unique_ptr<support::Tokenizer> tokenizer,
      std::vector<std::vector<float>> prompt_logits,
      EngineSettings engine_settings = MakeFakeEngineSettings())
      : tokenizer_(std::move(tokenizer)),
        prompt_logits_(std::move(prompt_logits)),
        engine_settings_(std::move(engine_settings)) {}

  void SetPrefillError(absl::Status status) {
    prefill_error_ = std::move(status);
  }
  // Fails prefills that do not start at step 0 (i.e. after a shared prefix).
  void SetSuffixPrefillError(absl::Status status) {
    suffix_prefill_error_ = std::move(status);
  }
  void SetRewindError(absl::Status status) {
    rewind_error_ = std::move(status);
  }
  // Task state reported by successful prefills.
  void SetPrefillTaskState(TaskState state) { prefill_task_state_ = state; }

  // Number of sessions created with `CreateSession` (clones not included).
  int sessions_created() const { return next_session_id_; }
  int clones_created() const { return clones_created_; }
  // Number of `RunTextScoring` calls on sessions that are not clones.
  int scorings_on_original_sessions() const {
    return scorings_on_original_sessions_;
  }
  const std::vector<PrefillRecord>& prefill_records() const {
    return prefill_records_;
  }

  absl::StatusOr<std::unique_ptr<SessionInterface>> CreateSession(
      const SessionConfig& session_config) override {
    const int session_id = next_session_id_++;
    return std::make_unique<FakeSession>(*this, session_id, session_config);
  }

  const EngineSettings& GetEngineSettings() const override {
    return engine_settings_;
  }

  const support::Tokenizer& GetTokenizer() const override {
    return *tokenizer_;
  }

  absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties()
      const override {
    return absl::UnimplementedError("No audio.");
  }

  absl::StatusOr<VisionExecutorProperties> GetVisionExecutorProperties()
      const override {
    return absl::UnimplementedError("No vision.");
  }

 private:
  class FakeSession : public SessionInterface {
   public:
    FakeSession(ScriptedScoringFakeEngine& parent, int session_id,
                SessionConfig config)
        : parent_(parent),
          session_id_(session_id),
          config_(std::move(config)) {}

    absl::StatusOr<Responses> GenerateContent(
        const std::vector<InputData>& contents) override {
      return absl::UnimplementedError("Not needed.");
    }
    absl::Status GenerateContentStream(
        const std::vector<InputData>& contents,
        absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override {
      return absl::UnimplementedError("Not needed.");
    }
    absl::Status GenerateContentStream(
        const std::vector<InputData>& contents,
        absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
        const DecodeConfig& decode_config) override {
      return absl::UnimplementedError("Not needed.");
    }
    absl::Status RunPrefill(const std::vector<InputData>& contents) override {
      return absl::UnimplementedError("Not needed.");
    }
    absl::StatusOr<Responses> RunDecode() override {
      return absl::UnimplementedError("Not needed.");
    }
    absl::StatusOr<Responses> RunDecode(
        const DecodeConfig& decode_config) override {
      return absl::UnimplementedError("Not needed.");
    }
    absl::StatusOr<BenchmarkInfo> GetBenchmarkInfo() override {
      return absl::UnimplementedError("Not needed.");
    }
    absl::StatusOr<BenchmarkInfo*> GetMutableBenchmarkInfo() override {
      return absl::UnimplementedError("Not needed.");
    }
    absl::Status WaitUntilDone() override { return absl::OkStatus(); }
    const SessionConfig& GetSessionConfig() const override { return config_; }

    absl::StatusOr<std::unique_ptr<TaskController>> PrefillPreprocessedContents(
        std::vector<InputData> preprocessed_contents,
        absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) override {
      if (!parent_.prefill_error_.ok()) {
        callback(parent_.prefill_error_);
        return std::make_unique<ImmediateTaskController>();
      }
      if (!tokens_.empty() && !parent_.suffix_prefill_error_.ok()) {
        callback(parent_.suffix_prefill_error_);
        return std::make_unique<ImmediateTaskController>();
      }
      for (const InputData& item : preprocessed_contents) {
        const auto* input_text = std::get_if<InputText>(&item);
        if (input_text == nullptr || !input_text->IsTensorBuffer()) {
          callback(absl::InvalidArgumentError("Expected TensorBuffer."));
          return std::make_unique<ImmediateTaskController>();
        }
        auto tensor_or = input_text->GetPreprocessedTextTensor();
        if (!tensor_or.ok() || *tensor_or == nullptr) {
          callback(absl::InvalidArgumentError("Missing token tensor."));
          return std::make_unique<ImmediateTaskController>();
        }
        auto token_rows =
            support::Tokenizer::TensorBufferToTokenIds(**tensor_or);
        if (!token_rows.ok() || token_rows->empty()) {
          callback(absl::InvalidArgumentError("Invalid token tensor."));
          return std::make_unique<ImmediateTaskController>();
        }
        const std::vector<int>& chunk = (*token_rows)[0];
        parent_.prefill_records_.push_back(PrefillRecord{
            .session_id = session_id_,
            .start_step = static_cast<int>(tokens_.size()),
            .tokens = chunk,
        });
        tokens_.insert(tokens_.end(), chunk.begin(), chunk.end());
      }
      callback(Responses(parent_.prefill_task_state_));
      return std::make_unique<ImmediateTaskController>();
    }

    absl::Status RewindToStep(int step) override {
      if (!parent_.rewind_error_.ok()) {
        return parent_.rewind_error_;
      }
      if (step < 0 || step > static_cast<int>(tokens_.size())) {
        return absl::InvalidArgumentError("Invalid rewind step.");
      }
      tokens_.resize(step);
      return absl::OkStatus();
    }

    absl::StatusOr<int> GetCurrentStep() const override {
      return static_cast<int>(tokens_.size());
    }

    absl::StatusOr<std::unique_ptr<SessionInterface>> Clone() override {
      auto clone =
          std::make_unique<FakeSession>(parent_, session_id_, config_);
      clone->tokens_ = tokens_;
      clone->is_clone_ = true;
      ++parent_.clones_created_;
      return clone;
    }

    // Returns `log P(target | prompt)` for a single-token `target`, and
    // appends the target token like the real `RunTextScoring`.
    absl::StatusOr<Responses> RunTextScoring(
        const std::vector<absl::string_view>& target_text,
        bool store_token_lengths) override {
      if (target_text.size() != 1) {
        return absl::InvalidArgumentError("Target text size should be 1.");
      }
      if (!is_clone_) {
        ++parent_.scorings_on_original_sessions_;
      }
      if (tokens_ != parent_.last_scored_prompt_) {
        parent_.last_scored_prompt_ = tokens_;
        ++parent_.prompt_index_;
      }
      if (parent_.prompt_index_ >=
          static_cast<int>(parent_.prompt_logits_.size())) {
        return absl::OutOfRangeError("No scripted logits remaining.");
      }
      const std::vector<float>& logits =
          parent_.prompt_logits_[parent_.prompt_index_];
      ABSL_ASSIGN_OR_RETURN(
          std::vector<int> ids,
          const_cast<support::Tokenizer&>(*parent_.tokenizer_)
              .TextToTokenIds(target_text[0]));
      if (ids.size() != 1 || ids[0] < 0 ||
          static_cast<size_t>(ids[0]) >= logits.size()) {
        return absl::InvalidArgumentError("Expected one in-vocab token.");
      }
      double max_logit = -std::numeric_limits<double>::infinity();
      for (float v : logits) {
        max_logit = std::max(max_logit, static_cast<double>(v));
      }
      double sum_exp = 0.0;
      for (float v : logits) {
        sum_exp += std::exp(static_cast<double>(v) - max_logit);
      }
      const double log_z = max_logit + std::log(sum_exp);
      tokens_.push_back(ids[0]);
      return Responses(
          TaskState::kDone, /*response_texts=*/{},
          std::vector<float>{static_cast<float>(logits[ids[0]] - log_z)});
    }

   private:
    ScriptedScoringFakeEngine& parent_;
    int session_id_ = 0;
    SessionConfig config_;
    std::vector<int> tokens_;
    bool is_clone_ = false;
  };

  std::unique_ptr<support::Tokenizer> tokenizer_;
  std::vector<std::vector<float>> prompt_logits_;
  EngineSettings engine_settings_;
  absl::Status prefill_error_ = absl::OkStatus();
  absl::Status suffix_prefill_error_ = absl::OkStatus();
  absl::Status rewind_error_ = absl::OkStatus();
  TaskState prefill_task_state_ = TaskState::kDone;
  int next_session_id_ = 0;
  int clones_created_ = 0;
  int scorings_on_original_sessions_ = 0;
  int prompt_index_ = -1;
  std::vector<int> last_scored_prompt_;
  std::vector<PrefillRecord> prefill_records_;
};

std::vector<float> MakeLogitsVector(
    int vocab_size,
    std::initializer_list<std::pair<int, float>> overrides = {}) {
  std::vector<float> logits(vocab_size, kMaskedLogit);
  for (const auto& [token_id, value] : overrides) {
    logits[token_id] = value;
  }
  return logits;
}

std::vector<InputData> MakeTextContents(
    std::initializer_list<absl::string_view> chunks) {
  std::vector<InputData> contents;
  contents.reserve(chunks.size());
  for (absl::string_view chunk : chunks) {
    contents.push_back(InputText(std::string(chunk)));
  }
  return contents;
}

void ExpectValidResponse(absl::Span<const DecisionQuestion> questions,
                         const DecisionResponse& response) {
  ASSERT_EQ(response.results.size(), questions.size());
  for (size_t i = 0; i < questions.size(); ++i) {
    const DecisionQuestion& q = questions[i];
    const DecisionResult& r = response.results[i];
    EXPECT_EQ(r.id, q.id);
    ASSERT_GE(r.probabilities.size(), 2u);
    float sum = 0.0f;
    bool found_selected = false;
    for (const DecisionOptionProbability& p : r.probabilities) {
      EXPECT_GE(p.probability, 0.0f);
      EXPECT_LE(p.probability, 1.0f);
      sum += p.probability;
      if (p.key == r.selected_key) {
        found_selected = true;
      }
    }
    EXPECT_NEAR(sum, 1.0f, 1e-3f);
    EXPECT_TRUE(found_selected);
    EXPECT_GE(r.confidence, 0.0f);
    EXPECT_LE(r.confidence, 1.0f);
    if (q.type == DecisionQuestionType::kChoice) {
      EXPECT_FALSE(r.expected_score.has_value());
    } else {
      EXPECT_TRUE(r.expected_score.has_value());
    }
  }
}

absl::StatusOr<EngineSettings> CreateRealEngineSettings(
    absl::string_view relative_model_path) {
  const std::filesystem::path model_path =
      std::filesystem::path(::testing::SrcDir()) /
      std::string(relative_model_path);
  ABSL_ASSIGN_OR_RETURN(ModelAssets assets,
                        ModelAssets::Create(model_path.string()));
  ABSL_ASSIGN_OR_RETURN(
      EngineSettings settings,
      EngineSettings::CreateDefault(std::move(assets), Backend::CPU));
  settings.GetMutableMainExecutorSettings().SetCacheDir(":nocache");
  return settings;
}

TEST(LlmDecisionEngineTest, RejectsInvalidQuestions) {
  auto fake_engine = std::make_shared<ScriptedScoringFakeEngine>(
      MakeLetterMockTokenizer(), std::vector<std::vector<float>>{});
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmDecisionEngine> engine,
                       LlmDecisionEngine::Create(*fake_engine));
  auto decide = [&](std::vector<DecisionQuestion> questions) {
    return engine->Decide({.contents = MakeTextContents({"Hello"}),
                           .questions = std::move(questions)});
  };

  EXPECT_THAT(decide({}), StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(decide({{.id = "", .type = DecisionQuestionType::kBoolean}}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(decide({{.id = "q1",
                       .type = DecisionQuestionType::kChoice,
                       .prompt = "Pick one",
                       .options = {{.key = "only_one"}}}}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // `kScore` questions must list at least 2 levels; there is no default scale.
  EXPECT_THAT(decide({{.id = "s0",
                       .type = DecisionQuestionType::kScore,
                       .prompt = "Rate it"}}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(decide({{.id = "s1",
                       .type = DecisionQuestionType::kScore,
                       .prompt = "Rate it",
                       .options = {{.key = "1"}}}}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  // `kBoolean` questions take 0 or exactly 2 options.
  EXPECT_THAT(
      decide({{.id = "b3",
               .type = DecisionQuestionType::kBoolean,
               .prompt = "Yes?",
               .options = {{.key = "no"}, {.key = "maybe"}, {.key = "yes"}}}}),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      decide({
          {.id = "dup", .type = DecisionQuestionType::kBoolean, .prompt = "Q1"},
          {.id = "dup", .type = DecisionQuestionType::kBoolean, .prompt = "Q2"},
      }),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("Duplicate")));

  std::vector<DecisionOption> too_many(27);
  for (int i = 0; i < 27; ++i) {
    too_many[i].key = absl::StrCat("opt_", i);
  }
  EXPECT_THAT(decide({{.id = "big",
                       .type = DecisionQuestionType::kChoice,
                       .prompt = "Pick one",
                       .options = std::move(too_many)}}),
              StatusIs(absl::StatusCode::kUnimplemented));
  // Nothing was prefilled for invalid requests.
  EXPECT_EQ(fake_engine->sessions_created(), 0);
}

TEST(LlmDecisionEngineTest,
     EvaluatesBooleanChoiceAndScoreWithIntraCallPrefixReuse) {
  // A Gemma 4 model: the shared prefix is reused and the label
  // log-probabilities are divided by its calibration temperature 3.5.
  // Labels follow the "Answer:" prefix, so " A", " B", " C" (40, 41, 42) are
  // scored.
  // Question 0 (kBoolean): " A" ("false") = 1.0, " B" ("true") = 3.0.
  // Question 1 (kChoice, 3 options): " B" is the most likely label.
  // Question 2 (kScore, keys 0 / 5 / 10): concentrated on " C" ("10").
  auto fake_engine = std::make_shared<ScriptedScoringFakeEngine>(
      MakeLetterMockTokenizer(),
      std::vector<std::vector<float>>{
          MakeLogitsVector(/*vocab_size=*/100, {{40, 1.0f}, {41, 3.0f}}),
          MakeLogitsVector(/*vocab_size=*/100,
                           {{40, 0.0f}, {41, 2.0f}, {11, 2.0f}, {42, 0.0f}}),
          MakeLogitsVector(/*vocab_size=*/100,
                           {{40, -5.0f}, {41, -5.0f}, {42, 8.0f}}),
      },
      MakeFakeEngineSettings(kSimpleChatTemplate, /*start_token_id=*/2,
                             /*gemma4=*/true));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmDecisionEngine> engine,
                       LlmDecisionEngine::Create(*fake_engine));

  const std::vector<DecisionQuestion> questions = {
      {.id = "refund",
       .type = DecisionQuestionType::kBoolean,
       .prompt = "Does the customer request a refund?"},
      {.id = "dept",
       .type = DecisionQuestionType::kChoice,
       .prompt = "Which department owns this ticket?",
       .options = {{.key = "sales"}, {.key = "billing"}, {.key = "shipping"}}},
      {.id = "urgency",
       .type = DecisionQuestionType::kScore,
       .prompt = "Rate urgency from 0 to 10.",
       .options = {{.key = "0"}, {.key = "5"}, {.key = "10"}}},
  };

  ASSERT_OK_AND_ASSIGN(
      DecisionResponse response,
      engine->Decide({.contents = MakeTextContents(
                          {"Customer message: I was charged twice for my order "
                           "and need an immediate refund."}),
                      .questions = questions}));
  ExpectValidResponse(questions, response);

  constexpr double kGemma4Temperature = 3.5;
  const float expected_p_true =
      static_cast<float>(1.0 / (1.0 + std::exp(-2.0 / kGemma4Temperature)));
  EXPECT_EQ(response.results[0].selected_key, "true");
  EXPECT_THAT(*response.results[0].expected_score,
              FloatNear(expected_p_true, kTolerance));

  // With the "Answer:" prefix only " B" (41) is scored, not "B" (11).
  EXPECT_EQ(response.results[1].selected_key, "billing");
  const double b_mass = std::exp(2.0 / kGemma4Temperature);
  const float expected_billing_p =
      static_cast<float>(b_mass / (1.0 + b_mass + 1.0));
  EXPECT_THAT(response.results[1].probabilities[1].probability,
              FloatNear(expected_billing_p, kTolerance));

  // `expected_score` is the mean level index in [0, 2]; the keys are only
  // identifiers.
  EXPECT_EQ(response.results[2].selected_key, "10");
  const double c_mass = std::exp(13.0 / kGemma4Temperature);
  const float expected_level =
      static_cast<float>((1.0 + 2.0 * c_mass) / (2.0 + c_mass));
  EXPECT_THAT(*response.results[2].expected_score,
              FloatNear(expected_level, kTolerance));

  // Shared prefix: 1 session created, 1 shared-prefix prefill from step 0 and
  // 3 question-suffix prefills all starting at the same `prefix_len`.
  EXPECT_EQ(fake_engine->sessions_created(), 1);
  ASSERT_THAT(fake_engine->prefill_records(), SizeIs(4));
  EXPECT_EQ(fake_engine->prefill_records()[0].start_step, 0);
  const int prefix_len =
      static_cast<int>(fake_engine->prefill_records()[0].tokens.size());
  EXPECT_GT(prefix_len, 20);
  EXPECT_EQ(fake_engine->prefill_records()[1].start_step, prefix_len);
  EXPECT_EQ(fake_engine->prefill_records()[2].start_step, prefix_len);
  EXPECT_EQ(fake_engine->prefill_records()[3].start_step, prefix_len);

  // Each label is scored once, on a clone; the prefilled session itself is
  // never scored.
  EXPECT_EQ(fake_engine->scorings_on_original_sessions(), 0);
  EXPECT_EQ(fake_engine->clones_created(), 2 + 3 + 3);
}

TEST(LlmDecisionEngineTest, BooleanWithCustomOptionsAndVerbalScoreLevels) {
  // Question 0 (kBoolean, options [no, yes]): " A" = 0.0, " B" = 2.0.
  // Question 1 (kScore, verbal levels poor / fair / good): concentrated on
  // " C" ("good").
  auto fake_engine = std::make_shared<ScriptedScoringFakeEngine>(
      MakeLetterMockTokenizer(),
      std::vector<std::vector<float>>{
          MakeLogitsVector(/*vocab_size=*/100, {{40, 0.0f}, {41, 2.0f}}),
          MakeLogitsVector(/*vocab_size=*/100,
                           {{40, -5.0f}, {41, -5.0f}, {42, 8.0f}}),
      });
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmDecisionEngine> engine,
                       LlmDecisionEngine::Create(*fake_engine));

  const std::vector<DecisionQuestion> questions = {
      {.id = "refund",
       .type = DecisionQuestionType::kBoolean,
       .prompt = "Does the customer request a refund?",
       .options = {{.key = "no"}, {.key = "yes"}}},
      {.id = "quality",
       .type = DecisionQuestionType::kScore,
       .prompt = "How good is the reply?",
       .options = {{.key = "poor"}, {.key = "fair"}, {.key = "good"}}},
  };
  ASSERT_OK_AND_ASSIGN(
      DecisionResponse response,
      engine->Decide({.contents = MakeTextContents({"I want my money back."}),
                      .questions = questions}));
  ExpectValidResponse(questions, response);

  // `expected_score` of a boolean question is the probability of the second
  // (positive) option.
  const float expected_p_yes = static_cast<float>(1.0 / (1.0 + std::exp(-2.0)));
  EXPECT_EQ(response.results[0].selected_key, "yes");
  EXPECT_THAT(*response.results[0].expected_score,
              FloatNear(expected_p_yes, kTolerance));

  // Verbal levels are weighted by their 0-based index, so the expected score
  // is in [0, 2] and close to 2 ("good").
  EXPECT_EQ(response.results[1].selected_key, "good");
  EXPECT_GT(*response.results[1].expected_score, 1.99f);
  EXPECT_LE(*response.results[1].expected_score, 2.0f);
}

// Two boolean questions whose first label is clearly preferred: "true" for
// the first, "false" for the second.
std::vector<std::vector<float>> TwoBooleanQuestionLogits() {
  return {MakeLogitsVector(100, {{40, 0.0f}, {41, 5.0f}}),
          MakeLogitsVector(100, {{40, 5.0f}, {41, 0.0f}})};
}

const std::vector<DecisionQuestion>& TwoBooleanQuestions() {
  static const auto* const kQuestions = new std::vector<DecisionQuestion>{
      {.id = "q1", .type = DecisionQuestionType::kBoolean, .prompt = "First?"},
      {.id = "q2", .type = DecisionQuestionType::kBoolean, .prompt = "Second?"},
  };
  return *kQuestions;
}

TEST(LlmDecisionEngineTest, ModelsWithoutStepRewindUseFreshSessionPerBranch) {
  auto fake_engine = std::make_shared<ScriptedScoringFakeEngine>(
      MakeLetterMockTokenizer(), TwoBooleanQuestionLogits());
  // A generic model: no prefix reuse.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmDecisionEngine> engine,
                       LlmDecisionEngine::Create(*fake_engine));
  ASSERT_OK_AND_ASSIGN(DecisionResponse response,
                       engine->Decide({.contents = MakeTextContents({"Hello"}),
                                       .questions = TwoBooleanQuestions()}));
  ExpectValidResponse(TwoBooleanQuestions(), response);
  EXPECT_EQ(response.results[0].selected_key, "true");
  EXPECT_EQ(response.results[1].selected_key, "false");
  EXPECT_EQ(fake_engine->sessions_created(), 2);
  ASSERT_THAT(fake_engine->prefill_records(), SizeIs(2));
  EXPECT_EQ(fake_engine->prefill_records()[0].start_step, 0);
  EXPECT_EQ(fake_engine->prefill_records()[1].start_step, 0);
  EXPECT_EQ(fake_engine->scorings_on_original_sessions(), 0);
}

TEST(LlmDecisionEngineTest, RawStringJinjaTemplateAndErrorPaths) {
  constexpr absl::string_view kStringConcatTemplate =
      "{% for m in messages %}{{ 'User: ' + m['content'] }}{% endfor %}";
  auto fake_engine = std::make_shared<ScriptedScoringFakeEngine>(
      MakeLetterMockTokenizer(),
      std::vector<std::vector<float>>{
          MakeLogitsVector(100, {{10, -5.0f}, {11, 10.0f}})},
      MakeFakeEngineSettings(kStringConcatTemplate, /*start_token_id=*/-1));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmDecisionEngine> engine,
                       LlmDecisionEngine::Create(*fake_engine));

  const std::vector<DecisionQuestion> questions = {
      {.id = "q", .type = DecisionQuestionType::kBoolean, .prompt = "Urgent?"}};
  ASSERT_OK_AND_ASSIGN(DecisionResponse response,
                       engine->Decide({.contents = MakeTextContents({"Urgent"}),
                                       .questions = questions}));
  EXPECT_EQ(response.results[0].selected_key, "true");

  // Multimodal input returns UnimplementedError.
  std::vector<InputData> image_contents;
  image_contents.push_back(InputImage("raw_bytes"));
  EXPECT_THAT(engine->Decide({.contents = std::move(image_contents),
                              .questions = questions}),
              StatusIs(absl::StatusCode::kUnimplemented));

  // A prefill that ends cancelled or failed is reported, not scored.
  fake_engine->SetPrefillTaskState(TaskState::kCancelled);
  EXPECT_THAT(engine->Decide({.contents = MakeTextContents({"Urgent"}),
                              .questions = questions}),
              StatusIs(absl::StatusCode::kCancelled));
  fake_engine->SetPrefillTaskState(TaskState::kFailed);
  EXPECT_THAT(engine->Decide({.contents = MakeTextContents({"Urgent"}),
                              .questions = questions}),
              StatusIs(absl::StatusCode::kInternal));
  fake_engine->SetPrefillTaskState(TaskState::kDone);

  // State-entry capacity error maps to ResourceExhaustedError.
  fake_engine->SetPrefillError(
      absl::FailedPreconditionError("Insufficient KV cache state entries."));
  EXPECT_THAT(engine->Decide({.contents = MakeTextContents({"Urgent"}),
                              .questions = questions}),
              StatusIs(absl::StatusCode::kResourceExhausted,
                       HasSubstr("does not fit in the context")));
}

TEST(LlmDecisionEngineTest, VocabularyRangeAndUnmappedLetterErrors) {
  // Label 'B' maps to token 11, out of range for vocab_size = 11.
  auto small_vocab_engine = std::make_shared<ScriptedScoringFakeEngine>(
      MakeLetterMockTokenizer(/*vocab_size=*/11),
      std::vector<std::vector<float>>{std::vector<float>(15, 0.0f)},
      MakeFakeEngineSettings(kSimpleChatTemplate, /*start_token_id=*/-1));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmDecisionEngine> engine1,
                       LlmDecisionEngine::Create(*small_vocab_engine));
  const std::vector<DecisionQuestion> q = {
      {.id = "q", .type = DecisionQuestionType::kBoolean, .prompt = "Yes?"}};
  EXPECT_THAT(
      engine1->Decide({.contents = MakeTextContents({"Hi"}), .questions = q}),
      StatusIs(absl::StatusCode::kOutOfRange));

  // Multi-token split for 'B' returns FailedPreconditionError.
  auto unmapped_tok = std::make_unique<NiceMock<MockTokenizer>>();
  ON_CALL(*unmapped_tok, GetVocabSize).WillByDefault(Return(100));
  ON_CALL(*unmapped_tok, TokenToId)
      .WillByDefault(Return(absl::NotFoundError("not set")));
  ON_CALL(*unmapped_tok, TextToTokenIds)
      .WillByDefault(
          [](absl::string_view text) -> absl::StatusOr<std::vector<int>> {
            if (text == "A" || text == " A") return std::vector<int>{10};
            if (text == "B" || text == " B") return std::vector<int>{11, 12};
            return std::vector<int>{5, 6};
          });
  auto unmapped_engine = std::make_shared<ScriptedScoringFakeEngine>(
      std::move(unmapped_tok),
      std::vector<std::vector<float>>{std::vector<float>(100, 0.0f)},
      MakeFakeEngineSettings(kSimpleChatTemplate, /*start_token_id=*/-1));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmDecisionEngine> engine2,
                       LlmDecisionEngine::Create(*unmapped_engine));
  EXPECT_THAT(
      engine2->Decide({.contents = MakeTextContents({"Hi"}), .questions = q}),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Choice label 'B'")));
}

TEST(LlmDecisionEngineTest, RealModelCreateFromSettingsAndSharedEngine) {
  ASSERT_OK_AND_ASSIGN(EngineSettings settings,
                       CreateRealEngineSettings(kTestLlmPath));

  // 1. Standalone `LlmDecisionEngine::Create(EngineSettings)`.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmDecisionEngine> standalone_decision,
                       LlmDecisionEngine::Create(settings));
  const std::vector<DecisionQuestion> questions = {
      {.id = "urgent",
       .type = DecisionQuestionType::kBoolean,
       .prompt = "Urgent?"},
      {.id = "route",
       .type = DecisionQuestionType::kChoice,
       .prompt = "Route?",
       .options = {{.key = "a"}, {.key = "b"}}},
  };
  ASSERT_OK_AND_ASSIGN(DecisionResponse standalone_resp,
                       standalone_decision->Decide(
                           {.contents = MakeTextContents({"Order delayed."}),
                            .questions = questions}));
  ExpectValidResponse(questions, standalone_resp);

  // 2. Shared `Engine` interleaved with a live chat session.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Engine> shared_engine,
                       EngineFactory::CreateDefault(settings));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SessionInterface> chat_session,
      shared_engine->CreateSession(SessionConfig::CreateDefault()));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmDecisionEngine> shared_decision,
                       LlmDecisionEngine::Create(*shared_engine));

  std::vector<InputData> chat_prompt;
  chat_prompt.push_back(InputText("Hello world"));
  ASSERT_OK(chat_session->RunPrefill(chat_prompt));

  ASSERT_OK_AND_ASSIGN(
      DecisionResponse shared_resp,
      shared_decision->Decide({.contents = MakeTextContents({"Order delayed."}),
                               .questions = questions}));
  ExpectValidResponse(questions, shared_resp);
  for (size_t i = 0; i < questions.size(); ++i) {
    EXPECT_EQ(shared_resp.results[i].selected_key,
              standalone_resp.results[i].selected_key);
    for (size_t j = 0; j < shared_resp.results[i].probabilities.size(); ++j) {
      EXPECT_NEAR(shared_resp.results[i].probabilities[j].probability,
                  standalone_resp.results[i].probabilities[j].probability,
                  1e-5f);
    }
  }

  // Verify `chat_session` continues decoding cleanly after `Decide()`.
  DecodeConfig decode_config = DecodeConfig::CreateDefault();
  decode_config.SetMaxOutputTokens(2);
  EXPECT_OK(chat_session->RunDecode(decode_config));
}

TEST(LlmDecisionEngineTest, SuffixPrefillCapacityErrorFallsBackToFreshSession) {
  // Models with only large static prefill signatures reject a short prefill
  // after the shared prefix.
  auto fake_engine = std::make_shared<ScriptedScoringFakeEngine>(
      MakeLetterMockTokenizer(), TwoBooleanQuestionLogits(),
      MakeFakeEngineSettings(kSimpleChatTemplate, /*start_token_id=*/2,
                             /*gemma4=*/true));
  fake_engine->SetSuffixPrefillError(absl::FailedPreconditionError(
      "Chosen prefill work group size exceeds available state entries."));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmDecisionEngine> engine,
                       LlmDecisionEngine::Create(*fake_engine));
  ASSERT_OK_AND_ASSIGN(DecisionResponse response,
                       engine->Decide({.contents = MakeTextContents({"Hello"}),
                                       .questions = TwoBooleanQuestions()}));
  ExpectValidResponse(TwoBooleanQuestions(), response);
  EXPECT_EQ(response.results[0].selected_key, "true");
  EXPECT_EQ(response.results[1].selected_key, "false");
  // The shared-prefix session plus one fresh session per question.
  EXPECT_EQ(fake_engine->sessions_created(), 3);
}

TEST(LlmDecisionEngineTest, RewindErrorFallsBackToFreshSession) {
  // E.g. an executor whose `SetCurrentStep` cannot rewind several tokens.
  auto fake_engine = std::make_shared<ScriptedScoringFakeEngine>(
      MakeLetterMockTokenizer(), TwoBooleanQuestionLogits(),
      MakeFakeEngineSettings(kSimpleChatTemplate, /*start_token_id=*/2,
                             /*gemma4=*/true));
  fake_engine->SetRewindError(
      absl::UnimplementedError("SetCurrentStep is not implemented."));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmDecisionEngine> engine,
                       LlmDecisionEngine::Create(*fake_engine));
  ASSERT_OK_AND_ASSIGN(DecisionResponse response,
                       engine->Decide({.contents = MakeTextContents({"Hello"}),
                                       .questions = TwoBooleanQuestions()}));
  ExpectValidResponse(TwoBooleanQuestions(), response);
  EXPECT_EQ(response.results[0].selected_key, "true");
  EXPECT_EQ(response.results[1].selected_key, "false");
  // The first question reuses the prefix session; the second needs a rewind,
  // which fails, so it uses a fresh session.
  EXPECT_EQ(fake_engine->sessions_created(), 2);
  ASSERT_THAT(fake_engine->prefill_records(), SizeIs(3));
  EXPECT_EQ(fake_engine->prefill_records()[2].start_step, 0);
}

TEST(LlmDecisionEngineTest, PromptLongerThanContextIsInvalidArgument) {
  ASSERT_OK_AND_ASSIGN(EngineSettings settings,
                       CreateRealEngineSettings(kTestLlmPath));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmDecisionEngine> engine,
                       LlmDecisionEngine::Create(std::move(settings)));
  std::string long_text;
  for (int i = 0; i < 200; ++i) {
    absl::StrAppend(&long_text, "word", i, " ");
  }
  EXPECT_THAT(
      engine->Decide({.contents = MakeTextContents({long_text}),
                      .questions = {{.id = "q",
                                     .type = DecisionQuestionType::kBoolean,
                                     .prompt = "Urgent?"}}}),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(LlmDecisionEngineTest, StripsLeadingBosStringEmittedByJinjaTemplate) {
  // A template that emits "<bos>" itself must not lead to a second BOS token.
  auto fake_engine = std::make_shared<ScriptedScoringFakeEngine>(
      MakeLetterMockTokenizer(),
      std::vector<std::vector<float>>{MakeLogitsVector(100, {{40, 1.0f}})},
      MakeFakeEngineSettings(absl::StrCat("<bos>", kSimpleChatTemplate)));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LlmDecisionEngine> engine,
                       LlmDecisionEngine::Create(*fake_engine));
  ASSERT_OK(engine->Decide(
      {.contents = MakeTextContents({"Hello"}),
       .questions = {{.id = "q",
                      .type = DecisionQuestionType::kBoolean,
                      .prompt = "Urgent?"}}}));
  ASSERT_THAT(fake_engine->prefill_records(), SizeIs(1));
  const std::vector<int>& tokens = fake_engine->prefill_records()[0].tokens;
  ASSERT_GE(tokens.size(), 2u);
  EXPECT_EQ(tokens[0], 2);
  EXPECT_NE(tokens[1], 2);
}

}  // namespace
}  // namespace litert::lm
