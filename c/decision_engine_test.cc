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

#include "c/decision_engine.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "c/engine.h"
#include "c/error_reporter.h"

namespace {

using EngineSettingsPtr =
    std::unique_ptr<LiteRtLmEngineSettings,
                    decltype(&litert_lm_engine_settings_delete)>;
using DecisionEnginePtr =
    std::unique_ptr<LiteRtLmDecisionEngine,
                    decltype(&litert_lm_experimental_decision_engine_delete)>;
using QuestionPtr =
    std::unique_ptr<LiteRtLmDecisionQuestion,
                    decltype(&litert_lm_experimental_decision_question_delete)>;
using OptionPtr =
    std::unique_ptr<LiteRtLmDecisionOption,
                    decltype(&litert_lm_experimental_decision_option_delete)>;
using RequestPtr =
    std::unique_ptr<LiteRtLmDecisionRequest,
                    decltype(&litert_lm_experimental_decision_request_delete)>;
using ResponsePtr =
    std::unique_ptr<LiteRtLmDecisionResponse,
                    decltype(&litert_lm_experimental_decision_response_delete)>;
using InputDataPtr =
    std::unique_ptr<LiteRtLmInputData, decltype(&litert_lm_input_data_delete)>;

std::string GetTestdataPath(const std::string& filename) {
  std::string srcdir = ::testing::SrcDir();
  std::replace(srcdir.begin(), srcdir.end(), '\\', '/');
  return srcdir + "/" + filename;
}

EngineSettingsPtr CreateTestSettings() {
  const std::string model_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm.litertlm");
  LiteRtLmEngineSettings* raw_settings = nullptr;
  EXPECT_EQ(litert_lm_engine_settings_create(model_path.c_str(), "cpu",
                                             /*vision_backend_str=*/nullptr,
                                             /*audio_backend_str=*/nullptr,
                                             &raw_settings),
            kLiteRtLmStatusOk);
  EngineSettingsPtr settings(raw_settings, &litert_lm_engine_settings_delete);
  if (settings != nullptr) {
    litert_lm_engine_settings_set_cache_dir(settings.get(), ":nocache");
    litert_lm_engine_settings_set_max_num_tokens(settings.get(), 1024);
  }
  return settings;
}

DecisionEnginePtr CreateTestEngine() {
  EngineSettingsPtr settings = CreateTestSettings();
  LiteRtLmDecisionEngine* engine = nullptr;
  if (settings != nullptr) {
    EXPECT_EQ(
        litert_lm_experimental_decision_engine_create(settings.get(), &engine),
        kLiteRtLmStatusOk)
        << litert_lm_get_last_error_message();
  }
  return DecisionEnginePtr(engine,
                           &litert_lm_experimental_decision_engine_delete);
}

RequestPtr CreateRequest() {
  LiteRtLmDecisionRequest* request = nullptr;
  EXPECT_EQ(litert_lm_experimental_decision_request_create(&request),
            kLiteRtLmStatusOk);
  return RequestPtr(request, &litert_lm_experimental_decision_request_delete);
}

QuestionPtr CreateNoulQuestion(const char* id,
                               const char* instructions = "Is it true?") {
  LiteRtLmDecisionQuestion* question = nullptr;
  EXPECT_EQ(litert_lm_experimental_decision_question_create_noul(
                id, instructions, &question),
            kLiteRtLmStatusOk);
  return QuestionPtr(question,
                     &litert_lm_experimental_decision_question_delete);
}

QuestionPtr CreateChoiceQuestion(const char* id,
                                 const char* instructions = "Pick one.") {
  LiteRtLmDecisionQuestion* question = nullptr;
  EXPECT_EQ(litert_lm_experimental_decision_question_create_choice(
                id, instructions, &question),
            kLiteRtLmStatusOk);
  return QuestionPtr(question,
                     &litert_lm_experimental_decision_question_delete);
}

QuestionPtr CreateScoreQuestion(const char* id,
                                const char* instructions = "Rate it.") {
  LiteRtLmDecisionQuestion* question = nullptr;
  EXPECT_EQ(litert_lm_experimental_decision_question_create_score(
                id, instructions, &question),
            kLiteRtLmStatusOk);
  return QuestionPtr(question,
                     &litert_lm_experimental_decision_question_delete);
}

InputDataPtr CreateInput(LiteRtLmInputDataType type, const std::string& data) {
  LiteRtLmInputData* input = nullptr;
  EXPECT_EQ(litert_lm_input_data_create(type, data.data(), data.size(), &input),
            kLiteRtLmStatusOk);
  return InputDataPtr(input, &litert_lm_input_data_delete);
}

// Decides `request` about the text `inputs`. On success, stores the response
// in `*response` (if not NULL). On failure, checks that the output is reset.
LiteRtLmStatusCode Decide(LiteRtLmDecisionEngine* engine,
                          const std::vector<std::string>& inputs,
                          const LiteRtLmDecisionRequest* request,
                          ResponsePtr* response = nullptr) {
  std::vector<InputDataPtr> owned;
  std::vector<const LiteRtLmInputData*> raw;
  for (const std::string& text : inputs) {
    owned.push_back(CreateInput(kLiteRtLmInputDataTypeText, text));
    raw.push_back(owned.back().get());
  }
  LiteRtLmDecisionResponse* raw_response =
      reinterpret_cast<LiteRtLmDecisionResponse*>(1);
  const LiteRtLmStatusCode status =
      litert_lm_experimental_decision_engine_decide(
          engine, raw.data(), raw.size(), request, &raw_response);
  if (status != kLiteRtLmStatusOk) {
    EXPECT_EQ(raw_response, nullptr);
    return status;
  }
  ResponsePtr owned_response(raw_response,
                             &litert_lm_experimental_decision_response_delete);
  if (response != nullptr) *response = std::move(owned_response);
  return status;
}

ResponsePtr DecideOrDie(LiteRtLmDecisionEngine* engine,
                        const std::vector<std::string>& inputs,
                        const LiteRtLmDecisionRequest* request) {
  ResponsePtr response(nullptr,
                       &litert_lm_experimental_decision_response_delete);
  EXPECT_EQ(Decide(engine, inputs, request, &response), kLiteRtLmStatusOk)
      << litert_lm_get_last_error_message();
  return response;
}

std::string AnswerId(const LiteRtLmDecisionResponse* response, size_t i) {
  const char* id = nullptr;
  EXPECT_EQ(litert_lm_experimental_decision_response_get_answer_id_at(response,
                                                                      i, &id),
            kLiteRtLmStatusOk);
  return id == nullptr ? "" : id;
}

float Confidence(const LiteRtLmDecisionResponse* response, size_t i) {
  float confidence = -1.0f;
  EXPECT_EQ(litert_lm_experimental_decision_response_get_confidence_at(
                response, i, &confidence),
            kLiteRtLmStatusOk);
  return confidence;
}

float Noul(const LiteRtLmDecisionResponse* response, size_t i) {
  float noul = -1.0f;
  EXPECT_EQ(
      litert_lm_experimental_decision_response_get_noul_at(response, i, &noul),
      kLiteRtLmStatusOk);
  return noul;
}

std::vector<float> Probabilities(const LiteRtLmDecisionResponse* response,
                                 size_t i) {
  size_t num = 0;
  EXPECT_EQ(litert_lm_experimental_decision_response_get_num_probabilities_at(
                response, i, &num),
            kLiteRtLmStatusOk);
  const float* probabilities = nullptr;
  EXPECT_EQ(litert_lm_experimental_decision_response_get_probabilities_at(
                response, i, &probabilities),
            kLiteRtLmStatusOk);
  if (probabilities == nullptr) return {};
  return std::vector<float>(probabilities, probabilities + num);
}

std::vector<std::string> OptionIds(const LiteRtLmDecisionResponse* response,
                                   size_t i) {
  size_t num = 0;
  EXPECT_EQ(litert_lm_experimental_decision_response_get_num_probabilities_at(
                response, i, &num),
            kLiteRtLmStatusOk);
  const char* const* ids = nullptr;
  EXPECT_EQ(litert_lm_experimental_decision_response_get_option_ids_at(response,
                                                                       i, &ids),
            kLiteRtLmStatusOk);
  if (ids == nullptr) return {};
  return std::vector<std::string>(ids, ids + num);
}

std::string Choice(const LiteRtLmDecisionResponse* response, size_t i) {
  const char* choice = nullptr;
  EXPECT_EQ(litert_lm_experimental_decision_response_get_choice_at(response, i,
                                                                   &choice),
            kLiteRtLmStatusOk);
  return choice == nullptr ? "" : choice;
}

// Appends the option `id` with `description` (NULL for none) to `question`.
LiteRtLmStatusCode AddOption(LiteRtLmDecisionQuestion* question, const char* id,
                             const char* description) {
  LiteRtLmDecisionOption* raw_option = nullptr;
  if (LiteRtLmStatusCode status =
          litert_lm_experimental_decision_option_create(id, &raw_option);
      status != kLiteRtLmStatusOk) {
    return status;
  }
  OptionPtr option(raw_option, &litert_lm_experimental_decision_option_delete);
  if (LiteRtLmStatusCode status =
          litert_lm_experimental_decision_option_set_description(option.get(),
                                                                 description);
      status != kLiteRtLmStatusOk) {
    return status;
  }
  return litert_lm_experimental_decision_question_add_option(question,
                                                             option.get());
}

float Sum(const std::vector<float>& values) {
  float sum = 0.0f;
  for (float v : values) sum += v;
  return sum;
}

// Adds a choice question `id` with the options `ids` and no descriptions.
void AddChoice(LiteRtLmDecisionRequest* request, const char* id,
               const std::vector<const char*>& ids) {
  QuestionPtr q = CreateChoiceQuestion(id);
  ASSERT_NE(q, nullptr);
  for (const char* option_id : ids) {
    ASSERT_EQ(AddOption(q.get(), option_id, nullptr), kLiteRtLmStatusOk);
  }
  ASSERT_EQ(
      litert_lm_experimental_decision_request_add_question(request, q.get()),
      kLiteRtLmStatusOk);
}

void AddScore(LiteRtLmDecisionRequest* request, const char* id,
              const std::vector<const char*>& levels,
              const char* instructions = "Rate it.") {
  QuestionPtr q = CreateScoreQuestion(id, instructions);
  ASSERT_NE(q, nullptr);
  for (const char* level : levels) {
    ASSERT_EQ(
        litert_lm_experimental_decision_question_add_level(q.get(), level),
        kLiteRtLmStatusOk);
  }
  ASSERT_EQ(
      litert_lm_experimental_decision_request_add_question(request, q.get()),
      kLiteRtLmStatusOk);
}

void AddNoul(LiteRtLmDecisionRequest* request, const char* id,
             const char* instructions = "Is it true?") {
  QuestionPtr q = CreateNoulQuestion(id, instructions);
  ASSERT_NE(q, nullptr);
  ASSERT_EQ(
      litert_lm_experimental_decision_request_add_question(request, q.get()),
      kLiteRtLmStatusOk);
}

TEST(DecisionEngineCTest, NullAndWrongVariantArguments) {
  LiteRtLmDecisionEngine* engine = nullptr;
  EXPECT_EQ(litert_lm_experimental_decision_engine_create(nullptr, &engine),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(litert_lm_experimental_decision_engine_create_from_engine(nullptr,
                                                                      &engine),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(litert_lm_experimental_decision_engine_create(nullptr, nullptr),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(litert_lm_experimental_decision_engine_create_from_engine(nullptr,
                                                                      nullptr),
            kLiteRtLmStatusInvalidArgument);

  RequestPtr request = CreateRequest();
  LiteRtLmDecisionResponse* response =
      reinterpret_cast<LiteRtLmDecisionResponse*>(1);
  EXPECT_EQ(litert_lm_experimental_decision_engine_decide(
                nullptr, nullptr, 0, request.get(), &response),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_STREQ(litert_lm_get_last_error_message(), "Invalid decision engine.");
  EXPECT_EQ(response, nullptr);

  // Question creation: NULL out_question, id, or instructions resets output.
  for (auto create_fn :
       {&litert_lm_experimental_decision_question_create_noul,
        &litert_lm_experimental_decision_question_create_choice,
        &litert_lm_experimental_decision_question_create_score}) {
    EXPECT_EQ(create_fn("q", "p?", nullptr), kLiteRtLmStatusInvalidArgument);
    LiteRtLmDecisionQuestion* raw_q =
        reinterpret_cast<LiteRtLmDecisionQuestion*>(1);
    EXPECT_EQ(create_fn(nullptr, "p?", &raw_q), kLiteRtLmStatusInvalidArgument);
    EXPECT_EQ(raw_q, nullptr);
    raw_q = reinterpret_cast<LiteRtLmDecisionQuestion*>(1);
    EXPECT_EQ(create_fn("q", nullptr, &raw_q), kLiteRtLmStatusInvalidArgument);
    EXPECT_EQ(raw_q, nullptr);
  }

  // Question setters and builders: NULL handles, NULL required strings, and
  // wrong question variants.
  QuestionPtr noul = CreateNoulQuestion("n");
  QuestionPtr choice = CreateChoiceQuestion("c");
  QuestionPtr score = CreateScoreQuestion("s");
  EXPECT_EQ(litert_lm_experimental_decision_question_set_true_description(
                nullptr, "yes"),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(litert_lm_experimental_decision_question_set_false_description(
                nullptr, "no"),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(litert_lm_experimental_decision_question_set_true_description(
                choice.get(), "yes"),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_STREQ(litert_lm_get_last_error_message(),
               "question is not a noul question.");
  EXPECT_EQ(litert_lm_experimental_decision_question_set_false_description(
                score.get(), "no"),
            kLiteRtLmStatusInvalidArgument);
  // NULL clears optional noul descriptions.
  EXPECT_EQ(litert_lm_experimental_decision_question_set_true_description(
                noul.get(), "yes"),
            kLiteRtLmStatusOk);
  EXPECT_EQ(litert_lm_experimental_decision_question_set_true_description(
                noul.get(), nullptr),
            kLiteRtLmStatusOk);
  EXPECT_EQ(litert_lm_experimental_decision_question_set_false_description(
                noul.get(), nullptr),
            kLiteRtLmStatusOk);

  LiteRtLmDecisionOption* raw_option =
      reinterpret_cast<LiteRtLmDecisionOption*>(0x1);
  EXPECT_EQ(litert_lm_experimental_decision_option_create(nullptr, &raw_option),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(raw_option, nullptr);
  EXPECT_EQ(litert_lm_experimental_decision_option_create("a", nullptr),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(
      litert_lm_experimental_decision_option_set_description(nullptr, "d"),
      kLiteRtLmStatusInvalidArgument);
  litert_lm_experimental_decision_option_delete(nullptr);
  ASSERT_EQ(litert_lm_experimental_decision_option_create("a", &raw_option),
            kLiteRtLmStatusOk);
  OptionPtr option(raw_option, &litert_lm_experimental_decision_option_delete);
  EXPECT_EQ(litert_lm_experimental_decision_question_add_option(nullptr,
                                                                option.get()),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(litert_lm_experimental_decision_question_add_option(choice.get(),
                                                                nullptr),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(litert_lm_experimental_decision_question_add_option(noul.get(),
                                                                option.get()),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_STREQ(litert_lm_get_last_error_message(),
               "question is not a choice question.");

  EXPECT_EQ(litert_lm_experimental_decision_question_add_level(nullptr, "low"),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(
      litert_lm_experimental_decision_question_add_level(score.get(), nullptr),
      kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(
      litert_lm_experimental_decision_question_add_level(choice.get(), "low"),
      kLiteRtLmStatusInvalidArgument);
  EXPECT_STREQ(litert_lm_get_last_error_message(),
               "question is not a score question.");

  // Request functions.
  EXPECT_EQ(litert_lm_experimental_decision_request_create(nullptr),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(litert_lm_experimental_decision_request_set_temperature(nullptr, 1),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(
      litert_lm_experimental_decision_request_add_question(nullptr, noul.get()),
      kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(litert_lm_experimental_decision_request_add_question(request.get(),
                                                                 nullptr),
            kLiteRtLmStatusInvalidArgument);

  size_t num_answers = 0;
  EXPECT_EQ(litert_lm_experimental_decision_response_get_num_answers(
                nullptr, &num_answers),
            kLiteRtLmStatusInvalidArgument);
  float value = 0.0f;
  EXPECT_EQ(
      litert_lm_experimental_decision_response_get_noul_at(nullptr, 0, &value),
      kLiteRtLmStatusInvalidArgument);

  litert_lm_experimental_decision_engine_delete(nullptr);
  litert_lm_experimental_decision_question_delete(nullptr);
  litert_lm_experimental_decision_request_delete(nullptr);
  litert_lm_experimental_decision_response_delete(nullptr);
}

TEST(DecisionEngineCTest, NullDecideArgumentsResetOutput) {
  DecisionEnginePtr engine = CreateTestEngine();
  ASSERT_NE(engine, nullptr);
  RequestPtr request = CreateRequest();
  AddNoul(request.get(), "q");
  LiteRtLmDecisionResponse* response =
      reinterpret_cast<LiteRtLmDecisionResponse*>(1);
  EXPECT_EQ(litert_lm_experimental_decision_engine_decide(
                engine.get(), nullptr, 0, nullptr, &response),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(response, nullptr);
  response = reinterpret_cast<LiteRtLmDecisionResponse*>(1);
  EXPECT_EQ(litert_lm_experimental_decision_engine_decide(
                engine.get(), nullptr, 1, request.get(), &response),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(response, nullptr);
  EXPECT_EQ(litert_lm_experimental_decision_engine_decide(
                engine.get(), nullptr, 0, request.get(), nullptr),
            kLiteRtLmStatusInvalidArgument);
}

TEST(DecisionEngineCTest, CreateFailsForMissingModel) {
  LiteRtLmEngineSettings* raw_settings = nullptr;
  ASSERT_EQ(
      litert_lm_engine_settings_create("/nonexistent/model.litertlm", "cpu",
                                       nullptr, nullptr, &raw_settings),
      kLiteRtLmStatusOk);
  EngineSettingsPtr settings(raw_settings, &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  LiteRtLmDecisionEngine* engine = reinterpret_cast<LiteRtLmDecisionEngine*>(1);
  EXPECT_NE(
      litert_lm_experimental_decision_engine_create(settings.get(), &engine),
      kLiteRtLmStatusOk);
  EXPECT_EQ(engine, nullptr);
}

// Adds the questions "topic" (choice), "refund" (noul) and "urgency" (score).
void AddTriageQuestions(LiteRtLmDecisionRequest* request) {
  QuestionPtr topic =
      CreateChoiceQuestion("topic", "What is the message about?");
  ASSERT_EQ(AddOption(topic.get(), "shipping", "Delivery and tracking"),
            kLiteRtLmStatusOk);
  ASSERT_EQ(AddOption(topic.get(), "billing", "Invoices and charges"),
            kLiteRtLmStatusOk);
  ASSERT_EQ(AddOption(topic.get(), "other", nullptr), kLiteRtLmStatusOk);
  ASSERT_EQ(litert_lm_experimental_decision_request_add_question(request,
                                                                 topic.get()),
            kLiteRtLmStatusOk);
  // `add_question` copies `topic`; mutating `topic` afterwards does not alter
  // the question already added to `request`.
  ASSERT_EQ(AddOption(topic.get(), "ignored_after_copy", nullptr),
            kLiteRtLmStatusOk);

  QuestionPtr refund =
      CreateNoulQuestion("refund", "Does the customer ask for a refund?");
  ASSERT_EQ(litert_lm_experimental_decision_question_set_true_description(
                refund.get(), "A refund is requested"),
            kLiteRtLmStatusOk);
  ASSERT_EQ(litert_lm_experimental_decision_question_set_false_description(
                refund.get(), "No refund is requested"),
            kLiteRtLmStatusOk);
  ASSERT_EQ(litert_lm_experimental_decision_request_add_question(request,
                                                                 refund.get()),
            kLiteRtLmStatusOk);

  AddScore(request, "urgency", {"Not urgent", "Somewhat urgent", "Very urgent"},
           "How urgent is this message?");
}

const std::vector<std::string>& TriageInput() {
  static const auto* const kInput = new std::vector<std::string>{
      "Customer support triage.", "I was charged twice. Please refund."};
  return *kInput;
}

TEST(DecisionEngineCTest, AnswersNoulChoiceAndScoreQuestions) {
  DecisionEnginePtr engine = CreateTestEngine();
  ASSERT_NE(engine, nullptr);
  RequestPtr request = CreateRequest();
  AddTriageQuestions(request.get());

  ResponsePtr response =
      DecideOrDie(engine.get(), TriageInput(), request.get());
  ASSERT_NE(response, nullptr);
  size_t num_answers = 0;
  ASSERT_EQ(litert_lm_experimental_decision_response_get_num_answers(
                response.get(), &num_answers),
            kLiteRtLmStatusOk);
  ASSERT_EQ(num_answers, 3);
  // Answers follow the order of the questions.
  EXPECT_EQ(AnswerId(response.get(), 0), "topic");
  EXPECT_EQ(AnswerId(response.get(), 1), "refund");
  EXPECT_EQ(AnswerId(response.get(), 2), "urgency");
  const LiteRtLmDecisionAnswerType kTypes[] = {
      kLiteRtLmDecisionAnswerTypeChoice, kLiteRtLmDecisionAnswerTypeNoul,
      kLiteRtLmDecisionAnswerTypeScore};
  for (size_t i = 0; i < 3; ++i) {
    LiteRtLmDecisionAnswerType type;
    ASSERT_EQ(litert_lm_experimental_decision_response_get_answer_type_at(
                  response.get(), i, &type),
              kLiteRtLmStatusOk);
    EXPECT_EQ(type, kTypes[i]);
  }

  // Choice: probabilities follow the order of the options (and exclude options
  // added to the builder after `add_question` copied it).
  EXPECT_EQ(OptionIds(response.get(), 0),
            (std::vector<std::string>{"shipping", "billing", "other"}));
  const std::vector<float> topic = Probabilities(response.get(), 0);
  ASSERT_EQ(topic.size(), 3);
  EXPECT_NEAR(Sum(topic), 1.0f, 1e-5);
  const std::string choice = Choice(response.get(), 0);
  const std::vector<std::string> ids = OptionIds(response.get(), 0);
  const size_t choice_index = absl::c_find(ids, choice) - ids.begin();
  ASSERT_LT(choice_index, 3) << choice;
  EXPECT_FLOAT_EQ(Confidence(response.get(), 0), topic[choice_index]);
  EXPECT_FLOAT_EQ(topic[choice_index],
                  *std::max_element(topic.begin(), topic.end()));

  // Noul.
  const float noul = Noul(response.get(), 1);
  EXPECT_GE(noul, 0.0f);
  EXPECT_LE(noul, 1.0f);
  EXPECT_FLOAT_EQ(Confidence(response.get(), 1), std::max(noul, 1.0f - noul));
  bool value = false;
  ASSERT_EQ(litert_lm_experimental_decision_response_get_noul_value_at(
                response.get(), 1, &value),
            kLiteRtLmStatusOk);
  EXPECT_EQ(value, noul >= 0.5f);

  // Score: the expected level and the most likely level.
  const std::vector<float> urgency = Probabilities(response.get(), 2);
  ASSERT_EQ(urgency.size(), 3);
  EXPECT_NEAR(Sum(urgency), 1.0f, 1e-5);
  float score = -1.0f;
  ASSERT_EQ(litert_lm_experimental_decision_response_get_score_at(
                response.get(), 2, &score),
            kLiteRtLmStatusOk);
  EXPECT_NEAR(score, urgency[1] + 2 * urgency[2], 1e-5);
  size_t level = 99;
  ASSERT_EQ(litert_lm_experimental_decision_response_get_most_likely_level_at(
                response.get(), 2, &level),
            kLiteRtLmStatusOk);
  EXPECT_EQ(level,
            std::max_element(urgency.begin(), urgency.end()) - urgency.begin());
  EXPECT_FLOAT_EQ(Confidence(response.get(), 2), urgency[level]);

  // No state is kept between calls, and the request can be reused.
  ResponsePtr second = DecideOrDie(engine.get(), TriageInput(), request.get());
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(Choice(second.get(), 0), choice);
  EXPECT_NEAR(Noul(second.get(), 1), noul, 1e-5);
  const std::vector<float> second_urgency = Probabilities(second.get(), 2);
  for (size_t k = 0; k < 3; ++k) {
    EXPECT_NEAR(second_urgency[k], urgency[k], 1e-5);
  }
}

TEST(DecisionEngineCTest, GettersRejectWrongTypesAndIndices) {
  DecisionEnginePtr engine = CreateTestEngine();
  ASSERT_NE(engine, nullptr);
  RequestPtr request = CreateRequest();
  AddTriageQuestions(request.get());
  ResponsePtr response =
      DecideOrDie(engine.get(), TriageInput(), request.get());
  ASSERT_NE(response, nullptr);
  const LiteRtLmDecisionResponse* r = response.get();

  float f = 0.0f;
  bool b = false;
  size_t level = 0;
  size_t num = 0;
  const char* s = "unchanged";
  const float* floats = reinterpret_cast<const float*>(1);
  const char* const* strings = reinterpret_cast<const char* const*>(1);
  // Answer 0 is a choice, 1 is a noul and 2 is a score answer.
  EXPECT_EQ(litert_lm_experimental_decision_response_get_noul_at(r, 0, &f),
            kLiteRtLmStatusNotFound);
  EXPECT_STREQ(litert_lm_get_last_error_message(),
               "No noul available at answer index 0.");
  EXPECT_EQ(
      litert_lm_experimental_decision_response_get_noul_value_at(r, 2, &b),
      kLiteRtLmStatusNotFound);
  EXPECT_EQ(litert_lm_experimental_decision_response_get_choice_at(r, 1, &s),
            kLiteRtLmStatusNotFound);
  EXPECT_EQ(s, nullptr);
  EXPECT_EQ(litert_lm_experimental_decision_response_get_score_at(r, 0, &f),
            kLiteRtLmStatusNotFound);
  EXPECT_EQ(litert_lm_experimental_decision_response_get_most_likely_level_at(
                r, 1, &level),
            kLiteRtLmStatusNotFound);
  EXPECT_EQ(litert_lm_experimental_decision_response_get_num_probabilities_at(
                r, 1, &num),
            kLiteRtLmStatusNotFound);
  EXPECT_EQ(litert_lm_experimental_decision_response_get_probabilities_at(
                r, 1, &floats),
            kLiteRtLmStatusNotFound);
  EXPECT_EQ(floats, nullptr);
  EXPECT_EQ(litert_lm_experimental_decision_response_get_option_ids_at(
                r, 2, &strings),
            kLiteRtLmStatusNotFound);
  EXPECT_EQ(strings, nullptr);

  // Index 3 is out of range for every getter.
  LiteRtLmDecisionAnswerType type;
  EXPECT_EQ(litert_lm_experimental_decision_response_get_answer_id_at(r, 3, &s),
            kLiteRtLmStatusOutOfRange);
  EXPECT_STREQ(litert_lm_get_last_error_message(),
               "Answer index 3 is out of range; the response has 3 answer(s).");
  EXPECT_EQ(
      litert_lm_experimental_decision_response_get_answer_type_at(r, 3, &type),
      kLiteRtLmStatusOutOfRange);
  EXPECT_EQ(
      litert_lm_experimental_decision_response_get_confidence_at(r, 3, &f),
      kLiteRtLmStatusOutOfRange);
  EXPECT_EQ(litert_lm_experimental_decision_response_get_noul_at(r, 3, &f),
            kLiteRtLmStatusOutOfRange);
  EXPECT_EQ(
      litert_lm_experimental_decision_response_get_noul_value_at(r, 3, &b),
      kLiteRtLmStatusOutOfRange);
  EXPECT_EQ(litert_lm_experimental_decision_response_get_choice_at(r, 3, &s),
            kLiteRtLmStatusOutOfRange);
  EXPECT_EQ(litert_lm_experimental_decision_response_get_score_at(r, 3, &f),
            kLiteRtLmStatusOutOfRange);
  EXPECT_EQ(litert_lm_experimental_decision_response_get_most_likely_level_at(
                r, 3, &level),
            kLiteRtLmStatusOutOfRange);
  EXPECT_EQ(litert_lm_experimental_decision_response_get_num_probabilities_at(
                r, 3, &num),
            kLiteRtLmStatusOutOfRange);
  EXPECT_EQ(litert_lm_experimental_decision_response_get_probabilities_at(
                r, 3, &floats),
            kLiteRtLmStatusOutOfRange);
  EXPECT_EQ(litert_lm_experimental_decision_response_get_option_ids_at(
                r, 3, &strings),
            kLiteRtLmStatusOutOfRange);

  // NULL output pointers.
  EXPECT_EQ(
      litert_lm_experimental_decision_response_get_num_answers(r, nullptr),
      kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(
      litert_lm_experimental_decision_response_get_answer_id_at(r, 0, nullptr),
      kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(
      litert_lm_experimental_decision_response_get_score_at(r, 2, nullptr),
      kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(litert_lm_experimental_decision_response_get_probabilities_at(
                r, 0, nullptr),
            kLiteRtLmStatusInvalidArgument);
}

TEST(DecisionEngineCTest, OptionalDescriptionsAndTemperature) {
  DecisionEnginePtr engine = CreateTestEngine();
  ASSERT_NE(engine, nullptr);
  // A very high temperature flattens the probabilities towards uniform.
  RequestPtr hot = CreateRequest();
  AddChoice(hot.get(), "topic", {"billing", "shipping", "other"});
  ASSERT_EQ(
      litert_lm_experimental_decision_request_set_temperature(hot.get(), 1e6f),
      kLiteRtLmStatusOk);
  ResponsePtr response = DecideOrDie(engine.get(), {"hi there"}, hot.get());
  ASSERT_NE(response, nullptr);
  for (float p : Probabilities(response.get(), 0)) {
    EXPECT_NEAR(p, 1.0f / 3, 1e-3);
  }

  for (float temperature : {0.0f, -1.0f}) {
    RequestPtr bad = CreateRequest();
    AddNoul(bad.get(), "q");
    ASSERT_EQ(litert_lm_experimental_decision_request_set_temperature(
                  bad.get(), temperature),
              kLiteRtLmStatusOk);
    EXPECT_EQ(Decide(engine.get(), {"hi"}, bad.get()),
              kLiteRtLmStatusInvalidArgument)
        << temperature;
  }
}

TEST(DecisionEngineCTest, NonAsciiTextAndOptionIdsRoundTrip) {
  DecisionEnginePtr engine = CreateTestEngine();
  ASSERT_NE(engine, nullptr);
  RequestPtr request = CreateRequest();
  QuestionPtr dept =
      CreateChoiceQuestion("department", "这条消息属于哪个部门？");
  ASSERT_EQ(AddOption(dept.get(), "物流", "配送"), kLiteRtLmStatusOk);
  ASSERT_EQ(AddOption(dept.get(), "账单", "扣款和发票"), kLiteRtLmStatusOk);
  ASSERT_EQ(litert_lm_experimental_decision_request_add_question(request.get(),
                                                                 dept.get()),
            kLiteRtLmStatusOk);
  ResponsePtr response =
      DecideOrDie(engine.get(), {"我被重复扣款了，请退款。"}, request.get());
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(OptionIds(response.get(), 0),
            (std::vector<std::string>{"物流", "账单"}));
  const std::string choice = Choice(response.get(), 0);
  EXPECT_TRUE(choice == "物流" || choice == "账单") << choice;

  // Question IDs must be ASCII.
  RequestPtr bad = CreateRequest();
  AddNoul(bad.get(), "问题");
  EXPECT_EQ(Decide(engine.get(), {"hi"}, bad.get()),
            kLiteRtLmStatusInvalidArgument);
}

TEST(DecisionEngineCTest, ImageAndAudioInputsAreUnimplemented) {
  DecisionEnginePtr engine = CreateTestEngine();
  ASSERT_NE(engine, nullptr);
  RequestPtr request = CreateRequest();
  AddNoul(request.get(), "q", "Is it a cat?");
  for (LiteRtLmInputDataType type :
       {kLiteRtLmInputDataTypeImage, kLiteRtLmInputDataTypeAudio}) {
    InputDataPtr text = CreateInput(kLiteRtLmInputDataTypeText, "Look:");
    InputDataPtr media = CreateInput(type, std::string(16, '\0'));
    const LiteRtLmInputData* inputs[] = {text.get(), media.get()};
    LiteRtLmDecisionResponse* response = nullptr;
    EXPECT_EQ(litert_lm_experimental_decision_engine_decide(
                  engine.get(), inputs, 2, request.get(), &response),
              kLiteRtLmStatusUnimplemented)
        << type;
    EXPECT_EQ(response, nullptr);
  }
}

TEST(DecisionEngineCTest, MoreThan26OptionsOrLevelsIsUnimplemented) {
  DecisionEnginePtr engine = CreateTestEngine();
  ASSERT_NE(engine, nullptr);
  std::vector<std::string> storage;
  for (int i = 0; i < 27; ++i) storage.push_back(absl::StrCat("o", i));
  std::vector<const char*> strings;
  for (const std::string& s : storage) strings.push_back(s.c_str());

  RequestPtr choice = CreateRequest();
  AddChoice(choice.get(), "q", strings);
  EXPECT_EQ(Decide(engine.get(), {"hi"}, choice.get()),
            kLiteRtLmStatusUnimplemented);

  RequestPtr score = CreateRequest();
  AddScore(score.get(), "q", strings);
  EXPECT_EQ(Decide(engine.get(), {"hi"}, score.get()),
            kLiteRtLmStatusUnimplemented);
}

TEST(DecisionEngineCTest, RejectsInvalidRequests) {
  DecisionEnginePtr engine = CreateTestEngine();
  ASSERT_NE(engine, nullptr);

  RequestPtr valid = CreateRequest();
  AddNoul(valid.get(), "q");
  // No input, and no non-empty text input.
  EXPECT_EQ(Decide(engine.get(), {}, valid.get()),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_EQ(Decide(engine.get(), {""}, valid.get()),
            kLiteRtLmStatusInvalidArgument);

  // No questions.
  RequestPtr empty = CreateRequest();
  EXPECT_EQ(Decide(engine.get(), {"hi"}, empty.get()),
            kLiteRtLmStatusInvalidArgument);

  std::vector<RequestPtr> bad;
  // Duplicate and invalid question IDs, empty instructions.
  bad.push_back(CreateRequest());
  AddNoul(bad.back().get(), "q");
  AddNoul(bad.back().get(), "q");
  bad.push_back(CreateRequest());
  AddNoul(bad.back().get(), "");
  bad.push_back(CreateRequest());
  AddNoul(bad.back().get(), "a b");
  bad.push_back(CreateRequest());
  AddNoul(bad.back().get(), "q", "");
  // Option counts and IDs.
  bad.push_back(CreateRequest());
  AddChoice(bad.back().get(), "q", {});
  bad.push_back(CreateRequest());
  AddChoice(bad.back().get(), "q", {"a"});
  bad.push_back(CreateRequest());
  AddChoice(bad.back().get(), "q", {"a", "a"});
  bad.push_back(CreateRequest());
  AddChoice(bad.back().get(), "q", {" a", "b"});
  bad.push_back(CreateRequest());
  AddChoice(bad.back().get(), "q", {"a", ""});
  // Level counts and descriptions.
  bad.push_back(CreateRequest());
  AddScore(bad.back().get(), "q", {});
  bad.push_back(CreateRequest());
  AddScore(bad.back().get(), "q", {"only"});
  bad.push_back(CreateRequest());
  AddScore(bad.back().get(), "q", {"low", ""});
  for (size_t i = 0; i < bad.size(); ++i) {
    EXPECT_EQ(Decide(engine.get(), {"hi"}, bad[i].get()),
              kLiteRtLmStatusInvalidArgument)
        << i;
  }
}

TEST(DecisionEngineCTest, CreateFromEngineSharesWeightsAndOutlivesHandle) {
  EngineSettingsPtr settings = CreateTestSettings();
  ASSERT_NE(settings, nullptr);
  LiteRtLmEngine* raw_base_engine = nullptr;
  ASSERT_EQ(litert_lm_engine_create(settings.get(), &raw_base_engine),
            kLiteRtLmStatusOk);
  std::unique_ptr<LiteRtLmEngine, decltype(&litert_lm_engine_delete)>
      base_engine(raw_base_engine, &litert_lm_engine_delete);

  LiteRtLmDecisionEngine* raw_decision_engine = nullptr;
  ASSERT_EQ(litert_lm_experimental_decision_engine_create_from_engine(
                base_engine.get(), &raw_decision_engine),
            kLiteRtLmStatusOk);
  DecisionEnginePtr decision_engine(
      raw_decision_engine, &litert_lm_experimental_decision_engine_delete);

  RequestPtr request = CreateRequest();
  AddNoul(request.get(), "refund", "Refund?");
  ResponsePtr before = DecideOrDie(decision_engine.get(),
                                   {"I was charged twice."}, request.get());
  ASSERT_NE(before, nullptr);

  // A generative session can be created on the shared engine at any time.
  LiteRtLmSession* raw_session = nullptr;
  EXPECT_EQ(
      litert_lm_engine_create_session(base_engine.get(), nullptr, &raw_session),
      kLiteRtLmStatusOk);
  litert_lm_session_delete(raw_session);

  // Decisions still work after the generative session; `decision_engine` is
  // destroyed before `base_engine` (reverse declaration order).
  ResponsePtr after = DecideOrDie(decision_engine.get(),
                                  {"I was charged twice."}, request.get());
  ASSERT_NE(after, nullptr);
  EXPECT_NEAR(Noul(after.get(), 0), Noul(before.get(), 0), 1e-5);
}

TEST(DecisionEngineCTest, InputLongerThanContextIsInvalidArgumentOnCpu) {
  DecisionEnginePtr engine = CreateTestEngine();
  ASSERT_NE(engine, nullptr);
  // The input alone does not fit in the context of the test model.
  std::string input;
  for (int i = 0; i < 200; ++i) {
    absl::StrAppend(&input, "word", i, " ");
  }
  RequestPtr request = CreateRequest();
  AddNoul(request.get(), "q", "Ok?");
  // The CPU backend rejects a prefill longer than the context as an invalid
  // argument.
  EXPECT_EQ(Decide(engine.get(), {input}, request.get()),
            kLiteRtLmStatusInvalidArgument);
  EXPECT_FALSE(std::string(litert_lm_get_last_error_message()).empty());
}

}  // namespace
