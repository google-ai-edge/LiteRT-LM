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

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "c/c_api_internal.h"
#include "c/engine.h"
#include "c/engine_internal.h"  // IWYU pragma: keep
#include "c/error_reporter.h"
#include "c/error_reporter_internal.h"
#include "runtime/core/llm_decision_engine.h"
#include "runtime/engine/decision_engine.h"
#include "runtime/engine/engine_settings.h"

struct LiteRtLmDecisionEngine {
  std::unique_ptr<litert::lm::DecisionEngine> engine;
};

struct LiteRtLmDecisionQuestion {
  litert::lm::DecisionQuestion question;
};

struct LiteRtLmDecisionOption {
  litert::lm::ChoiceQuestion::Criterion criterion;
};

// The questions and options of a `litert::lm::DecisionRequest`; the input is
// passed to `litert_lm_experimental_decision_engine_decide`, so that a request
// can be reused for several inputs.
struct LiteRtLmDecisionRequest {
  std::vector<litert::lm::DecisionQuestion> questions;
  std::optional<float> temperature;
};

struct LiteRtLmDecisionResponse {
  // Probabilities and option IDs of a choice answer, as C arrays.
  struct ChoiceArrays {
    std::vector<float> probabilities;
    std::vector<const char*> option_ids;
  };

  explicit LiteRtLmDecisionResponse(litert::lm::DecisionResponse r)
      : response(std::move(r)), choice_arrays(response.answers.size()) {
    for (size_t i = 0; i < response.answers.size(); ++i) {
      const auto* choice =
          std::get_if<litert::lm::ChoiceAnswer>(&response.answers[i].type);
      if (choice == nullptr) continue;
      for (const litert::lm::ChoiceAnswer::Probability& p :
           choice->probabilities) {
        choice_arrays[i].probabilities.push_back(p.probability);
        choice_arrays[i].option_ids.push_back(p.id.c_str());
      }
    }
  }

  LiteRtLmDecisionResponse(const LiteRtLmDecisionResponse&) = delete;
  LiteRtLmDecisionResponse& operator=(const LiteRtLmDecisionResponse&) = delete;
  LiteRtLmDecisionResponse(LiteRtLmDecisionResponse&&) = delete;
  LiteRtLmDecisionResponse& operator=(LiteRtLmDecisionResponse&&) = delete;

  litert::lm::DecisionResponse response;
  // Indexed like `response.answers`; empty for other answer types.
  std::vector<ChoiceArrays> choice_arrays;
};

namespace {

using ::litert::lm::ChoiceAnswer;
using ::litert::lm::ChoiceQuestion;
using ::litert::lm::DecisionAnswer;
using ::litert::lm::DecisionQuestion;
using ::litert::lm::NoulAnswer;
using ::litert::lm::NoulQuestion;
using ::litert::lm::ScoreAnswer;
using ::litert::lm::ScoreQuestion;
using ::litert::lm::c::ReturnError;

std::string OptionalString(const char* s) { return s == nullptr ? "" : s; }

template <typename T>
LiteRtLmStatusCode CreateQuestion(const char* id, const char* instructions,
                                  LiteRtLmDecisionQuestion** out_question) {
  LITERT_LM_C_RETURN_IF_NULL(out_question);
  *out_question = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(id);
  LITERT_LM_C_RETURN_IF_NULL(instructions);
  *out_question = new LiteRtLmDecisionQuestion{
      DecisionQuestion{.id = id, .instructions = instructions, .type = T{}}};
  return kLiteRtLmStatusOk;
}

struct AnswerTypeVisitor {
  LiteRtLmDecisionAnswerType operator()(const NoulAnswer&) const {
    return kLiteRtLmDecisionAnswerTypeNoul;
  }
  LiteRtLmDecisionAnswerType operator()(const ChoiceAnswer&) const {
    return kLiteRtLmDecisionAnswerTypeChoice;
  }
  LiteRtLmDecisionAnswerType operator()(const ScoreAnswer&) const {
    return kLiteRtLmDecisionAnswerTypeScore;
  }
};

// Returns the answer at `index` of `response`, or records an error and returns
// NULL if `response` is NULL or `index` is out of range.
const DecisionAnswer* GetAnswer(const LiteRtLmDecisionResponse* response,
                                size_t index, LiteRtLmStatusCode* status) {
  if (response == nullptr) {
    *status = ReturnError(absl::StatusCode::kInvalidArgument,
                          "response must not be NULL.");
    return nullptr;
  }
  const size_t num_answers = response->response.answers.size();
  if (index >= num_answers) {
    *status = ReturnError(
        absl::StatusCode::kOutOfRange,
        absl::StrFormat("Answer index %d is out of range; the response has %d "
                        "answer(s).",
                        index, num_answers));
    return nullptr;
  }
  return &response->response.answers[index];
}

// Like `GetAnswer`, but returns the typed answer `T` at `index`, or records a
// `kNotFound` error if the answer has another type.
template <typename T>
const T* GetTypedAnswer(const LiteRtLmDecisionResponse* response, size_t index,
                        absl::string_view what, LiteRtLmStatusCode* status) {
  const DecisionAnswer* answer = GetAnswer(response, index, status);
  if (answer == nullptr) return nullptr;
  const T* typed = std::get_if<T>(&answer->type);
  if (typed == nullptr) {
    *status = ReturnError(
        absl::StatusCode::kNotFound,
        absl::StrFormat("No %s available at answer index %d.", what, index));
  }
  return typed;
}

}  // namespace

extern "C" {

// -----------------------------------------------------------------------------
// Decision engine
// -----------------------------------------------------------------------------

LiteRtLmStatusCode litert_lm_experimental_decision_engine_create(
    const LiteRtLmEngineSettings* engine_settings,
    LiteRtLmDecisionEngine** out_engine) {
  LITERT_LM_C_RETURN_IF_NULL(out_engine);
  *out_engine = nullptr;
  if (!litert::lm::c::IsValidEngineSettings(engine_settings)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  LITERT_LM_C_ASSIGN_OR_RETURN(
      auto engine,
      litert::lm::LlmDecisionEngine::Create(*engine_settings->settings));
  *out_engine = new LiteRtLmDecisionEngine{std::move(engine)};
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_experimental_decision_engine_create_from_engine(
    LiteRtLmEngine* engine, LiteRtLmDecisionEngine** out_engine) {
  LITERT_LM_C_RETURN_IF_NULL(out_engine);
  *out_engine = nullptr;
  if (!litert::lm::c::IsValidEngine(engine)) {
    return kLiteRtLmStatusInvalidArgument;
  }
  LITERT_LM_C_ASSIGN_OR_RETURN(
      auto decision_engine,
      litert::lm::LlmDecisionEngine::Create(*engine->engine));
  *out_engine = new LiteRtLmDecisionEngine{std::move(decision_engine)};
  return kLiteRtLmStatusOk;
}

void litert_lm_experimental_decision_engine_delete(
    LiteRtLmDecisionEngine* engine) {
  delete engine;
}

LiteRtLmStatusCode litert_lm_experimental_decision_engine_decide(
    LiteRtLmDecisionEngine* engine, const LiteRtLmInputData* const* inputs,
    size_t num_inputs, const LiteRtLmDecisionRequest* request,
    LiteRtLmDecisionResponse** out_response) {
  LITERT_LM_C_RETURN_IF_NULL(out_response);
  *out_response = nullptr;
  if (engine == nullptr || engine->engine == nullptr) {
    return ReturnError(absl::StatusCode::kInvalidArgument,
                       "Invalid decision engine.");
  }
  LITERT_LM_C_RETURN_IF_NULL(request);
  litert::lm::DecisionRequest engine_request{
      .questions = request->questions,
      .temperature = request->temperature,
  };
  LITERT_LM_C_ASSIGN_OR_RETURN(
      engine_request.input,
      litert::lm::c::ToEngineInputData(inputs, num_inputs));
  LITERT_LM_C_ASSIGN_OR_RETURN(litert::lm::DecisionResponse response,
                               engine->engine->Decide(engine_request));
  *out_response = new LiteRtLmDecisionResponse(std::move(response));
  return kLiteRtLmStatusOk;
}

// -----------------------------------------------------------------------------
// Decision question
// -----------------------------------------------------------------------------

LiteRtLmStatusCode litert_lm_experimental_decision_question_create_noul(
    const char* id, const char* instructions,
    LiteRtLmDecisionQuestion** out_question) {
  return CreateQuestion<NoulQuestion>(id, instructions, out_question);
}

LiteRtLmStatusCode litert_lm_experimental_decision_question_create_choice(
    const char* id, const char* instructions,
    LiteRtLmDecisionQuestion** out_question) {
  return CreateQuestion<ChoiceQuestion>(id, instructions, out_question);
}

LiteRtLmStatusCode litert_lm_experimental_decision_question_create_score(
    const char* id, const char* instructions,
    LiteRtLmDecisionQuestion** out_question) {
  return CreateQuestion<ScoreQuestion>(id, instructions, out_question);
}

void litert_lm_experimental_decision_question_delete(
    LiteRtLmDecisionQuestion* question) {
  delete question;
}

LiteRtLmStatusCode
litert_lm_experimental_decision_question_set_true_description(
    LiteRtLmDecisionQuestion* question, const char* description) {
  LITERT_LM_C_RETURN_IF_NULL(question);
  auto* noul = std::get_if<NoulQuestion>(&question->question.type);
  if (noul == nullptr) {
    return ReturnError(absl::StatusCode::kInvalidArgument,
                       "question is not a noul question.");
  }
  noul->true_description = OptionalString(description);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode
litert_lm_experimental_decision_question_set_false_description(
    LiteRtLmDecisionQuestion* question, const char* description) {
  LITERT_LM_C_RETURN_IF_NULL(question);
  auto* noul = std::get_if<NoulQuestion>(&question->question.type);
  if (noul == nullptr) {
    return ReturnError(absl::StatusCode::kInvalidArgument,
                       "question is not a noul question.");
  }
  noul->false_description = OptionalString(description);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_experimental_decision_option_create(
    const char* id, LiteRtLmDecisionOption** out_option) {
  LITERT_LM_C_RETURN_IF_NULL(out_option);
  *out_option = nullptr;
  LITERT_LM_C_RETURN_IF_NULL(id);
  *out_option = new LiteRtLmDecisionOption{.criterion = {.id = id}};
  return kLiteRtLmStatusOk;
}

void litert_lm_experimental_decision_option_delete(
    LiteRtLmDecisionOption* option) {
  delete option;
}

LiteRtLmStatusCode litert_lm_experimental_decision_option_set_description(
    LiteRtLmDecisionOption* option, const char* description) {
  LITERT_LM_C_RETURN_IF_NULL(option);
  option->criterion.description = OptionalString(description);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_experimental_decision_question_add_option(
    LiteRtLmDecisionQuestion* question, const LiteRtLmDecisionOption* option) {
  LITERT_LM_C_RETURN_IF_NULL(question);
  LITERT_LM_C_RETURN_IF_NULL(option);
  auto* choice = std::get_if<ChoiceQuestion>(&question->question.type);
  if (choice == nullptr) {
    return ReturnError(absl::StatusCode::kInvalidArgument,
                       "question is not a choice question.");
  }
  choice->criteria.push_back(option->criterion);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_experimental_decision_question_add_level(
    LiteRtLmDecisionQuestion* question, const char* description) {
  LITERT_LM_C_RETURN_IF_NULL(question);
  LITERT_LM_C_RETURN_IF_NULL(description);
  auto* score = std::get_if<ScoreQuestion>(&question->question.type);
  if (score == nullptr) {
    return ReturnError(absl::StatusCode::kInvalidArgument,
                       "question is not a score question.");
  }
  score->criteria.emplace_back(description);
  return kLiteRtLmStatusOk;
}

// -----------------------------------------------------------------------------
// Decision request
// -----------------------------------------------------------------------------

LiteRtLmStatusCode litert_lm_experimental_decision_request_create(
    LiteRtLmDecisionRequest** out_request) {
  LITERT_LM_C_RETURN_IF_NULL(out_request);
  *out_request = new LiteRtLmDecisionRequest();
  return kLiteRtLmStatusOk;
}

void litert_lm_experimental_decision_request_delete(
    LiteRtLmDecisionRequest* request) {
  delete request;
}

LiteRtLmStatusCode litert_lm_experimental_decision_request_set_temperature(
    LiteRtLmDecisionRequest* request, float temperature) {
  LITERT_LM_C_RETURN_IF_NULL(request);
  request->temperature = temperature;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_experimental_decision_request_add_question(
    LiteRtLmDecisionRequest* request,
    const LiteRtLmDecisionQuestion* question) {
  LITERT_LM_C_RETURN_IF_NULL(request);
  LITERT_LM_C_RETURN_IF_NULL(question);
  request->questions.push_back(question->question);
  return kLiteRtLmStatusOk;
}

// -----------------------------------------------------------------------------
// Decision response
// -----------------------------------------------------------------------------

void litert_lm_experimental_decision_response_delete(
    LiteRtLmDecisionResponse* response) {
  delete response;
}

LiteRtLmStatusCode litert_lm_experimental_decision_response_get_num_answers(
    const LiteRtLmDecisionResponse* response, size_t* out_num_answers) {
  LITERT_LM_C_RETURN_IF_NULL(out_num_answers);
  LITERT_LM_C_RETURN_IF_NULL(response);
  *out_num_answers = response->response.answers.size();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_experimental_decision_response_get_answer_id_at(
    const LiteRtLmDecisionResponse* response, size_t index,
    const char** out_id) {
  LITERT_LM_C_RETURN_IF_NULL(out_id);
  *out_id = nullptr;
  LiteRtLmStatusCode status = kLiteRtLmStatusOk;
  const DecisionAnswer* answer = GetAnswer(response, index, &status);
  if (answer == nullptr) return status;
  *out_id = answer->id.c_str();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_experimental_decision_response_get_answer_type_at(
    const LiteRtLmDecisionResponse* response, size_t index,
    LiteRtLmDecisionAnswerType* out_type) {
  LITERT_LM_C_RETURN_IF_NULL(out_type);
  LiteRtLmStatusCode status = kLiteRtLmStatusOk;
  const DecisionAnswer* answer = GetAnswer(response, index, &status);
  if (answer == nullptr) return status;
  *out_type = std::visit(AnswerTypeVisitor{}, answer->type);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_experimental_decision_response_get_confidence_at(
    const LiteRtLmDecisionResponse* response, size_t index,
    float* out_confidence) {
  LITERT_LM_C_RETURN_IF_NULL(out_confidence);
  LiteRtLmStatusCode status = kLiteRtLmStatusOk;
  const DecisionAnswer* answer = GetAnswer(response, index, &status);
  if (answer == nullptr) return status;
  *out_confidence = answer->confidence;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_experimental_decision_response_get_noul_at(
    const LiteRtLmDecisionResponse* response, size_t index, float* out_noul) {
  LITERT_LM_C_RETURN_IF_NULL(out_noul);
  LiteRtLmStatusCode status = kLiteRtLmStatusOk;
  const auto* noul =
      GetTypedAnswer<NoulAnswer>(response, index, "noul", &status);
  if (noul == nullptr) return status;
  *out_noul = noul->noul;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_experimental_decision_response_get_noul_value_at(
    const LiteRtLmDecisionResponse* response, size_t index, bool* out_value) {
  LITERT_LM_C_RETURN_IF_NULL(out_value);
  LiteRtLmStatusCode status = kLiteRtLmStatusOk;
  const auto* noul =
      GetTypedAnswer<NoulAnswer>(response, index, "noul value", &status);
  if (noul == nullptr) return status;
  *out_value = noul->value;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_experimental_decision_response_get_choice_at(
    const LiteRtLmDecisionResponse* response, size_t index,
    const char** out_choice) {
  LITERT_LM_C_RETURN_IF_NULL(out_choice);
  *out_choice = nullptr;
  LiteRtLmStatusCode status = kLiteRtLmStatusOk;
  const auto* choice =
      GetTypedAnswer<ChoiceAnswer>(response, index, "choice", &status);
  if (choice == nullptr) return status;
  *out_choice = choice->choice.c_str();
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_experimental_decision_response_get_score_at(
    const LiteRtLmDecisionResponse* response, size_t index, float* out_score) {
  LITERT_LM_C_RETURN_IF_NULL(out_score);
  LiteRtLmStatusCode status = kLiteRtLmStatusOk;
  const auto* score =
      GetTypedAnswer<ScoreAnswer>(response, index, "score", &status);
  if (score == nullptr) return status;
  *out_score = score->score;
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode
litert_lm_experimental_decision_response_get_most_likely_level_at(
    const LiteRtLmDecisionResponse* response, size_t index,
    size_t* out_most_likely_level) {
  LITERT_LM_C_RETURN_IF_NULL(out_most_likely_level);
  LiteRtLmStatusCode status = kLiteRtLmStatusOk;
  const auto* score = GetTypedAnswer<ScoreAnswer>(response, index,
                                                  "most likely level", &status);
  if (score == nullptr) return status;
  *out_most_likely_level = static_cast<size_t>(score->most_likely_level);
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode
litert_lm_experimental_decision_response_get_num_probabilities_at(
    const LiteRtLmDecisionResponse* response, size_t index,
    size_t* out_num_probabilities) {
  LITERT_LM_C_RETURN_IF_NULL(out_num_probabilities);
  LiteRtLmStatusCode status = kLiteRtLmStatusOk;
  const DecisionAnswer* answer = GetAnswer(response, index, &status);
  if (answer == nullptr) return status;
  if (const auto* score = std::get_if<ScoreAnswer>(&answer->type)) {
    *out_num_probabilities = score->probabilities.size();
  } else if (const auto* choice = std::get_if<ChoiceAnswer>(&answer->type)) {
    *out_num_probabilities = choice->probabilities.size();
  } else {
    return ReturnError(
        absl::StatusCode::kNotFound,
        absl::StrFormat("No probabilities available at answer index %d.",
                        index));
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode
litert_lm_experimental_decision_response_get_probabilities_at(
    const LiteRtLmDecisionResponse* response, size_t index,
    const float** out_probabilities) {
  LITERT_LM_C_RETURN_IF_NULL(out_probabilities);
  *out_probabilities = nullptr;
  LiteRtLmStatusCode status = kLiteRtLmStatusOk;
  const DecisionAnswer* answer = GetAnswer(response, index, &status);
  if (answer == nullptr) return status;
  if (const auto* score = std::get_if<ScoreAnswer>(&answer->type)) {
    *out_probabilities = score->probabilities.data();
  } else if (std::holds_alternative<ChoiceAnswer>(answer->type)) {
    *out_probabilities = response->choice_arrays[index].probabilities.data();
  } else {
    return ReturnError(
        absl::StatusCode::kNotFound,
        absl::StrFormat("No probabilities available at answer index %d.",
                        index));
  }
  return kLiteRtLmStatusOk;
}

LiteRtLmStatusCode litert_lm_experimental_decision_response_get_option_ids_at(
    const LiteRtLmDecisionResponse* response, size_t index,
    const char* const** out_option_ids) {
  LITERT_LM_C_RETURN_IF_NULL(out_option_ids);
  *out_option_ids = nullptr;
  LiteRtLmStatusCode status = kLiteRtLmStatusOk;
  if (GetTypedAnswer<ChoiceAnswer>(response, index, "option IDs", &status) ==
      nullptr) {
    return status;
  }
  *out_option_ids = response->choice_arrays[index].option_ids.data();
  return kLiteRtLmStatusOk;
}

}  // extern "C"
