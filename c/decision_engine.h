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

#ifndef THIRD_PARTY_ODML_LITERT_LM_C_DECISION_ENGINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_C_DECISION_ENGINE_H_

#include <stdbool.h>
#include <stddef.h>

#if defined(__APPLE__)
#include "api_export.h"      // NOLINT
#include "engine.h"          // NOLINT
#include "error_reporter.h"  // NOLINT
#else
#include "c/api_export.h"
#include "c/engine.h"
#include "c/error_reporter.h"
#endif

// =============================================================================
// WARNING: The APIs declared in this header are EXPERIMENTAL and subject to
// change or removal without notice. API stability and backward compatibility
// are not guaranteed.
// =============================================================================
//
// TODO(shuangfeng): Drop the "experimental_" prefix from the function names
// once the API is stable.

// A decision engine answers typed questions about a shared input with
// calibrated probabilities instead of generated text. The questions and answers
// follow the concepts of the Gemini Decisions API:
//
// - A noul question asks whether something is true; its answer is the
//   probability `noul` that it is.
// - A choice question picks one of 2 to 26 options; its answer is the most
//   likely option and the probability of each option.
// - A score question rates the input on 2 to 26 ordered levels; its answer is
//   the expected level, the most likely level and the probability of each
//   level.
//
// Usage: build each `LiteRtLmDecisionQuestion` with the
// `litert_lm_experimental_decision_question_*` functions and add it to a
// `LiteRtLmDecisionRequest` with
// `litert_lm_experimental_decision_request_add_question`, pass the request with
// the input to `litert_lm_experimental_decision_engine_decide`, and read the
// answers of the returned `LiteRtLmDecisionResponse` with the
// `litert_lm_experimental_decision_response_*` functions.
//
// The create functions below currently create the LLM implementation
// (`LlmDecisionEngine`), which scores the options by the model's next-token
// probabilities over the option labels, without autoregressive decoding. All
// questions of one request share the prefill of the input where the model
// allows it. Other implementations may get their own create functions.
//
// No state is kept between calls: every call to
// `litert_lm_experimental_decision_engine_decide` starts from an empty context.
//
// Limits: the input must be text, and a question can have at most 26 options
// or levels.
//
// Threading follows "Threading model" in engine.h, which describes current
// behavior, not a commitment: calls on one decision engine are serialized
// internally, so one decision engine may be used from multiple threads, but its
// calls do not run in parallel. A decision engine created with
// `litert_lm_experimental_decision_engine_create_from_engine` is another user
// of the source engine: when single-threaded execution is enabled, its calls
// MUST NOT overlap with calls on the source engine or on any session or
// conversation created from it. Questions, requests and responses are not
// thread-safe.

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer for the LiteRT LM Decision Engine.
//
// Added in version 1.1.0.
typedef struct LiteRtLmDecisionEngine LiteRtLmDecisionEngine;

// Opaque pointer for a decision question, added to a `LiteRtLmDecisionRequest`
// with `litert_lm_experimental_decision_request_add_question`.
//
// Added in version 1.1.0.
typedef struct LiteRtLmDecisionQuestion LiteRtLmDecisionQuestion;

// Opaque pointer for an option of a choice question, added to the question
// with `litert_lm_experimental_decision_question_add_option`.
//
// Added in version 1.1.0.
typedef struct LiteRtLmDecisionOption LiteRtLmDecisionOption;

// Opaque pointer for a decision request: the questions to answer and the
// options of the decision.
//
// Added in version 1.1.0.
typedef struct LiteRtLmDecisionRequest LiteRtLmDecisionRequest;

// Opaque pointer for a decision response: one answer per question.
//
// Added in version 1.1.0.
typedef struct LiteRtLmDecisionResponse LiteRtLmDecisionResponse;

// The type of a decision answer, which is the type of its question. New values
// may be added in later versions; callers must handle unknown values.
//
// Added in version 1.1.0.
typedef enum {
  kLiteRtLmDecisionAnswerTypeNoul = 0,
  kLiteRtLmDecisionAnswerTypeChoice = 1,
  kLiteRtLmDecisionAnswerTypeScore = 2,
} LiteRtLmDecisionAnswerType;

// -----------------------------------------------------------------------------
// Decision engine
// -----------------------------------------------------------------------------

// Creates a LiteRT LM Decision Engine that scores the options with the LLM
// configured in `engine_settings`.
//
// @param engine_settings The engine settings, e.g. created with
//   `litert_lm_engine_settings_create`. The settings are copied; the caller
//   keeps ownership and may delete them after this call returns.
// @param out_engine On success, receives the created engine, owned by the
//   caller; release with `litert_lm_experimental_decision_engine_delete`. Set
//   to NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `out_engine` or `engine_settings` is NULL; otherwise the code of the error
//   that prevented the engine from being created (e.g. the model file could
//   not be opened).
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_engine_create(
    const LiteRtLmEngineSettings* engine_settings,
    LiteRtLmDecisionEngine** out_engine);

// Creates a LiteRT LM Decision Engine sharing an existing `LiteRtLmEngine`'s
// weights, tokenizer, and executor.
//
// Like sessions and conversations (see "Ownership and lifetime" in engine.h),
// the created decision engine MUST be destroyed before `engine`. `engine` may
// still be used for generative sessions meanwhile. The underlying engine runs
// the work of all its sessions one at a time, so decisions and generative
// sessions on the same engine may wait for each other. See the threading note
// at the top of this file.
//
// @param engine An initialized `LiteRtLmEngine`.
// @param out_engine On success, receives the created decision engine, owned by
//   the caller; release with `litert_lm_experimental_decision_engine_delete`.
//   Set to NULL on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h). kLiteRtLmStatusInvalidArgument if
//   `out_engine` or `engine` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_engine_create_from_engine(
    LiteRtLmEngine* engine, LiteRtLmDecisionEngine** out_engine);

// Destroys a LiteRT LM Decision Engine.
//
// @param engine The engine to destroy. NULL is a no-op.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_experimental_decision_engine_delete(
    LiteRtLmDecisionEngine* engine);

// Answers the questions of `request` about `inputs`.
//
// The request is validated here:
// - It has 1 to 1000 questions with unique IDs of 1 to 64 characters in
//   `[A-Za-z0-9_-]`, and non-empty instructions.
// - A choice question has 2 to 26 options with unique IDs. An option ID is
//   non-empty, at most 128 bytes, with no control characters and no leading or
//   trailing whitespace.
// - A score question has 2 to 26 non-empty level descriptions.
// - The temperature, if set, is finite and > 0.
//
// @param engine The decision engine.
// @param inputs The shared input, as an array of `num_inputs` input data. Text
//   inputs are joined with newlines. Image and audio inputs are not supported
//   yet.
// @param num_inputs The number of inputs; must be at least 1.
// @param request The questions to answer. Not modified; it may be reused.
// @param out_response On success, receives the answers, owned by the caller;
//   release with `litert_lm_experimental_decision_response_delete`. Set to NULL
//   on failure.
// @return kLiteRtLmStatusOk on success, or another LiteRtLmStatusCode on
//   failure (see error_reporter.h):
//   - kLiteRtLmStatusInvalidArgument if `engine`, `request` or `out_response`
//     is NULL, `inputs` is NULL while `num_inputs` is non-zero, there is no
//     non-empty text input, or the request breaks one of the rules above.
//   - kLiteRtLmStatusUnimplemented if a question has more than 26 options or
//     levels, or if an input is an image or audio.
//   - kLiteRtLmStatusInvalidArgument or kLiteRtLmStatusResourceExhausted,
//     depending on the backend, if the prompt of a question does not fit in
//     the context window of the model (InvalidArgument on CPU).
//   - Otherwise the code of the runtime error that prevented the decision.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_engine_decide(
    LiteRtLmDecisionEngine* engine, const LiteRtLmInputData* const* inputs,
    size_t num_inputs, const LiteRtLmDecisionRequest* request,
    LiteRtLmDecisionResponse** out_response);

// -----------------------------------------------------------------------------
// Decision question
// -----------------------------------------------------------------------------
//
// The `create_*`, `set_*` and `add_*` functions copy their string arguments and
// only check for NULL arguments and matching question types; semantic rules
// (ID format, non-empty instructions, option/level counts) are validated by
// `litert_lm_experimental_decision_engine_decide`.

// Creates a noul (true or false) question. Optional descriptions for the true
// and false outcomes can be configured with
// `litert_lm_experimental_decision_question_set_true_description` and
// `litert_lm_experimental_decision_question_set_false_description`.
//
// @param id The question ID, a NUL-terminated string.
// @param instructions The question, guidelines or rubric to evaluate.
// @param out_question On success, receives the created question, owned by the
//   caller; release with `litert_lm_experimental_decision_question_delete`. Set
//   to NULL on failure.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `out_question`, `id` or `instructions` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_question_create_noul(
    const char* id, const char* instructions,
    LiteRtLmDecisionQuestion** out_question);

// Creates a choice question over mutually exclusive options. Add options with
// `litert_lm_experimental_decision_question_add_option`.
//
// @param id The question ID, a NUL-terminated string.
// @param instructions The question, guidelines or rubric to evaluate.
// @param out_question On success, receives the created question, owned by the
//   caller; release with `litert_lm_experimental_decision_question_delete`. Set
//   to NULL on failure.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `out_question`, `id` or `instructions` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_question_create_choice(
    const char* id, const char* instructions,
    LiteRtLmDecisionQuestion** out_question);

// Creates a score question over ordered levels. Add levels from lowest to
// highest with `litert_lm_experimental_decision_question_add_level`.
//
// @param id The question ID, a NUL-terminated string.
// @param instructions The question, guidelines or rubric to evaluate.
// @param out_question On success, receives the created question, owned by the
//   caller; release with `litert_lm_experimental_decision_question_delete`. Set
//   to NULL on failure.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `out_question`, `id` or `instructions` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_question_create_score(
    const char* id, const char* instructions,
    LiteRtLmDecisionQuestion** out_question);

// Destroys a decision question.
//
// @param question The question to destroy. NULL is a no-op.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_experimental_decision_question_delete(
    LiteRtLmDecisionQuestion* question);

// Sets the optional description of when the answer of a noul question is true,
// shown to the model as the true option. Passing NULL or "" clears it.
//
// @param question The noul question to modify.
// @param description Optional description of when the answer is true, or NULL
//   to clear.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `question` is NULL or is not a noul question (leaving `question`
//   unchanged).
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_experimental_decision_question_set_true_description(
    LiteRtLmDecisionQuestion* question, const char* description);

// Sets the optional description of when the answer of a noul question is false,
// shown to the model as the false option. Passing NULL or "" clears it.
//
// @param question The noul question to modify.
// @param description Optional description of when the answer is false, or NULL
//   to clear.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `question` is NULL or is not a noul question (leaving `question`
//   unchanged).
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_experimental_decision_question_set_false_description(
    LiteRtLmDecisionQuestion* question, const char* description);

// Creates an option of a choice question.
//
// @param id The option ID, returned in the choice answer; a NUL-terminated
//   string.
// @param out_option On success, receives the created option, owned by the
//   caller; release with `litert_lm_experimental_decision_option_delete`. Set
//   to NULL on failure.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `id` or `out_option` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_option_create(
    const char* id, LiteRtLmDecisionOption** out_option);

// Destroys an option.
//
// @param option The option to destroy. NULL is a no-op.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_experimental_decision_option_delete(
    LiteRtLmDecisionOption* option);

// Sets the optional description of an option, shown to the model instead of
// its ID. Passing NULL or "" clears it.
//
// @param option The option to modify.
// @param description Optional description, or NULL to clear.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `option` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_option_set_description(
    LiteRtLmDecisionOption* option, const char* description);

// Appends `option` to a choice question. `option` is copied, so the caller
// keeps ownership of `option` and may delete or reuse it after this call
// returns. Options are shown to the model in the order they are added, labeled
// A to Z.
//
// @param question The choice question to modify.
// @param option The option to append.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `question` or `option` is NULL, or if `question` is not a choice question
//   (leaving `question` unchanged).
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_question_add_option(
    LiteRtLmDecisionQuestion* question, const LiteRtLmDecisionOption* option);

// Appends a level description to a score question, from the lowest level to the
// highest. Level `k` (0-based, in addition order) has the value `k`.
//
// @param question The score question to modify.
// @param description The level description.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `question` or `description` is NULL, or if `question` is not a score
//   question (leaving `question` unchanged).
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_question_add_level(
    LiteRtLmDecisionQuestion* question, const char* description);

// -----------------------------------------------------------------------------
// Decision request
// -----------------------------------------------------------------------------

// Creates an empty decision request.
//
// @param out_request On success, receives the request, owned by the caller;
//   release with `litert_lm_experimental_decision_request_delete`. Set to NULL
//   on failure.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `out_request` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_request_create(
    LiteRtLmDecisionRequest** out_request);

// Destroys a decision request.
//
// @param request The request to destroy. NULL is a no-op.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_experimental_decision_request_delete(
    LiteRtLmDecisionRequest* request);

// Sets the softmax temperature of all questions:
// `p_i = softmax(log_likelihood_i / temperature)`. If it is not set, the engine
// uses its calibrated temperature for the model.
//
// @param request The request.
// @param temperature The temperature; must be finite and > 0 (checked by
//   `litert_lm_experimental_decision_engine_decide`).
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `request` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_request_set_temperature(
    LiteRtLmDecisionRequest* request, float temperature);

// Appends `question` to `request`. `question` is copied, so the caller keeps
// ownership of `question` and may delete or reuse it after this call returns.
// Answers follow the order in which questions are added.
//
// @param request The request to modify.
// @param question The question to append.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `request` or `question` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_request_add_question(
    LiteRtLmDecisionRequest* request, const LiteRtLmDecisionQuestion* question);

// -----------------------------------------------------------------------------
// Decision response
// -----------------------------------------------------------------------------
//
// Answers are indexed in question order, from 0 to the number of answers minus
// 1. The `*_at` getters take the answer index and return:
// - kLiteRtLmStatusInvalidArgument if `response` or the output pointer is
//   NULL,
// - kLiteRtLmStatusOutOfRange if `index` is not less than the number of
//   answers,
// - kLiteRtLmStatusNotFound if the answer at `index` has no such value, i.e.
//   is of another type (e.g. the score of a noul answer).
// Scalar output values are left unchanged on failure, and output pointers are
// set to NULL. Strings and arrays are owned by `response` and valid until it is
// deleted.

// Destroys a decision response.
//
// @param response The response to destroy. NULL is a no-op.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
void litert_lm_experimental_decision_response_delete(
    LiteRtLmDecisionResponse* response);

// Returns the number of answers, which is the number of questions.
//
// @param out_num_answers On success, receives the number of answers. Left
//   unchanged on failure.
// @return kLiteRtLmStatusOk on success, or kLiteRtLmStatusInvalidArgument if
//   `response` or `out_num_answers` is NULL.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_response_get_num_answers(
    const LiteRtLmDecisionResponse* response, size_t* out_num_answers);

// Returns the ID of the question answered at `index`.
//
// @param out_id On success, receives the question ID, owned by `response`.
//   Set to NULL on failure.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_response_get_answer_id_at(
    const LiteRtLmDecisionResponse* response, size_t index,
    const char** out_id);

// Returns the type of the answer at `index`.
//
// @param out_type On success, receives the answer type. Left unchanged on
//   failure.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_response_get_answer_type_at(
    const LiteRtLmDecisionResponse* response, size_t index,
    LiteRtLmDecisionAnswerType* out_type);

// Returns the calibrated probability of the answer at `index`, in [0, 1]:
// `max(noul, 1 - noul)` for a noul answer, and the probability of the most
// likely option or level otherwise.
//
// @param out_confidence On success, receives the confidence. Left unchanged on
//   failure.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_response_get_confidence_at(
    const LiteRtLmDecisionResponse* response, size_t index,
    float* out_confidence);

// Returns the calibrated probability that the noul answer at `index` is true,
// in [0, 1].
//
// @param out_noul On success, receives the probability. Left unchanged on
//   failure.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_response_get_noul_at(
    const LiteRtLmDecisionResponse* response, size_t index, float* out_noul);

// Returns the most likely value of the noul answer at `index`: `noul >= 0.5`.
//
// @param out_value On success, receives the value. Left unchanged on failure.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_response_get_noul_value_at(
    const LiteRtLmDecisionResponse* response, size_t index, bool* out_value);

// Returns the ID of the most likely option of the choice answer at `index`
// (the first one on ties).
//
// @param out_choice On success, receives the option ID, owned by `response`.
//   Set to NULL on failure.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_response_get_choice_at(
    const LiteRtLmDecisionResponse* response, size_t index,
    const char** out_choice);

// Returns the expected level `sum_k k * p_k` of the score answer at `index`,
// in [0, K-1].
//
// @param out_score On success, receives the expected level. Left unchanged on
//   failure.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_response_get_score_at(
    const LiteRtLmDecisionResponse* response, size_t index, float* out_score);

// Returns the most likely level of the score answer at `index` (the lowest one
// on ties).
//
// @param out_most_likely_level On success, receives the 0-based level, an
//   index into the probabilities. Left unchanged on failure.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_experimental_decision_response_get_most_likely_level_at(
    const LiteRtLmDecisionResponse* response, size_t index,
    size_t* out_most_likely_level);

// Returns the number of probabilities of the choice or score answer at
// `index`: the number of options or levels of its question.
//
// @param out_num_probabilities On success, receives the number of
//   probabilities. Left unchanged on failure.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_experimental_decision_response_get_num_probabilities_at(
    const LiteRtLmDecisionResponse* response, size_t index,
    size_t* out_num_probabilities);

// Returns the calibrated probabilities of the choice or score answer at
// `index`, which sum to 1: one per option in the order of the question's
// options, or one per level indexed by level.
//
// @param out_probabilities On success, receives a pointer to an array whose
//   length is given by
//   `litert_lm_experimental_decision_response_get_num_probabilities_at`,
//   owned by `response`. Set to NULL on failure.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode
litert_lm_experimental_decision_response_get_probabilities_at(
    const LiteRtLmDecisionResponse* response, size_t index,
    const float** out_probabilities);

// Returns the option IDs of the choice answer at `index`, in the order of the
// question's options, matching `get_probabilities_at`.
//
// @param out_option_ids On success, receives a pointer to an array of
//   NUL-terminated strings whose length is given by
//   `litert_lm_experimental_decision_response_get_num_probabilities_at`,
//   owned by `response`. Set to NULL on failure.
//
// Added in version 1.1.0.
LITERT_LM_C_API_EXPORT
LiteRtLmStatusCode litert_lm_experimental_decision_response_get_option_ids_at(
    const LiteRtLmDecisionResponse* response, size_t index,
    const char* const** out_option_ids);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_C_DECISION_ENGINE_H_
