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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_DECISION_ENGINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_DECISION_ENGINE_H_

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/engine/io_types.h"

namespace litert::lm {

// Decision API: typed questions about one shared input, answered with
// calibrated probabilities instead of generated text.
//
// The concepts and field names follow the Gemini Decisions API, so its
// requests and responses map onto these types field by field. Questions and
// answers are ordered vectors with an `id` (instead of maps keyed by ID), and
// the question and answer types are `std::variant`s.

// A true / false question.
struct NoulQuestion {
  // Optional descriptions of when the answer is true and when it is false,
  // shown to the model as the two options.
  std::string true_description;
  std::string false_description;
};

// A single-select question over mutually exclusive options.
struct ChoiceQuestion {
  struct Criterion {
    // Option ID, returned in `ChoiceAnswer`. Non-empty, at most 128 bytes, no
    // control characters and no leading or trailing whitespace.
    std::string id;
    // Optional description shown to the model instead of `id`.
    std::string description;
  };
  // 2 to 26 options with unique IDs, shown to the model in this order (labeled
  // `A`..`Z`).
  std::vector<Criterion> criteria;
};

// A rubric question over ordered levels.
struct ScoreQuestion {
  // 2 to 26 non-empty level descriptions, from the lowest level to the
  // highest. Level `i` (0-based) has the value `i`.
  std::vector<std::string> criteria;
};

struct DecisionQuestion {
  // Unique within the request: 1 to 64 characters in `[A-Za-z0-9_-]`.
  std::string id;
  // The question, guidelines or rubric to evaluate against the input.
  // Required.
  std::string instructions;
  // The question type and its criteria.
  std::variant<NoulQuestion, ChoiceQuestion, ScoreQuestion> type;
};

struct DecisionRequest {
  // Shared input, evaluated once for all questions and treated as data, not
  // as instructions. Only `InputText` is supported for now; image and audio
  // inputs return `Unimplemented`.
  std::vector<InputData> input;
  // 1 to 1000 questions about `input`.
  std::vector<DecisionQuestion> questions;
  // Optional softmax temperature (finite, > 0) for all questions:
  // `p_i = softmax(log_likelihood_i / temperature)`. If unset, the
  // implementation uses its calibrated temperature for the model.
  std::optional<float> temperature;
};

// Answer to a `NoulQuestion`.
struct NoulAnswer {
  // `noul >= 0.5`.
  bool value = false;
  // Calibrated probability that the answer is true, in [0, 1].
  float noul = 0.0f;
};

// Answer to a `ChoiceQuestion`.
struct ChoiceAnswer {
  struct Probability {
    // `ChoiceQuestion::Criterion::id`.
    std::string id;
    float probability = 0.0f;
  };
  // ID of the most likely option (the first one on ties).
  std::string choice;
  // Calibrated probability of each option, in `ChoiceQuestion::criteria`
  // order; sums to 1.
  std::vector<Probability> probabilities;
};

// Answer to a `ScoreQuestion`.
struct ScoreAnswer {
  // Expected level `sum_k k * probabilities[k]`, in [0, K-1].
  float score = 0.0f;
  // Most likely level (the lowest one on ties).
  int most_likely_level = 0;
  // Calibrated probability of each level, indexed by level; sums to 1.
  std::vector<float> probabilities;
};

struct DecisionAnswer {
  // `DecisionQuestion::id` of the question answered.
  std::string id;
  // Calibrated probability of the answer: `max(noul, 1 - noul)` for noul
  // questions, and the probability of `choice` or `most_likely_level`
  // otherwise.
  float confidence = 0.0f;
  // Type-specific answer, matching the question type.
  std::variant<NoulAnswer, ChoiceAnswer, ScoreAnswer> type;
};

struct DecisionResponse {
  // One answer per question, in the order of `DecisionRequest::questions`.
  std::vector<DecisionAnswer> answers;
};

// Interface for decision engines. Implementations are created by their own
// factories, e.g. `LlmDecisionEngine::Create` in
// `runtime/core/llm_decision_engine.h`, which scores the options with an LLM.
//
// Example:
// ```cpp
// std::vector<InputData> input;
// input.push_back(InputText(
//     R"({"title": "Battery died after two weeks", )"
//     R"("review": "Sound is great, but stopped charging."})"));
// DecisionRequest request{
//     .input = std::move(input),
//     .questions = {
//         {.id = "defect",
//          .instructions = "Does the review report a product defect?",
//          .type = NoulQuestion{}},
//         {.id = "sentiment",
//          .instructions = "What is the overall sentiment of the review?",
//          .type = ChoiceQuestion{
//              .criteria = {{"positive", "Mostly satisfied"},
//                           {"mixed", "Praise and complaints in one review"},
//                           {"negative", "Mostly dissatisfied"}}}},
//         {.id = "severity",
//          .instructions = "How severe is the reported problem?",
//          .type = ScoreQuestion{
//              .criteria = {"Cosmetic", "Inconvenient", "Product unusable"}}},
//     },
// };
// ASSIGN_OR_RETURN(DecisionResponse response, engine->Decide(request));
//
// // answers[0]: confidence = 0.97f, NoulAnswer{.value = true, .noul = 0.97f}
// // answers[1]: confidence = 0.91f,
// //   ChoiceAnswer{.choice = "mixed",
// //                .probabilities = {{"positive", 0.03f}, {"mixed", 0.91f},
// //                                  {"negative", 0.06f}}}
// // answers[2]: confidence = 0.87f,
// //   ScoreAnswer{.score = 1.85f, .most_likely_level = 2,
// //               .probabilities = {0.02f, 0.11f, 0.87f}}
// if (const auto* sentiment =
//         std::get_if<ChoiceAnswer>(&response.answers[1].type)) {
//   ...
// }
// ```
class DecisionEngine {
 public:
  virtual ~DecisionEngine() = default;

  // Answers `request.questions` about `request.input`.
  virtual absl::StatusOr<DecisionResponse> Decide(
      const DecisionRequest& request) = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_DECISION_ENGINE_H_
