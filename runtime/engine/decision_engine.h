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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// Canonical type of a decision question.
enum class DecisionQuestionType {
  // Binary yes/no judgment (calibrated probability of "true"/"yes" in
  // [0.0, 1.0]). Equivalent to `"noul"` (Normalized Utility) in JEV and
  // Perplexity Decisions API.
  kBoolean,
  kNoul = kBoolean,
  // Categorical selection among K >= 2 unordered options.
  kChoice,
  // Ordinal rating across K >= 2 ordered levels.
  kScore,
};

// One candidate option (for `kChoice`) or rating criterion level (for
// `kScore`).
struct DecisionOption {
  // Unique identifier returned in `DecisionResult::selected_key` (e.g.,
  // "billing", "true", "5").
  std::string key;
  // Optional natural-language description or rubric criterion explaining this
  // option to the model.
  std::string description;
};

// A typed decision question evaluated against `DecisionRequest::contents`.
struct DecisionQuestion {
  // Unique question identifier within the request.
  std::string id;
  // Question type (`kBoolean` / `kNoul`, `kChoice`, or `kScore`).
  DecisionQuestionType type = DecisionQuestionType::kChoice;
  // Natural-language question, statement, or rating instructions to evaluate.
  std::string prompt;
  // Candidate options or rating criteria:
  // - `kBoolean` / `kNoul`: empty (defaults to ["false", "true"]) or 2 options
  //   [no, yes].
  // - `kChoice`: at least 2 options.
  // - `kScore`: empty (defaults to ["1".."5"]) or at least 2 ordered levels
  //   from lowest to highest.
  std::vector<DecisionOption> options;
};

// Request passed to `DecisionEngine::Decide`.
struct DecisionRequest {
  // Input contents / state to evaluate (e.g., text, serialized JSON state,
  // image, or audio; aligned with `EmbeddingEngine::ComputeEmbedding` and
  // `SessionInterface::RunPrefill`). Corresponds to `"state"` in JEV and
  // Perplexity Decisions API.
  std::vector<InputData> contents;
  // One or more typed questions evaluated over `contents`.
  std::vector<DecisionQuestion> questions;
};

// Calibrated probability of one option in `DecisionResult`.
struct DecisionOptionProbability {
  std::string key;
  float probability = 0.0f;
};

// Calibrated decision result for one `DecisionQuestion`.
struct DecisionResult {
  // Matches `DecisionQuestion::id`.
  std::string id;
  // Key of the winning option ("false"/"true" for default `kBoolean`/`kNoul`).
  std::string selected_key;
  // Full probability distribution over all options (sums to 1.0).
  std::vector<DecisionOptionProbability> probabilities;
  // Probability-weighted expected value:
  // - `kBoolean` / `kNoul`: P("true") in [0.0, 1.0] (0/1 Bernoulli expectation,
  //   equivalent to the scalar `"noul"` output in JEV / Perplexity).
  // - `kScore`: sum_i p_i * v_i over numeric option levels (default 1..K).
  // - `kChoice`: `std::nullopt`.
  std::optional<float> expected_score = std::nullopt;
  // Normalized Shannon certainty `1 - H(p) / log(K)` in [0.0, 1.0].
  float confidence = 0.0f;
};

// Response returned by `DecisionEngine::Decide`.
struct DecisionResponse {
  // One result per question in `request.questions`, in the same order.
  std::vector<DecisionResult> results;
};

// Interface for typed, calibrated multi-question decision making.
//
// `DecisionEngine` evaluates one or more typed questions (`kBoolean` / `kNoul`,
// `kChoice`, `kScore`) over `request.contents` (`std::vector<InputData>`). All
// questions in a `Decide()` call share a single prefill of `contents`, and the
// engine retains the KV cache of the shared prefix across consecutive calls so
// repeated or related requests reuse the longest common token prefix.
//
// Example query & response:
// ```cpp
// std::vector<InputData> contents;
// contents.push_back(InputText(
//     R"({"title": "Battery died after two weeks", )"
//     R"("review": "Sound is great, but stopped charging."})"));
// DecisionRequest request{
//     .contents = std::move(contents),
//     .questions = {
//         {.id = "defect",
//          .type = DecisionQuestionType::kBoolean,  // or kNoul
//          .prompt = "Does the review report a product defect?"},
//         {.id = "sentiment",
//          .type = DecisionQuestionType::kChoice,
//          .prompt = "What is the overall sentiment of the review?",
//          .options = {{"positive", "Mostly satisfied"},
//                      {"mixed", "Praise and complaints in one review"},
//                      {"negative", "Mostly dissatisfied"}}},
//         {.id = "severity",
//          .type = DecisionQuestionType::kScore,
//          .prompt = "How severe is the reported problem?",
//          .options = {{"1", "Cosmetic"},
//                      {"2", "Inconvenient"},
//                      {"3", "Product unusable"}}},
//     },
// };
// ASSIGN_OR_RETURN(DecisionResponse response, engine->Decide(request));
//
// // response.results[0] ("defect", kBoolean / kNoul):
// //   selected_key = "true", expected_score = 0.99f (P(true)),
// //   confidence = 0.92f, probabilities = {{"false", 0.01f}, {"true", 0.99f}}
// //
// // response.results[1] ("sentiment", kChoice):
// //   selected_key = "mixed", expected_score = std::nullopt,
// //   confidence = 0.78f,
// //   probabilities = {{"positive", 0.03f}, {"mixed", 0.91f},
// //                    {"negative", 0.06f}}
// //
// // response.results[2] ("severity", kScore):
// //   selected_key = "3", expected_score = 2.85f,
// //   confidence = 0.74f,
// //   probabilities = {{"1", 0.02f}, {"2", 0.11f}, {"3", 0.87f}}
// ```
class DecisionEngine {
 public:
  virtual ~DecisionEngine() = default;

  // Creates a `DecisionEngine` for the model configured in `engine_settings`.
  static absl::StatusOr<std::unique_ptr<DecisionEngine>> Create(
      EngineSettings engine_settings);

  // Evaluates `request.questions` about `request.contents` and returns
  // calibrated probability distributions and expected scores for each question.
  virtual absl::StatusOr<DecisionResponse> Decide(
      const DecisionRequest& request) = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_DECISION_ENGINE_H_
