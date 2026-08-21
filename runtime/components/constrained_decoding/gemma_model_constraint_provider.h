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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_GEMMA_MODEL_CONSTRAINT_PROVIDER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_GEMMA_MODEL_CONSTRAINT_PROVIDER_H_

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "sentencepiece_processor.h"  // from @sentencepiece

#ifndef GEMMA_MODEL_CONSTRAINT_PROVIDER_EXPORT
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || \
    defined(__NT__) || defined(_WIN64)
#ifdef GEMMA_MODEL_CONSTRAINT_PROVIDER_BUILD_DLL
#define GEMMA_MODEL_CONSTRAINT_PROVIDER_EXPORT __declspec(dllexport)
#else
#define GEMMA_MODEL_CONSTRAINT_PROVIDER_EXPORT __declspec(dllimport)
#endif  // GEMMA_MODEL_CONSTRAINT_PROVIDER_BUILD_DLL
#else
#define GEMMA_MODEL_CONSTRAINT_PROVIDER_EXPORT \
  __attribute__((visibility("default")))
#endif
#endif  // GEMMA_MODEL_CONSTRAINT_PROVIDER_EXPORT

extern "C" {

// Opaque pointer for litert::lm::GemmaModelConstraintProvider
typedef struct LiteRtLmGemmaModelConstraintProvider
    LiteRtLmGemmaModelConstraintProvider;

// Opaque pointer for litert::lm::Constraint
typedef struct LiteRtLmConstraint LiteRtLmConstraint;

// --- Enums & Structs ---

// Maps to litert::lm::GemmaFuncallFormat
typedef enum LiteRtLmGemmaFuncallFormat {
  kLiteRtLmGemmaFuncallFormatPythonStyle = 0,
  kLiteRtLmGemmaFuncallFormatFcStyle = 1,
} LiteRtLmGemmaFuncallFormat;

// Maps to litert::lm::GemmaConstraintMode
typedef enum LiteRtLmGemmaConstraintMode {
  kLiteRtLmGemmaConstraintModeTextAndOr = 0,
  kLiteRtLmGemmaConstraintModeFunctionCallOnly = 1,
} LiteRtLmGemmaConstraintMode;

// Maps to litert::lm::GemmaModelConstraintOptions
typedef struct LiteRtLmGemmaModelConstraintOptions {
  LiteRtLmGemmaFuncallFormat funcall_format;
  LiteRtLmGemmaConstraintMode constraint_mode;
  const char* code_fence_start;         // e.g. <start_function_call>
  const char* code_fence_end;           // e.g. <end_function_call>
  const char* open_quote;               // e.g. <escape>
  const char* close_quote;              // e.g. <escape>
  const char* function_response_start;  // e.g. <start_function_response>
} LiteRtLmGemmaModelConstraintOptions;

// --- C Functions ---

// Creates the GemmaModelConstraintProvider.
//
// @param serialized_sp_model_proto: Serialized SentencePiece model proto.
// @param serialized_sp_model_proto_len: Length of the serialized SentencePiece
// model proto.
// @param stop_token_ids: Array of arrays of ints. Can be NULL if num_stop_lists
// is 0.
// @param stop_token_lengths: Array containing length of each sub-array.
// @param num_stop_lists: Number of sub-arrays.
// @return Handle to the provider, or NULL if creation failed (e.g., status
// error).
GEMMA_MODEL_CONSTRAINT_PROVIDER_EXPORT LiteRtLmGemmaModelConstraintProvider*
LiteRtLmGemmaModelConstraintProvider_Create(
    const char* serialized_sp_model_proto, size_t serialized_sp_model_proto_len,
    const int** stop_token_ids, const size_t* stop_token_lengths,
    size_t num_stop_lists);

// Destroys the provider instance.
GEMMA_MODEL_CONSTRAINT_PROVIDER_EXPORT void
LiteRtLmGemmaModelConstraintProvider_Destroy(
    LiteRtLmGemmaModelConstraintProvider* provider);

// Creates a constraint from JSON tools and options.
//
// @param provider: The provider handle.
// @param json_tools_str: The tools defined in JSON format (string).
// @param options: Formatting options.
// @return Handle to the created Constraint, or NULL on failure (parsing or
// status error).
GEMMA_MODEL_CONSTRAINT_PROVIDER_EXPORT LiteRtLmConstraint*
LiteRtLmGemmaModelConstraintProvider_CreateConstraintFromTools(
    LiteRtLmGemmaModelConstraintProvider* provider, const char* json_tools_str,
    const LiteRtLmGemmaModelConstraintOptions* options);

// Destroys a generic Constraint instance created by the provider.
GEMMA_MODEL_CONSTRAINT_PROVIDER_EXPORT void LiteRtLmConstraint_Destroy(
    LiteRtLmConstraint* constraint);
}

namespace litert::lm {

// Supported function call formats for Gemma models.
enum class GemmaFuncallFormat {
  // Python-like funcall format.
  kPythonStyle,
  // Simplified JSON-based funcall format.
  kFcStyle,
};

// Supported constraint modes for Gemma models.
enum class GemmaConstraintMode {
  kTextAndOr,         // Both function call and text output are allowed.
  kFunctionCallOnly,  // Only function call is allowed.
};

// Options for formatting constraints regex.
struct GemmaModelConstraintOptions {
  GemmaFuncallFormat funcall_format = GemmaFuncallFormat::kPythonStyle;
  GemmaConstraintMode constraint_mode = GemmaConstraintMode::kTextAndOr;
  std::string code_fence_start;         // e.g. <start_function_call>
  std::string code_fence_end;           // e.g. <end_function_call>
  std::string open_quote;               // e.g. <escape>
  std::string close_quote;              // e.g. <escape>
  std::string function_response_start;  // e.g. <start_function_response>
};

// Provides constraints for Gemma models, leveraging the techniques described in
// https://arxiv.org/abs/2404.07362.
class GemmaModelConstraintProvider : public ConstraintProvider {
 public:
  static absl::StatusOr<std::unique_ptr<GemmaModelConstraintProvider>> Create(
      std::unique_ptr<sentencepiece::SentencePieceProcessor> processor,
      const std::vector<std::vector<int>>& stop_token_ids = {});

  // Creates a constraint based on the given tools and options. The constraint
  // will match single or multiple function calls or normal text.
  absl::StatusOr<std::unique_ptr<Constraint>> CreateConstraint(
      const nlohmann::ordered_json& tools,
      const GemmaModelConstraintOptions& options);

  absl::StatusOr<std::unique_ptr<Constraint>> CreateConstraint(
      ConstraintArg constraint_arg) const override;

 private:
  explicit GemmaModelConstraintProvider(
      std::unique_ptr<sentencepiece::SentencePieceProcessor> processor,
      std::unique_ptr<ConstraintProvider> internal_provider)
      : processor_(std::move(processor)),
        internal_provider_(std::move(internal_provider)) {};

  std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_;
  std::unique_ptr<ConstraintProvider> internal_provider_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_GEMMA_MODEL_CONSTRAINT_PROVIDER_H_
