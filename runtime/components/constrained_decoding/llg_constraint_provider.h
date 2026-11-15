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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LLG_CONSTRAINT_PROVIDER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LLG_CONSTRAINT_PROVIDER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/constraint_provider.h"
#include "runtime/components/constrained_decoding/constraint_provider_config.h"
#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "support/tokenizer/tokenizer.h"

struct LlgTokenizer;

namespace litert::lm {

using ::litert::support::Tokenizer;

class LlgConstraintProvider : public ConstraintProvider {
 public:
  // Creates a provider that uses `llg_config` as it is. For the tool call
  // constraints of a model data processor, use
  // `CreateLlgToolCallConstraintProvider` instead.
  static absl::StatusOr<std::unique_ptr<ConstraintProvider>> Create(
      const Tokenizer& tokenizer, LlGuidanceConfig llg_config);

  // LlgTokenizer must be valid. Takes ownership of LlgTokenizer.
  explicit LlgConstraintProvider(std::vector<uint32_t>&& token_lens,
                                 std::vector<uint8_t>&& token_bytes,
                                 LlgTokenizer* llg_tokenizer,
                                 LlGuidanceConfig llg_config)
      : token_lens_(std::move(token_lens)),
        token_bytes_(std::move(token_bytes)),
        llg_tokenizer_(std::move(llg_tokenizer)),
        llg_config_(std::move(llg_config)) {}

  ~LlgConstraintProvider() override;

  absl::StatusOr<std::unique_ptr<Constraint>> CreateConstraint(
      ConstraintArg constraint_arg) const override;

 private:
  const std::vector<uint32_t> token_lens_;
  const std::vector<uint8_t> token_bytes_;
  LlgTokenizer* llg_tokenizer_;  // Owned.
  LlGuidanceConfig llg_config_;
};

// Creates the provider for the tool call constraints of a model. Model data
// processors use this rather than `LlgConstraintProvider::Create`, so that they
// all set up the provider the same way and fail if it can't be created.
//
// The response ends with one of the model's end-of-sequence (EOS) tokens. They
// are the single-token sequences in `stop_token_ids`, or `fallback_eos_token`
// if there are none. `special_tokens` are the control tokens that the tool call
// grammars contain, e.g. the code fences around a call, which the constraints
// match as single tokens. Only those are needed: the EOS tokens are special
// tokens already, and the model's other control tokens are treated as text.
//
// Returns an error if `tokenizer` is null or the provider can't be created, so
// that a model never decodes without the constraints it was asked to use.
absl::StatusOr<std::unique_ptr<ConstraintProvider>>
CreateLlgToolCallConstraintProvider(
    const Tokenizer* tokenizer,
    const std::vector<std::vector<int>>& stop_token_ids,
    absl::string_view fallback_eos_token,
    std::vector<std::string> special_tokens);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LLG_CONSTRAINT_PROVIDER_H_
