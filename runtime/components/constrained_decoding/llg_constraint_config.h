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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LLG_CONSTRAINT_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LLG_CONSTRAINT_CONFIG_H_

#include <cstdint>
#include <optional>
#include <string>

namespace litert::lm {

enum class LlgConstraintType {
  // Constrain model output to follow the given regular expression.
  kRegex,
  // Constrain model output to follow the given JSON schema. See
  // https://github.com/guidance-ai/llguidance/blob/main/docs/json_schema.md for
  // more information.
  kJsonSchema,
  // Constrain model output to follow the given Lark grammar. See
  // https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md for more
  // information.
  kLark,
  // Constrain model output to follow the given LLGuidance-internal JSON-based
  // format. Deprecated in favor of the Lark format.
  kLlGuidanceInternal,
};

struct LlGuidanceConfig {
  std::optional<uint32_t> eos_id = std::nullopt;
};

struct LlGuidanceConstraintArg {
  LlgConstraintType constraint_type;
  std::string constraint_string;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LLG_CONSTRAINT_CONFIG_H_
