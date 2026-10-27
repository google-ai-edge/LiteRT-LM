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

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

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
  // Tokens that end generation. Any of them may end the output where the
  // constraint allows it to end, and none of them can appear anywhere else, not
  // even as text. If empty, the session's single-token stop sequences are used.
  std::vector<uint32_t> eos_ids;
  // Control tokens that should be matched as a single token, instead of treated
  // as bytes:
  // https://github.com/guidance-ai/llguidance/blob/main/docs/special_tokens.md
  std::vector<std::string> special_tokens;
};

// Appends the single-token sequences in `stop_token_ids` to `config.eos_ids`,
// skipping duplicates and negative IDs. Multi-token stop sequences are ignored.
inline void AddStopTokensAsEosIds(
    const std::vector<std::vector<int>>& stop_token_ids,
    LlGuidanceConfig& config) {
  for (const std::vector<int>& stop_sequence : stop_token_ids) {
    if (stop_sequence.size() != 1 || stop_sequence[0] < 0) {
      continue;
    }
    const uint32_t token_id = static_cast<uint32_t>(stop_sequence[0]);
    if (std::find(config.eos_ids.begin(), config.eos_ids.end(), token_id) ==
        config.eos_ids.end()) {
      config.eos_ids.push_back(token_id);
    }
  }
}

struct LlGuidanceConstraintArg {
  LlgConstraintType constraint_type;
  std::string constraint_string;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_LLG_CONSTRAINT_CONFIG_H_
