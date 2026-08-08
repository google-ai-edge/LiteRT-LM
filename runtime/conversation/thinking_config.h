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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_THINKING_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_THINKING_CONFIG_H_

namespace litert::lm {

// Configuration for thinking/reasoning generation.
class ThinkingConfig {
 public:
  // Default constructor: enabled with infinite budget (-1).
  ThinkingConfig() : enable_thinking_(true), thinking_token_budget_(-1) {}

  ThinkingConfig(bool enable_thinking, int thinking_token_budget)
      : enable_thinking_(enable_thinking),
        thinking_token_budget_(thinking_token_budget) {}

  bool enable_thinking() const { return enable_thinking_; }
  int thinking_token_budget() const { return thinking_token_budget_; }

 private:
  bool enable_thinking_;
  int thinking_token_budget_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_THINKING_CONFIG_H_
