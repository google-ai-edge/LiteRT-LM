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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_TOOL_CHOICE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_TOOL_CHOICE_H_

#include <string>
#include <variant>
#include <vector>

namespace litert::lm {

// Whether the model may, must, or must not call the tools it was given, and
// which of them it may call, like the OpenAI tool_choice parameter.
struct ToolChoice {
  // The model decides whether to answer in text, call tools, or both
  // ("auto").
  struct Auto {};
  // The model must answer in text only, with no tool calls ("none").
  struct None {};
  // The model must call at least one tool ("required").
  struct Required {};
  // The model must call the function `name`, which must be one of the tools
  // given to the model ({"type": "function", "function": {"name": ...}}).
  struct ForcedFunction {
    std::string name;
  };
  // The model may only call the functions in `names`, which must be among the
  // tools given to the model ({"type": "allowed_tools", ...}). If `required`
  // is true, it must call at least one of them. Otherwise it may also answer
  // in text.
  struct AllowedTools {
    std::vector<std::string> names;
    bool required = false;
  };

  std::variant<Auto, None, Required, ForcedFunction, AllowedTools> choice =
      Auto{};
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_TOOL_CHOICE_H_
