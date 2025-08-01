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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_TYPES_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_TYPES_H_

#include <variant>

#include "nlohmann/json.hpp"  // from @nlohmann_json

namespace litert::lm {

using JsonMessage = nlohmann::ordered_json;

// Message is the data container for a single turn of the conversation.
using Message = std::variant<JsonMessage>;

struct JsonContext {
  nlohmann::ordered_json messages;
  nlohmann::ordered_json tools;
  nlohmann::ordered_json extra_context;
};

// Context is the initial messages, tools and extra context for the
// conversation to begin with.
using Context = std::variant<nlohmann::ordered_json>;

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_TYPES_H_
