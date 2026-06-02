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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_PROMPT_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_PROMPT_UTILS_H_

#include <optional>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/components/prompt_template.h"
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"

namespace litert::lm {

// Fills the preface for the prompt template input.
// Args:
// - `preface`: The preface to be filled.
// - `model_data_processor`: The model data processor to be used.
// - `tmpl_input`: The prompt template input object reference to be filled.
// Returns:
// - An error status if the preface cannot be filled.
absl::Status FillPrefaceForPromptTemplateInput(
    const Preface& preface, const ModelDataProcessor* model_data_processor,
    PromptTemplateInput& tmpl_input);

// A utility function to render a single turn template for both incremental
// difference and appending logics.
// Args:
// - `processor`: The model data processor to be used.
// - `history`: The history of messages.
// - `preface`: The preface to be used.
// - `message`: The message to be rendered.
// - `prompt_template`: The prompt template to be used.
// - `current_is_appending_message`: Whether the data processor is in appending
//     state.
// - `append_message`: Whether the message is for appending. If false, the
//     message is either for incremental difference or the last part of
//     appending message.
// - `extra_context`: The extra context to be used.
// - `push_dummy_user_message_to_preface`: Whether to push a dummy user message
//     to the preface. It is used for Gemma3 templates.
// Returns:
// - The single turn template render result.
absl::StatusOr<ModelDataProcessor::SingleTurnTemplateRenderResult>
RenderSingleTurnTemplateCommon(
    const ModelDataProcessor& processor,
    std::vector<Message>& history, const Preface& preface,
    const Message& message, const PromptTemplate& prompt_template,
    bool current_is_appending_message, bool append_message,
    std::optional<nlohmann::ordered_json> extra_context,
    bool push_dummy_user_message_to_preface);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_PROMPT_UTILS_H_
