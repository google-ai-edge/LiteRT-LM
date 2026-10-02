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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_INTERNAL_CALLBACK_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_INTERNAL_CALLBACK_UTIL_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/conversation/io_types.h"
#include "runtime/conversation/model_data_processor/config_registry.h"
#include "runtime/conversation/model_data_processor/model_data_processor.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {

// Creates an internal callback that translates the Responses to Message and
// passes it to the user callback.
//
// Args:
// - model_data_processor: The ModelDataProcessor to use for translating
//     the Responses to Message.
// - processor_args: The arguments to use for the ModelDataProcessor.
// - user_callback: The callback to pass the translated Message to.
// - cancel_callback: Will be called in addition to user_callback when the
//     cancellation exception is caught.
// - complete_message_callback: Will also be called in addition to
//     user_callback when the task is completed.
absl::AnyInvocable<void(absl::StatusOr<Responses>)> CreateInternalCallback(
    const ModelDataProcessor& model_data_processor,
    DataProcessorArguments processor_args, const std::vector<Channel>& channels,
    absl::AnyInvocable<void(absl::StatusOr<Message>)> user_callback,
    absl::AnyInvocable<void()> cancel_callback = nullptr,
    absl::AnyInvocable<void(Message)> complete_message_callback = nullptr,
    const std::optional<std::string>& open_channel_name = std::nullopt,
    bool return_error_on_max_tokens_reached = false,
    bool stream_tool_calls = false,
    absl::string_view tool_call_channel_name = "tool_call",
    absl::AnyInvocable<void(int)> on_channel_tokens_callback = nullptr);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CONVERSATION_INTERNAL_CALLBACK_UTIL_H_
