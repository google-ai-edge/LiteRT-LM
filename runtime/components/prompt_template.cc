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

#include "runtime/components/prompt_template.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json

namespace litert::lm {

using json = nlohmann::ordered_json;
using minja_chat_template = ::minja::chat_template;
using minja_chat_template_inputs = ::minja::chat_template_inputs;

constexpr absl::string_view kIsAppendingToPrefill = "is_appending_to_prefill";

PromptTemplate::PromptTemplate(absl::string_view template_content) {
  minja_template_ = std::make_unique<minja_chat_template>(
      std::string(template_content), /*bos_token=*/"", /*eos_token=*/"");

  const auto& original_caps = minja_template_->original_caps();
  capabilities_ = PromptTemplateCapabilities{
      .supports_tools = original_caps.supports_tools,
      .supports_tool_calls = original_caps.supports_tool_calls,
      .supports_tool_responses = original_caps.supports_tool_responses,
      .supports_system_role = original_caps.supports_system_role,
      .supports_parallel_tool_calls =
          original_caps.supports_parallel_tool_calls,
      .supports_tool_call_id = original_caps.supports_tool_call_id,
      .requires_object_arguments = original_caps.requires_object_arguments,
      .requires_non_null_content = original_caps.requires_non_null_content,
      .requires_typed_content = original_caps.requires_typed_content,
      .supports_single_turn =
          absl::StrContains(template_content, kIsAppendingToPrefill)};
}

PromptTemplate::PromptTemplate(const PromptTemplate& other) {
  minja_template_ = std::make_unique<minja_chat_template>(
      other.minja_template_->source(), other.minja_template_->bos_token(),
      other.minja_template_->eos_token());
  capabilities_ = other.capabilities_;
}

PromptTemplate& PromptTemplate::operator=(const PromptTemplate& other) {
  minja_template_ = std::make_unique<minja_chat_template>(
      other.minja_template_->source(), other.minja_template_->bos_token(),
      other.minja_template_->eos_token());
  capabilities_ = other.capabilities_;
  return *this;
}

PromptTemplate::PromptTemplate(PromptTemplate&& other) {
  minja_template_ = std::move(other.minja_template_);
  capabilities_ = other.capabilities_;
}

PromptTemplate& PromptTemplate::operator=(PromptTemplate&& other) {
  minja_template_ = std::move(other.minja_template_);
  capabilities_ = other.capabilities_;
  return *this;
}

absl::StatusOr<std::string> PromptTemplate::Apply(
    const PromptTemplateInput& input) const {
  minja_chat_template_inputs minja_inputs;
  minja_inputs.messages = input.messages;
  minja_inputs.tools = input.tools;
  minja_inputs.add_generation_prompt = input.add_generation_prompt;
  minja_inputs.extra_context = input.extra_context;
  minja_inputs.now = absl::ToChronoTime(input.now);
  return minja_template_->apply(minja_inputs, {.apply_polyfills = false});
}

absl::string_view PromptTemplate::GetTemplateSource() const {
  return minja_template_->source();
}

}  // namespace litert::lm
