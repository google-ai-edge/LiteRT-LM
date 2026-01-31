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

#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/components/rust/minijinja_template.rs.h"
#include "re2/re2.h"  // from @com_googlesource_code_re2

namespace litert::lm {
namespace {

// Post-process the template to make it compatible with Minijinja.
//
// Minijinja is a Rust implementation of Jinja2, but it is not 100% compatible
// with the Python implementation. In particular, it does not support calling
// arbitrary Python methods on objects (e.g. `s.startswith("foo")`).
//
// This function uses regexes to rewrite common Python idioms found in
// tokenizer_config.json templates into Minijinja-compatible syntax.
std::string EditTemplateForMinijinja(absl::string_view template_content) {
  std::string modified_template = std::string(template_content);
  RE2::GlobalReplace(&modified_template, R"regex(\.startswith\((.*?)\))regex",
                     R"( is startingwith \1)");
  RE2::GlobalReplace(&modified_template, R"regex(\.endswith\((.*?)\))regex",
                     R"( is endingwith \1)");
  RE2::GlobalReplace(&modified_template,
                     R"regex(\.replace\((.*?),(.*?)\))regex",
                     R"( | replace(\1,\2))");
  RE2::GlobalReplace(&modified_template, R"regex(\.split\((.*?)\)\[0\])regex",
                     R"( | split(\1) | first)");
  RE2::GlobalReplace(&modified_template, R"regex(\.split\((.*?)\)\[-1\])regex",
                     R"( | split(\1) | last)");
  RE2::GlobalReplace(&modified_template, R"regex(\.split\((.*?)\))regex",
                     R"( | split(\1))");
  RE2::GlobalReplace(&modified_template, R"regex(\.join\((.*?)\))regex",
                     R"( | join(\1))");
  RE2::GlobalReplace(&modified_template, R"regex(\.[l,r]strip\(\))regex",
                     " | trim");
  RE2::GlobalReplace(&modified_template, R"regex(\.[l,r]strip\((.*?)\))regex",
                     R"( | trim(\1))");
  RE2::GlobalReplace(&modified_template, R"regex(\.items\(\))regex",
                     " | items");
  RE2::GlobalReplace(&modified_template, R"regex({% generation %})regex", "");
  RE2::GlobalReplace(&modified_template, R"regex({% endgeneration %})regex",
                     "");
  return modified_template;
}

}  // namespace

using json = nlohmann::ordered_json;

PromptTemplate::PromptTemplate(absl::string_view template_content,
                               bool edit_template_for_minijinja)
    : minijinja_template_(new_minijinja_template(
          edit_template_for_minijinja
              ? EditTemplateForMinijinja(template_content)
              : std::string(template_content))) {
  const auto caps = minijinja_template_->get_capabilities();
  capabilities_ = PromptTemplateCapabilities{
      .supports_tools = caps.supports_tools,
      .supports_tool_calls = caps.supports_tool_calls,
      .supports_system_role = caps.supports_system_role,
      .supports_parallel_tool_calls = caps.supports_parallel_tool_calls,
      .supports_tool_call_id = caps.supports_tool_call_id,
      .requires_typed_content = caps.requires_typed_content,
      .supports_single_turn = caps.supports_single_turn};
}

PromptTemplate::PromptTemplate(const PromptTemplate& other)
    : minijinja_template_(other.minijinja_template_->clone_template()) {
  capabilities_ = other.capabilities_;
}

PromptTemplate& PromptTemplate::operator=(const PromptTemplate& other) {
  minijinja_template_ = other.minijinja_template_->clone_template();
  capabilities_ = other.capabilities_;
  return *this;
}

PromptTemplate::PromptTemplate(PromptTemplate&& other)
    : minijinja_template_(std::move(other.minijinja_template_)) {
  capabilities_ = other.capabilities_;
}

PromptTemplate& PromptTemplate::operator=(PromptTemplate&& other) {
  minijinja_template_ = std::move(other.minijinja_template_);
  capabilities_ = other.capabilities_;
  return *this;
}

absl::StatusOr<std::string> PromptTemplate::Apply(
    const PromptTemplateInput& input) const {
  nlohmann::ordered_json minijinja_inputs;
  minijinja_inputs["messages"] = input.messages;
  minijinja_inputs["tools"] = input.tools;
  minijinja_inputs["add_generation_prompt"] = input.add_generation_prompt;
  minijinja_inputs["extra_context"] = input.extra_context;
  minijinja_inputs["now"] = absl::ToUnixSeconds(input.now);
  minijinja_inputs["bos_token"] = input.bos_token;
  minijinja_inputs["eos_token"] = input.eos_token;
  auto result = minijinja_template_->apply(minijinja_inputs.dump());
  if (!result.is_ok) {
    return absl::InternalError(
        absl::StrCat("Failed to apply template: ", std::string(result.error)));
  }
  return std::string(result.content);
}

absl::string_view PromptTemplate::GetTemplateSource() const {
  auto source = minijinja_template_->source();
  return absl::string_view(source.data(), source.size());
}

}  // namespace litert::lm
