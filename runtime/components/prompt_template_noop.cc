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

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/prompt_template.h"

namespace litert::lm {

struct PromptTemplate::MinijinjaTemplateImpl {
  std::string source;
};

PromptTemplate::PromptTemplate(absl::string_view template_content)
    : minijinja_template_(std::make_unique<MinijinjaTemplateImpl>(
          MinijinjaTemplateImpl{std::string(template_content)})),
      capabilities_({.supports_single_turn = false}) {}

PromptTemplate::~PromptTemplate() = default;

PromptTemplate::PromptTemplate(const PromptTemplate& other)
    : minijinja_template_(other.minijinja_template_
                              ? std::make_unique<MinijinjaTemplateImpl>(
                                    *other.minijinja_template_)
                              : nullptr),
      capabilities_(other.capabilities_) {}

PromptTemplate& PromptTemplate::operator=(const PromptTemplate& other) {
  if (this != &other) {
    minijinja_template_ = other.minijinja_template_
                              ? std::make_unique<MinijinjaTemplateImpl>(
                                    *other.minijinja_template_)
                              : nullptr;
    capabilities_ = other.capabilities_;
  }
  return *this;
}

PromptTemplate::PromptTemplate(PromptTemplate&& other)
    : minijinja_template_(std::move(other.minijinja_template_)),
      capabilities_(other.capabilities_) {}

PromptTemplate& PromptTemplate::operator=(PromptTemplate&& other) {
  if (this != &other) {
    minijinja_template_ = std::move(other.minijinja_template_);
    capabilities_ = other.capabilities_;
  }
  return *this;
}

absl::StatusOr<std::string> PromptTemplate::Apply(
    const PromptTemplateInput& input) const {
  return absl::UnimplementedError(
      "Minijinja prompt template engine is disabled in this build.");
}

absl::string_view PromptTemplate::GetTemplateSource() const {
  if (minijinja_template_ == nullptr) {
    return "";
  }
  return minijinja_template_->source;
}

}  // namespace litert::lm
