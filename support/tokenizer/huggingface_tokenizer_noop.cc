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

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/huggingface_tokenizer.h"

namespace litert::support {

// Note: ~HuggingFaceTokenizer() is intentionally not defined here. Because
// ~HuggingFaceTokenizer() is virtual and CreateFromFile / CreateFromJson always
// return UnimplementedError without constructing an instance,
// std::unique_ptr<HuggingFaceTokenizer> destruction dispatches via the vtable
// only when non-null and never references the destructor symbol directly.
// Defining the virtual destructor here would cause lld-link on Windows to emit
// a strong scalar deleting destructor that collides with
// huggingface_tokenizer.cc.

ABSL_ATTRIBUTE_WEAK absl::StatusOr<std::unique_ptr<HuggingFaceTokenizer>>
HuggingFaceTokenizer::CreateFromFile(absl::string_view json_path) {
  return absl::UnimplementedError(
      "HuggingFace tokenizer is disabled in this build.");
}

ABSL_ATTRIBUTE_WEAK absl::StatusOr<std::unique_ptr<HuggingFaceTokenizer>>
HuggingFaceTokenizer::CreateFromJson(const std::string& json) {
  return absl::UnimplementedError(
      "HuggingFace tokenizer is disabled in this build.");
}

}  // namespace litert::support
