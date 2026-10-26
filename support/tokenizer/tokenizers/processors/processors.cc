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
//
// C++ port of `src/processors/mod.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/processors/processors.h"

#include <cstddef>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/pre_tokenizers/byte_level.h"
#include "support/tokenizer/tokenizers/processors/bert.h"
#include "support/tokenizer/tokenizers/processors/roberta.h"
#include "support/tokenizer/tokenizers/processors/sequence.h"
#include "support/tokenizer/tokenizers/processors/template.h"
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {
namespace processors {

size_t PostProcessorWrapper::AddedTokens(bool is_pair) const {
  return std::visit(
      [is_pair](const auto& processor) {
        return processor.AddedTokens(is_pair);
      },
      variant_);
}

absl::StatusOr<std::vector<Encoding>> PostProcessorWrapper::ProcessEncodings(
    std::vector<Encoding> encodings, bool add_special_tokens) const {
  return std::visit(
      [&encodings, add_special_tokens](const auto& processor) {
        return processor.ProcessEncodings(std::move(encodings),
                                          add_special_tokens);
      },
      variant_);
}

absl::StatusOr<PostProcessorWrapper> PostProcessorWrapper::FromJson(
    const Json& json) {
  // Roberta must be before Bert for deserialization (serde does not validate
  // tags).
  if (auto p = RobertaProcessing::FromJson(json); p.ok()) {
    return PostProcessorWrapper(*std::move(p));
  }
  if (auto p = BertProcessing::FromJson(json); p.ok()) {
    return PostProcessorWrapper(*std::move(p));
  }
  if (auto p = ByteLevel::FromJson(json); p.ok()) {
    return PostProcessorWrapper(*std::move(p));
  }
  if (auto p = TemplateProcessing::FromJson(json); p.ok()) {
    return PostProcessorWrapper(*std::move(p));
  }
  if (auto p = Sequence::FromJson(json); p.ok()) {
    return PostProcessorWrapper(*std::move(p));
  }
  return absl::InvalidArgumentError(
      "data did not match any variant of untagged enum PostProcessorWrapper");
}

Json PostProcessorWrapper::ToJson() const {
  return std::visit([](const auto& processor) { return processor.ToJson(); },
                    variant_);
}

}  // namespace processors
}  // namespace tokenizers
