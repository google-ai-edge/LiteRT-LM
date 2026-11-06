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
// C++ port of `src/normalizers/replace.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_REPLACE_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_REPLACE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/regex.h"

namespace tokenizers::normalizers {

// Represents the different patterns that `Replace` can use
//
// `#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
// pub enum ReplacePattern { String(String), Regex(String) }` — an externally
// tagged enum, i.e. `{"String": "..."}` / `{"Regex": "..."}` in JSON.
struct ReplacePattern {
  enum class Kind { kString, kRegex };

  // `impl From<String> for ReplacePattern` / `impl From<&str>`.
  static ReplacePattern String(std::string s) {
    return ReplacePattern{Kind::kString, std::move(s)};
  }
  static ReplacePattern Regex(std::string r) {
    return ReplacePattern{Kind::kRegex, std::move(r)};
  }

  static absl::StatusOr<ReplacePattern> FromJson(const Json& json);
  Json ToJson() const;

  bool operator==(const ReplacePattern& other) const {
    return kind == other.kind && value == other.value;
  }
  bool operator!=(const ReplacePattern& other) const {
    return !(*this == other);
  }

  Kind kind;
  std::string value;
};

// This normalizer will take a `pattern` (for now only a String)
// and replace every occurrence with `content`.
//
// `#[derive(Debug, Serialize, Deserialize)]
// #[serde(tag = "type", try_from = "ReplaceDeserializer")]`
class Replace : public Normalizer, public Decoder {
 public:
  // `Replace::new<I: Into<ReplacePattern>, C: Into<String>>(pattern, content)`
  static absl::StatusOr<Replace> New(ReplacePattern pattern,
                                     std::string content);
  // `impl From<&str> for ReplacePattern` -> `ReplacePattern::String`.
  static absl::StatusOr<Replace> New(absl::string_view pattern,
                                     std::string content) {
    return New(ReplacePattern::String(std::string(pattern)),
               std::move(content));
  }

  static absl::StatusOr<Replace> FromJson(const Json& json);
  Json ToJson() const;

  const ReplacePattern& pattern() const { return pattern_; }
  // `pub content: String`
  const std::string& content() const { return content_; }
  void set_content(std::string content) { content_ = std::move(content); }

  // `impl PartialEq for Replace`: compares `pattern` and `content` only.
  bool operator==(const Replace& other) const {
    return pattern_ == other.pattern_ && content_ == other.content_;
  }
  bool operator!=(const Replace& other) const { return !(*this == other); }

  // `impl Normalizer for Replace`
  absl::Status Normalize(NormalizedString& normalized) const override;

  // `impl Decoder for Replace`
  absl::StatusOr<std::vector<std::string>> DecodeChain(
      std::vector<std::string> tokens) const override;

 private:
  Replace(ReplacePattern pattern, std::string content,
          std::shared_ptr<const SysRegex> regex)
      : pattern_(std::move(pattern)),
        content_(std::move(content)),
        regex_(std::move(regex)) {}

  ReplacePattern pattern_;
  std::string content_;
  // `#[serde(skip)] regex: SysRegex`
  // PORT NOTE: `impl Clone for Replace` re-compiles the regex from `pattern`
  // (and unwraps). A C++ copy constructor cannot report a compilation failure,
  // so copies share the immutable compiled regex instead (SysRegex is safe for
  // concurrent `FindIter` calls).
  std::shared_ptr<const SysRegex> regex_;
};

}  // namespace tokenizers::normalizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_REPLACE_H_
