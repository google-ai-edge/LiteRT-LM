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
// C++ port of `src/pre_tokenizers/split.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License
// 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line close to the
// Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_SPLIT_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_SPLIT_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/tokenizer/tokenizers/utils/regex.h"

namespace tokenizers {
namespace pre_tokenizers {

// Represents the different patterns that `Split` can use
//
// Rust: `enum SplitPattern { String(String), Regex(String) }`, serialized
// externally tagged: `{"String":"..."}` / `{"Regex":"..."}`.
class SplitPattern {
 public:
  enum class Kind {
    kString,
    kRegex,
  };

  // `impl From<&str> for SplitPattern` / `impl From<String> for SplitPattern`:
  // a plain string is a `String` pattern.
  SplitPattern(absl::string_view v)  // NOLINT(google-explicit-constructor)
      : kind_(Kind::kString), value_(v) {}
  SplitPattern(const char* v)  // NOLINT(google-explicit-constructor)
      : SplitPattern(absl::string_view(v)) {}
  SplitPattern(std::string v)  // NOLINT(google-explicit-constructor)
      : kind_(Kind::kString), value_(std::move(v)) {}

  static SplitPattern String(std::string v) {
    return SplitPattern(Kind::kString, std::move(v));
  }
  static SplitPattern Regex(std::string v) {
    return SplitPattern(Kind::kRegex, std::move(v));
  }

  Kind kind() const { return kind_; }
  const std::string& value() const { return value_; }

  static absl::StatusOr<SplitPattern> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const SplitPattern& a, const SplitPattern& b) {
    return a.kind_ == b.kind_ && a.value_ == b.value_;
  }
  friend bool operator!=(const SplitPattern& a, const SplitPattern& b) {
    return !(a == b);
  }

 private:
  SplitPattern(Kind kind, std::string value)
      : kind_(kind), value_(std::move(value)) {}

  Kind kind_;
  std::string value_;
};

// Splits on a pattern (string or regex) with the given behavior, optionally
// inverting the matches.
//
// serde (custom, `#[serde(tag = "type")]`):
// `{"type":"Split","pattern":{"String":"Hello"},"behavior":"Removed",
// "invert":true}`; the compiled regex is `#[serde(skip)]`.
class Split : public PreTokenizer {
 public:
  // `Split::new`: compiles the pattern (`regex::escape`d for `String`
  // patterns) and fails with the regex compilation error.
  static absl::StatusOr<Split> New(SplitPattern pattern,
                                   SplitDelimiterBehavior behavior,
                                   bool invert);

  const SplitPattern& pattern() const { return pattern_; }
  const SysRegex& regex() const { return *regex_; }
  SplitDelimiterBehavior behavior() const { return behavior_; }
  bool invert() const { return invert_; }

  absl::Status PreTokenize(PreTokenizedString& pretokenized) const override;

  static absl::StatusOr<Split> FromJson(const Json& json);
  Json ToJson() const;

  // `impl PartialEq for Split`: compares everything but the compiled regex.
  friend bool operator==(const Split& a, const Split& b) {
    return a.pattern_ == b.pattern_ && a.behavior_ == b.behavior_ &&
           a.invert_ == b.invert_;
  }
  friend bool operator!=(const Split& a, const Split& b) { return !(a == b); }

 private:
  Split(SplitPattern pattern, std::shared_ptr<const SysRegex> regex,
        SplitDelimiterBehavior behavior, bool invert)
      : pattern_(std::move(pattern)),
        regex_(std::move(regex)),
        behavior_(behavior),
        invert_(invert) {}

  SplitPattern pattern_;
  // PORT NOTE: Rust's `impl Clone for Split` recompiles the regex
  // (`Self::new(..).unwrap()`); here copies share the immutable compiled
  // regex instead, which cannot fail and is observably identical.
  std::shared_ptr<const SysRegex> regex_;
  SplitDelimiterBehavior behavior_;
  bool invert_;
};

}  // namespace pre_tokenizers
}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_SPLIT_H_
