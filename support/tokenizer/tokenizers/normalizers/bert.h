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
// C++ port of `src/normalizers/bert.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_BERT_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_BERT_H_

#include <optional>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers::normalizers {

// `#[derive(Copy, Clone, Debug, Deserialize, Serialize)] #[serde(tag = "type")]
// #[non_exhaustive]`
class BertNormalizer : public Normalizer {
 public:
  // `impl Default for BertNormalizer`
  BertNormalizer()
      : BertNormalizer(/*clean_text=*/true, /*handle_chinese_chars=*/true,
                       /*strip_accents=*/std::nullopt, /*lowercase=*/true) {}
  BertNormalizer(bool clean_text, bool handle_chinese_chars,
                 std::optional<bool> strip_accents, bool lowercase)
      : clean_text_(clean_text),
        handle_chinese_chars_(handle_chinese_chars),
        strip_accents_(strip_accents),
        lowercase_(lowercase) {}

  static absl::StatusOr<BertNormalizer> FromJson(const Json& json);
  Json ToJson() const;

  // Whether to do the bert basic cleaning:
  //   1. Remove any control characters
  //   2. Replace all sorts of whitespace by the classic one ` `
  bool clean_text() const { return clean_text_; }
  void set_clean_text(bool clean_text) { clean_text_ = clean_text; }
  // Whether to put spaces around chinese characters so they get split
  bool handle_chinese_chars() const { return handle_chinese_chars_; }
  void set_handle_chinese_chars(bool handle_chinese_chars) {
    handle_chinese_chars_ = handle_chinese_chars;
  }
  // Whether to strip accents
  std::optional<bool> strip_accents() const { return strip_accents_; }
  void set_strip_accents(std::optional<bool> strip_accents) {
    strip_accents_ = strip_accents;
  }
  // Whether to lowercase the input
  bool lowercase() const { return lowercase_; }
  void set_lowercase(bool lowercase) { lowercase_ = lowercase; }

  absl::Status Normalize(NormalizedString& normalized) const override;

 private:
  void DoCleanText(NormalizedString& normalized) const;
  void DoHandleChineseChars(NormalizedString& normalized) const;
  void DoStripAccents(NormalizedString& normalized) const;
  void DoLowercase(NormalizedString& normalized) const;

  bool clean_text_;
  bool handle_chinese_chars_;
  std::optional<bool> strip_accents_;
  bool lowercase_;
};

}  // namespace tokenizers::normalizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_BERT_H_
