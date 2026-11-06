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
// C++ port of `src/pre_tokenizers/byte_level.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_BYTE_LEVEL_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_BYTE_LEVEL_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {
namespace pre_tokenizers {

// Converts bytes to unicode characters.
// See https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
absl::flat_hash_map<uint8_t, char32_t> BytesChar();

// Provides all the necessary steps to handle the BPE tokenization at the
// byte-level. Takes care of all the required processing steps to transform a
// UTF-8 string as needed before and after the BPE model does its job.
//
// Serialized with `impl_serde_type!` (tag "type" = "ByteLevel"; `use_regex`
// defaults to true when missing).
class ByteLevel : public PreTokenizer, public Decoder, public PostProcessor {
 public:
  // `impl Default for ByteLevel`.
  ByteLevel()
      : ByteLevel(/*add_prefix_space=*/true, /*trim_offsets=*/true,
                  /*use_regex=*/true) {}
  // `ByteLevel::new`.
  ByteLevel(bool add_prefix_space, bool trim_offsets, bool use_regex)
      : add_prefix_space_(add_prefix_space),
        trim_offsets_(trim_offsets),
        use_regex_(use_regex) {}

  static absl::flat_hash_set<char32_t> Alphabet();

  // Builder-style setters (`#[must_use] fn add_prefix_space(mut self, v) ->
  // Self` etc.): return a modified copy.
  ByteLevel AddPrefixSpace(bool v) const {
    ByteLevel copy = *this;
    copy.add_prefix_space_ = v;
    return copy;
  }
  ByteLevel TrimOffsets(bool v) const {
    ByteLevel copy = *this;
    copy.trim_offsets_ = v;
    return copy;
  }
  ByteLevel UseRegex(bool v) const {
    ByteLevel copy = *this;
    copy.use_regex_ = v;
    return copy;
  }

  // Whether to add a leading space to the first word. This allows to treat the
  // leading word just as any other word.
  bool add_prefix_space() const { return add_prefix_space_; }
  // Whether the post processing step should trim offsets to avoid including
  // whitespaces.
  bool trim_offsets() const { return trim_offsets_; }
  // Whether to use the standard GPT2 regex for whitespace splitting
  // Set it to False if you want to use your own splitting.
  bool use_regex() const { return use_regex_; }

  // As a `PreTokenizer`, `ByteLevel` is in charge of transforming all the
  // unicode characters into their byte-level counterpart. It also splits the
  // input according to the configured regex.
  absl::Status PreTokenize(PreTokenizedString& pretokenized) const override;

  // As a `Decoder`, `ByteLevel` is in charge of converting any byte-level
  // characters to their unicode counterpart, before merging everything back
  // into a single String.
  // This decoder will consume the tokens and merge them in one step to
  // alleviate the fact that single token decoded might be a byte not
  // representable as as String.
  absl::StatusOr<std::vector<std::string>> DecodeChain(
      std::vector<std::string> tokens) const override;

  // As a `PostProcessor`, `ByteLevel` is in charge of trimming the offsets if
  // necessary.
  size_t AddedTokens(bool is_pair) const override;
  absl::StatusOr<std::vector<Encoding>> ProcessEncodings(
      std::vector<Encoding> encodings, bool add_special_tokens) const override;

  // serde: `{"type":"ByteLevel","add_prefix_space":..,"trim_offsets":..,
  // "use_regex":..}`.
  static absl::StatusOr<ByteLevel> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const ByteLevel& a, const ByteLevel& b) {
    return a.add_prefix_space_ == b.add_prefix_space_ &&
           a.trim_offsets_ == b.trim_offsets_ && a.use_regex_ == b.use_regex_;
  }
  friend bool operator!=(const ByteLevel& a, const ByteLevel& b) {
    return !(a == b);
  }

 private:
  bool add_prefix_space_;
  bool trim_offsets_;
  bool use_regex_;
};

// `pub fn process_offsets(encoding: &mut Encoding, add_prefix_space: bool)`;
// also used by `processors::roberta`.
void ProcessOffsets(Encoding& encoding, bool add_prefix_space);

}  // namespace pre_tokenizers
}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PRE_TOKENIZERS_BYTE_LEVEL_H_
