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
// C++ port of `src/processors/sequence.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License
// 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line close to the
// Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PROCESSORS_SEQUENCE_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PROCESSORS_SEQUENCE_H_

#include <cstddef>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/api.h"
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/utils/json.h"

namespace tokenizers {
namespace processors {

class PostProcessorWrapper;

// Applies a list of post-processors one after the other.
//
// serde: `impl_serde_type!` -> `{"type":"Sequence","processors":[...]}` (the
// `type` tag is validated when deserializing).
//
// `Sequence` holds a `std::vector<PostProcessorWrapper>` while
// `PostProcessorWrapper` is a variant containing `Sequence`: the special
// member functions are defined out of line (in sequence.cc, where both types
// are complete).
class Sequence : public PostProcessor {
 public:
  // `Sequence::new`.
  explicit Sequence(std::vector<PostProcessorWrapper> processors);
  Sequence(const Sequence& other);
  Sequence(Sequence&& other) noexcept;
  Sequence& operator=(const Sequence& other);
  Sequence& operator=(Sequence&& other) noexcept;
  ~Sequence() override;

  // `get(index) -> Option<&PostProcessorWrapper>` (null when out of range).
  const PostProcessorWrapper* Get(size_t index) const;
  // `get_mut(index) -> Option<&mut PostProcessorWrapper>`.
  PostProcessorWrapper* GetMut(size_t index);
  // `set_mut(index, post_proc)`.
  // PORT NOTE: Rust panics on an out-of-range index; this returns an
  // `OutOfRangeError` instead.
  absl::Status SetMut(size_t index, PostProcessorWrapper post_proc);

  // `impl AsRef<[PostProcessorWrapper]>` / `impl AsMut<[...]>` /
  // `impl IntoIterator`.
  const std::vector<PostProcessorWrapper>& AsRef() const;
  std::vector<PostProcessorWrapper>& AsMut();

  // `impl PostProcessor for Sequence`.
  size_t AddedTokens(bool is_pair) const override;
  absl::StatusOr<std::vector<Encoding>> ProcessEncodings(
      std::vector<Encoding> encodings, bool add_special_tokens) const override;

  static absl::StatusOr<Sequence> FromJson(const Json& json);
  Json ToJson() const;

  friend bool operator==(const Sequence& a, const Sequence& b);
  friend bool operator!=(const Sequence& a, const Sequence& b) {
    return !(a == b);
  }

 private:
  std::vector<PostProcessorWrapper> processors_;
};

}  // namespace processors
}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_PROCESSORS_SEQUENCE_H_
