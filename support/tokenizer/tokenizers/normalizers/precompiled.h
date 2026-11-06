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
// C++ port of `src/normalizers/precompiled.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_PRECOMPILED_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_PRECOMPILED_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"  // from @com_google_absl
// `pub use spm_precompiled::Precompiled;` — the struct itself (and its
// `Normalize` override, implemented in precompiled.cc) is declared there.
#include "support/tokenizer/tokenizers/normalizers/spm_precompiled.h"  // IWYU pragma: export

namespace tokenizers::normalizers {
// PORT NOTE: the module-private helper `fn replace(...)` of precompiled.rs is
// kept in a `precompiled` sub-namespace because the enclosing namespace
// already contains the `Replace` normalizer class.
namespace precompiled {

// Appends the transformations turning `old_part` into `new_part` to
// `transformations` (see `NormalizedString::Transform` for the meaning of the
// per-char offsets).
void Replace(std::vector<std::pair<char32_t, int64_t>>& transformations,
             absl::string_view old_part, absl::string_view new_part);

}  // namespace precompiled
}  // namespace tokenizers::normalizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_NORMALIZERS_PRECOMPILED_H_
