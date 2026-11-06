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
// C++ port of `src/tokenizer/mod.rs` (lines 1-220: the `Normalizer`,
// `PreTokenizer`, `Model`, `PostProcessor` and `Decoder` traits, `Result`,
// `ProcessorError` and `impl dyn PostProcessor`) from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.
//
// Represents a tokenization pipeline.
//
// A `Tokenizer` (tokenizer.h) is composed of some of the following parts.
//   - `Normalizer`: Takes care of the text normalization (like unicode
//     normalization).
//   - `PreTokenizer`: Takes care of the pre tokenization (ie. How to split
//     tokens and pre-process them.
//   - `Model`: A model encapsulates the tokenization algorithm (like BPE, Word
//     base, character based, ...).
//   - `PostProcessor`: Takes care of the processing after tokenization (like
//     truncating, padding, ...).
//
// PORT NOTE: `pub type Error = Box<dyn std::error::Error + Send + Sync>` and
// `pub type Result<T>` are `absl::Status` / `absl::StatusOr<T>`. `Offsets` and
// `Token` are defined in token.h (included here). The `Trainer` trait,
// `Model::save` and `Model::get_trainer` are out of scope (training is not
// ported).

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_API_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_API_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/tokenizer/encoding.h"
#include "support/tokenizer/tokenizers/tokenizer/normalizer.h"
#include "support/tokenizer/tokenizers/tokenizer/pre_tokenizer.h"
#include "support/tokenizer/tokenizers/tokenizer/token.h"  // IWYU pragma: export

namespace tokenizers {

// Takes care of pre-processing strings.
//
// `pub trait Normalizer: Sync`
class Normalizer {
 public:
  virtual ~Normalizer() = default;

  virtual absl::Status Normalize(NormalizedString& normalized) const = 0;
};

// The `PreTokenizer` is in charge of doing the pre-segmentation step. It splits
// the given string in multiple substrings, keeping track of the offsets of said
// substrings from the `NormalizedString`. In some occasions, the `PreTokenizer`
// might need to modify the given `NormalizedString` to ensure we can entirely
// keep track of the offsets and the mapping with the original string.
//
// `pub trait PreTokenizer`
class PreTokenizer {
 public:
  virtual ~PreTokenizer() = default;

  virtual absl::Status PreTokenize(PreTokenizedString& pretokenized) const = 0;
};

// Represents a model used during Tokenization (like BPE or Word or Unigram).
//
// `pub trait Model`
class Model {
 public:
  virtual ~Model() = default;

  // Tokenize the given sequence into multiple underlying `Token`. The `offsets`
  // on the `Token` are expected to be relative to the given sequence.
  virtual absl::StatusOr<std::vector<Token>> Tokenize(
      absl::string_view sequence) const = 0;
  // Find the ID associated to a string token
  virtual std::optional<uint32_t> TokenToId(absl::string_view token) const = 0;
  // Find the string token associated to an ID
  virtual std::optional<std::string> IdToToken(uint32_t id) const = 0;
  // Retrieve the entire vocabulary mapping (token -> ID)
  virtual absl::flat_hash_map<std::string, uint32_t> GetVocab() const = 0;
  // Retrieve the size of the vocabulary
  virtual size_t GetVocabSize() const = 0;
  // PORT NOTE: `fn save(&self, folder, prefix)` and `fn get_trainer(&self)`
  // are not ported (training / saving is out of scope).

  // Tokenize every pre-token within a `PreTokenizedString` in one call.
  //
  // When `truncation` is `Some((max_tokens, direction))`, the
  // `tokenize_with_limit` early-exit path is taken; otherwise every
  // pre-token is tokenized.
  //
  // The default calls `self.tokenize()` per pre-token, which is correct
  // for self-contained `Model` implementations.  Implementations that
  // wrap their inner model behind a lock (e.g. the `PyModel` and Node
  // `Model` bindings, both `Arc<RwLock<_>>`) can override this to
  // acquire the lock once for the whole sequence of pre-tokens; that
  // removes ~one atomic load/store pair per pre-token from the hot path
  // (~1 500 pre-tokens per ~10 KB document).  Both the truncated and
  // non-truncated paths benefit from the override because they share
  // this entry point.
  virtual absl::Status TokenizeInPretokenized(
      PreTokenizedString& pretokenized,
      std::optional<std::pair<size_t, TruncationDirection>> truncation) const;
};

// A `PostProcessor` has the responsibility to post process an encoded output of
// the `Tokenizer`. It adds any special tokens that a language model would
// require.
//
// `pub trait PostProcessor`
class PostProcessor {
 public:
  virtual ~PostProcessor() = default;

  // Returns the number of tokens that will be added during the processing step
  virtual size_t AddedTokens(bool is_pair) const = 0;
  // Process both encodings and returns a new merged one
  //
  // PORT NOTE: non-virtual; this is the Rust default method body which no
  // implementation overrides.
  absl::StatusOr<Encoding> Process(Encoding encoding,
                                   std::optional<Encoding> pair_encoding,
                                   bool add_special_tokens) const;
  // Process any amount of encodings and returns a series of encoding (might
  // merge them)
  virtual absl::StatusOr<std::vector<Encoding>> ProcessEncodings(
      std::vector<Encoding> encodings, bool add_special_tokens) const = 0;

  // `impl dyn PostProcessor { pub fn default_process(...) }`
  static absl::StatusOr<std::vector<Encoding>> DefaultProcess(
      std::vector<Encoding> encodings, bool add_special_tokens);
};

// `#[derive(thiserror::Error, Debug)] pub enum ProcessorError`
struct ProcessorError {
  // `#[error("encodings vector length must be either 1 or 2")]
  // InvalidEncodingsVecLength`
  static absl::Status InvalidEncodingsVecLength() {
    return absl::InvalidArgumentError(
        "encodings vector length must be either 1 or 2");
  }
};

// A `Decoder` changes the raw tokens into its more readable form.
//
// `pub trait Decoder`
class Decoder {
 public:
  virtual ~Decoder() = default;

  // Default implementation: joins the result of `DecodeChain` with "".
  virtual absl::StatusOr<std::string> Decode(
      std::vector<std::string> tokens) const;
  virtual absl::StatusOr<std::vector<std::string>> DecodeChain(
      std::vector<std::string> tokens) const = 0;
};

// PORT NOTE: `pub trait Trainer` is not ported (training is out of scope).

}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_TOKENIZER_API_H_
