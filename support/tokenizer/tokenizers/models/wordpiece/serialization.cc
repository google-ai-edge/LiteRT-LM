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
// C++ port of `src/models/wordpiece/serialization.rs` from the HuggingFace
// `tokenizers` crate v0.23.2 (https://github.com/huggingface/tokenizers,
// Apache License 2.0, Copyright 2020 The HuggingFace Team). Kept line-by-line
// close to the Rust original on purpose; see
// support/tokenizer/tokenizers/README.md.
//
// PORT NOTE: `impl Serialize for WordPiece` / `impl Deserialize for WordPiece`
// are the `WordPiece::ToJson` / `WordPiece::FromJson` members declared in
// wordpiece.h.

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/models.h"
#include "support/tokenizer/tokenizers/models/wordpiece/wordpiece.h"
#include "support/tokenizer/tokenizers/utils/json.h"
#include "support/util/status_macros.h"

namespace tokenizers::models::wordpiece {

Json WordPiece::ToJson() const {
  Json model = Json::object();

  // Small fields first
  model["type"] = "WordPiece";
  model["unk_token"] = unk_token_;
  model["continuing_subword_prefix"] = continuing_subword_prefix_;
  model["max_input_chars_per_word"] = max_input_chars_per_word_;

  // Then large ones
  const OrderedVocabIter ordered_vocab(vocab_r_);
  model["vocab"] = ordered_vocab.Serialize();

  return model;
}

absl::StatusOr<WordPiece> WordPiece::FromJson(const Json& json) {
  if (!json.is_object()) {
    return SerdeInvalidType(json, "struct WordPiece");
  }
  WordPieceBuilder builder;
  // PORT NOTE: Rust keeps the missing fields in an `AHashSet` and reports an
  // arbitrary one; this port reports the first one in declaration order.
  std::vector<absl::string_view> missing_fields = {
      // for retrocompatibility the "type" field is not mandatory
      "unk_token",
      "continuing_subword_prefix",
      "max_input_chars_per_word",
      "vocab",
  };

  for (const auto& [key, value] : json.items()) {
    if (key == "unk_token") {
      ASSIGN_OR_RETURN(std::string unk_token, StringFromJson(value));
      builder.UnkToken(std::move(unk_token));
    } else if (key == "continuing_subword_prefix") {
      ASSIGN_OR_RETURN(std::string prefix, StringFromJson(value));
      builder.ContinuingSubwordPrefix(std::move(prefix));
    } else if (key == "max_input_chars_per_word") {
      ASSIGN_OR_RETURN(const size_t max_input_chars_per_word,
                       UsizeFromJson(value));
      builder.MaxInputCharsPerWord(max_input_chars_per_word);
    } else if (key == "vocab") {
      ASSIGN_OR_RETURN(auto vocab, VocabFromJson(value));
      builder.Vocab(std::move(vocab));
    } else if (key == "type") {
      if (!value.is_string()) {
        return SerdeInvalidType(value, "a borrowed string");
      }
      if (value.get_ref<const std::string&>() != "WordPiece") {
        return SerdeInvalidValue(value, "WordPiece");
      }
    }
    for (auto it = missing_fields.begin(); it != missing_fields.end(); ++it) {
      if (*it == key) {
        missing_fields.erase(it);
        break;
      }
    }
  }

  if (!missing_fields.empty()) {
    return SerdeMissingField(missing_fields.front());
  } else {
    return builder.Build();
  }
}

}  // namespace tokenizers::models::wordpiece
