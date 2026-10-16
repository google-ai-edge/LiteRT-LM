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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_MECAB_LATTICE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_MECAB_LATTICE_H_

#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "omni/text2speech/kokoro/cjk_blob.h"
#include "omni/text2speech/kokoro/cjk_lattice.h"

namespace litert::omni::tts::kokoro {

// MeCab's cost model over the unidic-lite dictionary packed into `ja-lexicon`.
//
// This is a port of `Tokenizer::lookup` and `Connector::cost` from mecab
// 0.996. Everything it reads -- the double array, the token table's context
// ids and word costs, the connection matrix, and the character and unknown
// tables -- was copied byte for byte out of unidic-lite by the offline packer,
// so agreeing with MeCab is a matter of reproducing the traversal, not of
// retraining anything.
//
// Two parts of that traversal are worth knowing about before reading the
// implementation.
//
// Whitespace is not a word. MeCab skips a run of it and gives the bytes to the
// word that follows, which is why a node's surface can start after the offset
// it consumes from.
//
// The dictionary having an answer does not end the search. Each character
// category says whether to guess anyway (`invoke`), how many characters of the
// same category to guess across (`group`), and how many lengths to guess at
// (`length`). That is what keeps a long run of digits or katakana from being
// cut up into whichever short entries happen to be in the dictionary.
class MecabCostModel : public CostModel {
 public:
  // Builds the model from the sections of a `ja-lexicon` blob.
  //
  // args
  // - blob: Parsed `ja-lexicon` blob, which must outlive the result.
  //
  // returns
  // - The model, or an error if a section is missing or malformed, or the
  //   unknown-word dictionary lacks an entry for some character category.
  static absl::StatusOr<MecabCostModel> Create(const CjkBlob& blob);

  void CandidatesAt(absl::string_view text, int start,
                    std::vector<LatticeNode>& candidates) const override;
  double ConnectionCost(int right_id, int left_id) const override;

  // Returns the dictionary row a node came from, which is where its reading,
  // part of speech and pitch accent live.
  //
  // args
  // - node: A node from a `ViterbiSegment` run over this model.
  //
  // returns
  // - The node's token row, or an empty `MecabToken` if it has none.
  MecabToken Token(const LatticeNode& node) const;

  // Returns the character property table the model segments with.
  //
  // returns
  // - The `char.bin` table, valid for the lifetime of this model.
  const MecabCharProperty& Characters() const;

 private:
  // Where one character category's entries sit in the unknown-word table.
  struct UnknownEntries {
    int first = 0;
    int count = 0;
  };

  // The end of a run of characters and how many characters it spans.
  struct CategoryRun {
    int end = 0;
    int character_count = 0;
  };

  MecabCostModel(MecabDoubleArray dictionary_index, MecabTokenTable dictionary,
                 MecabTokenTable unknown, MecabConnectionMatrix connection,
                 MecabCharProperty characters,
                 std::vector<UnknownEntries> unknown_by_category);

  // Appends every dictionary entry beginning at `surface_start`.
  void AppendDictionaryMatches(absl::string_view text, int start,
                               int surface_start,
                               std::vector<LatticeNode>& candidates) const;

  // Appends `category`'s unknown-word entries for the span
  // [`surface_start`, `end`).
  void AppendUnknown(absl::string_view text, int start, int surface_start,
                     int end, int category,
                     std::vector<LatticeNode>& candidates) const;

  // Returns the run of characters starting at `offset` that keep sharing a
  // category with the one before them, `info` standing in for the character
  // before the first.
  CategoryRun SeekPastCategory(absl::string_view text, int offset,
                               MecabCharProperty::CharInfo info) const;

  MecabDoubleArray dictionary_index_;
  MecabTokenTable dictionary_;
  MecabTokenTable unknown_;
  MecabConnectionMatrix connection_;
  MecabCharProperty characters_;
  // Resolved once at load, so the tokenizer never searches the unknown-word
  // dictionary by category name.
  std::vector<UnknownEntries> unknown_by_category_;
  // What the character table says about a plain space, which is where the
  // whitespace skip starts.
  MecabCharProperty::CharInfo space_;
};

}  // namespace litert::omni::tts::kokoro

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_MECAB_LATTICE_H_
