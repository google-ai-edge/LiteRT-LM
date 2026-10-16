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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_CJK_LATTICE_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_CJK_LATTICE_H_

#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::omni::tts::kokoro {

// The word-lattice search behind Japanese segmentation.
//
// Japanese does not write spaces, so reading it starts by choosing, among the
// exponentially many ways a string can be cut into dictionary entries, the
// cutting a cost model likes best. The search needs two pieces from the model:
// the set of candidate words at each offset, and the price of putting one
// candidate after another. `MecabCostModel` charges a part-of-speech bigram
// read out of MeCab's connection matrix; keeping the search separate from the
// model lets it be tested against small hand-written models.

// A candidate word.
struct LatticeNode {
  // The span of input this node consumes, as byte offsets.
  int start = 0;
  int end = 0;

  // The word itself, pointing into the text that was segmented.
  //
  // This may be shorter than the consumed span. MeCab does not make a word out
  // of whitespace; it hands any run of it to the word that follows, so a node
  // can begin part way into what it consumes.
  absl::string_view surface;

  // The model's context ids, which only `CostModel::ConnectionCost` is
  // entitled to interpret. `left_id` describes this node as the right half of
  // a boundary, and `right_id` describes it as the left half.
  int left_id = 0;
  int right_id = 0;

  // What this node costs on its own, before any boundary cost.
  double word_cost = 0.0;

  // A handle the model can use to recover whatever else it knows about this
  // node -- for MeCab, the row of the token table it came from. Negative when
  // the model has nothing to point at.
  int token_index = -1;

  // Whether the model invented this node instead of reading it out of its
  // dictionary.
  bool is_unknown = false;
};

// The language-specific half of the search.
class CostModel {
 public:
  virtual ~CostModel();

  // Appends every node consuming input from byte `start`.
  //
  // Each node must have `start` as its `start` and must consume at least one
  // byte. A model has to offer something at every offset it is asked about, or
  // the sentence will have no path through it; MeCab does so by falling back
  // to its unknown-word table.
  //
  // args
  // - text: The whole UTF-8 input being segmented.
  // - start: Byte offset the candidates must start at.
  // - candidates: Output list the candidates are appended to.
  virtual void CandidatesAt(absl::string_view text, int start,
                            std::vector<LatticeNode>& candidates) const = 0;

  // Looks up the cost of one node boundary.
  //
  // The two ends of the sentence take part in this as context id 0, which is
  // how MeCab spells its virtual BOS and EOS nodes.
  //
  // args
  // - right_id: `right_id` of the left node.
  // - left_id: `left_id` of the right node.
  //
  // returns
  // - What it costs to follow the left node with the right one.
  virtual double ConnectionCost(int right_id, int left_id) const = 0;
};

// Finds the cheapest path through `text` under `model`.
//
// args
// - text: UTF-8 input to segment.
// - model: Cost model supplying candidates and boundary costs.
//
// returns
// - The cheapest path, in input order (empty for empty `text`), or an error
//   when `model` leaves a reachable offset with no candidates, so that no path
//   spans the input, or offers a node that does not consume the offset it was
//   asked about.
absl::StatusOr<std::vector<LatticeNode>> ViterbiSegment(absl::string_view text,
                                                        const CostModel& model);

}  // namespace litert::omni::tts::kokoro

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_TEXT2SPEECH_KOKORO_CJK_LATTICE_H_
