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
// C++ port of `src/models/unigram/lattice.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_UNIGRAM_LATTICE_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_UNIGRAM_LATTICE_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace tokenizers::models::unigram {

// A node from the lattice, that helps reconstruct the underlying `String`
//
// PORT NOTE: Rust shares nodes through `Rc<RefCell<Node>>`; the C++ `Lattice`
// owns every node and hands out stable `Node*` pointers instead (`NodeRef`).
// `PartialEq` on `Node` compares the vocabulary `id` only, and so does every
// `==` between node refs in the Rust code (see `NodeEq`).
struct Node {
  Node(size_t id, size_t node_id, size_t pos, size_t length, double score)
      : id(id),
        node_id(node_id),
        pos(pos),
        length(length),
        prev(nullptr),
        backtrace_score(0.0),
        score(score) {}

  // Vocabulary id
  size_t id;
  // Local lattice identifier
  size_t node_id;
  size_t pos;
  size_t length;
  Node* prev;
  double backtrace_score;
  double score;
};

using NodeRef = Node*;

inline bool NodeEq(const Node& a, const Node& b) { return a.id == b.id; }

// Returns log(exp(x) + exp(y)).
// if init_mode is true, returns log(exp(y)) == y.
// log(\sum_i exp(a[i])) can be computed as
// for (int i = 0; i < a.size(); ++i)
//   x = LogSumExp(x, a[i], i == 0);
double LogSumExp(double x, double y, bool init_mode);

// Structure to implement Viterbi algorithm to find the best encoding, or sample
// from all possible encodings of a given sentence.
//
// The lattice borrows `sentence`, which must outlive it.
class Lattice {
 public:
  static Lattice From(absl::string_view sentence, size_t bos_id, size_t eos_id);

  Lattice(const Lattice&) = delete;
  Lattice& operator=(const Lattice&) = delete;
  Lattice(Lattice&&) = default;
  Lattice& operator=(Lattice&&) = default;

  void Insert(size_t pos, size_t length, double score, size_t id);

  std::vector<NodeRef> Viterbi();

  std::string Piece(const Node& node) const;

  std::vector<std::string> Tokens();

  std::vector<std::vector<NodeRef>> Nbest(size_t n);

  std::vector<std::vector<std::string>> NbestTokens(size_t n);

  size_t Len() const { return len_; }

  bool IsEmpty() const { return len_ == 0; }

  NodeRef BosNode() const { return end_nodes_[0][0]; }
  NodeRef EosNode() const { return begin_nodes_[len_][0]; }

  absl::string_view Surface(size_t n) const;
  absl::string_view Sentence() const { return sentence_; }

  double PopulateMarginal(double freq, absl::Span<double> expected) const;

  // PORT NOTE: `rand::WeightedIndex::new(..).unwrap()` panics on invalid
  // weights (empty, negative, NaN or all-zero); this port reports the same
  // conditions as an error status.
  absl::StatusOr<std::vector<NodeRef>> Sample(double theta) const;

  absl::StatusOr<std::vector<std::string>> SampleToken(double theta) const;

  std::vector<NodeRef> SampleNbest(size_t n, double theta);

  // `impl Display for Lattice`
  std::string ToString() const;

  // `pub(super)` fields.
  absl::string_view sentence() const { return sentence_; }
  const std::vector<std::vector<NodeRef>>& begin_nodes() const {
    return begin_nodes_;
  }
  const std::vector<std::vector<NodeRef>>& end_nodes() const {
    return end_nodes_;
  }
  const std::vector<std::unique_ptr<Node>>& nodes() const { return nodes_; }

 private:
  Lattice(absl::string_view sentence, size_t bos_id, size_t eos_id);

  absl::string_view sentence_;
  size_t len_;
  std::vector<std::unique_ptr<Node>> nodes_;
  std::vector<std::vector<NodeRef>> begin_nodes_;
  std::vector<std::vector<NodeRef>> end_nodes_;
  size_t bos_id_;
  size_t eos_id_;
};

}  // namespace tokenizers::models::unigram

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_UNIGRAM_LATTICE_H_
