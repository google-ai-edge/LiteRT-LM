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

#include "support/tokenizer/tokenizers/models/unigram/lattice.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/random/discrete_distribution.h"  // from @com_google_absl
#include "absl/random/random.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/utils/utf8.h"

namespace tokenizers::models::unigram {

namespace {

struct Hypothesis;
using HypothesisRef = std::shared_ptr<Hypothesis>;

struct Hypothesis {
  Hypothesis(NodeRef node_ref, HypothesisRef next, double fx, double gx)
      : node_ref(node_ref), next(std::move(next)), fx(fx), gx(gx) {}

  NodeRef node_ref;
  HypothesisRef next;
  double fx;
  double gx;
};

// `impl Ord for Hypothesis`: ordered on `fx` (max-heap).
// TODO Maybe use Ordered Floats
// (https://docs.rs/ordered-float/1.0.2/ordered_float/)
struct HypothesisLess {
  bool operator()(const HypothesisRef& a, const HypothesisRef& b) const {
    return a->fx < b->fx;
  }
};

using Agenda = std::priority_queue<HypothesisRef, std::vector<HypothesisRef>,
                                   HypothesisLess>;

// `rand::distr::weighted::WeightedIndex::new(&weights)`: fails on an empty
// list, on a negative or NaN weight, and when all weights are zero.
std::optional<absl::discrete_distribution<size_t>> WeightedIndexNew(
    const std::vector<double>& weights) {
  if (weights.empty()) {
    return std::nullopt;
  }
  double total_weight = 0.0;
  for (const double w : weights) {
    if (!(w >= 0.0)) {
      return std::nullopt;
    }
    total_weight += w;
  }
  if (total_weight == 0.0) {
    return std::nullopt;
  }
  return absl::discrete_distribution<size_t>(weights.begin(), weights.end());
}

}  // namespace

double LogSumExp(double x, double y, bool init_mode) {
  if (init_mode) {
    return y;
  } else {
    const double vmin = x > y ? y : x;
    const double vmax = x > y ? x : y;
    const double k_minus_log_epsilon = 50.0;
    if (vmax > vmin + k_minus_log_epsilon) {
      return vmax;
    } else {
      return vmax + std::log(std::exp(vmin - vmax) + 1.0);
    }
  }
}

Lattice Lattice::From(absl::string_view sentence, size_t bos_id,
                      size_t eos_id) {
  return Lattice(sentence, bos_id, eos_id);
}

Lattice::Lattice(absl::string_view sentence, size_t bos_id, size_t eos_id)
    : sentence_(sentence),
      len_(sentence.size()),
      begin_nodes_(sentence.size() + 1),
      end_nodes_(sentence.size() + 1),
      bos_id_(bos_id),
      eos_id_(eos_id) {
  const size_t k_reserved_node_size = 16;
  // We are adding 2 tokens, bos and eos
  nodes_.reserve(k_reserved_node_size);

  auto bos = std::make_unique<Node>(bos_id, 0, 0, 0, 0.0);
  auto eos = std::make_unique<Node>(eos_id, 1, len_, 0, 0.0);

  begin_nodes_[len_].push_back(eos.get());
  end_nodes_[0].push_back(bos.get());

  nodes_.push_back(std::move(bos));
  nodes_.push_back(std::move(eos));
}

void Lattice::Insert(size_t pos, size_t length, double score, size_t id) {
  const size_t node_id = nodes_.size();
  auto node = std::make_unique<Node>(id, node_id, pos, length, score);

  begin_nodes_[pos].push_back(node.get());
  end_nodes_[pos + length].push_back(node.get());

  nodes_.push_back(std::move(node));
}

std::vector<NodeRef> Lattice::Viterbi() {
  const size_t len = len_;
  size_t pos = 0;
  while (pos <= len) {
    if (begin_nodes_[pos].empty()) {
      return {};
    }
    for (NodeRef rnode : begin_nodes_[pos]) {
      rnode->prev = nullptr;
      double best_score = 0.0;
      NodeRef best_node = nullptr;
      for (NodeRef lnode : end_nodes_[pos]) {
        const double score = lnode->backtrace_score + rnode->score;
        if (best_node == nullptr || score > best_score) {
          best_node = lnode;
          best_score = score;
        }
      }
      if (best_node != nullptr) {
        rnode->prev = best_node;
        rnode->backtrace_score = best_score;
      } else {
        return {};
      }
    }
    if (pos < sentence_.size()) {
      DecodeUtf8(sentence_, &pos);
    } else {
      break;
    }
  }

  std::vector<NodeRef> results;
  const Node& root = *begin_nodes_[len][0];
  NodeRef prev = root.prev;
  if (prev == nullptr) {
    return {};
  }
  NodeRef node = prev;
  while (node->prev != nullptr) {
    results.push_back(node);
    node = node->prev;
  }
  std::reverse(results.begin(), results.end());
  return results;
}

std::string Lattice::Piece(const Node& node) const {
  return std::string(sentence_.substr(node.pos, node.length));
}

std::vector<std::string> Lattice::Tokens() {
  std::vector<std::string> tokens;
  for (NodeRef node : Viterbi()) {
    tokens.push_back(Piece(*node));
  }
  return tokens;
}

std::vector<std::vector<NodeRef>> Lattice::Nbest(size_t n) {
  switch (n) {
    case 0:
      return {};
    case 1:
      return {Viterbi()};
    default: {
      // let k_reserved_hypothesis_size = 512;
      Agenda agenda;
      std::vector<std::vector<NodeRef>> hypotheses;
      NodeRef eos = EosNode();
      const double score = eos->score;
      agenda.push(std::make_shared<Hypothesis>(eos, nullptr, score, score));

      // Fill backtrace scores
      Viterbi();

      while (!agenda.empty()) {
        const HypothesisRef top = agenda.top();
        agenda.pop();
        NodeRef node = top->node_ref;
        if (node->id == BosNode()->id) {
          std::vector<NodeRef> hypothesis;
          HypothesisRef next = top->next;
          while (next != nullptr && next->next != nullptr) {
            hypothesis.push_back(next->node_ref);
            next = next->next;
          }
          hypotheses.push_back(std::move(hypothesis));
          if (hypotheses.size() == n) {
            return hypotheses;
          }
        } else {
          for (NodeRef lnode : end_nodes_[node->pos]) {
            const double top_gx = top->gx;
            const double fx = lnode->backtrace_score + top_gx;
            const double gx = lnode->score + top_gx;
            agenda.push(std::make_shared<Hypothesis>(lnode, top, fx, gx));
          }
          // When the input is too long or contains duplicated phrases,
          // `agenda` will get extremely big. Here we avoid this case by
          // dynamically shrinking the agenda.
          const size_t k_max_agenda_size = 100000;
          const size_t k_min_agenda_size = 512;
          if (agenda.size() > k_max_agenda_size) {
            Agenda new_agenda;
            const size_t len = std::min(k_min_agenda_size, n * 10);
            for (size_t i = 0; i < len; ++i) {
              new_agenda.push(agenda.top());
              agenda.pop();
            }
            agenda = std::move(new_agenda);
          }
        }
      }
      return hypotheses;
    }
  }
}

std::vector<std::vector<std::string>> Lattice::NbestTokens(size_t n) {
  std::vector<std::vector<std::string>> result;
  for (const std::vector<NodeRef>& path : Nbest(n)) {
    std::vector<std::string> tokens;
    for (NodeRef node : path) {
      tokens.push_back(Piece(*node));
    }
    result.push_back(std::move(tokens));
  }
  return result;
}

absl::string_view Lattice::Surface(size_t n) const {
  // `self.sentence.char_indices().nth(n)`
  size_t index = 0;
  for (const auto [pos, c] : CharIndices(sentence_)) {
    if (index == n) {
      return sentence_.substr(pos);
    }
    ++index;
  }
  return "";
}

double Lattice::PopulateMarginal(double freq,
                                 absl::Span<double> expected) const {
  const size_t len = Len();
  const size_t n_nodes = nodes_.size();
  std::vector<double> alpha(n_nodes, 0.0);
  std::vector<double> beta(n_nodes, 0.0);
  for (size_t pos = 0; pos <= len; ++pos) {
    for (NodeRef rnode : begin_nodes_[pos]) {
      for (NodeRef lnode : end_nodes_[pos]) {
        const size_t lid = lnode->node_id;
        const size_t rid = rnode->node_id;
        alpha[rid] = LogSumExp(alpha[rid], lnode->score + alpha[lid],
                               NodeEq(*lnode, *end_nodes_[pos][0]));
      }
    }
  }
  for (size_t pos = len + 1; pos-- > 0;) {
    // let rpos = len - pos;
    for (NodeRef lnode : end_nodes_[pos]) {
      for (NodeRef rnode : begin_nodes_[pos]) {
        const size_t lid = lnode->node_id;
        const size_t rid = rnode->node_id;
        beta[lid] = LogSumExp(beta[lid], rnode->score + beta[rid],
                              NodeEq(*rnode, *begin_nodes_[pos][0]));
      }
    }
  }

  const size_t eos_id = begin_nodes_[len][0]->node_id;
  const double z = alpha[eos_id];
  for (size_t pos = 0; pos < len; ++pos) {
    for (NodeRef node : begin_nodes_[pos]) {
      const size_t node_id = node->node_id;
      const size_t id = node->id;
      const double a = alpha[node_id];
      const double b = beta[node_id];
      const double total = a + node->score + b - z;
      const double update = freq * std::exp(total);
      // PORT NOTE: Rust panics on an `id` outside `expected`; skipped here.
      if (id < expected.size()) {
        expected[id] += update;
      }
    }
  }
  return freq * z;
}

absl::StatusOr<std::vector<NodeRef>> Lattice::Sample(double theta) const {
  const size_t len = Len();
  if (len == 0) {
    return std::vector<NodeRef>();
  }
  std::vector<double> alpha(nodes_.size(), 0.0);
  for (size_t pos = 0; pos <= len; ++pos) {
    for (NodeRef rnode : begin_nodes_[pos]) {
      for (NodeRef lnode : end_nodes_[pos]) {
        const size_t lid = lnode->node_id;
        const size_t rid = rnode->node_id;
        alpha[rid] = LogSumExp(alpha[rid], theta * (lnode->score + alpha[lid]),
                               NodeEq(*lnode, *end_nodes_[pos][0]));
      }
    }
  }

  absl::BitGen rng;
  std::vector<NodeRef> results;
  std::vector<double> probs;
  double z = alpha[EosNode()->node_id];
  NodeRef node = EosNode();
  while (true) {
    probs.clear();
    const size_t pos = node->pos;
    for (NodeRef lnode : end_nodes_[pos]) {
      const size_t lid = lnode->node_id;
      probs.push_back(std::exp(alpha[lid] + theta * lnode->score - z));
    }
    std::optional<absl::discrete_distribution<size_t>> dist =
        WeightedIndexNew(probs);
    if (!dist.has_value()) {
      return absl::InvalidArgumentError(
          "Lattice::sample: invalid weights for WeightedIndex");
    }
    const size_t index = (*dist)(rng);
    node = end_nodes_[pos][index];
    if (NodeEq(*node, *BosNode())) {
      break;
    }
    z = alpha[node->node_id];
    results.push_back(node);
  }
  std::reverse(results.begin(), results.end());
  return results;
}

absl::StatusOr<std::vector<std::string>> Lattice::SampleToken(
    double theta) const {
  absl::StatusOr<std::vector<NodeRef>> sampled = Sample(theta);
  if (!sampled.ok()) {
    return sampled.status();
  }
  std::vector<std::string> tokens;
  for (NodeRef node : *sampled) {
    tokens.push_back(Piece(*node));
  }
  return tokens;
}

std::vector<NodeRef> Lattice::SampleNbest(size_t n, double theta) {
  const std::vector<std::vector<NodeRef>> nbest_paths = Nbest(n);
  if (nbest_paths.empty()) {
    return Viterbi();
  }

  std::vector<double> probs;
  probs.reserve(nbest_paths.size());
  for (const std::vector<NodeRef>& p : nbest_paths) {
    double path_score = 0.0;
    for (NodeRef node : p) {
      path_score += node->score;
    }
    probs.push_back(std::exp(theta * path_score));
  }

  if (std::optional<absl::discrete_distribution<size_t>> dist =
          WeightedIndexNew(probs);
      dist.has_value()) {
    absl::BitGen rng;
    const size_t index = (*dist)(rng);
    return nbest_paths[index];
  } else {
    return nbest_paths[0];
  }
}

std::string Lattice::ToString() const {
  const auto display_pieces =
      [this](const std::vector<std::vector<NodeRef>>& nodes) {
        std::vector<std::string> lists;
        for (const std::vector<NodeRef>& l : nodes) {
          std::vector<std::string> pieces;
          for (NodeRef n : l) {
            pieces.push_back(absl::StrCat("\"", Piece(*n), "\""));
          }
          lists.push_back(absl::StrCat("[", absl::StrJoin(pieces, ", "), "]"));
        }
        return absl::StrCat("[", absl::StrJoin(lists, ", "), "]");
      };

  return absl::StrCat("Lattice { sentence: \"", sentence_,
                      "\", begin_nodes: ", display_pieces(begin_nodes_),
                      ", end_nodes: ", display_pieces(end_nodes_), " }");
}

}  // namespace tokenizers::models::unigram
