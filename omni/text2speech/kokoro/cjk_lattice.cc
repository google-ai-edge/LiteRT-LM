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

#include "omni/text2speech/kokoro/cjk_lattice.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::omni::tts::kokoro {
namespace {

// The context id MeCab gives the virtual nodes at the two ends of a sentence.
constexpr int kBoundaryContextId = 0;

// A candidate, plus the cheapest way the search found of reaching it.
struct PathEntry {
  LatticeNode node;
  // The cost of everything up to and including `node`, boundaries included.
  double total_cost = 0.0;
  // The entry `node` follows, or -1 when it follows the start of the sentence.
  int previous = -1;
};

}  // namespace

CostModel::~CostModel() = default;

absl::StatusOr<std::vector<LatticeNode>> ViterbiSegment(
    absl::string_view text, const CostModel& model) {
  const int length = static_cast<int>(text.size());
  std::vector<LatticeNode> path;
  if (length == 0) return path;

  std::vector<PathEntry> entries;
  // The entries ending at each offset. Those are exactly the nodes a candidate
  // starting there can follow, and keeping them per offset rather than per
  // position-pair is what makes this linear in the size of the lattice.
  std::vector<std::vector<int>> ending_at(length + 1);
  std::vector<LatticeNode> candidates;

  for (int start = 0; start < length; ++start) {
    // Offsets no node reaches are not offsets any node may start at. Skipping
    // them is not just an optimization: asking a model about the middle of a
    // UTF-8 sequence would make it decode garbage.
    if (start > 0 && ending_at[start].empty()) continue;

    candidates.clear();
    model.CandidatesAt(text, start, candidates);
    for (const LatticeNode& node : candidates) {
      if (node.start != start || node.end <= start || node.end > length) {
        return absl::InternalError(absl::StrFormat(
            "cost model offered a node spanning [%d, %d) when asked about "
            "offset %d of a %d byte input",
            node.start, node.end, start, length));
      }

      double best = model.ConnectionCost(kBoundaryContextId, node.left_id);
      int previous = -1;
      for (const int index : ending_at[start]) {
        const double cost =
            entries[index].total_cost +
            model.ConnectionCost(entries[index].node.right_id, node.left_id);
        if (previous < 0 || cost < best) {
          best = cost;
          previous = index;
        }
      }

      ending_at[node.end].push_back(static_cast<int>(entries.size()));
      entries.push_back({node, best + node.word_cost, previous});
    }
  }

  double best = std::numeric_limits<double>::infinity();
  int last = -1;
  for (const int index : ending_at[length]) {
    const double cost =
        entries[index].total_cost +
        model.ConnectionCost(entries[index].node.right_id, kBoundaryContextId);
    if (cost < best) {
      best = cost;
      last = index;
    }
  }
  if (last < 0) {
    return absl::InvalidArgumentError(
        "no path spans the input: the cost model left a reachable offset with "
        "no candidates");
  }

  for (int index = last; index >= 0; index = entries[index].previous) {
    path.push_back(entries[index].node);
  }
  std::reverse(path.begin(), path.end());
  return path;
}

}  // namespace litert::omni::tts::kokoro
