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
// C++ port of `src/models/bpe/word.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#include "support/tokenizer/tokenizers/models/bpe/word.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/random/random.h"  // from @com_google_absl
#include "support/tokenizer/tokenizers/models/bpe/bpe.h"

namespace tokenizers::models::bpe {

namespace {

// PORT NOTE: the private `Merge` struct lives in a nested namespace so that it
// does not clash with the `Word::Merge` member function in C++ name lookup.
namespace heap {

struct Merge {
  size_t pos;
  uint32_t rank;
  uint32_t new_id;
};

// By manually implementing this, we make the containing BinaryHeap a
// min-heap ordered first on the rank, and the pos otherwise
//
// PORT NOTE: `dary_heap::QuaternaryHeap` is a max-heap, exactly like
// `std::priority_queue`; this is the Rust `Ord::cmp` turned into `operator<`
// (the `PartialEq` impl only exists to satisfy Rust's trait bounds).
bool operator<(const Merge& a, const Merge& b) {
  if (a.rank != b.rank) {
    return b.rank < a.rank;
  }
  return b.pos < a.pos;
}

}  // namespace heap

}  // namespace

Word Word::WithCapacity(size_t capacity) {
  Word word;
  word.symbols_.reserve(capacity);
  return word;
}

void Word::Add(uint32_t c, size_t byte_len) {
  int64_t prev;
  int64_t next;
  {
    const int64_t len = static_cast<int64_t>(symbols_.size());
    if (!symbols_.empty()) {
      // Update `next` on the previous one
      symbols_.back().next = len;
      prev = len - 1;
      next = -1;
    } else {
      prev = -1;
      next = -1;
    }
  }
  symbols_.push_back(Symbol{
      /*c=*/c,
      /*prev=*/prev,
      /*next=*/next,
      /*len=*/byte_len,
  });
}

std::vector<std::pair<Pair, int32_t>> Word::Merge(uint32_t c1, uint32_t c2,
                                                  uint32_t replacement,
                                                  size_t max_length) {
  std::vector<std::pair<Pair, int32_t>> changes;
  size_t i = 0;
  while (true) {
    if (i >= symbols_.size()) {
      break;
    }

    // Found a pair
    if (symbols_[i].c == c1 && i + 1 < symbols_.size() &&
        symbols_[i + 1].c == c2) {
      const Symbol first = symbols_[i];
      const Symbol second = symbols_[i + 1];

      // Remove in place
      const Symbol new_s{
          /*c=*/replacement,
          /*prev=*/first.prev,
          /*next=*/second.next,
          /*len=*/first.len + second.len,
      };

      // If there are other characters before the pair
      if (i > 0) {
        changes.push_back({Pair(symbols_[i - 1].c, first.c), -1});
        if (symbols_[i - 1].len + new_s.len < max_length) {
          changes.push_back({Pair(symbols_[i - 1].c, replacement), 1});
        }
      }

      symbols_.insert(symbols_.begin() + i,
                      new_s);  // Insert replacement before first char of pair
      symbols_.erase(symbols_.begin() + i + 1);  // Remove first char of pair
      symbols_.erase(symbols_.begin() + i + 1);  // And then the second

      // If there are other characters after the pair
      if (i < symbols_.size() - 1) {
        changes.push_back({Pair(second.c, symbols_[i + 1].c), -1});
        if (symbols_[i + 1].len + new_s.len < max_length) {
          changes.push_back({Pair(replacement, symbols_[i + 1].c), 1});
        }
      }
    }

    i += 1;
  }

  return changes;
}

void Word::MergeAll(
    const absl::flat_hash_map<Pair, std::pair<uint32_t, uint32_t>>& merges,
    std::optional<float> dropout) {
  std::priority_queue<heap::Merge> queue;
  std::vector<heap::Merge> skip;
  skip.reserve(symbols_.size());

  // PORT NOTE: `rng()` is Rust's thread-local generator; a generator is only
  // created when dropout is actually enabled (a training-time feature).
  std::optional<absl::BitGen> rng;
  if (dropout.has_value()) {
    rng.emplace();
  }

  for (size_t index = 0; index + 1 < symbols_.size(); ++index) {
    const Pair pair(symbols_[index].c, symbols_[index + 1].c);
    auto m = merges.find(pair);
    if (m != merges.end()) {
      queue.push(heap::Merge{
          /*pos=*/index,
          /*rank=*/m->second.first,
          /*new_id=*/m->second.second,
      });
    }
  }

  while (!queue.empty()) {
    const heap::Merge top = queue.top();
    queue.pop();
    if (dropout.has_value() &&
        absl::Uniform<float>(*rng, 0.0f, 1.0f) < *dropout) {
      skip.push_back(top);
    } else {
      // Re-insert the skipped elements
      for (const heap::Merge& skipped : skip) {
        queue.push(skipped);
      }
      skip.clear();

      if (symbols_[top.pos].len == 0) {
        continue;
      }
      // Do nothing if we are the last symbol
      if (symbols_[top.pos].next == -1) {
        continue;
      }

      const size_t next_pos = static_cast<size_t>(symbols_[top.pos].next);
      const Symbol right = symbols_[next_pos];

      // Make sure we are not processing an expired queue entry
      const Pair target_new_pair(symbols_[top.pos].c, right.c);
      auto target = merges.find(target_new_pair);
      if (target == merges.end() || target->second.second != top.new_id) {
        continue;
      }

      // Otherwise, let's merge
      symbols_[top.pos].MergeWith(right, top.new_id);
      // Tag the right part as removed
      symbols_[next_pos].len = 0;

      // Update `prev` on the new `next` to the current pos
      if (right.next > -1 &&
          static_cast<size_t>(right.next) < symbols_.size()) {
        symbols_[static_cast<size_t>(right.next)].prev =
            static_cast<int64_t>(top.pos);
      }

      // Insert the new pair formed with the previous symbol
      const Symbol current = symbols_[top.pos];
      if (current.prev >= 0) {
        const size_t prev = static_cast<size_t>(current.prev);
        const Symbol prev_symbol = symbols_[prev];
        const Pair new_pair(prev_symbol.c, current.c);
        auto found = merges.find(new_pair);
        if (found != merges.end()) {
          queue.push(heap::Merge{
              /*pos=*/prev,
              /*rank=*/found->second.first,
              /*new_id=*/found->second.second,
          });
        }
      }

      // Insert the new pair formed with the next symbol
      const size_t next = static_cast<size_t>(current.next);
      if (next < symbols_.size()) {
        const Symbol next_symbol = symbols_[next];
        const Pair new_pair(current.c, next_symbol.c);
        auto found = merges.find(new_pair);
        if (found != merges.end()) {
          queue.push(heap::Merge{
              /*pos=*/top.pos,
              /*rank=*/found->second.first,
              /*new_id=*/found->second.second,
          });
        }
      }
    }
  }

  // Filter out the removed symbols
  std::vector<Symbol> kept;
  kept.reserve(symbols_.size());
  for (const Symbol& s : symbols_) {
    if (s.len != 0) {
      kept.push_back(s);
    }
  }
  symbols_ = std::move(kept);
}

std::vector<uint32_t> Word::GetChars() const {
  std::vector<uint32_t> chars;
  chars.reserve(symbols_.size());
  for (const Symbol& s : symbols_) {
    chars.push_back(s.c);
  }
  return chars;
}

std::vector<std::pair<size_t, size_t>> Word::GetOffsets() const {
  std::vector<std::pair<size_t, size_t>> offsets;
  offsets.reserve(symbols_.size());
  size_t pos = 0;
  for (const Symbol& symbol : symbols_) {
    const size_t new_pos = pos + symbol.len;
    offsets.push_back({pos, new_pos});
    pos = new_pos;
  }
  return offsets;
}

}  // namespace tokenizers::models::bpe
