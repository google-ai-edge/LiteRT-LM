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
// C++ port of `src/models/unigram/trie.rs` from the HuggingFace `tokenizers`
// crate v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_UNIGRAM_TRIE_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_UNIGRAM_TRIE_H_

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace tokenizers::models::unigram {

template <typename Label>
class Trie;

template <typename Label>
class TrieIterator;

template <typename Label>
class TrieBuilder {
 public:
  void Push(absl::Span<const Label> element) { trie_.Push(element); }

  Trie<Label> Build() && { return std::move(trie_); }
  Trie<Label> Build() const& { return trie_; }

 private:
  Trie<Label> trie_;
};

template <typename Label>
class Trie {
 public:
  // `Default`: a trie with an empty root.
  Trie() : nodes_(1) {}

  void Push(absl::Span<const Label> element) {
    size_t node = 0;
    for (const Label& label : element) {
      auto it = nodes_[node].children.find(label);
      if (it == nodes_[node].children.end()) {
        const size_t child = nodes_.size();
        nodes_.emplace_back();
        it = nodes_[node].children.emplace(label, child).first;
      }
      node = it->second;
    }
    nodes_[node].is_leaf = true;
  }

  TrieIterator<Label> CommonPrefixSearch(
      absl::Span<const Label> iterator) const {
    return TrieIterator<Label>(this, iterator);
  }

 private:
  friend class TrieIterator<Label>;

  // PORT NOTE: the Rust `Node<Label>` owns its children in an
  // `AHashMap<Label, Node<Label>>`; `absl::flat_hash_map` cannot hold the
  // recursive (incomplete) value type, so children are indices into the flat
  // `nodes_` arena (`nodes_[0]` is the root). Behaviour is identical.
  struct Node {
    bool is_leaf = false;
    absl::flat_hash_map<Label, size_t> children;
  };

  std::vector<Node> nodes_;
};

// PORT NOTE: the Rust iterator yields a clone of the matched prefix on every
// step; here `Next()` advances to the next leaf and `prefix()` exposes the
// matched labels without copying.
template <typename Label>
class TrieIterator {
 public:
  // Advances to the next leaf; returns false once the input is exhausted or
  // leaves the trie.
  bool Next() {
    while (true) {
      if (position_ >= iterator_.size()) {
        return false;
      }
      const Label label = iterator_[position_++];
      prefix_.push_back(label);
      auto child = trie_->nodes_[node_].children.find(label);
      if (child == trie_->nodes_[node_].children.end()) {
        return false;
      }
      node_ = child->second;
      if (trie_->nodes_[node_].is_leaf) {
        return true;
      }
    }
  }

  const std::vector<Label>& prefix() const { return prefix_; }

 private:
  friend class Trie<Label>;

  TrieIterator(const Trie<Label>* trie, absl::Span<const Label> iterator)
      : trie_(trie), node_(0), iterator_(iterator), position_(0) {}

  const Trie<Label>* trie_;
  size_t node_;
  std::vector<Label> prefix_;
  absl::Span<const Label> iterator_;
  size_t position_;
};

}  // namespace tokenizers::models::unigram

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_MODELS_UNIGRAM_TRIE_H_
