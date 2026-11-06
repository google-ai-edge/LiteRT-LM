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
// C++ port of `src/utils/cache.rs` from the HuggingFace `tokenizers` crate
// v0.23.2 (https://github.com/huggingface/tokenizers, Apache License 2.0,
// Copyright 2020 The HuggingFace Team). Kept line-by-line close to the Rust
// original on purpose; see
// support/tokenizer/tokenizers/README.md.

#ifndef THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UTILS_CACHE_H_
#define THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UTILS_CACHE_H_

#include <algorithm>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl

namespace tokenizers {

// The default capacity for a `BPE`'s internal cache.
inline constexpr size_t kDefaultCacheCapacity = 10'000;
// The maximum length we should cache in a model
// Strings that are too long have minimal chances to cache hit anyway
inline constexpr size_t kMaxLength = 256;

// Provides a simple multithread cache to speed up BPE tokenization that will
// try to read values concurrently but won't block if another thread is
// writing. The goal is clearly not the accuracy of the content, both get and
// set are not guaranteed to actually get or set.
//
// `RwLock::try_read` / `try_write` are `absl::Mutex::ReaderTryLock` /
// `TryLock`; the blocking `write()` is `Lock`. Heterogeneous lookup (Rust's
// `K: Borrow<Q>`) is what `absl::flat_hash_map` provides for the usual key
// types (`std::string` keys can be looked up with an `absl::string_view`).
template <typename K, typename V>
class Cache {
 public:
  using Map = absl::flat_hash_map<K, V>;

  // `impl Default for Cache`: `Self::new(DEFAULT_CACHE_CAPACITY)`.
  Cache() : Cache(kDefaultCacheCapacity) {}

  // Create new `Cache` with the given capacity.
  explicit Cache(size_t capacity) : capacity_(capacity) {
    map_.reserve(capacity);
  }

  // PORT NOTE: `Cache` is not `Clone` in Rust; the types holding one (`BPE`,
  // `Unigram`) implement `Clone` with `cache.fresh()`. The C++ copy
  // constructor does the same, so that those types can keep their defaulted
  // copy constructors: a copy is a fresh, empty cache with the same capacity.
  Cache(const Cache& other) : Cache(other.capacity_) {}
  Cache& operator=(const Cache& other) {
    if (this != &other) {
      absl::MutexLock lock(&mutex_);
      capacity_ = other.capacity_;
      map_.clear();
      map_.reserve(capacity_);
    }
    return *this;
  }
  // Moving requires exclusive access to `other`, like a Rust move.
  Cache(Cache&& other) noexcept {
    absl::MutexLock lock(&other.mutex_);
    capacity_ = other.capacity_;
    map_ = std::move(other.map_);
  }
  Cache& operator=(Cache&& other) noexcept {
    if (this != &other) {
      absl::MutexLock lock(&mutex_);
      absl::MutexLock other_lock(&other.mutex_);
      capacity_ = other.capacity_;
      map_ = std::move(other.map_);
    }
    return *this;
  }

  // We dont really care about Cache comparison, so let's make them always equal
  bool operator==(const Cache& /*other*/) const { return true; }
  bool operator!=(const Cache& /*other*/) const { return false; }

  // `pub capacity: usize`
  size_t capacity() const { return capacity_; }

  // Create a fresh `Cache` with the same configuration.
  Cache Fresh() const { return Cache(capacity_); }

  // Clear the cache.
  void Clear() {
    absl::MutexLock lock(&mutex_);
    map_.clear();
  }

  // `get_values(keys_iter)`: the cached values of `keys` (`nullopt` for a
  // miss), or `nullopt` as a whole if the read lock could not be acquired.
  template <typename Keys>
  std::optional<std::vector<std::optional<V>>> GetValues(
      const Keys& keys) const {
    if (mutex_.ReaderTryLock()) {
      std::vector<std::optional<V>> values;
      for (const auto& key : keys) {
        auto it = map_.find(key);
        values.push_back(it == map_.end() ? std::nullopt
                                          : std::optional<V>(it->second));
      }
      mutex_.ReaderUnlock();
      return values;
    } else {
      return std::nullopt;
    }
  }

  template <typename Q>
  std::optional<V> Get(const Q& key) const {
    if (mutex_.ReaderTryLock()) {
      auto it = map_.find(key);
      std::optional<V> value =
          it == map_.end() ? std::nullopt : std::optional<V>(it->second);
      mutex_.ReaderUnlock();
      return value;
    } else {
      return std::nullopt;
    }
  }

  // `set_values(entries)` for any iterable of `(K, V)` pairs (moved from when
  // `entries` is an rvalue).
  template <typename Entries>
  void SetValues(Entries&& entries) {
    // Before trying to acquire a write lock, we check if we are already at
    // capacity with a read handler.
    if (mutex_.ReaderTryLock()) {
      const bool at_capacity = map_.size() >= capacity_;
      mutex_.ReaderUnlock();
      if (at_capacity) {
        // At capacity, so do nothing.
        return;
      }
    } else {
      // If we couldn't acquire a read handle then we probably won't be able to
      // acquire a write handle one quadrillionth of a second later.
      return;
    }

    // Not at capacity, so try acquiring a write handle.
    if (mutex_.TryLock()) {
      // `cache.extend(entries.into_iter().take(free))`
      size_t free = capacity_ > map_.size() ? capacity_ - map_.size() : 0;
      for (auto&& entry : entries) {
        if (free == 0) break;
        --free;
        if constexpr (std::is_lvalue_reference_v<Entries>) {
          map_.insert_or_assign(entry.first, entry.second);
        } else {
          map_.insert_or_assign(std::move(entry.first),
                                std::move(entry.second));
        }
      }
      mutex_.Unlock();
    }
  }

  void Set(K key, V value) {
    std::vector<std::pair<K, V>> entries;
    entries.emplace_back(std::move(key), std::move(value));
    SetValues(std::move(entries));
  }

  // `resize(&mut self, capacity)`: like the Rust `&mut self`, requires that
  // no other thread uses the cache concurrently.
  void Resize(size_t capacity) {
    capacity_ = capacity;
    if (mutex_.TryLock()) {
      // `cache.shrink_to(capacity)`.
      // PORT NOTE: `absl::flat_hash_map` cannot shrink to a given size in
      // place; rebuild the table when it is larger than needed.
      const size_t wanted = std::max(capacity, map_.size());
      if (map_.capacity() > wanted) {
        Map shrunk;
        shrunk.reserve(wanted);
        for (auto& entry : map_) {
          shrunk.insert(std::move(entry));
        }
        map_ = std::move(shrunk);
      }
      mutex_.Unlock();
    }
  }

 private:
  mutable absl::Mutex mutex_;
  Map map_;
  size_t capacity_ = 0;
};

}  // namespace tokenizers

#endif  // THIRD_PARTY_ODML_LITERT_LM_SUPPORT_TOKENIZER_TOKENIZERS_UTILS_CACHE_H_
