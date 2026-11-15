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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_NO_REPEAT_NGRAM_CONFIG_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_NO_REPEAT_NGRAM_CONFIG_H_

#include <algorithm>

namespace litert::lm {

// A configuration storing the banning parameters for the
// `NoRepeatNgramConstraint`.
//
// Clamps the given parameters to their valid lower bounds during construction.
//
// Parameters are immutable once constructed to prevent unexpected divergence
// between the active banning settings and a running generative sequence's
// state.
class NoRepeatNgramConfig {
 public:
  // Constructs a config with the given bounds, clamping where necessary.
  //
  // @param no_repeat_ngram_size If set to an integer greater than 0, all ngrams
  // of that size can only occur once. The logits of the banned tokens will be
  // set to -inf. The value is clamped to [0, inf).
  // @param window_size The maximum number of recent tokens to consider for
  // banning. Tokens older than this are forgotten. A value of 0 means
  // track all infinite history. The value is clamped to [0, inf). If
  // set less than the ngram size, the window size will be set to the ngram size
  // to ensure that the ngram can be tracked.
  explicit NoRepeatNgramConfig(int no_repeat_ngram_size, int window_size)
      : no_repeat_ngram_size_(std::max(0, no_repeat_ngram_size)),
        window_size_(std::max(0, window_size)) {
    if (window_size_ != 0 && window_size_ < no_repeat_ngram_size_) {
      window_size_ = no_repeat_ngram_size_;
    }
  }

  // Returns a default config with banning disabled.
  static NoRepeatNgramConfig Default() { return NoRepeatNgramConfig(0, 0); }

  int no_repeat_ngram_size() const { return no_repeat_ngram_size_; }
  int window_size() const { return window_size_; }

  // Returns whether the banning config is enabled. If the ngram size is less
  // than or equal to 0, the config is disabled.
  bool enabled() const { return no_repeat_ngram_size_ > 0; }

 private:
  int no_repeat_ngram_size_;
  int window_size_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_NO_REPEAT_NGRAM_CONFIG_H_
